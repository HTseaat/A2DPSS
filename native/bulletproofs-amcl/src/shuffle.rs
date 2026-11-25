//! Neff verifiable secret shuffle (C. Andrew Neff, CCS'01).
//!
//! This module implements the shuffle stack used by ATCaaS mix-net role
//! assignment (paper Algorithm 6, `Shuffle.Mix` / `Shuffle.Verify`):
//!
//!   Layer 1  ILMPP            (2001 §3)  iterated logarithmic multiplication
//!   Layer 2  Simple k-shuffle (2001 §4)  Y_i^d = X_{π(i)}^c, P knows logs
//!   Layer 3  General k-shuffle (RGS)     single-key re-base X̄_i=X_{π(i)}^c,
//!                                         P need NOT know logs; built on the
//!                                         2004 ElGamal-pairs EGA structure
//!                                         (plan §3.7), final CP carries c
//!   Layer 4  DSA-key shuffle  (2001 §6)  Shuffle.Mix/Verify wrapper +
//!                                         output in-subgroup check
//!
//! Notation. The group is BLS12-381 `G1` of prime order `q` (amcl_wrapper).
//! Group operation is written additively in code (`+`, `-`, `point.mul(&s)`),
//! but multiplicatively in the comments to match the paper:
//!   `X^a`  == `X.mul(&a)`            (scalar multiplication / exponentiation)
//!   `X·Y`  == `X + &Y`              (group operation)
//!   `X/Y`  == `X - &Y`              (group operation with inverse)
//!   `g`    is a caller-supplied fixed generator (the SRS generator).
//!
//! All protocols are made non-interactive with the same Merlin transcript used
//! elsewhere in this crate (`TranscriptProtocol`); challenges are derived by
//! Fiat-Shamir, binding every public input and first-move commitment.

#![allow(non_snake_case)]

use amcl_wrapper::field_elem::FieldElement;
use amcl_wrapper::group_elem::GroupElement;
use amcl_wrapper::group_elem_g1::G1;
use merlin::Transcript;
use serde::{Deserialize, Serialize};
use std::ops::Mul;

use crate::transcript::TranscriptProtocol;

// ---------------------------------------------------------------------------
// small scalar helpers
// ---------------------------------------------------------------------------

/// `base^exp` for a field element and a small non-negative integer exponent.
fn fe_pow(base: &FieldElement, exp: usize) -> FieldElement {
    let mut acc = FieldElement::one();
    for _ in 0..exp {
        acc = acc * base;
    }
    acc
}

// ===========================================================================
// Layer 1: ILMPP  (paper §3, eqs (9)-(10))
// ===========================================================================
//
// Public:  X = [X_1..X_k], Y = [Y_1..Y_k] in G1, with all X_i, Y_i != 1.
// Witness: x_i = log_g X_i, y_i = log_g Y_i, satisfying  ∏ x_i = ∏ y_i.
//
// First move (eq 9):
//   A_1 = Y_1^{θ_1}
//   A_i = X_i^{θ_{i-1}} · Y_i^{θ_i}   (2 ≤ i ≤ k-1)
//   A_k = X_k^{θ_{k-1}}
// Challenge γ.  Responses r_1..r_{k-1} solve (eq 10), which the verifier checks:
//   Y_1^{r_1}                = A_1 · X_1^{-γ}
//   X_i^{r_{i-1}} · Y_i^{r_i} = A_i                         (2 ≤ i ≤ k-1)
//   X_k^{r_{k-1}}            = A_k · Y_k^{(-1)^{k-1} γ}
//
// Prover side: writing e_i := r_i - θ_i, the log form of (10) gives
//   e_1 = -γ · x_1 / y_1 ,   e_i = -(x_i / y_i) · e_{i-1}   (i ≥ 2),
// and the last (k-th) equation holds iff ∏ x_i = ∏ y_i.

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IlmppProof {
    pub a: Vec<G1>, // length k
    pub r: Vec<FieldElement>, // length k-1
}

fn ilmpp_absorb(tr: &mut Transcript, xs: &[G1], ys: &[G1], a: &[G1]) {
    for x in xs {
        tr.commit_point(b"ilmpp_x", x);
    }
    for y in ys {
        tr.commit_point(b"ilmpp_y", y);
    }
    for ai in a {
        tr.commit_point(b"ilmpp_a", ai);
    }
}

pub fn ilmpp_prove(
    tr: &mut Transcript,
    xs: &[G1],
    ys: &[G1],
    x_logs: &[FieldElement],
    y_logs: &[FieldElement],
) -> IlmppProof {
    let k = xs.len();
    assert!(k >= 2, "ILMPP requires k >= 2");
    assert_eq!(ys.len(), k);
    assert_eq!(x_logs.len(), k);
    assert_eq!(y_logs.len(), k);

    // θ_1..θ_{k-1}
    let thetas: Vec<FieldElement> = (0..k - 1).map(|_| FieldElement::random()).collect();

    // first move A_1..A_k
    let mut a: Vec<G1> = Vec::with_capacity(k);
    a.push(ys[0].clone().mul(&thetas[0])); // A_1 = Y_1^{θ_1}
    for i in 1..k - 1 {
        // A_i = X_i^{θ_{i-1}} · Y_i^{θ_i}
        let term = xs[i].clone().mul(&thetas[i - 1]) + &ys[i].clone().mul(&thetas[i]);
        a.push(term);
    }
    a.push(xs[k - 1].clone().mul(&thetas[k - 2])); // A_k = X_k^{θ_{k-1}}

    ilmpp_absorb(tr, xs, ys, &a);
    let gamma = tr.challenge_scalar(b"ilmpp_gamma");

    // responses via e_i recursion
    let mut e: Vec<FieldElement> = Vec::with_capacity(k - 1);
    // e_1 = -(γ · x_1 / y_1)
    let e1 = (gamma.clone() * &x_logs[0] * &y_logs[0].inverse()).negation();
    e.push(e1);
    for i in 1..k - 1 {
        // e_i = -(x_i / y_i) · e_{i-1}
        let factor = (x_logs[i].clone() * &y_logs[i].inverse()).negation();
        let ei = factor * &e[i - 1];
        e.push(ei);
    }
    // r_i = θ_i + e_i
    let r: Vec<FieldElement> = (0..k - 1)
        .map(|i| thetas[i].clone() + e[i].clone())
        .collect();

    IlmppProof { a, r }
}

pub fn ilmpp_verify(tr: &mut Transcript, xs: &[G1], ys: &[G1], proof: &IlmppProof) -> bool {
    let k = xs.len();
    if k < 2 || ys.len() != k || proof.a.len() != k || proof.r.len() != k - 1 {
        return false;
    }

    ilmpp_absorb(tr, xs, ys, &proof.a);
    let gamma = tr.challenge_scalar(b"ilmpp_gamma");

    // eq 1: Y_1^{r_1} == A_1 · X_1^{-γ}
    let lhs1 = ys[0].clone().mul(&proof.r[0]);
    let rhs1 = proof.a[0].clone() + &xs[0].clone().mul(&gamma.negation());
    if lhs1 != rhs1 {
        return false;
    }

    // eq i (2 ≤ i ≤ k-1): X_i^{r_{i-1}} · Y_i^{r_i} == A_i
    for i in 1..k - 1 {
        let lhs = xs[i].clone().mul(&proof.r[i - 1]) + &ys[i].clone().mul(&proof.r[i]);
        if lhs != proof.a[i] {
            return false;
        }
    }

    // eq k: X_k^{r_{k-1}} == A_k · Y_k^{(-1)^{k-1} γ}
    let sign = if (k - 1) % 2 == 0 {
        gamma.clone()
    } else {
        gamma.negation()
    };
    let lhsk = xs[k - 1].clone().mul(&proof.r[k - 2]);
    let rhsk = proof.a[k - 1].clone() + &ys[k - 1].clone().mul(&sign);
    lhsk == rhsk
}

// ===========================================================================
// Layer 2: Simple k-shuffle, Protocol II  (paper §4.1, eqs (19)-(22))
// ===========================================================================
//
// Public:  X = [X_i], Y = [Y_i], C = g^c, D = g^d.
// Witness: c, d and a permutation π with  Y_i^d = X_{π(i)}^c  for all i,
//          AND the discrete logs x_i = log X_i, y_i = log Y_i (this is the
//          "restrictive" condition that the general shuffle later lifts).
//
// Construction: with t a verifier challenge, set
//   U = D^t, W = C^t,  X̂_i = X_i / U,  Ŷ_i = Y_i / W,
// then ∏ x̂_i = (d/c)^k · ∏ ŷ_i, and (d/c)^k is committed by B = g^β, T = g^τ,
// s = β + λτ - (d/c)^k via the element  B·T^λ·g^{-s}  (log = (d/c)^k).
// The relation is proved by one ILMPP over the length-(k+1) augmented vectors
//   Φ = (X̂_1..X̂_k, g),   Ψ = (Ŷ_1..Ŷ_k, B·T^λ·g^{-s}).

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleShuffleProof {
    pub B: G1,
    pub T: G1,
    pub s: FieldElement,
    pub ilmpp: IlmppProof,
}

fn simple_absorb_head(tr: &mut Transcript, xs: &[G1], ys: &[G1], C: &G1, D: &G1) {
    for x in xs {
        tr.commit_point(b"ss_x", x);
    }
    for y in ys {
        tr.commit_point(b"ss_y", y);
    }
    tr.commit_point(b"ss_C", C);
    tr.commit_point(b"ss_D", D);
}

/// Build the augmented (Φ, Ψ) vectors shared by prover and verifier.
fn simple_augment(
    g: &G1,
    xs: &[G1],
    ys: &[G1],
    C: &G1,
    D: &G1,
    B: &G1,
    T: &G1,
    s: &FieldElement,
    lambda: &FieldElement,
    t: &FieldElement,
) -> (Vec<G1>, Vec<G1>) {
    let k = xs.len();
    let u = D.clone().mul(t); // U = D^t
    let w = C.clone().mul(t); // W = C^t

    let mut phi: Vec<G1> = Vec::with_capacity(k + 1);
    let mut psi: Vec<G1> = Vec::with_capacity(k + 1);
    for i in 0..k {
        phi.push(xs[i].clone() - &u); // X̂_i = X_i / U
        psi.push(ys[i].clone() - &w); // Ŷ_i = Y_i / W
    }
    phi.push(g.clone()); // Φ_{k+1} = g
    // Ψ_{k+1} = B · T^λ · g^{-s}
    let psi_last = B.clone() + &T.clone().mul(lambda) + &g.clone().mul(&s.negation());
    psi.push(psi_last);
    (phi, psi)
}

pub fn simple_shuffle_prove(
    tr: &mut Transcript,
    g: &G1,
    xs: &[G1],
    ys: &[G1],
    C: &G1,
    D: &G1,
    x_logs: &[FieldElement],
    y_logs: &[FieldElement],
    c: &FieldElement,
    d: &FieldElement,
) -> SimpleShuffleProof {
    let k = xs.len();
    assert!(k >= 2);

    simple_absorb_head(tr, xs, ys, C, D);

    let beta = FieldElement::random();
    let mut tau = FieldElement::random();
    while tau.is_zero() {
        tau = FieldElement::random();
    }
    let B = g.clone().mul(&beta);
    let T = g.clone().mul(&tau);
    tr.commit_point(b"ss_B", &B);
    tr.commit_point(b"ss_T", &T);
    let lambda = tr.challenge_scalar(b"ss_lambda");

    // s = β + λτ - (d/c)^k
    let dc = d.clone() * &c.inverse();
    let dck = fe_pow(&dc, k);
    let s = (beta.clone() + lambda.clone() * &tau).minus(&dck);
    tr.commit_scalar(b"ss_s", &s);
    let t = tr.challenge_scalar(b"ss_t");

    // augmented points
    let (phi, psi) = simple_augment(g, xs, ys, C, D, &B, &T, &s, &lambda, &t);

    // augmented logs
    let dt = d.clone() * &t;
    let ct = c.clone() * &t;
    let mut phi_logs: Vec<FieldElement> = Vec::with_capacity(k + 1);
    let mut psi_logs: Vec<FieldElement> = Vec::with_capacity(k + 1);
    for i in 0..k {
        phi_logs.push(x_logs[i].minus(&dt)); // x̂_i = x_i - d t
        psi_logs.push(y_logs[i].minus(&ct)); // ŷ_i = y_i - c t
    }
    phi_logs.push(FieldElement::one()); // log g = 1
    psi_logs.push(dck); // log(B T^λ g^{-s}) = (d/c)^k

    let ilmpp = ilmpp_prove(tr, &phi, &psi, &phi_logs, &psi_logs);
    SimpleShuffleProof { B, T, s, ilmpp }
}

pub fn simple_shuffle_verify(
    tr: &mut Transcript,
    g: &G1,
    xs: &[G1],
    ys: &[G1],
    C: &G1,
    D: &G1,
    proof: &SimpleShuffleProof,
) -> bool {
    let k = xs.len();
    if k < 2 || ys.len() != k {
        return false;
    }

    simple_absorb_head(tr, xs, ys, C, D);
    tr.commit_point(b"ss_B", &proof.B);
    tr.commit_point(b"ss_T", &proof.T);
    let lambda = tr.challenge_scalar(b"ss_lambda");
    tr.commit_scalar(b"ss_s", &proof.s);
    let t = tr.challenge_scalar(b"ss_t");

    let (phi, psi) = simple_augment(g, xs, ys, C, D, &proof.B, &proof.T, &proof.s, &lambda, &t);
    ilmpp_verify(tr, &phi, &psi, &proof.ilmpp)
}

// ===========================================================================
// Chaum-Pedersen DH-equality proof  (paper §5 step 9 primitive; §2 eq (1))
// ===========================================================================
//
// Proves knowledge of w with  X1 = G1^w  and  X2 = G2^w  (log_{G1} X1 =
// log_{G2} X2), non-interactively via Fiat-Shamir. This is the k=2 special
// case of the ILMPP (paper Remark 4) and is used as the consistency link in
// the general k-shuffle.

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpProof {
    pub t1: G1,
    pub t2: G1,
    pub z: FieldElement,
}

fn cp_absorb(tr: &mut Transcript, g1: &G1, x1: &G1, g2: &G1, x2: &G1) {
    tr.commit_point(b"cp_g1", g1);
    tr.commit_point(b"cp_x1", x1);
    tr.commit_point(b"cp_g2", g2);
    tr.commit_point(b"cp_x2", x2);
}

pub fn cp_prove(
    tr: &mut Transcript,
    g1: &G1,
    x1: &G1,
    g2: &G1,
    x2: &G1,
    w: &FieldElement,
) -> CpProof {
    cp_absorb(tr, g1, x1, g2, x2);
    let nu = FieldElement::random();
    let t1 = g1.clone().mul(&nu);
    let t2 = g2.clone().mul(&nu);
    tr.commit_point(b"cp_t1", &t1);
    tr.commit_point(b"cp_t2", &t2);
    let e = tr.challenge_scalar(b"cp_e");
    let z = nu + e * w;
    CpProof { t1, t2, z }
}

pub fn cp_verify(
    tr: &mut Transcript,
    g1: &G1,
    x1: &G1,
    g2: &G1,
    x2: &G1,
    proof: &CpProof,
) -> bool {
    cp_absorb(tr, g1, x1, g2, x2);
    tr.commit_point(b"cp_t1", &proof.t1);
    tr.commit_point(b"cp_t2", &proof.t2);
    let e = tr.challenge_scalar(b"cp_e");
    // G1^z == T1 · X1^e   and   G2^z == T2 · X2^e
    let l1 = g1.clone().mul(&proof.z);
    let r1 = proof.t1.clone() + &x1.clone().mul(&e);
    let l2 = g2.clone().mul(&proof.z);
    let r2 = proof.t2.clone() + &x2.clone().mul(&e);
    l1 == r1 && l2 == r2
}

// ===========================================================================
// Layer 3: General k-shuffle for single-key re-base  (RGS, see plan §3.7)
// ===========================================================================
//
// Statement (public):  g, {X_i}, {X̄_i}, C = g^c.
// Witness (P):  re-base exponent c, permutation π, with  X̄_i = X_{π(i)}^c.
//   P does NOT know log_g X_i — this is what lifts the simple-shuffle's
//   "P knows logs" restriction.
//
// Construction follows Neff 2004 (ElGamal pairs) §4 EGA specialized to a
// single component with a *re-base* (X̄_i = X_{π(i)}^c) instead of ElGamal
// re-encryption: drop the 2nd component / g^β prefix / τ, and replace the
// additive check Λ_1 g^τ = Φ_1 with the multiplicative Chaum-Pedersen check
//   Φ_out = Φ_in^c   (witness c).
// γ below is the simple-shuffle's internal blinding exponent (Neff's "γ");
// the re-base exponent c appears only in the final CP. Completeness is
// verified term-by-term in plan §3.7.5. Roles (all logs known to P, none V):
//   Γ=g^γ, A_i=g^{a_i}, C_i=A_{π(i)}^γ, U_i=g^{u_i}, W_i=g^{γ w_i},
//   Λ = ∏ X_i^{w_{π⁻¹(i)} - u_i}              (commits the π-permuted weights)
//   ρ_i (V), B_i=g^{ρ_i}/U_i, b_i=ρ_i-u_i, d_i=γ b_{π(i)}, D_i=B_{π(i)}^γ,
//   λ (V), r_i=a_i+λ b_i, s_i=γ r_{π(i)}, σ_i=w_i+b_{π(i)}.
// Checks: (i) Γ^{σ_i}=W_i D_i ;
//         (ii) Φ_out=∏X̄_i^{σ_i} = (Φ_in)^c, Φ_in=Λ·∏X_i^{ρ_i}   [CP] ;
//         (iii) simple-shuffle S_i=R_{π(i)}^γ, R_i=A_i B_i^λ, S_i=C_i D_i^λ.

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneralShuffleProof {
    pub gamma_pt: G1,           // Γ
    pub a: Vec<G1>,             // A_i
    pub cc: Vec<G1>,            // C_i
    pub u: Vec<G1>,             // U_i
    pub w: Vec<G1>,             // W_i
    pub lambda_pt: G1,          // Λ
    pub d: Vec<G1>,             // D_i
    pub sigma: Vec<FieldElement>,
    pub ss: SimpleShuffleProof,
    pub cp: CpProof,
}

fn gshuf_absorb_head(tr: &mut Transcript, g: &G1, xs: &[G1], xbars: &[G1], cc: &G1) {
    tr.commit_point(b"gs_g", g);
    for x in xs {
        tr.commit_point(b"gs_x", x);
    }
    for xb in xbars {
        tr.commit_point(b"gs_xbar", xb);
    }
    tr.commit_point(b"gs_C", cc);
}

#[allow(clippy::too_many_arguments)]
fn gshuf_absorb_round1(
    tr: &mut Transcript,
    gamma_pt: &G1,
    a: &[G1],
    cc: &[G1],
    u: &[G1],
    w: &[G1],
    lambda_pt: &G1,
) {
    tr.commit_point(b"gs_Gamma", gamma_pt);
    for ai in a {
        tr.commit_point(b"gs_A", ai);
    }
    for ci in cc {
        tr.commit_point(b"gs_Ci", ci);
    }
    for ui in u {
        tr.commit_point(b"gs_U", ui);
    }
    for wi in w {
        tr.commit_point(b"gs_W", wi);
    }
    tr.commit_point(b"gs_Lambda", lambda_pt);
}

/// `perm[i] = π(i)`. Returns `(xbars, C, proof)` where `xbars[i] = X_{π(i)}^c`
/// and `C = g^c`. The prover knows `c` and `perm` but NOT `log_g X_i`.
pub fn general_shuffle_prove(
    tr: &mut Transcript,
    g: &G1,
    xs: &[G1],
    c: &FieldElement,
    perm: &[usize],
) -> (Vec<G1>, G1, GeneralShuffleProof) {
    let k = xs.len();
    assert!(k >= 2, "general shuffle requires k >= 2");
    assert_eq!(perm.len(), k);
    // inverse permutation: inv[π(i)] = i
    let mut inv = vec![0usize; k];
    for (i, &pi) in perm.iter().enumerate() {
        inv[pi] = i;
    }

    // outputs: X̄_i = X_{π(i)}^c ; C = g^c
    let xbars: Vec<G1> = (0..k).map(|i| xs[perm[i]].clone().mul(c)).collect();
    let cc = g.clone().mul(c);

    gshuf_absorb_head(tr, g, xs, &xbars, &cc);

    // RGS.1 — P commits γ, a_i, u_i, w_i, derived points and Λ.
    let gamma = FieldElement::random();
    // a_i should be distinct (anti-grinding, 2004 §4.1); random over a large
    // field are distinct whp, which suffices for the benchmark/self-test.
    let a_s: Vec<FieldElement> = (0..k).map(|_| FieldElement::random()).collect();
    let u_s: Vec<FieldElement> = (0..k).map(|_| FieldElement::random()).collect();
    let w_s: Vec<FieldElement> = (0..k).map(|_| FieldElement::random()).collect();

    let gamma_pt = g.clone().mul(&gamma);
    let a_pts: Vec<G1> = a_s.iter().map(|ai| g.clone().mul(ai)).collect();
    let cc_pts: Vec<G1> = (0..k).map(|i| a_pts[perm[i]].clone().mul(&gamma)).collect();
    let u_pts: Vec<G1> = u_s.iter().map(|ui| g.clone().mul(ui)).collect();
    let w_pts: Vec<G1> = w_s
        .iter()
        .map(|wi| g.clone().mul(&(gamma.clone() * wi)))
        .collect();
    // Λ = ∏_i X_i^{ w_{π⁻¹(i)} − u_i }
    let mut lambda_pt = G1::identity();
    for i in 0..k {
        let exp = w_s[inv[i]].minus(&u_s[i]);
        lambda_pt = lambda_pt + &xs[i].clone().mul(&exp);
    }

    gshuf_absorb_round1(tr, &gamma_pt, &a_pts, &cc_pts, &u_pts, &w_pts, &lambda_pt);

    // RGS.2 — V challenge ρ_i (Fiat-Shamir).
    let rho: Vec<FieldElement> = (0..k).map(|_| tr.challenge_scalar(b"gs_rho")).collect();
    // b_i = ρ_i − u_i
    let b_s: Vec<FieldElement> = (0..k).map(|i| rho[i].minus(&u_s[i])).collect();

    // RGS.3 — D_i = g^{γ b_{π(i)}}, revealed.
    let d_pts: Vec<G1> = (0..k)
        .map(|i| g.clone().mul(&(gamma.clone() * &b_s[perm[i]])))
        .collect();
    for di in &d_pts {
        tr.commit_point(b"gs_D", di);
    }

    // RGS.4 — V challenge λ.
    let lam = tr.challenge_scalar(b"gs_lambda");

    // RGS.5 — r_i = a_i + λ b_i ; s_i = γ r_{π(i)} ; σ_i = w_i + b_{π(i)}.
    let r_s: Vec<FieldElement> = (0..k).map(|i| a_s[i].clone() + lam.clone() * &b_s[i]).collect();
    let s_s: Vec<FieldElement> = (0..k).map(|i| gamma.clone() * &r_s[perm[i]]).collect();
    let sigma: Vec<FieldElement> = (0..k).map(|i| w_s[i].clone() + b_s[perm[i]].clone()).collect();
    for si in &sigma {
        tr.commit_scalar(b"gs_sigma", si);
    }

    // RGS.6 — simple shuffle on (R, S): S_i = R_{π(i)}^γ ; C=Γ, D=g, c=γ, d=1.
    let r_pts: Vec<G1> = r_s.iter().map(|ri| g.clone().mul(ri)).collect();
    let s_pts: Vec<G1> = s_s.iter().map(|si| g.clone().mul(si)).collect();
    let one = FieldElement::one();
    let ss = simple_shuffle_prove(tr, g, &r_pts, &s_pts, &gamma_pt, g, &r_s, &s_s, &gamma, &one);

    // RGS.7(ii) — Φ_in = Λ·∏X_i^{ρ_i}, Φ_out = ∏X̄_i^{σ_i}, CP proves Φ_out=Φ_in^c.
    let mut phi_in = lambda_pt.clone();
    for i in 0..k {
        phi_in = phi_in + &xs[i].clone().mul(&rho[i]);
    }
    let mut phi_out = G1::identity();
    for i in 0..k {
        phi_out = phi_out + &xbars[i].clone().mul(&sigma[i]);
    }
    let cp = cp_prove(tr, g, &cc, &phi_in, &phi_out, c);

    let proof = GeneralShuffleProof {
        gamma_pt,
        a: a_pts,
        cc: cc_pts,
        u: u_pts,
        w: w_pts,
        lambda_pt,
        d: d_pts,
        sigma,
        ss,
        cp,
    };
    (xbars, cc, proof)
}

pub fn general_shuffle_verify(
    tr: &mut Transcript,
    g: &G1,
    xs: &[G1],
    xbars: &[G1],
    cc: &G1,
    proof: &GeneralShuffleProof,
) -> bool {
    let k = xs.len();
    if k < 2
        || xbars.len() != k
        || proof.a.len() != k
        || proof.cc.len() != k
        || proof.u.len() != k
        || proof.w.len() != k
        || proof.d.len() != k
        || proof.sigma.len() != k
    {
        return false;
    }

    gshuf_absorb_head(tr, g, xs, xbars, cc);
    gshuf_absorb_round1(
        tr,
        &proof.gamma_pt,
        &proof.a,
        &proof.cc,
        &proof.u,
        &proof.w,
        &proof.lambda_pt,
    );
    let rho: Vec<FieldElement> = (0..k).map(|_| tr.challenge_scalar(b"gs_rho")).collect();
    for di in &proof.d {
        tr.commit_point(b"gs_D", di);
    }
    let lam = tr.challenge_scalar(b"gs_lambda");
    for si in &proof.sigma {
        tr.commit_scalar(b"gs_sigma", si);
    }

    // check (i): Γ^{σ_i} == W_i · D_i
    for i in 0..k {
        let lhs = proof.gamma_pt.clone().mul(&proof.sigma[i]);
        let rhs = proof.w[i].clone() + &proof.d[i];
        if lhs != rhs {
            return false;
        }
    }

    // recompute R_i = A_i B_i^λ (B_i = g^{ρ_i}/U_i) and S_i = C_i D_i^λ
    let r_pts: Vec<G1> = (0..k)
        .map(|i| {
            let b_i = g.clone().mul(&rho[i]) - &proof.u[i];
            proof.a[i].clone() + &b_i.mul(&lam)
        })
        .collect();
    let s_pts: Vec<G1> = (0..k)
        .map(|i| proof.cc[i].clone() + &proof.d[i].clone().mul(&lam))
        .collect();

    // check (iii): simple shuffle (must run before CP to keep transcript order)
    if !simple_shuffle_verify(tr, g, &r_pts, &s_pts, &proof.gamma_pt, g, &proof.ss) {
        return false;
    }

    // check (ii): Φ_out = Φ_in^c via CP(g, C, Φ_in, Φ_out)
    let mut phi_in = proof.lambda_pt.clone();
    for i in 0..k {
        phi_in = phi_in + &xs[i].clone().mul(&rho[i]);
    }
    let mut phi_out = G1::identity();
    for i in 0..k {
        phi_out = phi_out + &xbars[i].clone().mul(&proof.sigma[i]);
    }
    cp_verify(tr, g, cc, &phi_in, &phi_out, &proof.cp)
}

// ===========================================================================
// Layer 4: DSA public-key shuffle wrapper  (Shuffle.Mix / Shuffle.Verify)
// ===========================================================================
//
// ATCaaS Algorithm 6 mix-net role assignment. `shuffle_mix` samples the
// shuffler's one-shot secrets — re-base exponent c and permutation π — and
// re-bases the input keys to
//     X̄_i = X_{π(i)}^c = C^{x_{π(i)}}        (new base C = g^c)
// so the private keys are unchanged (H=g^s  <=>  H'=C^s); it proves this with
// the general k-shuffle (RGS). `shuffle_verify` checks the RGS proof AND the
// in-subgroup property of the outputs (and the new base C): omitting the
// subgroup check breaks soundness (2004 Remark 2/5) and is not observable from
// the algebraic relation alone.
//
// The inputs `xs` are the agreed `inputSet` from the ledger and are assumed
// already validated at candidate-submission time; only the prover-supplied
// outputs and C are subgroup-checked here.

/// A uniformly random permutation of `0..k`, sampled via the group-element RNG
/// (sort indices by a random field-element key; byte ties are negligible).
fn random_permutation(k: usize) -> Vec<usize> {
    let mut keyed: Vec<(Vec<u8>, usize)> = (0..k)
        .map(|i| (FieldElement::random().to_bytes(), i))
        .collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    keyed.into_iter().map(|(_, idx)| idx).collect()
}

/// `Shuffle.Mix`: re-base + permute the input keys and produce a verifiable
/// transcript. Returns `(xbars, C, proof)` with `xbars[i] = X_{π(i)}^c`,
/// `C = g^c`. The secrets `c`, `π` are local one-shot values, dropped on return.
pub fn shuffle_mix(
    tr: &mut Transcript,
    g: &G1,
    xs: &[G1],
) -> (Vec<G1>, G1, GeneralShuffleProof) {
    let k = xs.len();
    let c = FieldElement::random();
    let perm = random_permutation(k);
    general_shuffle_prove(tr, g, xs, &c, &perm)
}

/// `Shuffle.Verify`: validate a `shuffle_mix` transcript. Enforces the RGS
/// proof and the in-subgroup property of every output `X̄_i` and the new base
/// `C` (required for soundness, 2004 Remark 2/5).
pub fn shuffle_verify(
    tr: &mut Transcript,
    g: &G1,
    xs: &[G1],
    xbars: &[G1],
    cc: &G1,
    proof: &GeneralShuffleProof,
) -> bool {
    // in-subgroup checks on the untrusted values (cheap, do first)
    if !cc.has_correct_order() {
        return false;
    }
    for xb in xbars {
        if !xb.has_correct_order() {
            return false;
        }
    }
    general_shuffle_verify(tr, g, xs, xbars, cc, proof)
}

// ===========================================================================
// Self-tests
// ===========================================================================
//
// These run as a cargo *example* (`cargo run --example shuffle_selftest`)
// rather than `cargo test`, because this crate's pre-existing `#[cfg(test)]`
// modules (e.g. src/ipp.rs) do not compile under the lib-test profile.
// Examples compile the library in normal mode, so they exercise only our code.

fn commit_all(g: &G1, logs: &[FieldElement]) -> Vec<G1> {
    logs.iter().map(|s| g.clone().mul(s)).collect()
}

fn selftest_ilmpp(g: &G1) -> Result<(), String> {
    let k = 5usize;
    let x: Vec<FieldElement> = (0..k).map(|_| FieldElement::random()).collect();
    let mut y: Vec<FieldElement> = (0..k).map(|_| FieldElement::random()).collect();
    // force ∏ x = ∏ y by fixing y_k = ∏x / ∏ y_{<k}
    let mut prod_x = FieldElement::one();
    for xi in &x {
        prod_x = prod_x * xi;
    }
    let mut prod_y_head = FieldElement::one();
    for i in 0..k - 1 {
        prod_y_head = prod_y_head * &y[i];
    }
    y[k - 1] = prod_x * &prod_y_head.inverse();

    let xpts = commit_all(g, &x);
    let ypts = commit_all(g, &y);

    let mut tp = Transcript::new(b"ilmpp_selftest");
    let proof = ilmpp_prove(&mut tp, &xpts, &ypts, &x, &y);
    let mut tv = Transcript::new(b"ilmpp_selftest");
    if !ilmpp_verify(&mut tv, &xpts, &ypts, &proof) {
        return Err("ILMPP: honest proof rejected".into());
    }
    // soundness: a vector with ∏x != ∏y must be rejected
    let mut ybad = ypts.clone();
    ybad[0] = ybad[0].clone() + g;
    let mut tb = Transcript::new(b"ilmpp_selftest");
    if ilmpp_verify(&mut tb, &xpts, &ybad, &proof) {
        return Err("ILMPP: tampered Y accepted".into());
    }
    Ok(())
}

fn selftest_simple_shuffle(g: &G1) -> Result<(), String> {
    let k = 6usize;
    let c = FieldElement::random();
    let d = FieldElement::random();
    let perm: Vec<usize> = (0..k).rev().collect(); // reverse permutation

    // choose x_i freely; set y_i so Y_i^d = X_{π(i)}^c  =>  y_i = c·x_{π(i)}/d
    let x: Vec<FieldElement> = (0..k).map(|_| FieldElement::random()).collect();
    let dinv = d.inverse();
    let y: Vec<FieldElement> = (0..k).map(|i| c.clone() * &x[perm[i]] * &dinv).collect();

    let xpts = commit_all(g, &x);
    let ypts = commit_all(g, &y);
    let cc = g.clone().mul(&c);
    let dd = g.clone().mul(&d);

    let mut tp = Transcript::new(b"ss_selftest");
    let proof = simple_shuffle_prove(&mut tp, g, &xpts, &ypts, &cc, &dd, &x, &y, &c, &d);
    let mut tv = Transcript::new(b"ss_selftest");
    if !simple_shuffle_verify(&mut tv, g, &xpts, &ypts, &cc, &dd, &proof) {
        return Err("simple-shuffle: honest proof rejected".into());
    }
    // soundness: tamper one output key
    let mut ybad = ypts.clone();
    ybad[2] = ybad[2].clone() + g;
    let mut tb = Transcript::new(b"ss_selftest");
    if simple_shuffle_verify(&mut tb, g, &xpts, &ybad, &cc, &dd, &proof) {
        return Err("simple-shuffle: tampered output accepted".into());
    }
    // soundness: wrong commitment C
    let cbad = g.clone().mul(&(c.clone() + FieldElement::one()));
    let mut tb2 = Transcript::new(b"ss_selftest");
    if simple_shuffle_verify(&mut tb2, g, &xpts, &ypts, &cbad, &dd, &proof) {
        return Err("simple-shuffle: wrong C accepted".into());
    }
    Ok(())
}

fn selftest_cp(g: &G1) -> Result<(), String> {
    let w = FieldElement::random();
    let g2 = G1::from_msg_hash(b"cp_selftest_second_base");
    let x1 = g.clone().mul(&w);
    let x2 = g2.clone().mul(&w);

    let mut tp = Transcript::new(b"cp_selftest");
    let proof = cp_prove(&mut tp, g, &x1, &g2, &x2, &w);
    let mut tv = Transcript::new(b"cp_selftest");
    if !cp_verify(&mut tv, g, &x1, &g2, &x2, &proof) {
        return Err("CP: honest proof rejected".into());
    }
    // soundness: unequal logs must be rejected
    let x2bad = x2.clone() + &g2;
    let mut tb = Transcript::new(b"cp_selftest");
    if cp_verify(&mut tb, g, &x1, &g2, &x2bad, &proof) {
        return Err("CP: unequal logs accepted".into());
    }
    Ok(())
}

fn selftest_general_shuffle(g: &G1) -> Result<(), String> {
    let k = 5usize;
    // secret "private keys" x_i (the prover never uses these)
    let xsec: Vec<FieldElement> = (0..k).map(|_| FieldElement::random()).collect();
    let xs = commit_all(g, &xsec);
    let c = FieldElement::random();
    let perm = vec![2usize, 4, 0, 3, 1]; // a permutation of 0..k

    let mut tp = Transcript::new(b"gs_selftest");
    let (xbars, cc, proof) = general_shuffle_prove(&mut tp, g, &xs, &c, &perm);

    // permutation/re-base correctness: X̄_i must equal g^{c·x_{π(i)}} = C^{x_{π(i)}}
    for i in 0..k {
        let want = g.clone().mul(&(c.clone() * &xsec[perm[i]]));
        if xbars[i] != want {
            return Err("general-shuffle: xbar != X_{π(i)}^c".into());
        }
    }

    let mut tv = Transcript::new(b"gs_selftest");
    if !general_shuffle_verify(&mut tv, g, &xs, &xbars, &cc, &proof) {
        return Err("general-shuffle: honest proof rejected".into());
    }

    // soundness 1: tamper an output key
    let mut xb_bad = xbars.clone();
    xb_bad[0] = xb_bad[0].clone() + g;
    let mut t1 = Transcript::new(b"gs_selftest");
    if general_shuffle_verify(&mut t1, g, &xs, &xb_bad, &cc, &proof) {
        return Err("general-shuffle: tampered output accepted".into());
    }

    // soundness 2: wrong re-base commitment C
    let cc_bad = cc.clone() + g;
    let mut t2 = Transcript::new(b"gs_selftest");
    if general_shuffle_verify(&mut t2, g, &xs, &xbars, &cc_bad, &proof) {
        return Err("general-shuffle: wrong C accepted".into());
    }

    // soundness 3: tamper an opened σ_i
    let mut p3 = proof.clone();
    p3.sigma[1] = p3.sigma[1].clone() + FieldElement::one();
    let mut t3 = Transcript::new(b"gs_selftest");
    if general_shuffle_verify(&mut t3, g, &xs, &xbars, &cc, &p3) {
        return Err("general-shuffle: tampered sigma accepted".into());
    }

    // soundness 4: tamper a D_i commitment
    let mut p4 = proof.clone();
    p4.d[2] = p4.d[2].clone() + g;
    let mut t4 = Transcript::new(b"gs_selftest");
    if general_shuffle_verify(&mut t4, g, &xs, &xbars, &cc, &p4) {
        return Err("general-shuffle: tampered D accepted".into());
    }

    Ok(())
}

fn selftest_shuffle_mix(g: &G1) -> Result<(), String> {
    let k = 6usize;
    let xsec: Vec<FieldElement> = (0..k).map(|_| FieldElement::random()).collect();
    let xs = commit_all(g, &xsec);

    let mut tp = Transcript::new(b"mix_selftest");
    let (xbars, cc, proof) = shuffle_mix(&mut tp, g, &xs);

    // outputs and new base must be in the prime-order subgroup
    if !cc.has_correct_order() {
        return Err("shuffle-mix: C not in subgroup".into());
    }
    for xb in &xbars {
        if !xb.has_correct_order() {
            return Err("shuffle-mix: output not in subgroup".into());
        }
    }

    // honest verify
    let mut tv = Transcript::new(b"mix_selftest");
    if !shuffle_verify(&mut tv, g, &xs, &xbars, &cc, &proof) {
        return Err("shuffle-mix: honest proof rejected".into());
    }

    // {X̄_i} must be a *bijective* re-base of inputs: each output equals
    // C^{x_j} = g^{c x_j} for a unique j (this is the "owner can claim its
    // role" property — a node holding x_j recomputes C^{x_j} and finds it).
    let cbase_pow: Vec<G1> = xsec.iter().map(|xj| cc.clone().mul(xj)).collect();
    let mut used = vec![false; k];
    for xb in &xbars {
        let mut found = false;
        for j in 0..k {
            if !used[j] && *xb == cbase_pow[j] {
                used[j] = true;
                found = true;
                break;
            }
        }
        if !found {
            return Err("shuffle-mix: output is not a bijective re-base of inputs".into());
        }
    }

    // tampering is still rejected through the wrapper
    let mut p_bad = proof.clone();
    p_bad.sigma[0] = p_bad.sigma[0].clone() + FieldElement::one();
    let mut tb = Transcript::new(b"mix_selftest");
    if shuffle_verify(&mut tb, g, &xs, &xbars, &cc, &p_bad) {
        return Err("shuffle-mix: tampered proof accepted".into());
    }

    Ok(())
}

/// Run all shuffle self-tests against a fixed generator. Returns Err(reason)
/// on the first failure.
pub fn selftest_all() -> Result<(), String> {
    let g = G1::from_msg_hash(b"atcaas_shuffle_selftest_generator");
    selftest_ilmpp(&g)?;
    selftest_simple_shuffle(&g)?;
    selftest_cp(&g)?;
    selftest_general_shuffle(&g)?;
    selftest_shuffle_mix(&g)?;
    Ok(())
}
