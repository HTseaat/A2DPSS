#![allow(non_snake_case)]

#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate amcl_wrapper;

extern crate serde;

#[macro_use]
extern crate serde_derive;

#[macro_use]
extern crate serde_json;

#[macro_use]
pub mod errors;

#[macro_use]
pub mod utils;

pub mod transcript;

pub mod ipp;

pub mod r1cs;
pub mod poseidon;

pub mod vrf;
pub use crate::vrf::{Secret, Public, Proof};

pub mod shuffle;

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use amcl_wrapper::field_elem::FieldElement;
use amcl_wrapper::group_elem::GroupElement;
use amcl_wrapper::group_elem_g1::{G1Vector, G1};
use crate::r1cs::{Prover, LinearCombination, ConstraintSystem, Variable, Verifier};
use crate::utils::get_generators;
use merlin::Transcript;
use serde::{Deserialize, Serialize};
use std::ops::Mul;
use hex;
use rayon::prelude::*;

use crate::poseidon::{PoseidonSponge, DuplexSpongeMode, PoseidonConfig, find_poseidon_ark_and_mds};
use crate::poseidon::{poseidon_permute_prover, poseidon_permute_verifier};
use crate::r1cs::proof::R1CSProof;
use amcl_wrapper::group_elem::GroupElementVector; // 确保导入 trait
use crate::poseidon::{prove_full, verify_full, FullProof};



#[derive(Deserialize)]
pub struct Witness {
    pub p: String,  // hex string
    pub q: String,
    pub r: String,
    pub p_blind: String,
    pub q_blind: String,
    pub r_blind: String,
}

#[derive(Deserialize)]
pub struct ProveInput {
    pub witnesses: Vec<Witness>,
    pub g: String,
    pub h: String,
}

#[derive(Serialize)]
pub struct ProofResult {
    pub proof: String,  // hex-encoded serialized proof
    pub commitments: Vec<String>,
}

#[no_mangle]
pub extern "C" fn pyProveFactors(json_input: *const c_char) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    // // 兼容旧接口调试用
    // let witnesses: Vec<Witness> = match serde_json::from_str(input) {
    //     Ok(w) => w,
    //     Err(_) => return std::ptr::null(),
    // };

    // 新结构解析生成元
    let parsed: ProveInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let witnesses = parsed.witnesses;
    // let g = G1::from_hex(parsed.g).unwrap();
    // let h = G1::from_hex(parsed.h).unwrap();
    let g_bytes = hex::decode(&parsed.g).unwrap();
    let g = G1::from_bytes(&g_bytes).unwrap();
    let h_bytes = hex::decode(&parsed.h).unwrap();
    let h = G1::from_bytes(&h_bytes).unwrap();

    let num_generators = witnesses.len(); // 每个乘法证明对应一对 G, H
    println!("num_generators {:?}", num_generators);
    let padded_num_generators = num_generators.next_power_of_two(); // ✨ 补足为2次幂
    let G: G1Vector = get_generators("G", padded_num_generators).into();
    let H: G1Vector = get_generators("H", padded_num_generators).into();
    // let g = G1::from_msg_hash("g".as_bytes());
    // let h = G1::from_msg_hash("h".as_bytes());
    let mut prover_transcript = Transcript::new(b"Factors");
    let mut commitment_hexes = vec![];
    let mut prover = Prover::new(&g, &h, &mut prover_transcript);

    for wit in &witnesses {
        let p = FieldElement::from_hex(wit.p.clone()).unwrap();
        let q = FieldElement::from_hex(wit.q.clone()).unwrap();
        let r = FieldElement::from_hex(wit.r.clone()).unwrap();

        let p_blind = FieldElement::from_hex(wit.p_blind.clone()).unwrap();
        let q_blind = FieldElement::from_hex(wit.q_blind.clone()).unwrap();
        let r_blind = FieldElement::from_hex(wit.r_blind.clone()).unwrap();

        // let (com_p, var_p) = prover.commit(p, FieldElement::random());
        // commitment_hexes.push(com_p.to_hex());
        // let (com_q, var_q) = prover.commit(q, FieldElement::random());
        // commitment_hexes.push(com_q.to_hex());

        let (com_p, var_p) = prover.commit(p, p_blind);
        commitment_hexes.push(com_p.to_hex());

        let (com_q, var_q) = prover.commit(q, q_blind);
        commitment_hexes.push(com_q.to_hex());
        let (_, _, o) = prover.multiply(var_p.into(), var_q.into());
        // let lc: LinearCombination = vec![(Variable::One(), r)].iter().collect();
        // prover.constrain(o - lc);
        let (com_r, var_r) = prover.commit(r, r_blind);
        commitment_hexes.push(com_r.to_hex());

        prover.constrain(o - var_r);
    }

    let proof = prover.prove(&G, &H).unwrap();
    let proof_bytes = bincode::serialize(&proof).unwrap();
    let proof_hex = hex::encode(proof_bytes);
    // let result = ProofResult { proof: proof_hex };
    let result = ProofResult {
        proof: proof_hex,
        commitments: commitment_hexes,
    };

    let json = serde_json::to_string(&result).unwrap();
    CString::new(json).unwrap().into_raw()
}

// ---------------------------------------------------------------------------
// ATCaaS Neff shuffle FFI  (Algorithm 6 Shuffle.Mix / Shuffle.Verify)
// ---------------------------------------------------------------------------
//
// Points are exchanged as AMCL `to_hex()` strings "(x,y)" — the canonical G1
// encoding used across the project (beaver/g1_backend.py `G1Point.value`).
// The proof is bincode-serialized then hex-encoded, mirroring pyProveFactors.
// Both Mix and Verify seed the same Fiat-Shamir transcript label.

#[derive(Deserialize)]
pub struct ShuffleMixInput {
    pub g: String,
    pub X: Vec<String>,
}

#[derive(Serialize)]
pub struct ShuffleMixOutput {
    pub Xbar: Vec<String>,
    pub C: String,
    pub proof: String,
}

#[derive(Deserialize)]
pub struct ShuffleVerifyInput {
    pub g: String,
    pub X: Vec<String>,
    pub Xbar: Vec<String>,
    pub C: String,
    pub proof: String,
}

#[derive(Serialize)]
pub struct ShuffleVerifyOutput {
    pub valid: bool,
}

/// Shuffle.Mix: re-base + permute the input DSA public keys, returning the new
/// keys, the new base C=g^c, and a verifiable transcript.
/// Input  JSON: {"g": hex, "X": [hex, ...]}
/// Output JSON: {"Xbar": [hex, ...], "C": hex, "proof": hex}   (null on bad input)
#[no_mangle]
pub extern "C" fn pyShuffleMix(json_input: *const c_char) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let parsed: ShuffleMixInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let g = match G1::from_hex(parsed.g.clone()) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let mut xs = Vec::with_capacity(parsed.X.len());
    for xh in &parsed.X {
        match G1::from_hex(xh.clone()) {
            Ok(v) => xs.push(v),
            Err(_) => return std::ptr::null(),
        }
    }
    if xs.len() < 2 {
        return std::ptr::null();
    }

    let mut tr = Transcript::new(b"atcaas_shuffle_v1");
    let (xbars, cc, proof) = shuffle::shuffle_mix(&mut tr, &g, &xs);

    let proof_hex = hex::encode(bincode::serialize(&proof).unwrap());
    let out = ShuffleMixOutput {
        Xbar: xbars.iter().map(|p| p.to_hex()).collect(),
        C: cc.to_hex(),
        proof: proof_hex,
    };
    let json = serde_json::to_string(&out).unwrap();
    CString::new(json).unwrap().into_raw()
}

/// Shuffle.Verify: validate a Shuffle.Mix transcript (RGS proof + output
/// subgroup checks). Output JSON: {"valid": bool}. Returns null only on
/// malformed JSON; any decode/verify failure maps to {"valid": false}.
#[no_mangle]
pub extern "C" fn pyShuffleVerify(json_input: *const c_char) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let parsed: ShuffleVerifyInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let valid = (|| -> Option<bool> {
        let g = G1::from_hex(parsed.g.clone()).ok()?;
        let mut xs = Vec::with_capacity(parsed.X.len());
        for xh in &parsed.X {
            xs.push(G1::from_hex(xh.clone()).ok()?);
        }
        let mut xbars = Vec::with_capacity(parsed.Xbar.len());
        for xh in &parsed.Xbar {
            xbars.push(G1::from_hex(xh.clone()).ok()?);
        }
        let cc = G1::from_hex(parsed.C.clone()).ok()?;
        let proof: shuffle::GeneralShuffleProof =
            bincode::deserialize(&hex::decode(&parsed.proof).ok()?).ok()?;
        let mut tr = Transcript::new(b"atcaas_shuffle_v1");
        Some(shuffle::shuffle_verify(&mut tr, &g, &xs, &xbars, &cc, &proof))
    })()
    .unwrap_or(false);

    let json = serde_json::to_string(&ShuffleVerifyOutput { valid }).unwrap();
    CString::new(json).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn pyFreeString(ptr: *mut c_char) {
    if !ptr.is_null() {
        // try-catch like handling for double free (unsafe, but defensive)
        let _ = unsafe { CString::from_raw(ptr) };
    }
}

// Expose verifier interface to Python
#[derive(Deserialize)]
pub struct VerificationInput {
    pub proof: String,            // hex-encoded proof string
    pub commitments: Vec<String>, // hex-encoded G1 commitments
    pub g: String,                // hex-encoded G1 generator
    pub h: String                 // hex-encoded H generator
}

#[derive(Serialize)]
pub struct VerificationResult {
    pub verified: bool,
}

#[no_mangle]
pub extern "C" fn pyVerifyFactors(json_input: *const c_char) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let parsed: VerificationInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    // let g = G1::from_hex(parsed.g).unwrap();
    // let h = G1::from_hex(parsed.h).unwrap();
    let g_bytes = hex::decode(&parsed.g).unwrap();
    let g = G1::from_bytes(&g_bytes).unwrap();
    let h_bytes = hex::decode(&parsed.h).unwrap();
    let h = G1::from_bytes(&h_bytes).unwrap();
    let num_generators = parsed.commitments.len() / 3;
    println!("num_generators {:?}", num_generators);
    let padded_num_generators = num_generators.next_power_of_two(); // ✨ 补足为2次幂
    println!("padded_num_generators {:?}", padded_num_generators);
    let G: G1Vector = get_generators("G", padded_num_generators).into();
    let H: G1Vector = get_generators("H", padded_num_generators).into();
    // let G: G1Vector = get_generators("G", num_generators).into();
    // let H: G1Vector = get_generators("H", num_generators).into();
    let mut verifier_transcript = Transcript::new(b"Factors");
    let mut verifier = crate::r1cs::Verifier::new(&mut verifier_transcript);

    let mut commitments: Vec<G1> = parsed
        .commitments
        .iter()
        .map(|c| G1::from_hex(c.to_string()).unwrap())
        .collect();

    for _ in 0..(parsed.commitments.len() / 3) {
        let var_p = verifier.commit(commitments.remove(0));
        let var_q = verifier.commit(commitments.remove(0));
        let (_, _, o) = verifier.multiply(var_p.into(), var_q.into());
        let var_r = verifier.commit(commitments.remove(0));
        verifier.constrain(o - var_r);
    }

    let proof_bytes = match hex::decode(parsed.proof) {
        Ok(b) => b,
        Err(_) => return std::ptr::null(),
    };

    let proof = match bincode::deserialize(&proof_bytes) {
        Ok(p) => p,
        Err(_) => return std::ptr::null(),
    };

    let verified = verifier.verify(&proof, &g, &h, &G, &H).is_ok();
    let result = VerificationResult { verified };

    let json = serde_json::to_string(&result).unwrap();
    CString::new(json).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn pyGetGenerators(prefix: *const c_char, n: usize) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(prefix) };
    let prefix = c_str.to_str().unwrap();

    let gens = get_generators(prefix, n);
    let hex_vec: Vec<String> = gens.iter().map(|g| g.to_hex()).collect();

    let json = serde_json::to_string(&hex_vec).unwrap();
    CString::new(json).unwrap().into_raw()
}

#[derive(Deserialize)]
pub struct VectorCommitInput {
    pub scalars: Vec<String>,
    pub bases: Vec<String>,
}

#[no_mangle]
pub extern "C" fn pyComputeInnerProductInput(json_input: *const c_char) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = c_str.to_str().unwrap();
    let parsed: VectorCommitInput = serde_json::from_str(input).unwrap();

    let scalars: Vec<FieldElement> = parsed.scalars
        .iter()
        .map(|s| FieldElement::from_hex(s.clone()).unwrap())
        .collect();

    let bases: Vec<G1> = parsed.bases
        .iter()
        .map(|b| G1::from_hex(b.clone()).unwrap())
        .collect();

    let mut acc = G1::identity();
    for (s, g) in scalars.iter().zip(bases.iter()) {
        acc = acc + g.mul(s);
    }

    let json = serde_json::to_string(&acc.to_hex()).unwrap();
    CString::new(json).unwrap().into_raw()
}


// #[derive(Deserialize)]
// pub struct IPProofInput {
//     pub g_vec: Vec<String>,
//     pub h_vec: Vec<String>,
//     pub in_commit: String,
//     pub out_commit: String,
// }

// #[no_mangle]
// pub extern "C" fn pyProveInnerProduct(json_input: *const c_char) -> *const c_char {
//     let c_str = unsafe { CStr::from_ptr(json_input) };
//     let input = c_str.to_str().unwrap();
//     let parsed: IPProofInput = serde_json::from_str(input).unwrap();

//     let g_vec: Vec<G1> = parsed.g_vec.iter().map(|s| G1::from_hex(s.clone()).unwrap()).collect();
//     let h_vec: Vec<G1> = parsed.h_vec.iter().map(|s| G1::from_hex(s.clone()).unwrap()).collect();
//     let in_commit = G1::from_hex(parsed.in_commit).unwrap();
//     let out_commit = G1::from_hex(parsed.out_commit).unwrap();

//     let mut transcript = Transcript::new(b"InnerProduct");
//     let proof = crate::ipp::ProveInnerProduct(&g_vec, &h_vec, &in_commit, &out_commit, &mut transcript).unwrap();
//     let proof_bytes = bincode::serialize(&proof).unwrap();
//     let proof_hex = hex::encode(proof_bytes);

//     let json = serde_json::to_string(&proof_hex).unwrap();
//     CString::new(json).unwrap().into_raw()
// }

// #[no_mangle]
// pub extern "C" fn pyFreeString(ptr: *mut c_char) {
//     if !ptr.is_null() {
//         unsafe {
//             CString::from_raw(ptr);
//         }
//     }
// }


// --- FFI for prove_full and verify_full ---
#[derive(Deserialize)]
pub struct PvTransferProveInput {
    pub g: String,
    pub h: String,
    pub pk: String,
    pub C1: String,
    pub C2: String,
    pub W: Vec<String>,
    pub r: String,
    pub r_prime: String,
    pub m: Vec<String>,
    pub m_prime: Vec<String>,
}

#[derive(Serialize)]
pub struct PvTransferProofOutput {
    pub proof: String,
}

#[no_mangle]
pub extern "C" fn pyProveFull(json_input: *const c_char) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let parsed: PvTransferProveInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let g = G1::from_bytes(&hex::decode(&parsed.g).unwrap()).unwrap();
    let h = G1::from_bytes(&hex::decode(&parsed.h).unwrap()).unwrap();
    let pk = G1::from_bytes(&hex::decode(&parsed.pk).unwrap()).unwrap();
    let C1 = G1::from_bytes(&hex::decode(&parsed.C1).unwrap()).unwrap();
    let C2 = G1::from_bytes(&hex::decode(&parsed.C2).unwrap()).unwrap();
    let W: Vec<G1> = parsed.W.iter().map(|s| G1::from_bytes(&hex::decode(s).unwrap()).unwrap()).collect();
    let r = FieldElement::from_hex(parsed.r).unwrap();
    let r_prime = FieldElement::from_hex(parsed.r_prime).unwrap();
    let m: Vec<FieldElement> = parsed.m.iter().map(|s| FieldElement::from_hex(s.clone()).unwrap()).collect();
    let m_prime: Vec<FieldElement> = parsed.m_prime.iter().map(|s| FieldElement::from_hex(s.clone()).unwrap()).collect();

    let (ark, mds) = poseidon::find_poseidon_ark_and_mds(255, 2, 8, 57, 0);
    let cfg = poseidon::PoseidonConfig::new(8, 57, 5, mds.clone(), ark.clone(), 2, 1);
    let G_vec = get_generators("G", 4096);
    let H_vec = get_generators("H", 4096);

    let proof = prove_full(&g, &h, &pk, &C1, &C2, &W, &r, &r_prime, &m, &m_prime, &cfg, &ark, &mds, &G_vec, &H_vec);
    let proof_bytes = bincode::serialize(&proof).unwrap();
    let proof_hex = hex::encode(proof_bytes);
    let result = PvTransferProofOutput { proof: proof_hex };
    let json = serde_json::to_string(&result).unwrap();
    CString::new(json).unwrap().into_raw()
}

#[derive(Deserialize)]
pub struct PvTransferVerifyInput {
    pub g: String,
    pub h: String,
    pub pk: String,
    pub C1: String,
    pub C2: String,
    pub W: Vec<String>,
    pub proof: String,
}

#[derive(Serialize)]
pub struct PvTransferVerifyOutput {
    pub verified: bool,
}

#[no_mangle]
pub extern "C" fn pyVerifyFull(json_input: *const c_char) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let parsed: PvTransferVerifyInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let g = G1::from_bytes(&hex::decode(&parsed.g).unwrap()).unwrap();
    let h = G1::from_bytes(&hex::decode(&parsed.h).unwrap()).unwrap();
    let pk = G1::from_bytes(&hex::decode(&parsed.pk).unwrap()).unwrap();
    let C1 = G1::from_bytes(&hex::decode(&parsed.C1).unwrap()).unwrap();
    let C2 = G1::from_bytes(&hex::decode(&parsed.C2).unwrap()).unwrap();
    let W: Vec<G1> = parsed.W.iter().map(|s| G1::from_bytes(&hex::decode(s).unwrap()).unwrap()).collect();
    let proof_bytes = hex::decode(&parsed.proof).unwrap();
    let proof: FullProof = bincode::deserialize(&proof_bytes).unwrap();

    let (ark, mds) = poseidon::find_poseidon_ark_and_mds(255, 2, 8, 57, 0);
    let cfg = poseidon::PoseidonConfig::new(8, 57, 5, mds.clone(), ark.clone(), 2, 1);
    let G_vec = get_generators("G", 4096);
    let H_vec = get_generators("H", 4096);

    let verified = verify_full(&g, &h, &pk, &C1, &C2, &W, &proof, &cfg, &ark, &mds, &G_vec, &H_vec);
    let result = PvTransferVerifyOutput { verified };
    let json = serde_json::to_string(&result).unwrap();
    CString::new(json).unwrap().into_raw()
}

// --- ElGamal encryption FFI ---
#[derive(Deserialize)]
pub struct ElGamalEncryptInput {
    pub g: String,   // G1 base point (hex of uncompressed)
    pub pk: String,  // public key (hex)
    pub r: String,   // randomness (hex scalar)
    pub k: String    // message (hex scalar, here g^k)
}

#[derive(Serialize)]
pub struct ElGamalEncryptOutput {
    pub C1: String,
    pub C2: String,
}

#[derive(Deserialize)]
pub struct ElGamalDecryptInput {
    pub C1: String,  // hex string of G1 point
    pub C2: String,  // hex string of G1 point
    pub sk: String   // hex string of scalar
}

#[derive(Serialize)]
pub struct ElGamalDecryptOutput {
    pub message: String,  // hex string of G1 point
}

#[no_mangle]
pub extern "C" fn pyElGamalEncrypt(json_input: *const std::os::raw::c_char) -> *const std::os::raw::c_char {
    let c_str = unsafe { std::ffi::CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let parsed: ElGamalEncryptInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let g = G1::from_bytes(&hex::decode(&parsed.g).unwrap()).unwrap();
    let pk = G1::from_bytes(&hex::decode(&parsed.pk).unwrap()).unwrap();
    let r = FieldElement::from_hex(parsed.r).unwrap();
    let k = FieldElement::from_hex(parsed.k).unwrap();

    let C1 = g.clone().mul(&r);
    let gk = g.mul(&k);
    let pkr = pk.mul(&r);
    let C2 = gk + &pkr;

    let output = ElGamalEncryptOutput {
        C1: C1.to_hex(),
        C2: C2.to_hex(),
    };

    let json = serde_json::to_string(&output).unwrap();
    std::ffi::CString::new(json).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn pyElGamalDecrypt(json_input: *const c_char) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let parsed: ElGamalDecryptInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let C1 = G1::from_bytes(&hex::decode(&parsed.C1).unwrap()).unwrap();
    let C2 = G1::from_bytes(&hex::decode(&parsed.C2).unwrap()).unwrap();
    let sk = FieldElement::from_hex(parsed.sk).unwrap();

    // Compute shared secret: C1^sk = g^{r*sk}
    let shared = C1.mul(&sk);

    // Decrypt: message = C2 - shared
    let msg = C2 - &shared;

    let output = ElGamalDecryptOutput {
        message: msg.to_hex(),
    };

    let json = serde_json::to_string(&output).unwrap();
    CString::new(json).unwrap().into_raw()
}
// --------- pyComputeCommitmentGH: compute W = g^m * h^m' for each (m, m') pair from Python input ---------

#[derive(Deserialize)]
pub struct GHCommitmentInput {
    pub g: String,
    pub h: String,
    pub m: Vec<String>,
    pub m_prime: Vec<String>,
}

#[no_mangle]
pub extern "C" fn pyComputeCommitmentGH(json_input: *const std::os::raw::c_char) -> *const std::os::raw::c_char {
    use std::ffi::{CStr, CString};
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let parsed: GHCommitmentInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let g = G1::from_bytes(&hex::decode(&parsed.g).unwrap()).unwrap();
    let h = G1::from_bytes(&hex::decode(&parsed.h).unwrap()).unwrap();
    let m_vec: Vec<FieldElement> = parsed.m.iter().map(|s| FieldElement::from_hex(s.clone()).unwrap()).collect();
    let m_prime_vec: Vec<FieldElement> = parsed.m_prime.iter().map(|s| FieldElement::from_hex(s.clone()).unwrap()).collect();

    if m_vec.len() != m_prime_vec.len() {
        return std::ptr::null();
    }

    let mut result: Vec<String> = Vec::with_capacity(m_vec.len());
    for i in 0..m_vec.len() {
        let W_i = g.clone().mul(&m_vec[i]) + &h.clone().mul(&m_prime_vec[i]);
        result.push(W_i.to_hex());
    }

    let json = serde_json::to_string(&result).unwrap();
    CString::new(json).unwrap().into_raw()
}

fn parse_g1_for_ffi(value: &str) -> Option<G1> {
    let trimmed = value.trim();
    if trimmed == "infinity" || (trimmed.starts_with('(') && trimmed.ends_with(')')) {
        return G1::from_hex(trimmed.to_string()).ok();
    }

    let without_prefix = trimmed.strip_prefix("0x").unwrap_or(trimmed);
    match hex::decode(without_prefix) {
        Ok(bytes) => G1::from_bytes(&bytes).ok(),
        Err(_) => G1::from_hex(trimmed.to_string()).ok(),
    }
}

fn parse_scalar_for_ffi(value: &str) -> Option<FieldElement> {
    let trimmed = value.trim();
    let without_prefix = trimmed.strip_prefix("0x").unwrap_or(trimmed);
    FieldElement::from_hex(without_prefix.to_string()).ok()
}

fn g1_to_json(point: G1) -> *const c_char {
    CString::new(serde_json::to_string(&point.to_hex()).unwrap())
        .unwrap()
        .into_raw()
}

#[derive(Deserialize)]
pub struct G1UnaryInput {
    pub point: String,
}

#[derive(Deserialize)]
pub struct G1BinaryInput {
    pub left: String,
    pub right: String,
}

#[derive(Deserialize)]
pub struct G1MulInput {
    pub point: String,
    pub scalar: String,
}

#[derive(Deserialize)]
pub struct G1LinearCombinationInput {
    pub points: Vec<String>,
    pub scalars: Vec<String>,
}

#[derive(Deserialize)]
pub struct G1LinearRelationInput {
    pub announcement: String,
    pub target: String,
    pub points: Vec<String>,
    pub scalars: Vec<String>,
}

#[derive(Deserialize)]
pub struct G1LinearRelationsVerifyInput {
    pub challenge: String,
    pub relations: Vec<G1LinearRelationInput>,
}

#[derive(Deserialize)]
pub struct G1ChunkProofVerifyInput {
    pub key_base: String,
    pub msg_base: String,
    pub y0: String,
    pub B: Vec<String>,
    pub C: Vec<String>,
    pub D0: String,
    pub D: Vec<Vec<String>>,
    pub Y: String,
    pub rpks: Vec<String>,
    pub c1: Vec<Vec<String>>,
    pub c2: Vec<Vec<String>>,
    pub x_powers: Vec<String>,
    pub challenge_weights: Vec<Vec<String>>,
    pub z_s_weighted: String,
    pub z_r: Vec<Vec<String>>,
    pub rpk_weights: Vec<String>,
    pub z_b: String,
}

#[derive(Serialize)]
pub struct G1ChunkProofVerifyOutput {
    pub valid: bool,
}

#[derive(Deserialize)]
pub struct G1CiphertextInput {
    pub C1: String,
    pub C2: String,
}

#[derive(Deserialize)]
pub struct G1ChunkPairAggregationInput {
    pub weights: Vec<String>,
    pub enc_s_c1: Vec<Vec<String>>,
    pub enc_s_c2: Vec<Vec<String>>,
    pub enc_hat_c1: Vec<Vec<String>>,
    pub enc_hat_c2: Vec<Vec<String>>,
    pub claimed_c_v: Option<Vec<G1CiphertextInput>>,
}

#[derive(Serialize)]
pub struct G1ChunkPairAggregationOutput {
    pub valid: bool,
    pub r_s: Vec<String>,
    pub d_s: Vec<String>,
    pub r_hat: Vec<String>,
    pub d_hat: Vec<String>,
}

#[derive(Serialize)]
pub struct G1EqualOutput {
    pub equal: bool,
}

#[no_mangle]
pub extern "C" fn pyG1Identity() -> *const c_char {
    g1_to_json(G1::identity())
}

#[no_mangle]
pub extern "C" fn pyG1Normalize(json_input: *const c_char) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let parsed: G1UnaryInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let point = match parse_g1_for_ffi(&parsed.point) {
        Some(v) => v,
        None => return std::ptr::null(),
    };
    g1_to_json(point)
}

#[no_mangle]
pub extern "C" fn pyG1Add(json_input: *const c_char) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let parsed: G1BinaryInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let left = match parse_g1_for_ffi(&parsed.left) {
        Some(v) => v,
        None => return std::ptr::null(),
    };
    let right = match parse_g1_for_ffi(&parsed.right) {
        Some(v) => v,
        None => return std::ptr::null(),
    };
    g1_to_json(left + &right)
}

#[no_mangle]
pub extern "C" fn pyG1Sub(json_input: *const c_char) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let parsed: G1BinaryInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let left = match parse_g1_for_ffi(&parsed.left) {
        Some(v) => v,
        None => return std::ptr::null(),
    };
    let right = match parse_g1_for_ffi(&parsed.right) {
        Some(v) => v,
        None => return std::ptr::null(),
    };
    g1_to_json(left - &right)
}

#[no_mangle]
pub extern "C" fn pyG1Mul(json_input: *const c_char) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let parsed: G1MulInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let point = match parse_g1_for_ffi(&parsed.point) {
        Some(v) => v,
        None => return std::ptr::null(),
    };
    let scalar = match parse_scalar_for_ffi(&parsed.scalar) {
        Some(v) => v,
        None => return std::ptr::null(),
    };
    g1_to_json(point.mul(&scalar))
}

#[no_mangle]
pub extern "C" fn pyG1LinearCombination(json_input: *const c_char) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let parsed: G1LinearCombinationInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    if parsed.points.len() != parsed.scalars.len() {
        return std::ptr::null();
    }

    let mut acc = G1::identity();
    for i in 0..parsed.points.len() {
        let point = match parse_g1_for_ffi(&parsed.points[i]) {
            Some(v) => v,
            None => return std::ptr::null(),
        };
        let scalar = match parse_scalar_for_ffi(&parsed.scalars[i]) {
            Some(v) => v,
            None => return std::ptr::null(),
        };
        acc += point.mul(&scalar);
    }
    g1_to_json(acc)
}

/// Verify each linear-representation relation independently in parallel.
#[no_mangle]
pub extern "C" fn pyG1VerifyLinearRelations(
    json_input: *const c_char,
) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let parsed: G1LinearRelationsVerifyInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let challenge = parse_scalar_for_ffi(&parsed.challenge);
    let valid = match challenge {
        None => false,
        Some(challenge) => {
            !parsed.relations.is_empty()
                && parsed.relations.par_iter().all(|relation| {
                    if relation.points.is_empty()
                        || relation.points.len() != relation.scalars.len()
                    {
                        return false;
                    }
                    let announcement = parse_g1_for_ffi(&relation.announcement);
                    let target = parse_g1_for_ffi(&relation.target);
                    let points = relation
                        .points
                        .iter()
                        .map(|point| parse_g1_for_ffi(point))
                        .collect::<Option<Vec<_>>>();
                    let scalars = relation
                        .scalars
                        .iter()
                        .map(|scalar| parse_scalar_for_ffi(scalar))
                        .collect::<Option<Vec<_>>>();
                    match (announcement, target, points, scalars) {
                        (Some(announcement), Some(target), Some(points), Some(scalars)) => {
                            let left = points
                                .iter()
                                .zip(scalars.iter())
                                .fold(G1::identity(), |acc, (point, scalar)| {
                                    acc + &point.mul(scalar)
                                });
                            let right = announcement + &target.mul(&challenge);
                            left == right
                        }
                        _ => false,
                    }
                })
        }
    };

    CString::new(
        serde_json::to_string(&G1ChunkProofVerifyOutput { valid }).unwrap(),
    )
    .unwrap()
    .into_raw()
}

fn parallel_g1_linear_combination(points: &[G1], scalars: &[FieldElement]) -> Option<G1> {
    if points.len() != scalars.len() {
        return None;
    }
    Some(
        points
            .par_iter()
            .zip(scalars.par_iter())
            .map(|(point, scalar)| point.mul(scalar))
            .reduce(G1::identity, |left, right| left + &right),
    )
}

fn aggregate_chunk_rows(
    rows: &[Vec<String>],
    weights: &[FieldElement],
) -> Option<Vec<G1>> {
    let chunks = weights.len();
    if chunks == 0 || rows.iter().any(|row| row.len() != chunks) {
        return None;
    }
    let aggregates = rows
        .par_iter()
        .map(|row| -> Option<G1> {
            let points = row
                .iter()
                .map(|point| parse_g1_for_ffi(point))
                .collect::<Option<Vec<_>>>()?;
            let mut aggregate = G1::identity();
            for (point, weight) in points.iter().zip(weights.iter()) {
                aggregate += point * weight;
            }
            Some(aggregate)
        })
        .collect::<Vec<_>>();
    aggregates.into_iter().collect::<Option<Vec<_>>>()
}

/// Aggregate both chunk ciphertext families once and optionally compare their
/// pairwise sum with the transcript's explicit derived c_v vector.
#[no_mangle]
pub extern "C" fn pyG1AggregateChunkPairs(
    json_input: *const c_char,
) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let parsed: G1ChunkPairAggregationInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let result = (|| -> Option<G1ChunkPairAggregationOutput> {
        let n = parsed.enc_s_c1.len();
        if n == 0
            || parsed.enc_s_c2.len() != n
            || parsed.enc_hat_c1.len() != n
            || parsed.enc_hat_c2.len() != n
            || parsed
                .claimed_c_v
                .as_ref()
                .map_or(false, |claimed| claimed.len() != n)
        {
            return None;
        }
        let weights = parsed
            .weights
            .iter()
            .map(|weight| parse_scalar_for_ffi(weight))
            .collect::<Option<Vec<_>>>()?;
        let r_s = aggregate_chunk_rows(&parsed.enc_s_c1, &weights)?;
        let d_s = aggregate_chunk_rows(&parsed.enc_s_c2, &weights)?;
        let r_hat = aggregate_chunk_rows(&parsed.enc_hat_c1, &weights)?;
        let d_hat = aggregate_chunk_rows(&parsed.enc_hat_c2, &weights)?;

        let valid = if let Some(claimed) = parsed.claimed_c_v.as_ref() {
            (0..n).into_par_iter().all(|i| {
                let claimed_c1 = parse_g1_for_ffi(&claimed[i].C1);
                let claimed_c2 = parse_g1_for_ffi(&claimed[i].C2);
                match (claimed_c1, claimed_c2) {
                    (Some(c1), Some(c2)) => {
                        r_s[i].clone() + &r_hat[i] == c1
                            && d_s[i].clone() + &d_hat[i] == c2
                    }
                    _ => false,
                }
            })
        } else {
            true
        };

        Some(G1ChunkPairAggregationOutput {
            valid,
            r_s: r_s.iter().map(|point| point.to_hex()).collect(),
            d_s: d_s.iter().map(|point| point.to_hex()).collect(),
            r_hat: r_hat.iter().map(|point| point.to_hex()).collect(),
            d_hat: d_hat.iter().map(|point| point.to_hex()).collect(),
        })
    })()
    .unwrap_or(G1ChunkPairAggregationOutput {
        valid: false,
        r_s: Vec::new(),
        d_s: Vec::new(),
        r_hat: Vec::new(),
        d_hat: Vec::new(),
    });

    CString::new(serde_json::to_string(&result).unwrap())
        .unwrap()
        .into_raw()
}


/// Verify every group equation of one Fiat-Shamir chunk proof in one FFI call.
///
/// The Python verifier derives the transcript challenges exactly as before and
/// folds the repeated challenge powers into one scalar per ciphertext chunk.
/// This function only evaluates the original group equations.  It neither
/// batches them probabilistically nor changes their acceptance condition.
#[no_mangle]
pub extern "C" fn pyG1VerifyChunkingEquations(
    json_input: *const c_char,
) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let parsed: G1ChunkProofVerifyInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let valid = (|| -> Option<bool> {
        let n = parsed.rpks.len();
        if n == 0
            || parsed.B.is_empty()
            || parsed.B.len() != parsed.C.len()
            || parsed.B.len() != parsed.x_powers.len()
            || parsed.D.len() != n
            || parsed.c1.len() != n
            || parsed.c2.len() != n
            || parsed.challenge_weights.len() != n
            || parsed.z_r.len() != n
            || parsed.rpk_weights.len() != n
        {
            return Some(false);
        }
        let chunks = parsed.D[0].len();
        if chunks == 0
            || parsed.D.iter().any(|row| row.len() != chunks)
            || parsed.c1.iter().any(|row| row.len() != chunks)
            || parsed.c2.iter().any(|row| row.len() != chunks)
            || parsed.challenge_weights.iter().any(|row| row.len() != chunks)
            || parsed.z_r.iter().any(|row| row.len() != chunks)
        {
            return Some(false);
        }

        let key_base = parse_g1_for_ffi(&parsed.key_base)?;
        let msg_base = parse_g1_for_ffi(&parsed.msg_base)?;
        let y0 = parse_g1_for_ffi(&parsed.y0)?;
        let d0 = parse_g1_for_ffi(&parsed.D0)?;
        let y = parse_g1_for_ffi(&parsed.Y)?;
        let b_points = parsed
            .B
            .iter()
            .map(|point| parse_g1_for_ffi(point))
            .collect::<Option<Vec<_>>>()?;
        let c_points = parsed
            .C
            .iter()
            .map(|point| parse_g1_for_ffi(point))
            .collect::<Option<Vec<_>>>()?;
        let rpk_points = parsed
            .rpks
            .iter()
            .map(|point| parse_g1_for_ffi(point))
            .collect::<Option<Vec<_>>>()?;

        let parse_point_rows = |rows: &[Vec<String>]| -> Option<Vec<G1>> {
            rows.iter()
                .flat_map(|row| row.iter())
                .map(|point| parse_g1_for_ffi(point))
                .collect::<Option<Vec<_>>>()
        };
        let parse_scalar_rows = |rows: &[Vec<String>]| -> Option<Vec<FieldElement>> {
            rows.iter()
                .flat_map(|row| row.iter())
                .map(|scalar| parse_scalar_for_ffi(scalar))
                .collect::<Option<Vec<_>>>()
        };

        let d_points = parse_point_rows(&parsed.D)?;
        let c1_points = parse_point_rows(&parsed.c1)?;
        let c2_points = parse_point_rows(&parsed.c2)?;
        let challenge_weights = parse_scalar_rows(&parsed.challenge_weights)?;
        let z_r = parse_scalar_rows(&parsed.z_r)?;
        let x_powers = parsed
            .x_powers
            .iter()
            .map(|scalar| parse_scalar_for_ffi(scalar))
            .collect::<Option<Vec<_>>>()?;
        let rpk_weights = parsed
            .rpk_weights
            .iter()
            .map(|scalar| parse_scalar_for_ffi(scalar))
            .collect::<Option<Vec<_>>>()?;
        let z_s_weighted = parse_scalar_for_ffi(&parsed.z_s_weighted)?;
        let z_b = parse_scalar_for_ffi(&parsed.z_b)?;

        // D_ij + C1_ij * sum_k(e_ijk*x^k) == key_base * z_r_ij.
        // Every equation is checked independently; parallel execution does not
        // replace them with a randomized aggregate equation.
        let chunk_equations_hold = (0..n * chunks).into_par_iter().all(|idx| {
            let left = d_points[idx].clone() + &(&c1_points[idx] * &challenge_weights[idx]);
            let right = &key_base * &z_r[idx];
            left == right
        });
        if !chunk_equations_hold {
            return Some(false);
        }

        let left2 = d0 + &parallel_g1_linear_combination(&b_points, &x_powers)?;
        if left2 != &key_base * &z_b {
            return Some(false);
        }

        // Fold repeated appearances of each C2_ij before entering Rust.  This
        // is exactly the same linear combination as the original nested loop.
        let c_sum = parallel_g1_linear_combination(&c_points, &x_powers)?;
        let c2_sum = parallel_g1_linear_combination(&c2_points, &challenge_weights)?;
        let left3 = y + &c_sum + &c2_sum;

        let rpk_sum = parallel_g1_linear_combination(&rpk_points, &rpk_weights)?;
        let right3 = &y0 * &z_b + &(&msg_base * &z_s_weighted) + &rpk_sum;
        Some(left3 == right3)
    })()
    .unwrap_or(false);

    CString::new(
        serde_json::to_string(&G1ChunkProofVerifyOutput { valid }).unwrap(),
    )
    .unwrap()
    .into_raw()
}

#[no_mangle]
pub extern "C" fn pyG1Equal(json_input: *const c_char) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let parsed: G1BinaryInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let left = match parse_g1_for_ffi(&parsed.left) {
        Some(v) => v,
        None => return std::ptr::null(),
    };
    let right = match parse_g1_for_ffi(&parsed.right) {
        Some(v) => v,
        None => return std::ptr::null(),
    };
    let json = serde_json::to_string(&G1EqualOutput { equal: left == right }).unwrap();
    CString::new(json).unwrap().into_raw()
}


// --- VRF prove and verify FFI ---

#[derive(Deserialize)]
pub struct VrfProveInput {
    pub sk: String,       // hex-encoded secret key scalar
    pub msg: String,      // hex-encoded message bytes
    pub g:  String,
}

#[derive(Serialize)]
pub struct VrfProveOutput {
    pub beta: String,     // hex-encoded VRF output point
    pub c: String,        // hex-encoded proof challenge scalar
    pub s: String,        // hex-encoded proof response scalar
    
}

/// pyVrfProve: takes { "sk": "<hex>", "msg": "<hex>" }
/// returns { "beta": "<hex>", "c": "<hex>", "s": "<hex>" } or null on error
#[no_mangle]
pub extern "C" fn pyVrfProve(json_input: *const std::os::raw::c_char) -> *const std::os::raw::c_char {
    let c_str = unsafe { std::ffi::CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let parsed: VrfProveInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    // Decode g and install as global generator
    let g_bytes = hex::decode(parsed.g.clone()).unwrap();
    let g_point = G1::from_bytes(&g_bytes).unwrap();
    crate::vrf::set_generator(&g_point);

    // Decode secret key scalar
    let sk = match FieldElement::from_hex(parsed.sk.clone()) {
        Ok(fe) => fe,
        Err(_) => return std::ptr::null(),
    };
    // let secret = Secret { sk };
    let secret = Secret::from_scalar(sk);


    // Decode message from hex
    let msg_bytes = match hex::decode(parsed.msg.clone()) {
        Ok(bytes) => bytes,
        Err(_) => return std::ptr::null(),
    };

    // Perform VRF prove
    let (beta, proof) = secret.prove(&msg_bytes);

    // Serialize outputs to hex strings
    let beta_hex = beta.to_hex();
    let c_hex = proof.c.to_hex();
    let s_hex = proof.s.to_hex();

    let result = VrfProveOutput {
        beta: beta_hex,
        c: c_hex,
        s: s_hex,
    };

    let json = match serde_json::to_string(&result) {
        Ok(s) => s,
        Err(_) => return std::ptr::null(),
    };
    std::ffi::CString::new(json).unwrap().into_raw()
}

#[derive(Deserialize)]
pub struct VrfVerifyInput {
    pub pk: String,       // hex-encoded public key point
    pub msg: String,      // hex-encoded message bytes
    pub beta: String,     // hex-encoded VRF output point
    pub c: String,        // hex-encoded proof challenge
    pub s: String,        // hex-encoded proof response
    pub g: String,
}

#[derive(Serialize)]
pub struct VrfVerifyOutput {
    pub valid: bool,
}

/// pyVrfVerify: takes { "pk": "<hex>", "msg": "<hex>", "beta": "<hex>", "c": "<hex>", "s": "<hex>" }
/// returns { "valid": true/false } or null on error
#[no_mangle]
pub extern "C" fn pyVrfVerify(json_input: *const std::os::raw::c_char) -> *const std::os::raw::c_char {
    let c_str = unsafe { std::ffi::CStr::from_ptr(json_input) };
    let input = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let parsed: VrfVerifyInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    // Decode g and install as global generator
    let g_bytes = hex::decode(parsed.g.clone()).unwrap();
    let g_point = G1::from_bytes(&g_bytes).unwrap();
    crate::vrf::set_generator(&g_point);

    // Decode public key point
    let pk_bytes = match hex::decode(parsed.pk.clone()) {
        Ok(b) => b,
        Err(_) => return std::ptr::null(),
    };
    let pk_point = match G1::from_bytes(&pk_bytes) {
        Ok(p) => p,
        Err(_) => return std::ptr::null(),
    };
    // let public = Public { pk: pk_point };
    let public = Public::from_point(pk_point);

    // Decode message
    let msg_bytes = match hex::decode(parsed.msg.clone()) {
        Ok(bytes) => bytes,
        Err(_) => return std::ptr::null(),
    };

    // Decode beta point
    let beta_bytes = match hex::decode(parsed.beta.clone()) {
        Ok(b) => b,
        Err(_) => return std::ptr::null(),
    };
    let beta_point = match G1::from_bytes(&beta_bytes) {
        Ok(p) => p,
        Err(_) => return std::ptr::null(),
    };

    // Decode proof scalars
    let c_scalar = match FieldElement::from_hex(parsed.c.clone()) {
        Ok(fe) => fe,
        Err(_) => return std::ptr::null(),
    };
    let s_scalar = match FieldElement::from_hex(parsed.s.clone()) {
        Ok(fe) => fe,
        Err(_) => return std::ptr::null(),
    };
    let proof = Proof { c: c_scalar, s: s_scalar };

    // Verify VRF proof
    let is_valid = public.verify(&msg_bytes, &beta_point, &proof);

    let result = VrfVerifyOutput { valid: is_valid };
    let json = match serde_json::to_string(&result) {
        Ok(s) => s,
        Err(_) => return std::ptr::null(),
    };
    std::ffi::CString::new(json).unwrap().into_raw()
}


// === Poseidon-派生密钥的对称加/解密 =======================================

#[derive(Deserialize)]
pub struct SymEncryptInput {
    pub pk: String,      // hex-encoded G1 公钥
    pub k: String,       // hex-encoded 标量
    pub m: String,       // hex-encoded 标量
    pub m_prime: String, // hex-encoded 标量
}

#[derive(Serialize)]
pub struct SymEncryptOutput {
    pub c: String,       // hex-encoded 标量 = m − K
    pub c_prime: String, // hex-encoded 标量 = m′ − K
}

#[no_mangle]
pub extern "C" fn pySymEncrypt(json_input: *const c_char) -> *const c_char {
    use std::ffi::{CStr, CString};

    // -------- 解析输入 --------------------------------------------------
    let input = match unsafe { CStr::from_ptr(json_input) }.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let parsed: SymEncryptInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let pk = match G1::from_bytes(&hex::decode(&parsed.pk).unwrap()) {
        Ok(p) => p,
        Err(_) => return std::ptr::null(),
    };
    let k_fe        = FieldElement::from_hex(parsed.k).unwrap();
    let m_fe        = FieldElement::from_hex(parsed.m).unwrap();
    let m_prime_fe  = FieldElement::from_hex(parsed.m_prime).unwrap();

    // -------- 计算共享点 pk^k ------------------------------------------
    let shared_pt   = pk.mul(&k_fe);
    let shared_bytes = shared_pt.to_bytes(true);
    let shared_fe   = FieldElement::from_bytes(&shared_bytes[1..]).unwrap();

    // -------- Poseidon 导出密钥 K --------------------------------------
    let (ark, mds) = find_poseidon_ark_and_mds(255, 2, 8, 57, 0);
    let poseidon_cfg = PoseidonConfig::new(8, 57, 5, mds.clone(), ark.clone(), 2, 1);
    let mut sponge = PoseidonSponge {
        parameters: poseidon_cfg,
        state: vec![FieldElement::zero(); 3],
        mode: DuplexSpongeMode::Absorbing { next_absorb_index: 0 },
    };
    sponge.absorb(0, &[shared_fe.clone()]);
    sponge.permute();
    let mut out = vec![FieldElement::zero(); 1];
    sponge.squeeze(0, &mut out);
    let K = out[0].clone();

    // -------- 输出密文 --------------------------------------------------
    let c        = m_fe        - &K;
    let c_prime  = m_prime_fe  - &K;

    let result = SymEncryptOutput { c: c.to_hex(), c_prime: c_prime.to_hex() };
    let json   = serde_json::to_string(&result).unwrap();
    CString::new(json).unwrap().into_raw()
}

#[derive(Deserialize)]
pub struct SymDecryptInput {
    pub gk: String,      // hex-encoded G1 点 = g^k
    pub sk: String,      // hex-encoded 标量（secret key）
    pub c: String,       // hex-encoded 标量
    pub c_prime: String, // hex-encoded 标量
}

#[derive(Serialize)]
pub struct SymDecryptOutput {
    pub m: String,       // hex-encoded 标量
    pub m_prime: String, // hex-encoded 标量
}

#[no_mangle]
pub extern "C" fn pySymDecrypt(json_input: *const c_char) -> *const c_char {
    use std::ffi::{CStr, CString};

    // -------- 解析输入 --------------------------------------------------
    let input = match unsafe { CStr::from_ptr(json_input) }.to_str() {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };
    let parsed: SymDecryptInput = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return std::ptr::null(),
    };

    let gk = match G1::from_bytes(&hex::decode(&parsed.gk).unwrap()) {
        Ok(p) => p,
        Err(_) => return std::ptr::null(),
    };
    let sk_fe       = FieldElement::from_hex(parsed.sk).unwrap();
    let c_fe        = FieldElement::from_hex(parsed.c).unwrap();
    let c_prime_fe  = FieldElement::from_hex(parsed.c_prime).unwrap();

    // -------- 计算共享点 (g^k)^sk --------------------------------------
    let shared_pt   = gk.mul(&sk_fe);
    let shared_bytes = shared_pt.to_bytes(true);
    let shared_fe   = FieldElement::from_bytes(&shared_bytes[1..]).unwrap();

    // -------- Poseidon 导出密钥 K --------------------------------------
    let (ark, mds) = find_poseidon_ark_and_mds(255, 2, 8, 57, 0);
    let poseidon_cfg = PoseidonConfig::new(8, 57, 5, mds.clone(), ark.clone(), 2, 1);
    let mut sponge = PoseidonSponge {
        parameters: poseidon_cfg,
        state: vec![FieldElement::zero(); 3],
        mode: DuplexSpongeMode::Absorbing { next_absorb_index: 0 },
    };
    sponge.absorb(0, &[shared_fe.clone()]);
    sponge.permute();
    let mut out = vec![FieldElement::zero(); 1];
    sponge.squeeze(0, &mut out);
    let K = out[0].clone();

    // -------- 还原明文 --------------------------------------------------
    let m        = c_fe       + &K;
    let m_prime  = c_prime_fe + &K;

    let result = SymDecryptOutput { m: m.to_hex(), m_prime: m_prime.to_hex() };
    let json   = serde_json::to_string(&result).unwrap();
    CString::new(json).unwrap().into_raw()
}
// ======================================================================
