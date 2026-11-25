"""beaver/vrf.py — VRF utility functions for ATCaaS committee sortition.

Provides the hash-to-integer step and threshold comparison that are absent
from the Rust FFI layer (pyVrfProve / pyVrfVerify only handle the EC-VRF
cryptographic proof; thresholding must be done in Python).

These are pure functions with no state. Import them wherever VRF-based
committee selection is needed (YosoRBC, DagTob, role assignment, …).
"""

import hashlib
import json
from ctypes import CDLL, c_char_p, c_void_p, string_at
from typing import Optional, Sequence, Tuple

from Crypto.Util.number import long_to_bytes

MAX = 1 << 256  # SHA256 output space: r ∈ [0, MAX)

BERNOULLI_COMMITTEE_MODE = "bernoulli"
FIXED_SIZE_COMMITTEE_MODE = "fixed-size"
COMMITTEE_SELECTION_MODES = frozenset(
    {BERNOULLI_COMMITTEE_MODE, FIXED_SIZE_COMMITTEE_MODE}
)


try:
    lib_bulletproof = CDLL("./libbulletproofs_amcl.so")
    lib_bulletproof.pyVrfProve.argtypes = [c_char_p]
    lib_bulletproof.pyVrfProve.restype = c_void_p
    lib_bulletproof.pyVrfVerify.argtypes = [c_char_p]
    lib_bulletproof.pyVrfVerify.restype = c_void_p
    lib_bulletproof.pyFreeString.argtypes = [c_void_p]
    lib_bulletproof.pyFreeString.restype = None
except OSError:
    lib_bulletproof = None


def vrf_backend_available() -> bool:
    """Return whether the cryptographic VRF FFI is available."""

    return lib_bulletproof is not None


def _beta_to_canonical_bytes(beta_str: str) -> bytes:
    """Convert pyVrfProve's '(X_hex,Y_hex)' → uncompressed 04||X||Y bytes.

    pyVrfProve (Rust lib.rs) serialises the G1 output point via G1::to_hex(),
    which produces the string "(X_hex,Y_hex)".
    pyVrfVerify expects G1::from_bytes(), which parses the 04||X||Y encoding.

    This function bridges that asymmetry and also provides the canonical byte
    string that is hashed to obtain the integer VRF output r = SHA256(β).
    Both prover and verifier call this function so they always hash the same
    byte string and obtain the same r.
    """
    x_hex, y_hex = beta_str.strip("()").split(",")
    return b'\x04' + bytes.fromhex(x_hex) + bytes.fromhex(y_hex)


def _r_from_beta(beta_str: str) -> int:
    """VRF output integer: r = SHA256(canonical_bytes(β)) ∈ [0, MAX).

    Algorand §4.2 / ATCaaS: the raw VRF output is a curve point β; the
    integer r used for threshold comparison is derived by hashing β.
    """
    return int.from_bytes(
        hashlib.sha256(_beta_to_canonical_bytes(beta_str)).digest(), "big"
    )


def _passes(r: int, threshold_fraction: float) -> bool:
    """Return True iff r ≤ threshold_fraction * MAX.

    Corresponds to Algorand Algorithm 1 line 3:
        if r ≤ p · MAX then b ← 1

    For ATCaaS ordered-agreement committees: threshold_fraction = n / M
    (expected committee size n out of M long-lived parties).

    threshold_fraction = 1.0 means every node is always selected, which
    is the correct behaviour for the legacy M == n demo case (backward
    compatibility: existing callers that do not pass threshold_fraction
    get the default 1.0 and see no change in behaviour).
    """
    return r <= int(threshold_fraction * MAX)


def normalize_committee_selection_mode(mode: str) -> str:
    """Return the canonical experiment committee-selection mode."""

    normalized = str(mode).strip().lower()
    aliases = {
        "fixed": FIXED_SIZE_COMMITTEE_MODE,
        "fixed_size": FIXED_SIZE_COMMITTEE_MODE,
        "bernoulli-vrf": BERNOULLI_COMMITTEE_MODE,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in COMMITTEE_SELECTION_MODES:
        raise ValueError(
            "committee selection mode must be one of "
            f"{sorted(COMMITTEE_SELECTION_MODES)}"
        )
    return normalized


def fixed_size_committee(
    *,
    label: bytes,
    population_size: int,
    committee_size: int,
    seed: str = "",
) -> Tuple[int, ...]:
    """Derive an exact-size public role-assignment fixture.

    This helper is intentionally an *experiment fixture*, not a realization
    of anonymous VRF self-selection.  It ranks every long-lived role using
    public deterministic material and returns exactly ``committee_size``
    roles.  Selected roles still attach cryptographic VRF proofs when they
    speak, but the fixture (and therefore membership) is public in advance.

    The returned order is the canonical role namespace used by common-coin
    leader selection; it is ordered by fixture score rather than node id.
    """

    if isinstance(label, str):
        label = label.encode("utf-8")
    label = bytes(label)
    population_size = int(population_size)
    committee_size = int(committee_size)
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if not 1 <= committee_size <= population_size:
        raise ValueError(
            "committee_size must satisfy 1 <= value <= population_size"
        )
    seed_bytes = str(seed).encode("utf-8")
    domain = b"ATCAAS:FIXED-SIZE-COMMITTEE:v1"
    ranked = []
    for role in range(population_size):
        material = b"|".join(
            (
                domain,
                seed_bytes,
                label,
                int(role).to_bytes(8, byteorder="big", signed=False),
            )
        )
        ranked.append((hashlib.sha256(material).digest(), role))
    ranked.sort()
    return tuple(role for _, role in ranked[:committee_size])


def generator_hex_from_srs(srs: dict) -> str:
    """Return the uncompressed G1 generator encoding expected by the VRF FFI."""
    pk_dict = json.loads(srs["Pk"].decode("utf-8"))
    g0 = pk_dict["G1_g"][0]
    gx = long_to_bytes(int(g0["X"]), 48)
    gy = long_to_bytes(int(g0["Y"]), 48)
    return (b"\x04" + gx + gy).hex()


def vrf_prove_sortition(
    *,
    label: bytes,
    vrf_sk,
    g_hex: str,
    threshold_fraction: float,
) -> Tuple[bool, Optional[dict]]:
    """Produce a VRF proof for ``label`` and apply Bernoulli sortition."""
    if lib_bulletproof is None:
        return True, {}
    sk_hex = hex(int(vrf_sk))[2:]
    payload = {
        "sk": sk_hex,
        "msg": label.hex(),
        "g": g_hex,
    }
    ptr = lib_bulletproof.pyVrfProve(json.dumps(payload).encode("utf-8"))
    try:
        proof = json.loads(string_at(ptr).decode("utf-8"))
    finally:
        lib_bulletproof.pyFreeString(ptr)
    selected = _passes(_r_from_beta(proof["beta"]), threshold_fraction)
    if not selected:
        return False, None
    return True, proof


def vrf_verify_sortition(
    *,
    label: bytes,
    proof: dict,
    sender_id: int,
    vrf_pks: Sequence,
    g_hex: str,
    threshold_fraction: float,
) -> bool:
    """Verify a VRF proof and the same Bernoulli threshold check."""
    if lib_bulletproof is None:
        return True
    if not proof:
        return False

    pk_entry = vrf_pks[sender_id]
    if isinstance(pk_entry, dict):
        gx_bytes = long_to_bytes(int(pk_entry["X"]), 48)
        gy_bytes = long_to_bytes(int(pk_entry["Y"]), 48)
        pk_hex = (b"\x04" + gx_bytes + gy_bytes).hex()
    else:
        pk_hex = pk_entry

    payload = {
        "pk": pk_hex,
        "msg": label.hex(),
        "beta": _beta_to_canonical_bytes(proof["beta"]).hex(),
        "c": proof["c"],
        "s": proof["s"],
        "g": g_hex,
    }
    ptr = lib_bulletproof.pyVrfVerify(json.dumps(payload).encode("utf-8"))
    try:
        raw = string_at(ptr).decode("utf-8")
    finally:
        lib_bulletproof.pyFreeString(ptr)

    try:
        res = json.loads(raw)
        valid = bool(res.get("valid", res.get("ok", False)))
    except Exception:
        valid = raw.strip().lower() in ("true", "1")
    if not valid:
        return False
    return _passes(_r_from_beta(proof["beta"]), threshold_fraction)
