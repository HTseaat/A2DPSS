"""PVSS primitive layer for YOSO DPSS.

This module combines Pedersen/Shamir helpers and pair-output VE into the
Algorithm-1 shaped PVSS surface used by the DPSS justifier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from beaver.chunk_elgamal import (
    combine_elgamal_ciphertexts,
    decrypt_elgamal_ciphertext,
    normalize_elgamal_ciphertext,
)
from beaver.g1_backend import G1Point, g1_linear_combination
from beaver.pedersen_shamir import (
    batch_pedersen_commit,
    eval_poly,
    interpolate_at_zero,
    lagrange_coeff_at_zero,
    pedersen_deg_check,
    sample_poly,
)
from beaver.ve_proof import (
    derive_c_v_vector,
    ve_benc_prove,
    ve_bverify,
    ve_dec,
)


LEGACY_TRANSCRIPT_VERSION = 1
TRANSCRIPT_VERSION = 2
SUPPORTED_TRANSCRIPT_VERSIONS = frozenset(
    {LEGACY_TRANSCRIPT_VERSION, TRANSCRIPT_VERSION}
)


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _normalize_point(value: str) -> str:
    return G1Point.from_hex(value).to_hex()


@dataclass(frozen=True)
class PVSSTranscript:
    dealer_id: int
    dealer_eval_point: int
    commitment_vector: list[str]
    enc_results: list[dict]
    pi_ve: dict
    derived_c_v_vector: list[dict] | None = None
    version: int = LEGACY_TRANSCRIPT_VERSION

    def to_obj(self) -> dict:
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("PVSS transcript version must be an integer")
        version = self.version
        if version not in SUPPORTED_TRANSCRIPT_VERSIONS:
            raise ValueError("unsupported PVSS transcript version")
        obj = {
            "version": version,
            "dealer_id": int(self.dealer_id),
            "dealer_eval_point": int(self.dealer_eval_point),
            "commitment_vector": list(self.commitment_vector),
            "enc_results": self.enc_results,
            "pi_ve": self.pi_ve,
        }
        if version == TRANSCRIPT_VERSION:
            if self.derived_c_v_vector is None:
                raise ValueError("PVSS transcript v2 requires derived_c_v_vector")
            obj["derived_c_v_vector"] = [
                normalize_elgamal_ciphertext(ciphertext)
                for ciphertext in self.derived_c_v_vector
            ]
        elif self.derived_c_v_vector is not None:
            raise ValueError(
                "legacy PVSS transcript cannot carry derived_c_v_vector"
            )
        return obj

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_obj())

    @staticmethod
    def from_bytes(data: bytes | bytearray | str) -> "PVSSTranscript":
        if isinstance(data, str):
            data = data.encode("utf-8")
        obj = json.loads(
            bytes(data).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(obj, dict):
            raise ValueError("PVSS transcript must encode a JSON object")
        version = obj.get("version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError("PVSS transcript version must be an integer")
        if version not in SUPPORTED_TRANSCRIPT_VERSIONS:
            raise ValueError("unsupported PVSS transcript version")
        expected_fields = {
            "version",
            "dealer_id",
            "dealer_eval_point",
            "commitment_vector",
            "enc_results",
            "pi_ve",
        }
        if version == TRANSCRIPT_VERSION:
            expected_fields.add("derived_c_v_vector")
        missing = expected_fields - set(obj)
        extra = set(obj) - expected_fields
        if missing:
            raise ValueError(
                f"PVSS transcript is missing fields: {sorted(missing)}"
            )
        if extra:
            raise ValueError(
                f"PVSS transcript has unknown fields: {sorted(extra)}"
            )
        return PVSSTranscript(
            dealer_id=int(obj["dealer_id"]),
            dealer_eval_point=int(obj["dealer_eval_point"]),
            commitment_vector=list(obj["commitment_vector"]),
            enc_results=list(obj["enc_results"]),
            pi_ve=dict(obj["pi_ve"]),
            derived_c_v_vector=(
                [
                    normalize_elgamal_ciphertext(ciphertext)
                    for ciphertext in obj["derived_c_v_vector"]
                ]
                if version == TRANSCRIPT_VERSION
                else None
            ),
            version=version,
        )


def pvss_deal(
    g_hex: str,
    h_hex: str,
    rpk_list: Sequence[str],
    s: int,
    s_hat: int,
    t: int,
    dealer_id: int,
    dealer_eval_point: int,
    key_base_hex: str | None = None,
) -> PVSSTranscript:
    """Create a PVSS transcript for one dealer."""

    n = len(rpk_list)
    if n == 0:
        raise ValueError("rpk_list must be non-empty")
    if t < 0:
        raise ValueError("t must be non-negative")
    if n < t + 1:
        raise ValueError("need at least t+1 recipients")

    phi = sample_poly(s, t)
    phi_hat = sample_poly(s_hat, t)
    eval_points = list(range(1, n + 1))
    shares = [eval_poly(phi, point) for point in eval_points]
    hat_shares = [eval_poly(phi_hat, point) for point in eval_points]
    commitment_vector = batch_pedersen_commit(
        g_hex,
        h_hex,
        [s] + shares,
        [s_hat] + hat_shares,
    )
    share_pairs = list(zip(shares, hat_shares))
    enc_results, pi_ve = ve_benc_prove(
        g_hex, h_hex, rpk_list, share_pairs, commitment_vector, key_base_hex=key_base_hex
    )
    derived_c_v_vector = derive_c_v_vector(enc_results)
    return PVSSTranscript(
        dealer_id=int(dealer_id),
        dealer_eval_point=int(dealer_eval_point),
        commitment_vector=commitment_vector,
        enc_results=enc_results,
        pi_ve=pi_ve,
        derived_c_v_vector=derived_c_v_vector,
        version=TRANSCRIPT_VERSION,
    )


def pvss_verify(
    g_hex: str,
    h_hex: str,
    rpk_list: Sequence[str],
    transcript: PVSSTranscript,
    t: int,
    key_base_hex: str | None = None,
) -> bool:
    """Verify transcript internal consistency, excluding transfer anchor check."""

    try:
        n = len(rpk_list)
        if n == 0 or t < 0 or n < t + 1:
            return False
        if len(transcript.commitment_vector) != n + 1:
            return False
        if len(transcript.enc_results) != n:
            return False
        if (
            isinstance(transcript.version, bool)
            or not isinstance(transcript.version, int)
            or transcript.version not in SUPPORTED_TRANSCRIPT_VERSIONS
        ):
            return False
        claimed_c_v_vector = None
        if transcript.version == TRANSCRIPT_VERSION:
            if transcript.derived_c_v_vector is None:
                return False
            if len(transcript.derived_c_v_vector) != n:
                return False
            claimed_c_v_vector = transcript.derived_c_v_vector
        elif transcript.derived_c_v_vector is not None:
            return False
        if not pedersen_deg_check(
            transcript.commitment_vector,
            t,
            eval_points=list(range(0, n + 1)),
        ):
            return False
        return ve_bverify(
            g_hex,
            h_hex,
            rpk_list,
            transcript.commitment_vector,
            transcript.enc_results,
            transcript.pi_ve,
            key_base_hex=key_base_hex,
            _claimed_c_v_vector=claimed_c_v_vector,
        )
    except (KeyError, TypeError, ValueError):
        return False


def pvss_verify_anchor(
    transcript: PVSSTranscript,
    old_commitment_vector: Sequence[str],
) -> bool:
    """Verify transcript constant commitment against the old sharing anchor."""

    try:
        expected = old_commitment_vector[int(transcript.dealer_eval_point)]
        return G1Point.from_hex(transcript.commitment_vector[0]) == G1Point.from_hex(expected)
    except (IndexError, TypeError, ValueError):
        return False


def pvss_deliver(
    g_hex: str,
    h_hex: str,
    transcript: PVSSTranscript,
    dk: int,
    role_idx: int,
) -> tuple[int, int]:
    """Decrypt the share pair for ``role_idx``."""

    role_idx = int(role_idx)
    if not 0 <= role_idx < len(transcript.enc_results):
        raise IndexError("role_idx out of range")
    return ve_dec(g_hex, h_hex, dk, transcript.enc_results[role_idx])


def pvss_decrypt_c_v(
    transcript: PVSSTranscript,
    dk: int,
    role_idx: int,
) -> str:
    """Decrypt one v2 transcript's explicit ``c_v`` to a G1 commitment."""

    if transcript.version != TRANSCRIPT_VERSION:
        raise ValueError("explicit c_v requires a PVSS transcript v2")
    if transcript.derived_c_v_vector is None:
        raise ValueError("PVSS transcript has no derived_c_v_vector")
    role_idx = int(role_idx)
    if not 0 <= role_idx < len(transcript.derived_c_v_vector):
        raise IndexError("role_idx out of range")
    return decrypt_elgamal_ciphertext(
        dk,
        transcript.derived_c_v_vector[role_idx],
    )


def pvss_interpolate(
    dealer_eval_points: Sequence[int],
    share_pairs: Sequence[tuple[int, int]],
    t: int,
) -> tuple[int, int]:
    """Recover the constant pair from at least ``t+1`` dealer shares."""

    if t < 0:
        raise ValueError("t must be non-negative")
    if len(dealer_eval_points) != len(share_pairs):
        raise ValueError("dealer_eval_points and share_pairs must have the same length")
    if len(share_pairs) < t + 1:
        raise ValueError("need at least t+1 share pairs to interpolate")

    points = list(dealer_eval_points[: t + 1])
    s_values = [pair[0] for pair in share_pairs[: t + 1]]
    hat_values = [pair[1] for pair in share_pairs[: t + 1]]
    return (
        interpolate_at_zero(points, s_values),
        interpolate_at_zero(points, hat_values),
    )


def interpolate_c_v_vector(
    dealer_eval_points: Sequence[int],
    c_v_vectors: Sequence[Sequence[dict]],
    t: int,
) -> list[dict]:
    """Combine explicit commitment ciphertext vectors at interpolation point 0."""

    if t < 0:
        raise ValueError("t must be non-negative")
    if len(dealer_eval_points) != len(c_v_vectors):
        raise ValueError(
            "dealer_eval_points and c_v_vectors must have the same length"
        )
    if len(c_v_vectors) < t + 1:
        raise ValueError("need at least t+1 c_v vectors to interpolate")

    points = [int(point) for point in dealer_eval_points[: t + 1]]
    if len(set(points)) != len(points):
        raise ValueError("dealer evaluation points must be distinct")
    vectors = [list(vector) for vector in c_v_vectors[: t + 1]]
    width = len(vectors[0])
    if width == 0 or any(len(vector) != width for vector in vectors):
        raise ValueError("c_v vectors must be non-empty and equal length")

    coefficients = [
        lagrange_coeff_at_zero(points, point)
        for point in points
    ]
    return [
        combine_elgamal_ciphertexts(
            [vector[recipient] for vector in vectors],
            coefficients,
        )
        for recipient in range(width)
    ]


def interpolate_commitment_vector(
    dealer_eval_points: Sequence[int],
    commitment_vectors: Sequence[Sequence[str]],
    t: int,
) -> list[str]:
    """Derive successor commitments from one fixed resharing transcript set.

    Algorithm 4 A²DPSS uses this primitive to derive ``vc'`` from its fixed
    deterministic set ``T``.  Algorithm 5 OCDPSS composes that result and uses
    the same ``T`` for operation-share combination.  The primitive was also
    used by the older draft's resharing-only compatibility path.

    Each transcript ``j`` carries ``commitment_vector_j[m] = g^{phi_j(m)} h^{phi_hat_j(m)}``
    where ``phi_j(0)`` is old holder ``j``'s share. The successor polynomial is
    ``phi'(x) = sum_j lambda_j * phi_j(x)`` with ``lambda_j`` the Lagrange-at-0
    coefficients for the old holders' eval points -- exactly the coefficients
    ``pvss_interpolate`` uses to recover shares. By Pedersen homomorphism the
    successor commitment vector is the same Lagrange combination *in the
    exponent*::

        v_{s,u'}[m] = sum_j lambda_j * commitment_vector_j[m]   (on G1)

    The result anchors the next handoff: ``v_{s,u'}[e_i]`` equals the Pedersen
    commitment to successor role ``i``'s reconstructed share, so the next
    transcript's :func:`pvss_verify_anchor` will pass. The same ``t+1`` dealer
    set must be used here as in :func:`pvss_interpolate`.
    """

    if t < 0:
        raise ValueError("t must be non-negative")
    if len(dealer_eval_points) != len(commitment_vectors):
        raise ValueError(
            "dealer_eval_points and commitment_vectors must have the same length"
        )
    if len(commitment_vectors) < t + 1:
        raise ValueError("need at least t+1 transcripts to interpolate")

    points = [int(p) for p in dealer_eval_points[: t + 1]]
    vectors = [list(v) for v in commitment_vectors[: t + 1]]
    width = len(vectors[0])
    if width == 0 or any(len(v) != width for v in vectors):
        raise ValueError("commitment vectors must be non-empty and equal length")

    coeffs = [lagrange_coeff_at_zero(points, x_i) for x_i in points]
    return [
        g1_linear_combination([vec[m] for vec in vectors], coeffs).to_hex()
        for m in range(width)
    ]
