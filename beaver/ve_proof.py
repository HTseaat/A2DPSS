"""Pair-output verifiable encryption backend for YOSO DPSS PVSS."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from beaver.chunk_elgamal import (
    NUM_CHUNKS,
    aggregate_chunk_ciphertext,
    build_lookup_table,
    chunk_weights,
    combine_elgamal_ciphertexts,
    dec_chunks,
    enc_chunks,
)
from beaver.chunk_proof import prove_chunking, verify_chunking
from beaver.g1_backend import (
    G1Point,
    field_add,
    field_mul,
    field_normalize,
    g1_aggregate_chunk_pairs,
    g1_linear_combination,
)
from beaver.sigma import hash_to_scalar, prove_linear_relations, verify_linear_relations


PROOF_VERSION = 1


def _normalize_point(value: str) -> str:
    return G1Point.from_hex(value).to_hex()


def _normalize_rpks(rpk_list: Sequence[str]) -> list[str]:
    return [_normalize_point(rpk) for rpk in rpk_list]


def _recipient_commitments(commitment_vector: Sequence[str], n: int) -> list[str]:
    if len(commitment_vector) == n:
        return [_normalize_point(commitment) for commitment in commitment_vector]
    if len(commitment_vector) == n + 1:
        return [_normalize_point(commitment) for commitment in commitment_vector[1:]]
    raise ValueError("commitment_vector must have length n or n+1")


def _normalize_commitment_vector(commitment_vector: Sequence[str]) -> list[str]:
    return [_normalize_point(commitment) for commitment in commitment_vector]


def _chunk_weights() -> list[int]:
    """Compatibility wrapper for the now-public chunk weight primitive."""

    return list(chunk_weights())


def _extract_enc_lists(enc_results: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    enc_s = []
    enc_hat = []
    for idx, entry in enumerate(enc_results):
        if entry.get("recipient", idx) != idx:
            raise ValueError("enc_results recipient indexes must be dense and ordered")
        enc_s.append(entry["enc_s"])
        enc_hat.append(entry["enc_hat_s"])
    return enc_s, enc_hat


def _normalize_enc_results(enc_results: Sequence[dict], rpks: Sequence[str]) -> list[dict]:
    normalized = []
    for idx, entry in enumerate(enc_results):
        rpk = _normalize_point(entry.get("rpk", rpks[idx]))
        if rpk != rpks[idx]:
            raise ValueError(f"enc_result rpk mismatch at recipient {idx}")
        enc_s = entry["enc_s"]
        enc_hat = entry["enc_hat_s"]
        normalized.append(
            {
                "recipient": int(entry.get("recipient", idx)),
                "rpk": rpk,
                "enc_s": enc_s,
                "enc_hat_s": enc_hat,
            }
        )
    _extract_enc_lists(normalized)
    return normalized


def _public_statement(
    g_hex: str,
    h_hex: str,
    rpks: Sequence[str],
    commitment_vector: Sequence[str],
    enc_results: Sequence[dict],
) -> dict:
    return {
        "version": PROOF_VERSION,
        "g": _normalize_point(g_hex),
        "h": _normalize_point(h_hex),
        "rpks": list(rpks),
        "commitment_vector": list(commitment_vector),
        "enc_results": enc_results,
    }


def _batch_scalars(statement: dict, n: int) -> list[int]:
    x = hash_to_scalar("ve-proof-batch-x", statement, nonzero=True)
    scalars = []
    current = x
    for _ in range(n):
        scalars.append(current)
        current = field_mul(current, x)
    return scalars


def _aggregate_chunks(ciphertext: dict) -> tuple[G1Point, G1Point]:
    """Compatibility wrapper returning points instead of a wire object."""

    aggregate = aggregate_chunk_ciphertext(ciphertext)
    return (
        G1Point.from_hex(aggregate["C1"]),
        G1Point.from_hex(aggregate["C2"]),
    )


def derive_c_v(enc_result_for_i: dict) -> dict:
    """Derive the complete encrypted Pedersen commitment for one recipient.

    ``enc_s`` encrypts ``g^s`` and ``enc_hat_s`` encrypts ``h^hat_s`` under
    the same recipient key and key base.  Combining their aggregate
    ciphertexts therefore encrypts ``g^s h^hat_s`` without introducing a
    second message or independent encryption randomness.
    """

    if not isinstance(enc_result_for_i, dict):
        raise TypeError("VE recipient result must be an object")
    try:
        enc_s = enc_result_for_i["enc_s"]
        enc_hat = enc_result_for_i["enc_hat_s"]
    except KeyError as exc:
        raise ValueError("VE recipient result is missing pair ciphertexts") from exc

    recipient = int(enc_result_for_i.get("recipient", enc_s.get("recipient", 0)))
    if int(enc_s.get("recipient", recipient)) != recipient:
        raise ValueError("enc_s recipient does not match VE recipient")
    if int(enc_hat.get("recipient", recipient)) != recipient:
        raise ValueError("enc_hat_s recipient does not match VE recipient")

    outer_rpk = enc_result_for_i.get("rpk")
    inner_rpks = [enc_s.get("rpk"), enc_hat.get("rpk")]
    known_rpks = [rpk for rpk in [outer_rpk, *inner_rpks] if rpk is not None]
    if known_rpks:
        normalized_rpks = [_normalize_point(rpk) for rpk in known_rpks]
        if len(set(normalized_rpks)) != 1:
            raise ValueError("pair ciphertexts do not use the same recipient key")

    return combine_elgamal_ciphertexts(
        [
            aggregate_chunk_ciphertext(enc_s),
            aggregate_chunk_ciphertext(enc_hat),
        ],
        [1, 1],
    )


def derive_c_v_vector(enc_results: Sequence[dict]) -> list[dict]:
    """Derive one explicit commitment ciphertext per dense recipient."""

    derived = []
    for idx, enc_result in enumerate(enc_results):
        if int(enc_result.get("recipient", idx)) != idx:
            raise ValueError("enc_results recipient indexes must be dense and ordered")
        derived.append(derive_c_v(enc_result))
    return derived


def _aggregate_chunk_pairs(
    enc_results: Sequence[dict],
    claimed_c_v_vector: Sequence[dict] | None = None,
) -> dict | None:
    """Aggregate paired chunks once in Rust and optionally validate c_v."""

    try:
        enc_s, enc_hat = _extract_enc_lists(enc_results)
        if claimed_c_v_vector is not None:
            if len(claimed_c_v_vector) != len(enc_results):
                return None
            for ciphertext in claimed_c_v_vector:
                if (
                    not isinstance(ciphertext, dict)
                    or set(ciphertext) != {"C1", "C2"}
                ):
                    return None

        def coordinates(ciphertexts, coordinate):
            return [
                [chunk[coordinate] for chunk in ciphertext["chunks"]]
                for ciphertext in ciphertexts
            ]

        result = g1_aggregate_chunk_pairs(
            {
                "weights": [format(weight, "x") for weight in _chunk_weights()],
                "enc_s_c1": coordinates(enc_s, "C1"),
                "enc_s_c2": coordinates(enc_s, "C2"),
                "enc_hat_c1": coordinates(enc_hat, "C1"),
                "enc_hat_c2": coordinates(enc_hat, "C2"),
                "claimed_c_v": (
                    list(claimed_c_v_vector)
                    if claimed_c_v_vector is not None
                    else None
                ),
            }
        )
        if not result["valid"]:
            return None
        n = len(enc_results)
        if any(
            not isinstance(result[name], list) or len(result[name]) != n
            for name in ("r_s", "d_s", "r_hat", "d_hat")
        ):
            return None
        return result
    except (KeyError, TypeError, ValueError):
        return None


def verify_c_v_vector_and_collect(
    enc_results: Sequence[dict],
    claimed_c_v_vector: Sequence[dict],
) -> dict | None:
    """Verify c_v and retain the aggregates needed by VE verification."""

    return _aggregate_chunk_pairs(enc_results, claimed_c_v_vector)


def verify_c_v_vector(
    enc_results: Sequence[dict],
    claimed_c_v_vector: Sequence[dict],
) -> bool:
    """Check that every explicit ``c_v`` is exactly derived from its chunks."""

    try:
        return (
            verify_c_v_vector_and_collect(enc_results, claimed_c_v_vector)
            is not None
        )
    except (KeyError, TypeError, ValueError):
        return False


def _aggregate_randomness(witness: dict) -> int:
    randomness = witness["randomness"]
    if len(randomness) != NUM_CHUNKS:
        raise ValueError(f"expected {NUM_CHUNKS} randomness values")
    total = 0
    for rho, weight in zip(randomness, _chunk_weights()):
        total = field_add(total, field_mul(rho, weight))
    return total


def _weighted_sum(values: Sequence[int], weights: Sequence[int]) -> int:
    total = 0
    for value, weight in zip(values, weights):
        total = field_add(total, field_mul(value, weight))
    return total


def _share_relations(
    g_hex: str,
    h_hex: str,
    rpks: Sequence[str],
    recipient_commitments: Sequence[str],
    enc_s: Sequence[dict],
    enc_hat: Sequence[dict],
    alphas: Sequence[int],
    key_base_hex: str | None = None,
    chunk_aggregates: dict | None = None,
) -> tuple[dict, list[dict]]:
    g = _normalize_point(g_hex)
    h = _normalize_point(h_hex)
    # ElGamal randomness base: C1 = key_base^rho (= g, unless re-based to C — P4 §5.1)
    key_base = _normalize_point(key_base_hex) if key_base_hex is not None else g
    if chunk_aggregates is None:
        r_s_points = []
        d_s_points = []
        r_hat_points = []
        d_hat_points = []
        for enc_s_i, enc_hat_i in zip(enc_s, enc_hat):
            r_i, d_i = _aggregate_chunks(enc_s_i)
            r_h_i, d_h_i = _aggregate_chunks(enc_hat_i)
            r_s_points.append(r_i)
            d_s_points.append(d_i)
            r_hat_points.append(r_h_i)
            d_hat_points.append(d_h_i)
        r_s = [point.to_hex() for point in r_s_points]
        d_s = [point.to_hex() for point in d_s_points]
        r_hat = [point.to_hex() for point in r_hat_points]
        d_hat = [point.to_hex() for point in d_hat_points]
    else:
        r_s = list(chunk_aggregates["r_s"])
        d_s = list(chunk_aggregates["d_s"])
        r_hat = list(chunk_aggregates["r_hat"])
        d_hat = list(chunk_aggregates["d_hat"])
        if any(len(points) != len(rpks) for points in (r_s, d_s, r_hat, d_hat)):
            raise ValueError("chunk aggregate length does not match recipients")

    v_agg = g1_linear_combination(recipient_commitments, alphas).to_hex()
    d_s_agg = g1_linear_combination(d_s, alphas).to_hex()
    d_hat_agg = g1_linear_combination(d_hat, alphas).to_hex()

    statement = {
        "V": v_agg,
        "D_s": d_s_agg,
        "D_hat": d_hat_agg,
        "R_s": r_s,
        "R_hat": r_hat,
    }
    relations = [
        {
            "name": "commitment-batch",
            "target": v_agg,
            "terms": [
                {"secret": "S", "base": g},
                {"secret": "H", "base": h},
            ],
        },
        {
            "name": "enc-s-batch",
            "target": d_s_agg,
            "terms": [{"secret": "S", "base": g}]
            + [
                {"secret": f"rho_s_{i}", "base": rpks[i], "coeff": alphas[i]}
                for i in range(len(rpks))
            ],
        },
        {
            "name": "enc-hat-batch",
            "target": d_hat_agg,
            "terms": [{"secret": "H", "base": h}]
            + [
                {"secret": f"rho_hat_{i}", "base": rpks[i], "coeff": alphas[i]}
                for i in range(len(rpks))
            ],
        },
    ]
    for i in range(len(rpks)):
        relations.append(
            {
                "name": f"enc-s-randomness-{i}",
                "target": r_s[i],
                "terms": [{"secret": f"rho_s_{i}", "base": key_base}],
            }
        )
        relations.append(
            {
                "name": f"enc-hat-randomness-{i}",
                "target": r_hat[i],
                "terms": [{"secret": f"rho_hat_{i}", "base": key_base}],
            }
        )
    return statement, relations


def ve_benc_prove(
    g_hex: str,
    h_hex: str,
    rpk_list: Sequence[str],
    share_pairs: Sequence[tuple[int, int]],
    commitment_vector: Sequence[str],
    key_base_hex: str | None = None,
) -> tuple[list[dict], dict]:
    """Return pair-output VE ciphertexts and proof.

    ``key_base_hex`` is the ElGamal key base — the base under which the recipient
    role pubkeys are formed (``rpk = key_base^rsk``). Defaults to ``g``; pass the
    re-base ``C`` for RA role pubkeys ``rpk = C^{x_i}`` (P4 §5.1). The message
    bases (g for s, h for ŝ) and the Pedersen commitment vector are unaffected."""

    n = len(rpk_list)
    if n == 0:
        raise ValueError("rpk_list must be non-empty")
    if len(share_pairs) != n:
        raise ValueError("share_pairs length must match rpk_list")

    g = _normalize_point(g_hex)
    h = _normalize_point(h_hex)
    key_base = _normalize_point(key_base_hex) if key_base_hex is not None else g
    rpks = _normalize_rpks(rpk_list)
    full_commitments = _normalize_commitment_vector(commitment_vector)
    recipient_commitments = _recipient_commitments(full_commitments, n)
    s_values = [field_normalize(pair[0]) for pair in share_pairs]
    h_values = [field_normalize(pair[1]) for pair in share_pairs]

    def encrypt_and_prove(values: Sequence[int], msg_base: str):
        ciphertexts, witnesses = enc_chunks(
            rpks,
            values,
            msg_base,
            key_base_hex=key_base,
        )
        proof = prove_chunking(
            rpks,
            ciphertexts,
            witnesses,
            msg_base,
            key_base_hex=key_base,
        )
        return ciphertexts, witnesses, proof

    # The s and hat-s branches have no shared witnesses.  Keep the pool local
    # so secret shares and encryption randomness never cross a process boundary.
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="pvss-prove",
    ) as executor:
        s_future = executor.submit(encrypt_and_prove, s_values, g)
        hat_future = executor.submit(encrypt_and_prove, h_values, h)
        enc_s, wit_s, pi_chunk_s = s_future.result()
        enc_hat, wit_hat, pi_chunk_hat = hat_future.result()
    enc_results = [
        {
            "recipient": i,
            "rpk": rpks[i],
            "enc_s": enc_s[i],
            "enc_hat_s": enc_hat[i],
        }
        for i in range(n)
    ]
    normalized_enc_results = _normalize_enc_results(enc_results, rpks)
    public = _public_statement(g, h, rpks, full_commitments, normalized_enc_results)
    alphas = _batch_scalars(public, n)

    share_statement, relations = _share_relations(
        g, h, rpks, recipient_commitments, enc_s, enc_hat, alphas, key_base_hex=key_base
    )
    secrets = {
        "S": _weighted_sum(s_values, alphas),
        "H": _weighted_sum(h_values, alphas),
    }
    for i in range(n):
        secrets[f"rho_s_{i}"] = _aggregate_randomness(wit_s[i])
        secrets[f"rho_hat_{i}"] = _aggregate_randomness(wit_hat[i])

    pi_ve = {
        "version": PROOF_VERSION,
        "pi_chunk_s": pi_chunk_s,
        "pi_chunk_hat_s": pi_chunk_hat,
        "pi_share": prove_linear_relations(
            {"public": public, "share": share_statement}, relations, secrets
        ),
    }
    return normalized_enc_results, pi_ve


def ve_bverify(
    g_hex: str,
    h_hex: str,
    rpk_list: Sequence[str],
    commitment_vector: Sequence[str],
    enc_results: Sequence[dict],
    pi_ve: dict,
    key_base_hex: str | None = None,
    _claimed_c_v_vector: Sequence[dict] | None = None,
) -> bool:
    try:
        if pi_ve.get("version") != PROOF_VERSION:
            return False
        n = len(rpk_list)
        if n == 0 or len(enc_results) != n:
            return False
        g = _normalize_point(g_hex)
        h = _normalize_point(h_hex)
        key_base = _normalize_point(key_base_hex) if key_base_hex is not None else g
        rpks = _normalize_rpks(rpk_list)
        full_commitments = _normalize_commitment_vector(commitment_vector)
        recipient_commitments = _recipient_commitments(full_commitments, n)
        normalized_enc_results = _normalize_enc_results(enc_results, rpks)
        enc_s, enc_hat = _extract_enc_lists(normalized_enc_results)
        chunk_aggregates = _aggregate_chunk_pairs(
            normalized_enc_results,
            _claimed_c_v_vector,
        )
        if chunk_aggregates is None:
            return False
        if not verify_chunking(rpks, enc_s, pi_ve["pi_chunk_s"], g, key_base_hex=key_base):
            return False
        if not verify_chunking(rpks, enc_hat, pi_ve["pi_chunk_hat_s"], h, key_base_hex=key_base):
            return False

        public = _public_statement(g, h, rpks, full_commitments, normalized_enc_results)
        alphas = _batch_scalars(public, n)
        share_statement, relations = _share_relations(
            g,
            h,
            rpks,
            recipient_commitments,
            enc_s,
            enc_hat,
            alphas,
            key_base_hex=key_base,
            chunk_aggregates=chunk_aggregates,
        )
        return verify_linear_relations(
            {"public": public, "share": share_statement},
            relations,
            pi_ve["pi_share"],
        )
    except (KeyError, TypeError, ValueError):
        return False


def ve_dec(
    g_hex: str,
    h_hex: str,
    rsk: int,
    enc_result_for_i: dict,
) -> tuple[int, int]:
    """Decrypt one pair-output VE ciphertext into ``(s_i, hat_s_i)``."""

    s_value = dec_chunks(rsk, enc_result_for_i["enc_s"], build_lookup_table(g_hex))
    h_value = dec_chunks(rsk, enc_result_for_i["enc_hat_s"], build_lookup_table(h_hex))
    return s_value, h_value
