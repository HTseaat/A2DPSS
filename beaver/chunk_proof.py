"""Fiat-Shamir chunk proof for YOSO DPSS chunked ElGamal.

The proof follows the batching idea used by the ADKR prototype: random
challenge coefficients compress all encrypted chunks into a few group
relations, while range slack checks keep each chunk within the configured
``CHUNK_BYTES`` range without revealing the chunks themselves.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from beaver.chunk_elgamal import CHUNK_BASE, CHUNK_BYTES, CHUNK_MASK, NUM_CHUNKS
from beaver.g1_backend import (
    FIELD_ORDER,
    G1Point,
    field_add,
    field_mul,
    field_normalize,
    field_random,
    g1_verify_chunking_equations,
)


SECURITY_LEVEL = 256
NUM_ZK_REPETITIONS = 32
CHALLENGE_BITS = (SECURITY_LEVEL + NUM_ZK_REPETITIONS - 1) // NUM_ZK_REPETITIONS
CHALLENGE_MASK = (1 << CHALLENGE_BITS) - 1
PROOF_VERSION = 1


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_int(label: str, value, *, modulo: int | None = None) -> int:
    digest = hashlib.sha256(label.encode("utf-8") + b":" + _canonical_json(value)).digest()
    number = int.from_bytes(digest, byteorder="big")
    return number % modulo if modulo else number


def _normalize_point(value: str) -> str:
    return G1Point.from_hex(value).to_hex()


def _normalize_ciphertexts(ciphertext) -> list[dict]:
    if isinstance(ciphertext, dict) and "ciphertexts" in ciphertext:
        ciphertext = ciphertext["ciphertexts"]
    elif isinstance(ciphertext, dict) and "chunks" in ciphertext:
        ciphertext = [ciphertext]
    if not isinstance(ciphertext, list):
        raise ValueError("ciphertext must be a ciphertext dict or a list of them")

    normalized = []
    for recipient, entry in enumerate(ciphertext):
        if not isinstance(entry, dict):
            raise ValueError(f"ciphertext entry {recipient} must be a dict")
        chunks = entry.get("chunks")
        if not isinstance(chunks, list) or len(chunks) != NUM_CHUNKS:
            raise ValueError(f"ciphertext entry {recipient} must contain {NUM_CHUNKS} chunks")
        normalized_chunks = []
        for chunk_idx, chunk_entry in enumerate(chunks):
            try:
                c1 = _normalize_point(chunk_entry["C1"])
                c2 = _normalize_point(chunk_entry["C2"])
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"invalid ciphertext chunk {recipient}:{chunk_idx}"
                ) from exc
            normalized_chunks.append({"C1": c1, "C2": c2})
        normalized.append(
            {
                "recipient": int(entry.get("recipient", recipient)),
                "rpk": _normalize_point(entry["rpk"]) if "rpk" in entry else None,
                "chunks": normalized_chunks,
            }
        )
    return normalized


def _normalize_witnesses(witness) -> list[dict]:
    if isinstance(witness, dict) and "witnesses" in witness:
        witness = witness["witnesses"]
    elif isinstance(witness, dict) and "chunks" in witness:
        witness = [witness]
    if not isinstance(witness, list):
        raise ValueError("witness must be a witness dict or a list of them")

    normalized = []
    for recipient, entry in enumerate(witness):
        if not isinstance(entry, dict):
            raise ValueError(f"witness entry {recipient} must be a dict")
        chunks = entry.get("chunks")
        randomness = entry.get("randomness")
        if not isinstance(chunks, list) or len(chunks) != NUM_CHUNKS:
            raise ValueError(f"witness entry {recipient} must contain {NUM_CHUNKS} chunks")
        if not isinstance(randomness, list) or len(randomness) != NUM_CHUNKS:
            raise ValueError(
                f"witness entry {recipient} must contain {NUM_CHUNKS} randomness values"
            )
        normalized_chunks = []
        normalized_randomness = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk = int(chunk)
            if not 0 <= chunk < CHUNK_BASE:
                raise ValueError(f"witness chunk {recipient}:{chunk_idx} is out of range")
            normalized_chunks.append(chunk)
        for rho in randomness:
            normalized_randomness.append(field_normalize(rho))
        normalized.append(
            {
                "recipient": int(entry.get("recipient", recipient)),
                "chunks": normalized_chunks,
                "randomness": normalized_randomness,
            }
        )
    return normalized


def _public_instance(
    rpks: Sequence[str],
    ciphertexts: Sequence[dict],
    msg_base: str,
    key_base: str,
) -> dict:
    return {
        "version": PROOF_VERSION,
        "chunk_bytes": CHUNK_BYTES,
        "num_chunks": NUM_CHUNKS,
        "challenge_bits": CHALLENGE_BITS,
        "repetitions": NUM_ZK_REPETITIONS,
        "msg_base": msg_base,
        "key_base": key_base,
        "rpks": list(rpks),
        "ciphertexts": ciphertexts,
    }


def _derive_e(
    instance: dict,
    y0: str,
    b_list: Sequence[str],
    c_list: Sequence[str],
    *,
    parallel: bool = False,
) -> list[list[list[int]]]:
    seed_material = {
        "instance": instance,
        "y0": y0,
        "B": list(b_list),
        "C": list(c_list),
    }
    # ``sort_keys=True`` places the outer keys in the fixed order i,j,k,seed.
    # Serialize the large invariant seed once, while hashing exactly the same
    # bytes as the original per-challenge ``_hash_int`` calls.
    seed_json = _canonical_json(seed_material)
    hash_prefix = hashlib.sha256()
    hash_prefix.update(b"chunk-proof-e:")
    n = len(instance["rpks"])
    def derive_row(index: tuple[int, int]) -> tuple[int, int, list[int]]:
        i, j = index
        row_prefix = (
            b'{"i":'
            + str(i).encode("ascii")
            + b',"j":'
            + str(j).encode("ascii")
            + b',"k":'
        )
        row = []
        for k in range(NUM_ZK_REPETITIONS):
            digest = hash_prefix.copy()
            digest.update(row_prefix)
            digest.update(str(k).encode("ascii"))
            digest.update(b',"seed":')
            digest.update(seed_json)
            digest.update(b"}")
            row.append(
                int.from_bytes(digest.digest(), byteorder="big")
                & CHALLENGE_MASK
            )
        return i, j, row

    indexes = [(i, j) for i in range(n) for j in range(NUM_CHUNKS)]
    if parallel and len(indexes) > 1:
        # Three OpenSSL hashing workers give the best throughput here: a
        # fourth worker competes for the large invariant transcript in cache.
        worker_count = min(3, os.cpu_count() or 1, len(indexes))
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="pvss-fiat-shamir",
        ) as executor:
            derived_rows = executor.map(derive_row, indexes)
            e = [[None] * NUM_CHUNKS for _ in range(n)]
            for i, j, row in derived_rows:
                e[i][j] = row
            return e

    e = [[None] * NUM_CHUNKS for _ in range(n)]
    for i, j in indexes:
        _, _, row = derive_row((i, j))
        e[i][j] = row
    return e


def _derive_x(e, z_s: Sequence[int], d0: str, d_list, y: str) -> int:
    return _hash_int(
        "chunk-proof-x",
        {"e": e, "z_s": list(z_s), "D0": d0, "D": d_list, "Y": y},
        modulo=FIELD_ORDER,
    )


def _x_powers(x: int) -> list[int]:
    powers = []
    current = field_normalize(x)
    for _ in range(NUM_ZK_REPETITIONS):
        powers.append(current)
        current = field_mul(current, x)
    return powers


def _range_bounds(n: int) -> tuple[int, int]:
    s_bound = n * NUM_CHUNKS * CHUNK_MASK * CHALLENGE_MASK
    z_bound = max(1, 4 * NUM_ZK_REPETITIONS * s_bound)
    return s_bound, z_bound


def prove_chunking(
    rpk_list: Sequence[str],
    ciphertext,
    witness,
    msg_base_hex: str,
    *,
    key_base_hex: str | None = None,
) -> dict:
    """Prove that chunk ciphertexts encrypt bounded chunks under ``msg_base_hex``."""

    rpks = [_normalize_point(rpk) for rpk in rpk_list]
    ciphertexts = _normalize_ciphertexts(ciphertext)
    witnesses = _normalize_witnesses(witness)
    msg_base = _normalize_point(msg_base_hex)
    key_base = _normalize_point(key_base_hex or msg_base_hex)
    if len(rpks) != len(ciphertexts) or len(rpks) != len(witnesses):
        raise ValueError("rpk_list, ciphertext, and witness lengths must match")
    if not rpks:
        raise ValueError("chunk proof requires at least one recipient")

    for i, (rpk, ctxt, wit) in enumerate(zip(rpks, ciphertexts, witnesses)):
        if ctxt["recipient"] != i or wit["recipient"] != i:
            raise ValueError("recipient indexes must be dense and ordered")
        if ctxt["rpk"] is not None and ctxt["rpk"] != rpk:
            raise ValueError(f"ciphertext rpk mismatch at recipient {i}")

    msg_base_point = G1Point.from_hex(msg_base)
    key_base_point = G1Point.from_hex(key_base)
    rpk_points = [G1Point.from_hex(rpk) for rpk in rpks]
    c1_points = [
        [G1Point.from_hex(entry["C1"]) for entry in ctxt["chunks"]]
        for ctxt in ciphertexts
    ]
    c2_points = [
        [G1Point.from_hex(entry["C2"]) for entry in ctxt["chunks"]]
        for ctxt in ciphertexts
    ]

    for i, wit in enumerate(witnesses):
        for j, (chunk, rho) in enumerate(zip(wit["chunks"], wit["randomness"])):
            expected_c2 = (msg_base_point * chunk) + (rpk_points[i] * rho)
            expected_c1 = key_base_point * rho
            if expected_c1 != c1_points[i][j] or expected_c2 != c2_points[i][j]:
                raise ValueError(f"witness does not match ciphertext chunk {i}:{j}")

    n = len(rpks)
    s_bound, z_bound = _range_bounds(n)
    eta = field_random()
    y0 = key_base_point * eta

    beta = [field_random() for _ in range(NUM_ZK_REPETITIONS)]
    sigma = [secrets.randbelow(z_bound - s_bound + 1) for _ in range(NUM_ZK_REPETITIONS)]
    b_points = [(key_base_point * beta_k).to_hex() for beta_k in beta]
    c_points = [
        ((y0 * beta_k) + (msg_base_point * sigma_k)).to_hex()
        for beta_k, sigma_k in zip(beta, sigma)
    ]

    instance = _public_instance(rpks, ciphertexts, msg_base, key_base)
    e = _derive_e(instance, y0.to_hex(), b_points, c_points)

    z_s = []
    for k in range(NUM_ZK_REPETITIONS):
        total = sigma[k]
        for i in range(n):
            for j in range(NUM_CHUNKS):
                total += e[i][j][k] * witnesses[i]["chunks"][j]
        if not 0 <= total <= z_bound:
            raise ValueError("internal range proof bound failed")
        z_s.append(total)

    delta0 = field_random()
    delta = [
        [field_random() for _ in range(NUM_CHUNKS)]
        for _ in range(n)
    ]
    d0 = (key_base_point * delta0).to_hex()
    d_points = [
        [(key_base_point * delta[i][j]).to_hex() for j in range(NUM_CHUNKS)]
        for i in range(n)
    ]
    y = y0 * delta0
    for i in range(n):
        for j in range(NUM_CHUNKS):
            y = y + (rpk_points[i] * delta[i][j])
    y_hex = y.to_hex()

    x = _derive_x(e, z_s, d0, d_points, y_hex)
    x_powers = _x_powers(x)
    z_r = []
    for i in range(n):
        row = []
        for j in range(NUM_CHUNKS):
            total = delta[i][j]
            for k in range(NUM_ZK_REPETITIONS):
                term = field_mul(e[i][j][k], witnesses[i]["randomness"][j])
                total = field_add(total, field_mul(term, x_powers[k]))
            row.append(total)
        z_r.append(row)

    z_b = delta0
    for beta_k, x_k in zip(beta, x_powers):
        z_b = field_add(z_b, field_mul(beta_k, x_k))

    return {
        "version": PROOF_VERSION,
        "y0": y0.to_hex(),
        "B": b_points,
        "C": c_points,
        "D0": d0,
        "D": d_points,
        "Y": y_hex,
        "z_s": z_s,
        "z_r": z_r,
        "z_b": z_b,
    }


def verify_chunking(
    rpk_list: Sequence[str],
    ciphertext,
    proof: dict,
    msg_base_hex: str,
    *,
    key_base_hex: str | None = None,
) -> bool:
    """Verify a chunk proof.  Returns ``False`` on malformed input."""

    try:
        if proof.get("version") != PROOF_VERSION:
            return False
        rpks = [_normalize_point(rpk) for rpk in rpk_list]
        ciphertexts = _normalize_ciphertexts(ciphertext)
        msg_base = _normalize_point(msg_base_hex)
        key_base = _normalize_point(key_base_hex or msg_base_hex)
        n = len(rpks)
        if n != len(ciphertexts):
            return False
        if n == 0:
            return False
        if len(proof.get("B", [])) != NUM_ZK_REPETITIONS:
            return False
        if len(proof.get("C", [])) != NUM_ZK_REPETITIONS:
            return False
        if len(proof.get("D", [])) != n:
            return False
        if len(proof.get("z_s", [])) != NUM_ZK_REPETITIONS:
            return False
        if len(proof.get("z_r", [])) != n:
            return False

        for i, (rpk, ctxt) in enumerate(zip(rpks, ciphertexts)):
            if ctxt["recipient"] != i:
                return False
            if ctxt["rpk"] is not None and ctxt["rpk"] != rpk:
                return False
            if len(proof["D"][i]) != NUM_CHUNKS or len(proof["z_r"][i]) != NUM_CHUNKS:
                return False

        _, z_bound = _range_bounds(n)
        z_s = [int(value) for value in proof["z_s"]]
        if any(value < 0 or value > z_bound for value in z_s):
            return False

        instance = _public_instance(rpks, ciphertexts, msg_base, key_base)
        e = _derive_e(
            instance,
            proof["y0"],
            proof["B"],
            proof["C"],
            parallel=True,
        )
        x = _derive_x(e, z_s, proof["D0"], proof["D"], proof["Y"])
        x_powers = _x_powers(x)

        z_r = [
            [field_normalize(int(value)) for value in row]
            for row in proof["z_r"]
        ]
        challenge_weights = [
            [
                field_normalize(
                    sum(e[i][j][k] * x_powers[k] for k in range(NUM_ZK_REPETITIONS))
                )
                for j in range(NUM_CHUNKS)
            ]
            for i in range(n)
        ]
        z_s_weighted = field_normalize(
            sum(z_s_k * x_k for z_s_k, x_k in zip(z_s, x_powers))
        )
        rpk_weights = [field_normalize(sum(row)) for row in z_r]

        scalar_hex = lambda value: format(field_normalize(value), "x")
        return g1_verify_chunking_equations(
            {
                "key_base": key_base,
                "msg_base": msg_base,
                "y0": proof["y0"],
                "B": proof["B"],
                "C": proof["C"],
                "D0": proof["D0"],
                "D": proof["D"],
                "Y": proof["Y"],
                "rpks": rpks,
                "c1": [
                    [entry["C1"] for entry in ctxt["chunks"]]
                    for ctxt in ciphertexts
                ],
                "c2": [
                    [entry["C2"] for entry in ctxt["chunks"]]
                    for ctxt in ciphertexts
                ],
                "x_powers": [scalar_hex(value) for value in x_powers],
                "challenge_weights": [
                    [scalar_hex(value) for value in row]
                    for row in challenge_weights
                ],
                "z_s_weighted": scalar_hex(z_s_weighted),
                "z_r": [[scalar_hex(value) for value in row] for row in z_r],
                "rpk_weights": [scalar_hex(value) for value in rpk_weights],
                "z_b": scalar_hex(int(proof["z_b"])),
            }
        )
    except (KeyError, TypeError, ValueError):
        return False
