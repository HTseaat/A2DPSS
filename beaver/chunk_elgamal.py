"""Chunked ElGamal encryption for YOSO DPSS PVSS.

This module encrypts scalar shares chunk-by-chunk in G1.  It is intentionally
proof-free; the witnesses returned by ``enc_chunks`` are meant for the Step 1.3
chunk proof module.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import mmap
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from beaver.g1_backend import (
    FIELD_ORDER,
    G1Point,
    field_normalize,
    field_random,
    g1_linear_combination,
)


CHUNK_BYTES = 2
NUM_CHUNKS = 16
CHUNK_BITS = 8 * CHUNK_BYTES
CHUNK_BASE = 1 << CHUNK_BITS
CHUNK_MASK = CHUNK_BASE - 1
SCALAR_CHUNK_CAPACITY = 1 << (CHUNK_BITS * NUM_CHUNKS)
LOOKUP_TABLE_ARTIFACT_SCHEMA = "pvss-chunk-lookup-table/v1"
LOOKUP_TABLE_SET_SCHEMA = "pvss-chunk-lookup-table-set/v1"
LOOKUP_POINT_ENCODING = "amcl-g1-affine-fixed-v1"
_LOOKUP_KEY_BYTES = 97
_LOOKUP_VALUE_BYTES = 2
_LOOKUP_RECORD_BYTES = _LOOKUP_KEY_BYTES + _LOOKUP_VALUE_BYTES
_PRECOMPUTED_LOOKUP_TABLES = {}
_PRECOMPUTED_LOOKUP_LOCK = threading.RLock()


def chunk_weights() -> tuple[int, ...]:
    """Return the little-endian scalar weight of every ciphertext chunk."""

    weights = []
    current = 1
    for _ in range(NUM_CHUNKS):
        weights.append(current)
        current = field_normalize(current * CHUNK_BASE)
    return tuple(weights)


def normalize_elgamal_ciphertext(ciphertext: dict) -> dict:
    """Return one canonical two-point ElGamal ciphertext."""

    if not isinstance(ciphertext, dict):
        raise TypeError("ElGamal ciphertext must be an object")
    if set(ciphertext) != {"C1", "C2"}:
        raise ValueError("ElGamal ciphertext must contain exactly C1 and C2")
    return {
        "C1": G1Point.from_hex(ciphertext["C1"]).to_hex(),
        "C2": G1Point.from_hex(ciphertext["C2"]).to_hex(),
    }


def aggregate_chunk_ciphertext(ciphertext_for_i: dict) -> dict:
    """Collapse chunk ciphertexts into an ElGamal encryption of the scalar.

    If chunk ``k`` encrypts ``base^(value_k)`` with randomness ``rho_k``,
    the returned ciphertext uses weights ``CHUNK_BASE^k`` and encrypts the
    complete scalar under aggregate randomness.  The operation is public and
    deterministic; it does not perform bounded discrete-log recovery.
    """

    if not isinstance(ciphertext_for_i, dict):
        raise TypeError("chunk ciphertext must be an object")
    chunks = ciphertext_for_i.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != NUM_CHUNKS:
        raise ValueError(f"ciphertext must contain {NUM_CHUNKS} chunks")
    try:
        c1 = g1_linear_combination(
            [entry["C1"] for entry in chunks],
            chunk_weights(),
        )
        c2 = g1_linear_combination(
            [entry["C2"] for entry in chunks],
            chunk_weights(),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("malformed chunk ciphertext") from exc
    return {"C1": c1.to_hex(), "C2": c2.to_hex()}


def combine_elgamal_ciphertexts(
    ciphertexts: Sequence[dict],
    coefficients: Sequence[int],
) -> dict:
    """Return the public linear combination of two-point ciphertexts."""

    if len(ciphertexts) != len(coefficients):
        raise ValueError("ciphertexts and coefficients must have the same length")
    if not ciphertexts:
        raise ValueError("need at least one ElGamal ciphertext")
    normalized = [
        normalize_elgamal_ciphertext(ciphertext)
        for ciphertext in ciphertexts
    ]
    scalars = [field_normalize(coefficient) for coefficient in coefficients]
    return {
        "C1": g1_linear_combination(
            [ciphertext["C1"] for ciphertext in normalized],
            scalars,
        ).to_hex(),
        "C2": g1_linear_combination(
            [ciphertext["C2"] for ciphertext in normalized],
            scalars,
        ).to_hex(),
    }


def decrypt_elgamal_ciphertext(rsk: int, ciphertext: dict) -> str:
    """Decrypt a complete ElGamal ciphertext to its encoded G1 message."""

    normalized = normalize_elgamal_ciphertext(ciphertext)
    c1 = G1Point.from_hex(normalized["C1"])
    c2 = G1Point.from_hex(normalized["C2"])
    return (c2 - (c1 * field_normalize(rsk))).to_hex()


def split_scalar(value: int) -> list[int]:
    """Split a scalar into little-endian ``CHUNK_BYTES`` chunks."""

    value = int(value)
    if not 0 <= value < FIELD_ORDER:
        raise ValueError("value must be a canonical scalar in [0, FIELD_ORDER)")

    chunks = []
    remaining = value
    for _ in range(NUM_CHUNKS):
        chunks.append(remaining & CHUNK_MASK)
        remaining >>= CHUNK_BITS
    if remaining:
        raise ValueError("value does not fit into NUM_CHUNKS")
    return chunks


def restore_scalar(chunks: Sequence[int]) -> int:
    """Restore a scalar from little-endian chunks."""

    if len(chunks) != NUM_CHUNKS:
        raise ValueError(f"expected {NUM_CHUNKS} chunks")

    value = 0
    for idx, chunk in enumerate(chunks):
        chunk = int(chunk)
        if not 0 <= chunk < CHUNK_BASE:
            raise ValueError(f"chunk {idx} is outside [0, {CHUNK_BASE})")
        value += chunk << (CHUNK_BITS * idx)
    if not 0 <= value < FIELD_ORDER:
        raise ValueError("restored value is not a canonical scalar")
    return value


def _normalize_point_hex(point_hex: str) -> str:
    return G1Point.from_hex(point_hex).to_hex()


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _lookup_point_key(point_hex: str) -> bytes:
    """Encode a canonical AMCL G1 string as one fixed-width lookup key."""

    if point_hex == "infinity":
        return b"\x00" * _LOOKUP_KEY_BYTES
    if not isinstance(point_hex, str) or not (
        point_hex.startswith("(") and point_hex.endswith(")")
    ):
        raise ValueError("lookup point is not a canonical affine G1 string")
    coordinates = point_hex[1:-1].split(",")
    if len(coordinates) != 2 or any(len(value) != 96 for value in coordinates):
        raise ValueError("lookup point must contain two fixed-width coordinates")
    try:
        return b"\x01" + bytes.fromhex("".join(coordinates))
    except ValueError as exc:
        raise ValueError("lookup point contains a non-hex coordinate") from exc


def _lookup_artifact_header(msg_base_norm: str) -> dict:
    return {
        "schema": LOOKUP_TABLE_ARTIFACT_SCHEMA,
        "curve": "BLS12-381-G1",
        "point_encoding": LOOKUP_POINT_ENCODING,
        "chunk_bytes": CHUNK_BYTES,
        "chunk_base": CHUNK_BASE,
        "entry_count": CHUNK_BASE,
        "record_bytes": _LOOKUP_RECORD_BYTES,
        "base": msg_base_norm,
    }


class PrecomputedLookupTable:
    """Read-only mmap-backed bounded discrete-log lookup table."""

    def __init__(self, path: Path, *, data_offset: int, entry_count: int):
        self.path = Path(path)
        self.entry_count = int(entry_count)
        self._file = self.path.open("rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._data_offset = int(data_offset)

    def __len__(self) -> int:
        return self.entry_count

    def __getitem__(self, point_hex: str) -> int:
        key = _lookup_point_key(point_hex)
        low = 0
        high = self.entry_count
        while low < high:
            middle = (low + high) // 2
            offset = self._data_offset + middle * _LOOKUP_RECORD_BYTES
            candidate = self._mmap[offset : offset + _LOOKUP_KEY_BYTES]
            if candidate < key:
                low = middle + 1
            else:
                high = middle
        if low >= self.entry_count:
            raise KeyError(point_hex)
        offset = self._data_offset + low * _LOOKUP_RECORD_BYTES
        if self._mmap[offset : offset + _LOOKUP_KEY_BYTES] != key:
            raise KeyError(point_hex)
        value_offset = offset + _LOOKUP_KEY_BYTES
        return int.from_bytes(
            self._mmap[value_offset : value_offset + _LOOKUP_VALUE_BYTES],
            byteorder="big",
        )

    def close(self) -> None:
        mapping = getattr(self, "_mmap", None)
        if mapping is not None:
            mapping.close()
            self._mmap = None
        handle = getattr(self, "_file", None)
        if handle is not None:
            handle.close()
            self._file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def write_lookup_table_artifact(
    msg_base_hex: str,
    output_path: str | Path,
    *,
    artifact_path: str,
) -> dict:
    """Generate one versioned public lookup-table artifact."""

    if CHUNK_BASE > 1 << (8 * _LOOKUP_VALUE_BYTES):
        raise ValueError("lookup-table value encoding is too small")
    if (
        not isinstance(artifact_path, str)
        or not artifact_path
        or Path(artifact_path).is_absolute()
        or ".." in Path(artifact_path).parts
    ):
        raise ValueError("artifact_path must be a safe relative path")
    msg_base_norm = _normalize_point_hex(msg_base_hex)
    header = _lookup_artifact_header(msg_base_norm)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    records = []
    msg_base = G1Point.from_hex(msg_base_norm)
    point = G1Point.identity()
    for chunk in range(CHUNK_BASE):
        records.append(
            _lookup_point_key(point.to_hex())
            + chunk.to_bytes(_LOOKUP_VALUE_BYTES, byteorder="big")
        )
        point = point + msg_base
    records.sort()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    byte_size = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            header_line = _canonical_json(header) + b"\n"
            handle.write(header_line)
            digest.update(header_line)
            byte_size += len(header_line)
            for record in records:
                handle.write(record)
                digest.update(record)
                byte_size += len(record)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        **header,
        "artifact_path": artifact_path,
        "artifact_sha256": digest.hexdigest(),
        "artifact_bytes": byte_size,
    }


def generate_lookup_table_artifacts(
    output_directory: str | Path,
    *,
    g_hex: str,
    h_hex: str,
    relative_directory: str = "lookup_tables",
) -> dict:
    """Generate the public g/h lookup artifacts during configuration setup."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    # Normalize before starting workers so the shared native library is loaded.
    bases = {"g": _normalize_point_hex(g_hex), "h": _normalize_point_hex(h_hex)}
    started = time.perf_counter()

    def generate(label: str) -> tuple[str, dict]:
        filename = f"{label}.lookup"
        return label, write_lookup_table_artifact(
            bases[label],
            output / filename,
            artifact_path=(Path(relative_directory) / filename).as_posix(),
        )

    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="pvss-lookup-setup",
    ) as executor:
        generated = dict(executor.map(generate, ("g", "h")))
    return {
        "schema": LOOKUP_TABLE_SET_SCHEMA,
        "offline_generation_seconds": time.perf_counter() - started,
        "tables": generated,
    }


def _safe_artifact_location(base_directory: Path, artifact_path: str) -> Path:
    relative = Path(artifact_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("lookup artifact path must remain inside the config directory")
    root = base_directory.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("lookup artifact resolves outside the config directory") from exc
    return candidate


def load_lookup_table_artifact(
    manifest: Mapping,
    *,
    base_directory: str | Path,
) -> tuple[str, PrecomputedLookupTable]:
    """Authenticate and mmap one configured lookup-table artifact."""

    if not isinstance(manifest, Mapping):
        raise TypeError("lookup artifact manifest must be a mapping")
    header_fields = set(_lookup_artifact_header(""))
    expected_fields = header_fields | {
        "artifact_path",
        "artifact_sha256",
        "artifact_bytes",
    }
    if set(manifest) != expected_fields:
        raise ValueError("lookup artifact manifest fields do not match its schema")
    base_norm = _normalize_point_hex(manifest["base"])
    expected_header = _lookup_artifact_header(base_norm)
    if any(manifest.get(name) != value for name, value in expected_header.items()):
        raise ValueError("lookup artifact parameters do not match this implementation")
    expected_digest = manifest["artifact_sha256"]
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ValueError("lookup artifact digest must be lowercase SHA-256 hex")
    try:
        bytes.fromhex(expected_digest)
    except ValueError as exc:
        raise ValueError("lookup artifact digest is not hex") from exc
    if expected_digest.lower() != expected_digest:
        raise ValueError("lookup artifact digest must be lowercase")
    path = _safe_artifact_location(
        Path(base_directory),
        manifest["artifact_path"],
    )
    if path.stat().st_size != manifest["artifact_bytes"]:
        raise ValueError("lookup artifact size does not match its manifest")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        header_line = handle.readline()
        digest.update(header_line)
        try:
            actual_header = json.loads(header_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("lookup artifact header is malformed") from exc
        if actual_header != expected_header:
            raise ValueError("lookup artifact header does not match its manifest")
        previous_key = None
        seen_values = bytearray(CHUNK_BASE)
        for _ in range(CHUNK_BASE):
            record = handle.read(_LOOKUP_RECORD_BYTES)
            if len(record) != _LOOKUP_RECORD_BYTES:
                raise ValueError("lookup artifact is truncated")
            digest.update(record)
            key = record[:_LOOKUP_KEY_BYTES]
            if previous_key is not None and key <= previous_key:
                raise ValueError("lookup artifact keys are not strictly sorted")
            previous_key = key
            value = int.from_bytes(record[_LOOKUP_KEY_BYTES:], byteorder="big")
            if value >= CHUNK_BASE or seen_values[value]:
                raise ValueError("lookup artifact values are not a permutation")
            seen_values[value] = 1
        if handle.read(1):
            raise ValueError("lookup artifact contains trailing bytes")
    if not hmac.compare_digest(digest.hexdigest(), expected_digest):
        raise ValueError("lookup artifact SHA-256 mismatch")

    return base_norm, PrecomputedLookupTable(
        path,
        data_offset=len(header_line),
        entry_count=CHUNK_BASE,
    )


def install_lookup_table_artifacts(
    configured: Mapping,
    *,
    base_directory: str | Path,
) -> None:
    """Load an authenticated g/h table set before online protocol timing."""

    if not isinstance(configured, Mapping) or set(configured) != {
        "schema",
        "offline_generation_seconds",
        "tables",
    }:
        raise ValueError("lookup table set has an invalid shape")
    if configured["schema"] != LOOKUP_TABLE_SET_SCHEMA:
        raise ValueError("unsupported lookup table set schema")
    generation_seconds = configured["offline_generation_seconds"]
    if (
        isinstance(generation_seconds, bool)
        or not isinstance(generation_seconds, (int, float))
        or generation_seconds < 0
    ):
        raise ValueError("lookup offline generation time must be non-negative")
    tables = configured["tables"]
    if not isinstance(tables, Mapping) or set(tables) != {"g", "h"}:
        raise ValueError("lookup table set must contain exactly g and h")

    loaded = []
    try:
        for label in ("g", "h"):
            loaded.append(
                load_lookup_table_artifact(
                    tables[label],
                    base_directory=base_directory,
                )
            )
    except Exception:
        for _base, table in loaded:
            table.close()
        raise

    with _PRECOMPUTED_LOOKUP_LOCK:
        for base, table in loaded:
            previous = _PRECOMPUTED_LOOKUP_TABLES.get(base)
            _PRECOMPUTED_LOOKUP_TABLES[base] = table
            if previous is not None and previous is not table:
                previous.close()


def clear_precomputed_lookup_tables() -> None:
    """Release installed mmap tables; intended for tests and shutdown."""

    with _PRECOMPUTED_LOOKUP_LOCK:
        tables = list(_PRECOMPUTED_LOOKUP_TABLES.values())
        _PRECOMPUTED_LOOKUP_TABLES.clear()
    for table in tables:
        table.close()
    _build_lookup_table_cached.cache_clear()


def is_lookup_table_precomputed(msg_base_hex: str) -> bool:
    msg_base_norm = _normalize_point_hex(msg_base_hex)
    with _PRECOMPUTED_LOOKUP_LOCK:
        return msg_base_norm in _PRECOMPUTED_LOOKUP_TABLES


@lru_cache(maxsize=8)
def _build_lookup_table_cached(msg_base_norm: str) -> dict[str, int]:
    msg_base = G1Point.from_hex(msg_base_norm)
    table = {}
    point = G1Point.identity()
    for chunk in range(CHUNK_BASE):
        table[point.to_hex()] = chunk
        point = point + msg_base
    return table


def build_lookup_table(msg_base_hex: str):
    """Return a bounded discrete-log table for one chunk base.

    Configured deployments use an authenticated mmap-backed artifact generated
    offline.  Legacy configs retain the cached in-memory construction fallback.
    """

    msg_base_norm = _normalize_point_hex(msg_base_hex)
    with _PRECOMPUTED_LOOKUP_LOCK:
        precomputed = _PRECOMPUTED_LOOKUP_TABLES.get(msg_base_norm)
    if precomputed is not None:
        return precomputed
    return _build_lookup_table_cached(msg_base_norm)


def derive_public_key(secret_key: int, msg_base_hex: str) -> str:
    """Return ``msg_base^secret_key`` as a normalized G1 string."""

    return (G1Point.from_hex(msg_base_hex) * field_normalize(secret_key)).to_hex()


def enc_chunks(
    rpk_list: Sequence[str],
    values: Sequence[int],
    msg_base_hex: str,
    *,
    key_base_hex: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Encrypt one scalar value per recipient.

    For each recipient public key ``rpk`` and scalar ``value``, each chunk is
    encrypted as ``C1 = key_base^rho`` and
    ``C2 = msg_base^chunk + rpk^rho``.  ``key_base_hex`` defaults to
    ``msg_base_hex`` for the standalone Step 1.2 API; pair-VE passes
    ``key_base=g`` and ``msg_base=h`` for ``hat_s``.
    """

    if len(rpk_list) != len(values):
        raise ValueError("rpk_list and values must have the same length")

    msg_base = G1Point.from_hex(msg_base_hex)
    key_base = G1Point.from_hex(key_base_hex or msg_base_hex)
    ciphertexts = []
    witnesses = []

    for recipient, (rpk_hex, value) in enumerate(zip(rpk_list, values)):
        rpk = G1Point.from_hex(rpk_hex)
        normalized_value = int(value)
        chunks = split_scalar(normalized_value)
        chunk_ciphertexts = []
        randomness = []

        for chunk in chunks:
            rho = field_random()
            c1 = key_base * rho
            c2 = (msg_base * chunk) + (rpk * rho)
            chunk_ciphertexts.append({"C1": c1.to_hex(), "C2": c2.to_hex()})
            randomness.append(rho)

        ciphertexts.append(
            {
                "recipient": recipient,
                "rpk": rpk.to_hex(),
                "chunks": chunk_ciphertexts,
            }
        )
        witnesses.append(
            {
                "recipient": recipient,
                "value": normalized_value,
                "chunks": chunks,
                "randomness": randomness,
            }
        )

    return ciphertexts, witnesses


def dec_chunks(
    rsk: int,
    ciphertext_for_i: dict,
    lookup_table,
) -> int:
    """Decrypt one recipient's chunk ciphertexts and restore the scalar."""

    chunks = ciphertext_for_i.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != NUM_CHUNKS:
        raise ValueError(f"ciphertext must contain {NUM_CHUNKS} chunks")

    sk = field_normalize(rsk)
    recovered_chunks = []
    for idx, entry in enumerate(chunks):
        try:
            c1 = G1Point.from_hex(entry["C1"])
            c2 = G1Point.from_hex(entry["C2"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid chunk ciphertext at index {idx}") from exc

        message_point = (c2 - (c1 * sk)).to_hex()
        try:
            recovered_chunks.append(lookup_table[message_point])
        except KeyError as exc:
            raise ValueError(f"chunk {idx} is not in the lookup table") from exc

    return restore_scalar(recovered_chunks)
