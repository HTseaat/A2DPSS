"""Rust-backed BLS12-381 G1 helpers for PVSS/DPSS.

The project already uses ``libbulletproofs_amcl.so`` for Pedersen commitments.
This module keeps the Python side deliberately thin: G1 parsing, addition,
subtraction, scalar multiplication, and linear combinations are delegated to
the same AMCL-backed Rust library so every PVSS component shares one point
encoding and one group implementation.
"""

from __future__ import annotations

import json
import secrets
from ctypes import CDLL, c_char_p, c_void_p, string_at
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


FIELD_ORDER = (
    52435875175126190479447740508185965837690552500527637822603658699938581184513
)


def field_normalize(value: int) -> int:
    return int(value) % FIELD_ORDER


def field_random() -> int:
    return secrets.randbelow(FIELD_ORDER - 1) + 1


def field_inv(value: int) -> int:
    value = field_normalize(value)
    if value == 0:
        raise ZeroDivisionError("cannot invert zero in the scalar field")
    return pow(value, FIELD_ORDER - 2, FIELD_ORDER)


def field_add(left: int, right: int) -> int:
    return (int(left) + int(right)) % FIELD_ORDER


def field_sub(left: int, right: int) -> int:
    return (int(left) - int(right)) % FIELD_ORDER


def field_mul(left: int, right: int) -> int:
    return (int(left) * int(right)) % FIELD_ORDER


def _default_lib_path() -> Path:
    return Path(__file__).resolve().parents[1] / "libbulletproofs_amcl.so"


_LIB = None


def _load_lib(path: str | Path | None = None):
    global _LIB
    if _LIB is not None and path is None:
        return _LIB

    lib = CDLL(str(path or _default_lib_path()))
    for name in (
        "pyG1Identity",
        "pyG1Normalize",
        "pyG1Add",
        "pyG1Sub",
        "pyG1Mul",
        "pyG1LinearCombination",
        "pyG1VerifyLinearRelations",
        "pyG1AggregateChunkPairs",
        "pyG1VerifyChunkingEquations",
        "pyG1Equal",
    ):
        fn = getattr(lib, name)
        fn.restype = c_void_p
    lib.pyG1Normalize.argtypes = [c_char_p]
    lib.pyG1Add.argtypes = [c_char_p]
    lib.pyG1Sub.argtypes = [c_char_p]
    lib.pyG1Mul.argtypes = [c_char_p]
    lib.pyG1LinearCombination.argtypes = [c_char_p]
    lib.pyG1VerifyLinearRelations.argtypes = [c_char_p]
    lib.pyG1AggregateChunkPairs.argtypes = [c_char_p]
    lib.pyG1VerifyChunkingEquations.argtypes = [c_char_p]
    lib.pyG1Equal.argtypes = [c_char_p]
    lib.pyFreeString.argtypes = [c_void_p]
    lib.pyFreeString.restype = None

    if path is None:
        _LIB = lib
    return lib


def _ffi_json(function_name: str, payload: dict | None = None):
    lib = _load_lib()
    fn = getattr(lib, function_name)
    if payload is None:
        ptr = fn()
    else:
        ptr = fn(json.dumps(payload).encode("utf-8"))
    if not ptr:
        raise ValueError(f"{function_name} rejected its input")
    try:
        return json.loads(string_at(ptr).decode("utf-8"))
    finally:
        lib.pyFreeString(ptr)


def _scalar_hex(value: int) -> str:
    return format(field_normalize(value), "x")


@dataclass(frozen=True)
class G1Point:
    """G1 point wrapper around the Rust AMCL ``G1`` encoding.

    ``from_hex`` accepts both existing project encodings:
    - raw AMCL bytes hex, including the ``04||X||Y`` values stored in configs;
    - AMCL ``to_hex()`` strings such as ``(x,y)`` or ``infinity``.

    ``to_hex`` returns the normalized AMCL ``to_hex()`` string.
    """

    value: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "value", _ffi_json("pyG1Normalize", {"point": self.value})
        )

    @staticmethod
    def identity() -> "G1Point":
        return G1Point(_ffi_json("pyG1Identity"))

    @staticmethod
    def from_hex(data: str) -> "G1Point":
        return G1Point(data)

    @staticmethod
    def from_dict(data: dict) -> "G1Point":
        raw = (
            b"\x04"
            + int(data["X"]).to_bytes(48, byteorder="big")
            + int(data["Y"]).to_bytes(48, byteorder="big")
        )
        return G1Point(raw.hex())

    def is_identity(self) -> bool:
        return self.value == "infinity"

    def to_hex(self) -> str:
        return self.value

    def __add__(self, other: "G1Point") -> "G1Point":
        if not isinstance(other, G1Point):
            return NotImplemented
        return G1Point(
            _ffi_json("pyG1Add", {"left": self.value, "right": other.value})
        )

    def __sub__(self, other: "G1Point") -> "G1Point":
        if not isinstance(other, G1Point):
            return NotImplemented
        return G1Point(
            _ffi_json("pyG1Sub", {"left": self.value, "right": other.value})
        )

    def __mul__(self, scalar: int) -> "G1Point":
        return G1Point(
            _ffi_json(
                "pyG1Mul",
                {"point": self.value, "scalar": _scalar_hex(scalar)},
            )
        )

    def __rmul__(self, scalar: int) -> "G1Point":
        return self * scalar

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, G1Point):
            return False
        return bool(
            _ffi_json("pyG1Equal", {"left": self.value, "right": other.value})["equal"]
        )


def g1_identity() -> G1Point:
    return G1Point.identity()


def g1_from_hex(data: str) -> G1Point:
    return G1Point.from_hex(data)


def g1_linear_combination(
    points: Sequence[G1Point] | Sequence[str],
    scalars: Sequence[int],
) -> G1Point:
    if len(points) != len(scalars):
        raise ValueError("points and scalars must have the same length")

    point_values = [
        p.to_hex() if isinstance(p, G1Point) else G1Point.from_hex(p).to_hex()
        for p in points
    ]
    result = _ffi_json(
        "pyG1LinearCombination",
        {
            "points": point_values,
            "scalars": [_scalar_hex(scalar) for scalar in scalars],
        },
    )
    return G1Point(result)


def g1_verify_chunking_equations(payload: dict) -> bool:
    """Evaluate all group equations for one chunk proof in native code."""

    result = _ffi_json("pyG1VerifyChunkingEquations", payload)
    if not isinstance(result, dict) or not isinstance(result.get("valid"), bool):
        raise ValueError("pyG1VerifyChunkingEquations returned malformed output")
    return result["valid"]


def g1_verify_linear_relations(payload: dict) -> bool:
    """Verify exact Sigma linear relations in one native call."""

    result = _ffi_json("pyG1VerifyLinearRelations", payload)
    if not isinstance(result, dict) or not isinstance(result.get("valid"), bool):
        raise ValueError("pyG1VerifyLinearRelations returned malformed output")
    return result["valid"]


def g1_aggregate_chunk_pairs(payload: dict) -> dict:
    """Aggregate paired chunk ciphertexts and optionally validate c_v."""

    result = _ffi_json("pyG1AggregateChunkPairs", payload)
    required = {"valid", "r_s", "d_s", "r_hat", "d_hat"}
    if not isinstance(result, dict) or not required.issubset(result):
        raise ValueError("pyG1AggregateChunkPairs returned malformed output")
    if not isinstance(result["valid"], bool):
        raise ValueError("pyG1AggregateChunkPairs returned malformed validity")
    return result


def g1_sum(points: Iterable[G1Point]) -> G1Point:
    acc = G1Point.identity()
    for point in points:
        acc = acc + point
    return acc
