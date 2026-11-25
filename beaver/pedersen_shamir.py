"""Pedersen/Shamir helpers for the YOSO DPSS PVSS implementation.

The Pedersen commitment wrapper uses the existing AMCL FFI and returns the
same normalized G1 strings as ``beaver.g1_backend.G1Point.to_hex()``.
"""

from __future__ import annotations

import json
import secrets
from ctypes import CDLL, c_char_p, c_void_p, string_at
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence


FIELD_ORDER = (
    52435875175126190479447740508185965837690552500527637822603658699938581184513
)
G1_UNCOMPRESSED_BYTES = 97
G1_COORD_BYTES = 48


def field_normalize(value: int) -> int:
    return int(value) % FIELD_ORDER


def field_random() -> int:
    return secrets.randbelow(FIELD_ORDER)


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


def sample_poly(secret: int, degree: int) -> list[int]:
    """Return coefficients for a random degree-``degree`` polynomial.

    The coefficient list is in little-endian order: ``coeffs[i]`` is the
    coefficient of ``x^i``.  The constant term is fixed to ``secret``.
    """

    if degree < 0:
        raise ValueError("degree must be non-negative")
    return [field_normalize(secret)] + [field_random() for _ in range(degree)]


def eval_poly(coeffs: Sequence[int], x: int) -> int:
    """Evaluate a scalar polynomial at ``x`` using Horner's rule."""

    x = field_normalize(x)
    result = 0
    for coeff in reversed(coeffs):
        result = field_add(field_mul(result, x), coeff)
    return result


def _normalize_points(points: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(field_normalize(point) for point in points)
    if len(normalized) == 0:
        raise ValueError("at least one interpolation point is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("interpolation points must be distinct")
    return normalized


def lagrange_coeff(points: Sequence[int], x_i: int, at: int = 0) -> int:
    """Return the Lagrange basis coefficient l_i(at)."""

    normalized_points = _normalize_points(points)
    x_i = field_normalize(x_i)
    at = field_normalize(at)
    if x_i not in normalized_points:
        raise ValueError("x_i must be one of the interpolation points")

    numerator = 1
    denominator = 1
    for x_j in normalized_points:
        if x_j == x_i:
            continue
        numerator = field_mul(numerator, field_sub(at, x_j))
        denominator = field_mul(denominator, field_sub(x_i, x_j))
    return field_mul(numerator, field_inv(denominator))


def lagrange_coeff_at_zero(points: Sequence[int], x_i: int) -> int:
    return lagrange_coeff(points, x_i, at=0)


def interpolate_at(points: Sequence[int], values: Sequence[int], at: int) -> int:
    """Interpolate a scalar polynomial from points and evaluate it at ``at``."""

    normalized_points = _normalize_points(points)
    if len(normalized_points) != len(values):
        raise ValueError("points and values must have the same length")

    result = 0
    for x_i, y_i in zip(normalized_points, values):
        coeff = lagrange_coeff(normalized_points, x_i, at=at)
        result = field_add(result, field_mul(y_i, coeff))
    return result


def interpolate_at_zero(points: Sequence[int], values: Sequence[int]) -> int:
    return interpolate_at(points, values, at=0)


def _scalar_to_hex(value: int) -> str:
    return format(field_normalize(value), "x")


def _as_uncompressed_g1_hex(point: str) -> str:
    data = point.strip()
    if data.startswith("0x"):
        data = data[2:]
    if data.startswith("(") and data.endswith(")"):
        x_hex, y_hex = data.strip("()").split(",", 1)
        x = int(x_hex.strip(), 16)
        y = int(y_hex.strip(), 16)
        return (
            b"\x04"
            + x.to_bytes(G1_COORD_BYTES, byteorder="big")
            + y.to_bytes(G1_COORD_BYTES, byteorder="big")
        ).hex()

    raw = bytes.fromhex(data)
    if len(raw) != G1_UNCOMPRESSED_BYTES or raw[0] != 0x04:
        raise ValueError("expected an uncompressed G1 point encoded as 04||X||Y")
    return raw.hex()


def _default_bulletproof_lib_path() -> Path:
    return Path(__file__).resolve().parents[1] / "libbulletproofs_amcl.so"


@lru_cache(maxsize=4)
def _load_bulletproof_lib(lib_path: str) -> CDLL:
    lib = CDLL(lib_path)
    lib.pyComputeCommitmentGH.argtypes = [c_char_p]
    lib.pyComputeCommitmentGH.restype = c_void_p
    lib.pyFreeString.argtypes = [c_void_p]
    lib.pyFreeString.restype = None
    return lib


def batch_pedersen_commit(
    g_hex: str,
    h_hex: str,
    values: Sequence[int],
    blinds: Sequence[int],
    *,
    lib_path: str | None = None,
) -> list[str]:
    """Compute ``g^value h^blind`` for each pair using ``pyComputeCommitmentGH``.

    ``g_hex`` and ``h_hex`` may be either raw ``04||X||Y`` hex strings or AMCL
    ``(x,y)`` strings.  Returned commitments are normalized AMCL G1 strings.
    """

    if len(values) != len(blinds):
        raise ValueError("values and blinds must have the same length")

    resolved_lib_path = str(lib_path or _default_bulletproof_lib_path())
    lib = _load_bulletproof_lib(resolved_lib_path)
    payload = {
        "g": _as_uncompressed_g1_hex(g_hex),
        "h": _as_uncompressed_g1_hex(h_hex),
        "m": [_scalar_to_hex(value) for value in values],
        "m_prime": [_scalar_to_hex(blind) for blind in blinds],
    }
    ptr = lib.pyComputeCommitmentGH(json.dumps(payload).encode("utf-8"))
    if not ptr:
        raise ValueError("pyComputeCommitmentGH rejected the Pedersen payload")
    try:
        raw_points = json.loads(string_at(ptr).decode("utf-8"))
    finally:
        lib.pyFreeString(ptr)

    if not isinstance(raw_points, list) or not all(
        isinstance(point, str) for point in raw_points
    ):
        raise ValueError("pyComputeCommitmentGH returned an invalid point list")
    return raw_points


def pedersen_commit(
    g_hex: str,
    h_hex: str,
    value: int,
    blind: int,
    *,
    lib_path: str | None = None,
) -> str:
    return batch_pedersen_commit(
        g_hex, h_hex, [value], [blind], lib_path=lib_path
    )[0]


def _load_default_g1_backend():
    try:
        from beaver.g1_backend import G1Point, g1_linear_combination
    except ModuleNotFoundError as exc:
        raise NotImplementedError(
            "pedersen_deg_check requires beaver.g1_backend or an explicit "
            "backend with from_hex()/identity() support"
        ) from exc
    return G1Point, g1_linear_combination


def _backend_identity(backend):
    return backend.identity()


def _backend_from_hex(backend, value: str):
    return backend.from_hex(value)


def _point_add(left, right):
    return left + right


def _point_mul(point, scalar: int):
    return point * field_normalize(scalar)


def pedersen_deg_check(
    commitment_vector: Sequence[str],
    degree: int,
    *,
    eval_points: Sequence[int] | None = None,
    backend=None,
) -> bool:
    """Check that commitments lie on a degree-``degree`` G1 polynomial.

    If ``eval_points`` is omitted, points are assumed to be
    ``1, 2, ..., len(commitment_vector)``.  That keeps the default aligned with
    Shamir shares, where point 0 is the secret and is not distributed.

    The actual G1 arithmetic is delegated to ``backend``.  By default this
    uses the Rust-backed ``beaver.g1_backend.G1Point`` implementation.
    """

    if degree < 0:
        raise ValueError("degree must be non-negative")
    if len(commitment_vector) == 0:
        raise ValueError("commitment_vector must be non-empty")
    if backend is None:
        backend, linear_combination = _load_default_g1_backend()
    else:
        linear_combination = None
    points = (
        tuple(range(1, len(commitment_vector) + 1))
        if eval_points is None
        else _normalize_points(eval_points)
    )
    if len(points) != len(commitment_vector):
        raise ValueError("eval_points and commitment_vector must have the same length")

    parsed = [_backend_from_hex(backend, value) for value in commitment_vector]
    if len(commitment_vector) <= degree + 1:
        return True

    basis_points = points[: degree + 1]
    basis_commitments = parsed[: degree + 1]

    for target_x, target_commitment in zip(points[degree + 1 :], parsed[degree + 1 :]):
        coeffs = [
            lagrange_coeff(basis_points, basis_x, at=target_x)
            for basis_x in basis_points
        ]
        if linear_combination is not None:
            predicted = linear_combination(basis_commitments, coeffs)
        else:
            predicted = _backend_identity(backend)
            for basis_commitment, coeff in zip(basis_commitments, coeffs):
                predicted = _point_add(predicted, _point_mul(basis_commitment, coeff))
        if predicted != target_commitment:
            return False
    return True
