"""Operation-coupled threshold-share evaluation for YOSO DPSS.

Algorithm 5 needs a holder to publish an operation share together with a
zero-knowledge proof that the exponent used for that operation share is the
same exponent hidden in the holder's Pedersen commitment.  This module is the
cryptographic adapter layer for that step; it deliberately knows nothing
about PVSS, YOSO-RBC, transcript selection, or successor reconstruction.

For an operation base ``X`` and holder commitment ``C_i`` it proves

    C_i = s_i * g + s_hat_i * h    and    y_i = s_i * X

using one Fiat-Shamir transformed, two-relation Schnorr proof.  Additive group
notation is used because :mod:`beaver.g1_backend` exposes G1 that way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from beaver.g1_backend import (
    FIELD_ORDER,
    G1Point,
    field_add,
    field_mul,
    field_normalize,
    field_random,
    g1_from_hex,
    g1_linear_combination,
)
from beaver.pedersen_shamir import lagrange_coeff_at_zero
from beaver.sigma import canonical_json, hash_to_scalar


_PROOF_VERSION = 1
_FS_DOMAIN = "atcaas-operation-share-v1"


def _operation_input_bytes(value: bytes | bytearray | memoryview | str) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError("operation_input must be bytes-like or str")


def _operation_id(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("operation_id must be non-empty")
    return normalized


def _point(value: str | G1Point, *, name: str, nonidentity: bool = False) -> G1Point:
    result = value if isinstance(value, G1Point) else g1_from_hex(str(value))
    if nonidentity and result.is_identity():
        raise ValueError(f"{name} must not be the identity point")
    return result


@dataclass(frozen=True)
class OperationShareProof:
    """NIZK proof tying an operation share to a Pedersen-committed share."""

    t1: str  # a*g + b*h
    t2: str  # a*X
    z1: int  # a + challenge*s_i
    z2: int  # b + challenge*s_hat_i

    def to_dict(self) -> dict[str, str | int]:
        return {
            "version": _PROOF_VERSION,
            "t1": self.t1,
            "t2": self.t2,
            "z1": str(field_normalize(self.z1)),
            "z2": str(field_normalize(self.z2)),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OperationShareProof":
        version = int(value.get("version", _PROOF_VERSION))
        if version != _PROOF_VERSION:
            raise ValueError(f"unsupported operation-share proof version: {version}")
        z1 = int(value["z1"])
        z2 = int(value["z2"])
        if not (0 <= z1 < FIELD_ORDER and 0 <= z2 < FIELD_ORDER):
            raise ValueError("operation-share proof responses must be canonical scalars")
        return cls(
            t1=str(value["t1"]),
            t2=str(value["t2"]),
            z1=z1,
            z2=z2,
        )

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_dict())

    @classmethod
    def from_bytes(cls, value: bytes | bytearray | memoryview) -> "OperationShareProof":
        import json

        decoded = json.loads(bytes(value).decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("operation-share proof must encode a JSON object")
        return cls.from_dict(decoded)


def _challenge(
    *,
    operation_id: str,
    operation_input: bytes,
    g: G1Point,
    h: G1Point,
    operation_base: G1Point,
    commitment: G1Point,
    operation_share: G1Point,
    t1: G1Point,
    t2: G1Point,
) -> int:
    """Bind the proof to the complete public operation statement."""

    return hash_to_scalar(
        _FS_DOMAIN,
        {
            "version": _PROOF_VERSION,
            "operation_id": _operation_id(operation_id),
            "operation_input": operation_input.hex(),
            "g": g.to_hex(),
            "h": h.to_hex(),
            "operation_base": operation_base.to_hex(),
            "commitment": commitment.to_hex(),
            "operation_share": operation_share.to_hex(),
            "t1": t1.to_hex(),
            "t2": t2.to_hex(),
        },
        nonzero=True,
    )


def partial_eval_exponentiation(
    *,
    operation_id: str,
    operation_input: bytes | bytearray | memoryview | str,
    g: str | G1Point,
    h: str | G1Point,
    operation_base: str | G1Point,
    commitment_i: str | G1Point,
    s_i: int,
    s_hat_i: int,
) -> tuple[str, OperationShareProof]:
    """Compute ``y_i = s_i*X`` and prove consistency with ``commitment_i``.

    A mismatched local share/commitment is rejected before producing a proof;
    this catches stale or incorrectly indexed DPSS state at the holder.
    """

    op_id = _operation_id(operation_id)
    op_input = _operation_input_bytes(operation_input)
    g_point = _point(g, name="g", nonidentity=True)
    h_point = _point(h, name="h", nonidentity=True)
    if g_point == h_point:
        raise ValueError("g and h must be distinct generators")
    base = _point(operation_base, name="operation_base", nonidentity=True)
    commitment = _point(commitment_i, name="commitment_i")
    share = field_normalize(s_i)
    blind_share = field_normalize(s_hat_i)

    if g_point * share + h_point * blind_share != commitment:
        raise ValueError("share pair does not open commitment_i")

    operation_share = base * share
    a = field_random()
    b = field_random()
    t1 = g_point * a + h_point * b
    t2 = base * a
    challenge = _challenge(
        operation_id=op_id,
        operation_input=op_input,
        g=g_point,
        h=h_point,
        operation_base=base,
        commitment=commitment,
        operation_share=operation_share,
        t1=t1,
        t2=t2,
    )
    proof = OperationShareProof(
        t1=t1.to_hex(),
        t2=t2.to_hex(),
        z1=field_add(a, field_mul(challenge, share)),
        z2=field_add(b, field_mul(challenge, blind_share)),
    )
    return operation_share.to_hex(), proof


def verify_exponentiation_share(
    *,
    operation_id: str,
    operation_input: bytes | bytearray | memoryview | str,
    g: str | G1Point,
    h: str | G1Point,
    operation_base: str | G1Point,
    commitment_i: str | G1Point,
    operation_share_i: str | G1Point,
    proof: OperationShareProof | Mapping[str, Any],
) -> bool:
    """Verify an operation-share proof.  Malformed adversarial input is false."""

    try:
        op_id = _operation_id(operation_id)
        op_input = _operation_input_bytes(operation_input)
        g_point = _point(g, name="g", nonidentity=True)
        h_point = _point(h, name="h", nonidentity=True)
        if g_point == h_point:
            return False
        base = _point(operation_base, name="operation_base", nonidentity=True)
        commitment = _point(commitment_i, name="commitment_i")
        operation_share = _point(operation_share_i, name="operation_share_i")
        parsed = proof if isinstance(proof, OperationShareProof) else OperationShareProof.from_dict(proof)
        t1 = _point(parsed.t1, name="proof.t1")
        t2 = _point(parsed.t2, name="proof.t2")
        challenge = _challenge(
            operation_id=op_id,
            operation_input=op_input,
            g=g_point,
            h=h_point,
            operation_base=base,
            commitment=commitment,
            operation_share=operation_share,
            t1=t1,
            t2=t2,
        )
        z1 = int(parsed.z1)
        z2 = int(parsed.z2)
        if not (0 <= z1 < FIELD_ORDER and 0 <= z2 < FIELD_ORDER):
            return False
        commitment_relation = g_point * z1 + h_point * z2 == t1 + commitment * challenge
        operation_relation = base * z1 == t2 + operation_share * challenge
        return bool(commitment_relation and operation_relation)
    except Exception:
        return False


def combine_exponentiation_shares(
    eval_points: Sequence[int], operation_shares: Sequence[str | G1Point]
) -> G1Point:
    """Interpolate operation shares at zero in G1."""

    points = [field_normalize(point) for point in eval_points]
    if not points:
        raise ValueError("at least one operation share is required")
    if len(points) != len(operation_shares):
        raise ValueError("eval_points and operation_shares must have the same length")
    if len(set(points)) != len(points):
        raise ValueError("eval_points must be distinct")
    if 0 in points:
        raise ValueError("eval_points must be non-zero")
    coefficients = [lagrange_coeff_at_zero(points, point) for point in points]
    shares = [_point(value, name="operation_share") for value in operation_shares]
    return g1_linear_combination(shares, coefficients)


class OperationAdapter(ABC):
    """Protocol-facing operation extension required by Algorithm 5."""

    @abstractmethod
    def public_context(self, operation_input: bytes | str) -> Mapping[str, Any]:
        """Return the canonical public statement used for this operation."""

    @abstractmethod
    def partial_eval(
        self,
        operation_input: bytes | str,
        *,
        commitment_i: str,
        s_i: int,
        s_hat_i: int,
    ) -> tuple[str, OperationShareProof]:
        """Produce one operation share and its consistency proof."""

    @abstractmethod
    def verify_share(
        self,
        operation_input: bytes | str,
        *,
        commitment_i: str,
        operation_share_i: str,
        proof: OperationShareProof | Mapping[str, Any],
    ) -> bool:
        """Verify one operation share against one holder commitment."""

    @abstractmethod
    def combine(
        self,
        operation_input: bytes | str,
        *,
        eval_points: Sequence[int],
        operation_shares: Sequence[str],
    ) -> Any:
        """Combine a qualified subset into the public operation result."""


BaseDeriver = Callable[[bytes], Union[str, G1Point]]
Postprocessor = Callable[[bytes, G1Point], Any]


class ThresholdExponentiationOperation(OperationAdapter):
    """Reusable adapter for operations whose partial result is ``s_i*X(x)``.

    ``base_deriver`` supplies ``X(x)``.  With ``X=H(message)`` the result is a
    threshold BLS-style signature; with ``X=C1`` and a postprocessor that
    removes the combined point from ``C2``, it is threshold ElGamal/ECIES-style
    decryption.
    """

    def __init__(
        self,
        *,
        operation_id: str,
        g_hex: str,
        h_hex: str,
        base_deriver: BaseDeriver,
        postprocessor: Optional[Postprocessor] = None,
    ) -> None:
        self.operation_id = _operation_id(operation_id)
        self.g = _point(g_hex, name="g", nonidentity=True)
        self.h = _point(h_hex, name="h", nonidentity=True)
        if self.g == self.h:
            raise ValueError("g and h must be distinct generators")
        if not callable(base_deriver):
            raise TypeError("base_deriver must be callable")
        if postprocessor is not None and not callable(postprocessor):
            raise TypeError("postprocessor must be callable")
        self._base_deriver = base_deriver
        self._postprocessor = postprocessor

    def _base(self, operation_input: bytes) -> G1Point:
        return _point(
            self._base_deriver(operation_input),
            name="operation_base",
            nonidentity=True,
        )

    def public_context(self, operation_input: bytes | str) -> Mapping[str, Any]:
        op_input = _operation_input_bytes(operation_input)
        return {
            "version": 1,
            "operation_id": self.operation_id,
            "operation_input": op_input.hex(),
            "g": self.g.to_hex(),
            "h": self.h.to_hex(),
            "operation_base": self._base(op_input).to_hex(),
        }

    def partial_eval(
        self,
        operation_input: bytes | str,
        *,
        commitment_i: str,
        s_i: int,
        s_hat_i: int,
    ) -> tuple[str, OperationShareProof]:
        op_input = _operation_input_bytes(operation_input)
        return partial_eval_exponentiation(
            operation_id=self.operation_id,
            operation_input=op_input,
            g=self.g,
            h=self.h,
            operation_base=self._base(op_input),
            commitment_i=commitment_i,
            s_i=s_i,
            s_hat_i=s_hat_i,
        )

    def verify_share(
        self,
        operation_input: bytes | str,
        *,
        commitment_i: str,
        operation_share_i: str,
        proof: OperationShareProof | Mapping[str, Any],
    ) -> bool:
        try:
            op_input = _operation_input_bytes(operation_input)
            base = self._base(op_input)
        except Exception:
            return False
        return verify_exponentiation_share(
            operation_id=self.operation_id,
            operation_input=op_input,
            g=self.g,
            h=self.h,
            operation_base=base,
            commitment_i=commitment_i,
            operation_share_i=operation_share_i,
            proof=proof,
        )

    def combine(
        self,
        operation_input: bytes | str,
        *,
        eval_points: Sequence[int],
        operation_shares: Sequence[str],
    ) -> Any:
        op_input = _operation_input_bytes(operation_input)
        # Resolve the base even though interpolation itself does not need it:
        # malformed operation input must fail consistently across all methods.
        self._base(op_input)
        combined = combine_exponentiation_shares(eval_points, operation_shares)
        if self._postprocessor is not None:
            return self._postprocessor(op_input, combined)
        return combined.to_hex()


__all__ = [
    "OperationAdapter",
    "OperationShareProof",
    "ThresholdExponentiationOperation",
    "combine_exponentiation_shares",
    "partial_eval_exponentiation",
    "verify_exponentiation_share",
]
