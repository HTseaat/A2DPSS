"""Common vk-to-operation-share DLEQ proof for RU22 and BHK+22.

For a holder share ``s_i`` this module proves, in additive notation,

    vk_i = s_i * G_vk
    y_i  = s_i * X

with one Fiat-Shamir transformed two-relation Schnorr proof.  The proof is
protocol-independent cryptography, while its canonical statement explicitly
binds the protocol, trial, qid, parent state, operation context, role, and
evaluation point.  Protocol wire bundles and runtime integration live in later
steps.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from beaver.dag_tob import Qid
from beaver.g1_backend import (
    FIELD_ORDER,
    G1Point,
    field_add,
    field_mul,
    field_random,
    g1_linear_combination,
)
from beaver.pedersen_shamir import lagrange_coeff_at_zero
from beaver.sigma import canonical_json, hash_to_scalar
from beaver.threshold_operation_contract import (
    BHK22_OPERATION_QID_KINDS,
    OPERATION_ID,
    OPERATION_VK_BASE_HEX,
    RU22_OPERATION_QID_KIND,
    ThresholdOperationContext,
)


STATEMENT_SCHEMA = "vk-operation-share-statement/v1"
PROOF_KIND = "vk-to-operation-share/v1"
PROOF_VERSION = 1
SUPPORTED_PROTOCOLS = ("ru22", "bhk22")

_STATEMENT_DIGEST_DOMAIN = b"ATCaaS/vk-operation-share/statement/v1\x00"
_FS_DOMAIN = "atcaas-vk-operation-share-dleq-v1"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_SCALAR_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _decode_object(
    value: bytes | bytearray | memoryview | str, *, name: str
) -> dict[str, Any]:
    try:
        raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed {name} JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must encode a JSON object")
    return decoded


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{name} has unknown fields: {sorted(extra)}")


def _nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a canonical non-empty string")
    return value


def _digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase sha256 hex")
    return value


def _strict_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _canonical_scalar_text(value: Any, *, name: str) -> int:
    if not isinstance(value, str) or _SCALAR_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical decimal scalar string")
    scalar = int(value)
    if scalar >= FIELD_ORDER:
        raise ValueError(f"{name} must be smaller than the scalar-field order")
    return scalar


def _witness_scalar(value: Any) -> int:
    scalar = _strict_int(value, name="share", minimum=1)
    if scalar >= FIELD_ORDER:
        raise ValueError("share must be a canonical non-zero scalar")
    return scalar


def _canonical_point(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a canonical G1 string")
    try:
        point = G1Point.from_hex(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a valid G1 point") from exc
    if point.is_identity():
        raise ValueError(f"{name} must not be the identity point")
    if point.to_hex() != value:
        raise ValueError(f"{name} must use canonical AMCL G1 encoding")
    return value


def _canonical_qid(value: Any, *, protocol: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("qid must be a canonical structured qid string")
    decoded = Qid.decode(value)
    if decoded.encode() != value:
        raise ValueError("qid must be a canonical structured qid string")
    if protocol == "ru22" and decoded.kind != RU22_OPERATION_QID_KIND:
        raise ValueError("RU22 statement must use the RU22 operation qid kind")
    if (
        protocol == "bhk22"
        and decoded.kind not in BHK22_OPERATION_QID_KINDS.values()
    ):
        raise ValueError("BHK22 statement must use a BHK22 operation qid kind")
    return value


@dataclass(frozen=True)
class VKOperationShareStatement:
    """Complete public statement bound by the common DLEQ proof."""

    protocol: str
    trial_id: str
    qid: str
    parent_state_digest: str
    arm_context_digest: str
    operation_context_digest: str
    operation_id: str
    operation_input_digest: str
    holder_role_id: int
    evaluation_point: int
    operation_base_hex: str
    operation_vk_base_hex: str
    operation_vk_hex: str
    operation_share_hex: str
    statement_digest: str = field(init=False)

    def __post_init__(self) -> None:
        protocol = _nonempty_text(self.protocol, name="protocol")
        if protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError("unsupported vk-operation-share protocol")
        trial_id = _nonempty_text(self.trial_id, name="trial_id")
        qid = _canonical_qid(self.qid, protocol=protocol)
        parent_digest = _digest(
            self.parent_state_digest, name="parent_state_digest"
        )
        arm_digest = _digest(self.arm_context_digest, name="arm_context_digest")
        operation_digest = _digest(
            self.operation_context_digest,
            name="operation_context_digest",
        )
        if self.operation_id != OPERATION_ID:
            raise ValueError("unsupported operation_id")
        input_digest = _digest(
            self.operation_input_digest,
            name="operation_input_digest",
        )
        holder_role_id = _strict_int(
            self.holder_role_id, name="holder_role_id", minimum=0
        )
        evaluation_point = _strict_int(
            self.evaluation_point, name="evaluation_point", minimum=1
        )
        if evaluation_point >= FIELD_ORDER:
            raise ValueError("evaluation_point must be in the non-zero scalar field")
        operation_base = _canonical_point(
            self.operation_base_hex, name="operation_base_hex"
        )
        vk_base = _canonical_point(
            self.operation_vk_base_hex, name="operation_vk_base_hex"
        )
        if vk_base != OPERATION_VK_BASE_HEX:
            raise ValueError("operation_vk_base_hex must equal the frozen G_vk")
        operation_vk = _canonical_point(
            self.operation_vk_hex, name="operation_vk_hex"
        )
        operation_share = _canonical_point(
            self.operation_share_hex, name="operation_share_hex"
        )

        for name, value in (
            ("protocol", protocol),
            ("trial_id", trial_id),
            ("qid", qid),
            ("parent_state_digest", parent_digest),
            ("arm_context_digest", arm_digest),
            ("operation_context_digest", operation_digest),
            ("operation_input_digest", input_digest),
            ("holder_role_id", holder_role_id),
            ("evaluation_point", evaluation_point),
            ("operation_base_hex", operation_base),
            ("operation_vk_base_hex", vk_base),
            ("operation_vk_hex", operation_vk),
            ("operation_share_hex", operation_share),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "statement_digest", self.digest())

    def to_obj(self) -> dict[str, str | int]:
        return {
            "schema": STATEMENT_SCHEMA,
            "protocol": self.protocol,
            "trial_id": self.trial_id,
            "qid": self.qid,
            "parent_state_digest": self.parent_state_digest,
            "arm_context_digest": self.arm_context_digest,
            "operation_context_digest": self.operation_context_digest,
            "operation_id": self.operation_id,
            "operation_input_digest": self.operation_input_digest,
            "holder_role_id": self.holder_role_id,
            "evaluation_point": self.evaluation_point,
            "operation_base_hex": self.operation_base_hex,
            "operation_vk_base_hex": self.operation_vk_base_hex,
            "operation_vk_hex": self.operation_vk_hex,
            "operation_share_hex": self.operation_share_hex,
        }

    def digest(self) -> str:
        return hashlib.sha256(
            _STATEMENT_DIGEST_DOMAIN + canonical_json(self.to_obj())
        ).hexdigest()

    def to_bytes(self) -> bytes:
        value = self.to_obj()
        value["statement_digest"] = self.statement_digest
        return canonical_json(value)

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "VKOperationShareStatement":
        expected = {
            "schema",
            "protocol",
            "trial_id",
            "qid",
            "parent_state_digest",
            "arm_context_digest",
            "operation_context_digest",
            "operation_id",
            "operation_input_digest",
            "holder_role_id",
            "evaluation_point",
            "operation_base_hex",
            "operation_vk_base_hex",
            "operation_vk_hex",
            "operation_share_hex",
            "statement_digest",
        }
        _exact_fields(value, expected, name="vk-operation-share statement")
        if value["schema"] != STATEMENT_SCHEMA:
            raise ValueError("unsupported vk-operation-share statement schema")
        supplied_digest = _digest(
            value["statement_digest"], name="statement_digest"
        )
        statement = cls(
            protocol=value["protocol"],
            trial_id=value["trial_id"],
            qid=value["qid"],
            parent_state_digest=value["parent_state_digest"],
            arm_context_digest=value["arm_context_digest"],
            operation_context_digest=value["operation_context_digest"],
            operation_id=value["operation_id"],
            operation_input_digest=value["operation_input_digest"],
            holder_role_id=value["holder_role_id"],
            evaluation_point=value["evaluation_point"],
            operation_base_hex=value["operation_base_hex"],
            operation_vk_base_hex=value["operation_vk_base_hex"],
            operation_vk_hex=value["operation_vk_hex"],
            operation_share_hex=value["operation_share_hex"],
        )
        if not hmac.compare_digest(statement.statement_digest, supplied_digest):
            raise ValueError("statement_digest does not match the public statement")
        return statement

    @classmethod
    def from_bytes(
        cls, value: bytes | bytearray | memoryview | str
    ) -> "VKOperationShareStatement":
        return cls.from_obj(_decode_object(value, name="vk-operation-share statement"))


@dataclass(frozen=True)
class VKOperationShareProof:
    """Strict canonical wire object for the two-relation Schnorr proof."""

    statement_digest: str
    t_vk_hex: str
    t_operation_hex: str
    response: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "statement_digest",
            _digest(self.statement_digest, name="statement_digest"),
        )
        object.__setattr__(
            self, "t_vk_hex", _canonical_point(self.t_vk_hex, name="t_vk_hex")
        )
        object.__setattr__(
            self,
            "t_operation_hex",
            _canonical_point(self.t_operation_hex, name="t_operation_hex"),
        )
        response = _strict_int(self.response, name="response", minimum=0)
        if response >= FIELD_ORDER:
            raise ValueError("response must be a canonical scalar")
        object.__setattr__(self, "response", response)

    def to_obj(self) -> dict[str, str | int]:
        return {
            "kind": PROOF_KIND,
            "version": PROOF_VERSION,
            "statement_digest": self.statement_digest,
            "t_vk_hex": self.t_vk_hex,
            "t_operation_hex": self.t_operation_hex,
            "response": str(self.response),
        }

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "VKOperationShareProof":
        expected = {
            "kind",
            "version",
            "statement_digest",
            "t_vk_hex",
            "t_operation_hex",
            "response",
        }
        _exact_fields(value, expected, name="vk-operation-share proof")
        if value["kind"] != PROOF_KIND:
            raise ValueError("unexpected vk-operation-share proof kind")
        if type(value["version"]) is not int or value["version"] != PROOF_VERSION:
            raise ValueError("unsupported vk-operation-share proof version")
        return cls(
            statement_digest=value["statement_digest"],
            t_vk_hex=value["t_vk_hex"],
            t_operation_hex=value["t_operation_hex"],
            response=_canonical_scalar_text(value["response"], name="response"),
        )

    @classmethod
    def from_bytes(
        cls, value: bytes | bytearray | memoryview | str
    ) -> "VKOperationShareProof":
        return cls.from_obj(_decode_object(value, name="vk-operation-share proof"))


def _challenge(
    statement: VKOperationShareStatement,
    *,
    t_vk_hex: str,
    t_operation_hex: str,
) -> int:
    return hash_to_scalar(
        _FS_DOMAIN,
        {
            "proof_kind": PROOF_KIND,
            "proof_version": PROOF_VERSION,
            "statement": statement.to_obj(),
            "statement_digest": statement.statement_digest,
            "t_vk_hex": t_vk_hex,
            "t_operation_hex": t_operation_hex,
        },
        nonzero=True,
    )


def prove_vk_operation_share(
    statement: VKOperationShareStatement,
    *,
    share: int,
) -> VKOperationShareProof:
    """Prove the statement after checking that ``share`` opens both points."""

    if not isinstance(statement, VKOperationShareStatement):
        raise TypeError("statement must be a VKOperationShareStatement")
    witness = _witness_scalar(share)
    vk_base = G1Point.from_hex(statement.operation_vk_base_hex)
    operation_base = G1Point.from_hex(statement.operation_base_hex)
    if vk_base * witness != G1Point.from_hex(statement.operation_vk_hex):
        raise ValueError("share does not open operation_vk_hex")
    if operation_base * witness != G1Point.from_hex(
        statement.operation_share_hex
    ):
        raise ValueError("share does not open operation_share_hex")

    blinding = field_random()
    t_vk = vk_base * blinding
    t_operation = operation_base * blinding
    challenge = _challenge(
        statement,
        t_vk_hex=t_vk.to_hex(),
        t_operation_hex=t_operation.to_hex(),
    )
    return VKOperationShareProof(
        statement_digest=statement.statement_digest,
        t_vk_hex=t_vk.to_hex(),
        t_operation_hex=t_operation.to_hex(),
        response=field_add(blinding, field_mul(challenge, witness)),
    )


def verify_vk_operation_share(
    statement: VKOperationShareStatement,
    proof: VKOperationShareProof | Mapping[str, Any],
) -> bool:
    """Verify one proof; malformed or adversarial public input returns False."""

    try:
        if not isinstance(statement, VKOperationShareStatement):
            return False
        parsed = (
            proof
            if isinstance(proof, VKOperationShareProof)
            else VKOperationShareProof.from_obj(proof)
        )
        if not hmac.compare_digest(
            statement.statement_digest, parsed.statement_digest
        ):
            return False
        vk_base = G1Point.from_hex(statement.operation_vk_base_hex)
        operation_base = G1Point.from_hex(statement.operation_base_hex)
        operation_vk = G1Point.from_hex(statement.operation_vk_hex)
        operation_share = G1Point.from_hex(statement.operation_share_hex)
        t_vk = G1Point.from_hex(parsed.t_vk_hex)
        t_operation = G1Point.from_hex(parsed.t_operation_hex)
        challenge = _challenge(
            statement,
            t_vk_hex=parsed.t_vk_hex,
            t_operation_hex=parsed.t_operation_hex,
        )
        vk_relation = (
            vk_base * parsed.response == t_vk + operation_vk * challenge
        )
        operation_relation = (
            operation_base * parsed.response
            == t_operation + operation_share * challenge
        )
        return bool(vk_relation and operation_relation)
    except Exception:
        return False


def evaluate_and_prove_vk_operation_share(
    *,
    protocol: str,
    trial_id: str,
    qid: str,
    parent_state_digest: str,
    arm_context_digest: str,
    operation_context: ThresholdOperationContext,
    holder_role_id: int,
    evaluation_point: int,
    share: int,
) -> tuple[VKOperationShareStatement, VKOperationShareProof]:
    """Compute ``vk_i``/``y_i``, build the statement, and prove it."""

    if not isinstance(operation_context, ThresholdOperationContext):
        raise TypeError("operation_context must be a ThresholdOperationContext")
    witness = _witness_scalar(share)
    vk_base = G1Point.from_hex(operation_context.operation_vk_base_hex)
    operation_base = G1Point.from_hex(operation_context.operation_base_hex)
    statement = VKOperationShareStatement(
        protocol=protocol,
        trial_id=trial_id,
        qid=qid,
        parent_state_digest=parent_state_digest,
        arm_context_digest=arm_context_digest,
        operation_context_digest=operation_context.context_digest,
        operation_id=operation_context.operation_id,
        operation_input_digest=operation_context.operation_input_digest,
        holder_role_id=holder_role_id,
        evaluation_point=evaluation_point,
        operation_base_hex=operation_base.to_hex(),
        operation_vk_base_hex=vk_base.to_hex(),
        operation_vk_hex=(vk_base * witness).to_hex(),
        operation_share_hex=(operation_base * witness).to_hex(),
    )
    return statement, prove_vk_operation_share(statement, share=witness)


def combine_vk_operation_shares(
    evaluation_points: Sequence[int],
    operation_shares: Sequence[str],
) -> G1Point:
    """Interpolate a non-empty qualified subset of operation shares at zero."""

    if isinstance(evaluation_points, (str, bytes)) or not isinstance(
        evaluation_points, Sequence
    ):
        raise TypeError("evaluation_points must be a sequence")
    if isinstance(operation_shares, (str, bytes)) or not isinstance(
        operation_shares, Sequence
    ):
        raise TypeError("operation_shares must be a sequence")
    if not evaluation_points:
        raise ValueError("at least one operation share is required")
    if len(evaluation_points) != len(operation_shares):
        raise ValueError(
            "evaluation_points and operation_shares must have the same length"
        )
    points = []
    for value in evaluation_points:
        point = _strict_int(value, name="evaluation_point", minimum=1)
        if point >= FIELD_ORDER:
            raise ValueError("evaluation_point must be in the non-zero scalar field")
        points.append(point)
    if len(set(points)) != len(points):
        raise ValueError("evaluation_points must be distinct")
    shares = [
        G1Point.from_hex(_canonical_point(value, name="operation_share"))
        for value in operation_shares
    ]
    coefficients = [
        lagrange_coeff_at_zero(points, point) for point in points
    ]
    return g1_linear_combination(shares, coefficients)


__all__ = [
    "PROOF_KIND",
    "PROOF_VERSION",
    "STATEMENT_SCHEMA",
    "SUPPORTED_PROTOCOLS",
    "VKOperationShareProof",
    "VKOperationShareStatement",
    "combine_vk_operation_shares",
    "evaluate_and_prove_vk_operation_share",
    "prove_vk_operation_share",
    "verify_vk_operation_share",
]
