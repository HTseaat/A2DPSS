"""Frozen Step-0 contract for the operation-coupled YOSO experiment.

This module deliberately contains no protocol prover, wire bundle, runtime, or
fixture secret.  It is the shared public contract that the A2DPSS/OCDPSS,
RU22, and BHK+22 application arms must import before their protocol-specific
implementation is added.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from beaver.baselines.bhk22_context import hash_to_g1
from beaver.baselines.bhk22_contract import (
    CARRY_FORWARD_PATH_PHASES,
    PATH_ONE_ACCUSATION,
    PATH_OPTIMISTIC,
)
from beaver.g1_backend import G1Point


# New application experiment schemas.  The pure-resharing schemas remain
# unchanged and are intentionally not aliases of these values.
WORKLOAD_SCHEMA = "yoso-threshold-service-workload/v1"
ARM_SCHEMA = "yoso-threshold-service-arm/v1"
PRIVATE_SHARD_SCHEMA = "yoso-threshold-service-node-private/v1"
OPERATION_CONTEXT_SCHEMA = "yoso-threshold-service-operation-context/v1"
RU22_OPERATION_CONTEXT_SCHEMA = "ru22-operation-context/v1"
RU22_OPERATION_PVR_SCHEMA = "ru22-operation-pvr/v1"
BHK22_OPERATION_CONTEXT_SCHEMA = "bhk22-operation-context/v1"
BHK22_OPERATION_DEALER_BATCH_SCHEMA = "bhk22-operation-dealer-batch/v1"

ARM_ORDER = ("a2dpss", "ru22", "bhk22")

# The operation is fixed to threshold exponentiation/signing for this
# experiment.  ``hash_to_g1`` hashes candidate x coordinates, chooses the even
# square root, clears the BLS12-381 G1 cofactor, and normalizes with AMCL.  In
# particular, it does not expose a known discrete-log relation to G_vk.
OPERATION_KIND = "threshold-sign"
OPERATION_ID = "atcaas-threshold-sign-v1"
MESSAGE_ENCODING = "raw-bytes-hex-v1"
HASH_TO_GROUP_SUITE = (
    "bls12-381-g1-sha256-try-and-increment-even-y-clear-cofactor-amcl-v1"
)
OPERATION_HASH_TO_GROUP_DOMAIN = b"ATCaaS/threshold-service/message-to-g1/v1"
OPERATION_VK_BASE_DOMAIN = b"ATCaaS/threshold-service/vk-base-to-g1/v1"
OPERATION_VK_BASE_MESSAGE = b"G_vk"
HASH_TO_GROUP_TEST_VECTOR_MESSAGE = b"ATCaaS Step-0 HashToGroup test vector"
HASH_TO_GROUP_TEST_VECTOR_POINT_HEX = (
    "(115AB1FB7764B9F038970C5B964B824C7F9F34546432695AA0106B02E7877291E04FC"
    "215E9E8371A33FCA0FEFAA3C035,146363C1642D629765AE33E0137A7A00BEC6F41C74"
    "D73148F4D832B129F8E167F90FBE2636E2F7F8FB21277BD3DDBD5A)"
)

# Qid kinds are disjoint from pure resharing (a2dpss, ru22-dpss, and
# bhk22-dag-handover).  A2DPSS keeps the already implemented Algorithm-5 kind.
A2DPSS_OPERATION_QID_KIND = "ocdpss"
RU22_OPERATION_QID_KIND = "ru22-ocdpss"
BHK22_OPERATION_QID_KINDS = MappingProxyType(
    {
        "deal": "bhk22-ocdpss-deal",
        "verify": "bhk22-ocdpss-verify",
        "response": "bhk22-ocdpss-response",
    }
)
PURE_RESHARING_QID_KINDS = frozenset(
    {"a2dpss", "ru22-dpss", "bhk22-dag-handover"}
)

# In strict YOSO the BHK application arm carries ciphertext forward and has no
# plaintext-finalize speech.  Response is supplementary on the accusation
# path, hence 2 versus 3 TOB blocks.
BHK22_ENCRYPTED_CARRY_FORWARD_BLOCK_COUNTS = MappingProxyType(
    {PATH_OPTIMISTIC: 2, PATH_ONE_ACCUSATION: 3}
)

# RU22 and BHK+22 each carry two separate new operation proofs.  Their entries
# intentionally have different identifiers and must not be serialized as one
# aggregate proof in the first implementation.
RU22_OPERATION_PROOF_KINDS = (
    "ru22-encrypted-commitment-to-vk/v1",
    "vk-to-operation-share/v1",
)
BHK22_OPERATION_PROOF_KINDS = (
    "bhk22-encrypted-public-value-to-vk/v1",
    "vk-to-operation-share/v1",
)

_CONTEXT_DIGEST_DOMAIN = b"ATCaaS/threshold-service/operation-context/v1\x00"
_INPUT_DIGEST_DOMAIN = b"ATCaaS/threshold-service/operation-input/v1\x00"
_APPLICATION_KEY_DIGEST_DOMAIN = (
    b"ATCaaS/threshold-service/application-public-key/v1\x00"
)
_OUTPUT_DIGEST_DOMAIN = b"ATCaaS/threshold-service/operation-output/v1\x00"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> bytes:
    """Encode public contract data in the repository's canonical JSON form."""

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


def _digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase sha256 hex")
    return value


def _input_bytes(value: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("operation_input must be bytes-like")
    return bytes(value)


def _canonical_nonidentity_point(value: str | G1Point, *, name: str) -> str:
    try:
        point = value if isinstance(value, G1Point) else G1Point.from_hex(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a canonical G1 point") from exc
    if point.is_identity():
        raise ValueError(f"{name} must not be the identity point")
    normalized = point.to_hex()
    if isinstance(value, str) and value != normalized:
        raise ValueError(f"{name} must use canonical AMCL G1 encoding")
    return normalized


def derive_operation_base(operation_input: bytes | bytearray | memoryview) -> str:
    """Return the frozen real HashToGroup result for an operation message."""

    return hash_to_g1(OPERATION_HASH_TO_GROUP_DOMAIN, _input_bytes(operation_input))


# G_vk is independent of each request's X and is fixed as a literal test
# vector, rather than silently changing if the derivation helper is edited.
OPERATION_VK_BASE_HEX = (
    "(003EE6267D60BB35FBA16F84B9B1E39E4D04B53C977D0C9EC8FB787A93D2E1494EBB"
    "6313225183C262577531AF63C2A9,0D2F97D66B5044E8FEA9F7712AB0F9E50E77DFC4"
    "F3CF4DD6D6D2D512FDA426314BF0161F144832073C2BB52E616F386F)"
)


def operation_input_digest(
    operation_input: bytes | bytearray | memoryview,
) -> str:
    return hashlib.sha256(_INPUT_DIGEST_DOMAIN + _input_bytes(operation_input)).hexdigest()


def application_public_key_digest(public_key: str | G1Point) -> str:
    canonical = _canonical_nonidentity_point(
        public_key, name="maintained_global_public_key"
    )
    return hashlib.sha256(
        _APPLICATION_KEY_DIGEST_DOMAIN + canonical.encode("ascii")
    ).hexdigest()


def operation_output_digest(
    *,
    operation_id: str,
    operation_input: bytes | bytearray | memoryview,
    operation_output: str | G1Point,
) -> str:
    """Publish a request-bound digest without publishing fixture secrets."""

    if operation_id != OPERATION_ID:
        raise ValueError("unsupported operation_id")
    input_bytes = _input_bytes(operation_input)
    output = _canonical_nonidentity_point(
        operation_output, name="operation_output"
    )
    material = {
        "operation_id": operation_id,
        "operation_input_digest": operation_input_digest(input_bytes),
        "operation_output_hex": output,
    }
    return hashlib.sha256(_OUTPUT_DIGEST_DOMAIN + canonical_json(material)).hexdigest()


def operation_context_digest(material: Mapping[str, Any]) -> str:
    """Digest context statement fields independently of insertion order."""

    if not isinstance(material, Mapping):
        raise TypeError("operation context material must be a mapping")
    return hashlib.sha256(
        _CONTEXT_DIGEST_DOMAIN + canonical_json(dict(material))
    ).hexdigest()


@dataclass(frozen=True)
class ThresholdOperationContext:
    """Canonical public operation statement shared by all three arms."""

    operation_input: bytes
    maintained_global_public_key: str
    expected_operation_output_digest: str
    coin_public_key_digest: str
    operation_kind: str = OPERATION_KIND
    operation_id: str = OPERATION_ID
    message_encoding: str = MESSAGE_ENCODING
    hash_to_group_suite: str = HASH_TO_GROUP_SUITE
    operation_input_digest: str = field(init=False)
    operation_base_hex: str = field(init=False)
    operation_vk_base_hex: str = field(init=False)
    application_public_key_digest: str = field(init=False)
    context_digest: str = field(init=False)

    def __post_init__(self) -> None:
        operation_input = _input_bytes(self.operation_input)
        if self.operation_kind != OPERATION_KIND:
            raise ValueError("unsupported operation_kind")
        if self.operation_id != OPERATION_ID:
            raise ValueError("unsupported operation_id")
        if self.message_encoding != MESSAGE_ENCODING:
            raise ValueError("unsupported operation message encoding")
        if self.hash_to_group_suite != HASH_TO_GROUP_SUITE:
            raise ValueError("unsupported hash_to_group_suite")

        public_key = _canonical_nonidentity_point(
            self.maintained_global_public_key,
            name="maintained_global_public_key",
        )
        expected_digest = _digest(
            self.expected_operation_output_digest,
            name="expected_operation_output_digest",
        )
        coin_digest = _digest(
            self.coin_public_key_digest,
            name="coin_public_key_digest",
        )
        input_digest = operation_input_digest(operation_input)
        operation_base = derive_operation_base(operation_input)
        application_digest = application_public_key_digest(public_key)
        if hmac.compare_digest(application_digest, coin_digest):
            raise ValueError(
                "application and common-coin public-key digests must differ"
            )

        for name, value in (
            ("operation_input", operation_input),
            ("maintained_global_public_key", public_key),
            ("expected_operation_output_digest", expected_digest),
            ("coin_public_key_digest", coin_digest),
            ("operation_input_digest", input_digest),
            ("operation_base_hex", operation_base),
            ("operation_vk_base_hex", OPERATION_VK_BASE_HEX),
            ("application_public_key_digest", application_digest),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "context_digest", operation_context_digest(self.digest_material())
        )

    def digest_material(self) -> dict[str, str]:
        return {
            "schema": OPERATION_CONTEXT_SCHEMA,
            "operation_kind": self.operation_kind,
            "operation_id": self.operation_id,
            "operation_input_hex": self.operation_input.hex(),
            "operation_input_digest": self.operation_input_digest,
            "message_encoding": self.message_encoding,
            "hash_to_group_suite": self.hash_to_group_suite,
            "operation_base_hex": self.operation_base_hex,
            "operation_vk_base_hex": self.operation_vk_base_hex,
            "maintained_global_public_key": self.maintained_global_public_key,
            "application_public_key_digest": self.application_public_key_digest,
            "coin_public_key_digest": self.coin_public_key_digest,
            "expected_operation_output_digest": self.expected_operation_output_digest,
        }

    def to_obj(self) -> dict[str, str]:
        result = self.digest_material()
        result["operation_context_digest"] = self.context_digest
        return result

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "ThresholdOperationContext":
        expected_fields = {
            "schema",
            "operation_kind",
            "operation_id",
            "operation_input_hex",
            "operation_input_digest",
            "message_encoding",
            "hash_to_group_suite",
            "operation_base_hex",
            "operation_vk_base_hex",
            "maintained_global_public_key",
            "application_public_key_digest",
            "coin_public_key_digest",
            "expected_operation_output_digest",
            "operation_context_digest",
        }
        _exact_fields(value, expected_fields, name="threshold operation context")
        if value["schema"] != OPERATION_CONTEXT_SCHEMA:
            raise ValueError("unsupported threshold operation context schema")
        if not isinstance(value["operation_input_hex"], str):
            raise ValueError("operation_input_hex must be lowercase hexadecimal")
        try:
            operation_input = bytes.fromhex(value["operation_input_hex"])
        except ValueError as exc:
            raise ValueError(
                "operation_input_hex must be lowercase hexadecimal"
            ) from exc
        if operation_input.hex() != value["operation_input_hex"]:
            raise ValueError("operation_input_hex must be lowercase hexadecimal")

        supplied_context_digest = _digest(
            value["operation_context_digest"], name="operation_context_digest"
        )
        context = cls(
            operation_input=operation_input,
            maintained_global_public_key=value["maintained_global_public_key"],
            expected_operation_output_digest=value[
                "expected_operation_output_digest"
            ],
            coin_public_key_digest=value["coin_public_key_digest"],
            operation_kind=value["operation_kind"],
            operation_id=value["operation_id"],
            message_encoding=value["message_encoding"],
            hash_to_group_suite=value["hash_to_group_suite"],
        )
        derived_fields = {
            "operation_input_digest": context.operation_input_digest,
            "operation_base_hex": context.operation_base_hex,
            "operation_vk_base_hex": context.operation_vk_base_hex,
            "application_public_key_digest": context.application_public_key_digest,
        }
        for name, expected in derived_fields.items():
            if not hmac.compare_digest(str(value[name]), expected):
                raise ValueError(f"{name} does not match the operation statement")
        if not hmac.compare_digest(context.context_digest, supplied_context_digest):
            raise ValueError(
                "operation_context_digest does not match the operation statement"
            )
        return context

    @classmethod
    def from_bytes(
        cls, value: bytes | bytearray | memoryview | str
    ) -> "ThresholdOperationContext":
        return cls.from_obj(_decode_object(value, name="threshold operation context"))


# Exact checks used by protocol-specific justifiers.  The pure function below
# makes missing, unknown, or non-boolean decisions fail closed.
ACCEPTANCE_REQUIREMENTS = MappingProxyType(
    {
        "a2dpss": (
            "base_resharing_valid",
            "pedersen_to_operation_share_proof_valid",
            "same_selected_set",
            "operation_output_matches_expected",
        ),
        "ru22": (
            "base_resharing_valid",
            "encrypted_commitment_to_vk_proof_valid",
            "vk_to_operation_share_proof_valid",
            "same_selected_set",
            "operation_output_matches_expected",
        ),
        "bhk22": (
            "base_resharing_valid",
            "encrypted_public_value_to_vk_proof_valid",
            "vk_to_operation_share_proof_valid",
            "figure7_resharing_anchored_to_same_vk",
            "terminal_encrypted_state_matches",
            "same_selected_set",
            "operation_output_matches_expected",
        ),
    }
)


def accepts_operation_candidate(protocol: str, checks: Mapping[str, bool]) -> bool:
    """Evaluate the frozen protocol-independent acceptance truth table."""

    requirements = ACCEPTANCE_REQUIREMENTS.get(protocol)
    if requirements is None or not isinstance(checks, Mapping):
        return False
    if set(checks) != set(requirements):
        return False
    return all(type(checks[name]) is bool and checks[name] for name in requirements)


def operation_qid_kind(protocol: str, *, phase: str | None = None) -> str:
    """Resolve the frozen application qid kind without constructing wire data."""

    if protocol == "a2dpss" and phase is None:
        return A2DPSS_OPERATION_QID_KIND
    if protocol == "ru22" and phase is None:
        return RU22_OPERATION_QID_KIND
    if protocol == "bhk22" and phase in BHK22_OPERATION_QID_KINDS:
        return BHK22_OPERATION_QID_KINDS[phase]
    raise ValueError("unsupported protocol/phase for operation qid")


def is_operation_qid_kind(kind: str) -> bool:
    operation_kinds = {
        A2DPSS_OPERATION_QID_KIND,
        RU22_OPERATION_QID_KIND,
        *BHK22_OPERATION_QID_KINDS.values(),
    }
    return isinstance(kind, str) and kind in operation_kinds


# Catch accidental divergence from the already frozen BHK encrypted phases at
# import/test time, while continuing to leave the pure BHK contract untouched.
assert {
    path: len(CARRY_FORWARD_PATH_PHASES[path])
    for path in BHK22_ENCRYPTED_CARRY_FORWARD_BLOCK_COUNTS
} == dict(BHK22_ENCRYPTED_CARRY_FORWARD_BLOCK_COUNTS)
assert not PURE_RESHARING_QID_KINDS.intersection(
    {
        A2DPSS_OPERATION_QID_KIND,
        RU22_OPERATION_QID_KIND,
        *BHK22_OPERATION_QID_KINDS.values(),
    }
)
assert hash_to_g1(
    OPERATION_VK_BASE_DOMAIN, OPERATION_VK_BASE_MESSAGE
) == OPERATION_VK_BASE_HEX
assert (
    derive_operation_base(HASH_TO_GROUP_TEST_VECTOR_MESSAGE)
    == HASH_TO_GROUP_TEST_VECTOR_POINT_HEX
)


__all__ = [
    "A2DPSS_OPERATION_QID_KIND",
    "ACCEPTANCE_REQUIREMENTS",
    "ARM_ORDER",
    "ARM_SCHEMA",
    "BHK22_ENCRYPTED_CARRY_FORWARD_BLOCK_COUNTS",
    "BHK22_OPERATION_CONTEXT_SCHEMA",
    "BHK22_OPERATION_DEALER_BATCH_SCHEMA",
    "BHK22_OPERATION_PROOF_KINDS",
    "BHK22_OPERATION_QID_KINDS",
    "HASH_TO_GROUP_SUITE",
    "HASH_TO_GROUP_TEST_VECTOR_MESSAGE",
    "HASH_TO_GROUP_TEST_VECTOR_POINT_HEX",
    "MESSAGE_ENCODING",
    "OPERATION_CONTEXT_SCHEMA",
    "OPERATION_HASH_TO_GROUP_DOMAIN",
    "OPERATION_ID",
    "OPERATION_KIND",
    "OPERATION_VK_BASE_HEX",
    "PRIVATE_SHARD_SCHEMA",
    "PURE_RESHARING_QID_KINDS",
    "RU22_OPERATION_CONTEXT_SCHEMA",
    "RU22_OPERATION_PROOF_KINDS",
    "RU22_OPERATION_PVR_SCHEMA",
    "RU22_OPERATION_QID_KIND",
    "ThresholdOperationContext",
    "WORKLOAD_SCHEMA",
    "accepts_operation_candidate",
    "application_public_key_digest",
    "canonical_json",
    "derive_operation_base",
    "is_operation_qid_kind",
    "operation_context_digest",
    "operation_input_digest",
    "operation_output_digest",
    "operation_qid_kind",
]
