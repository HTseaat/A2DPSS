"""Local protocol layer for Algorithm-5 operation-coupled DPSS.

The module owns the canonical context/bundle wire model, one-shot holder
action, self-contained full-bundle justifier, public finalization, and private
successor reconstruction.  Network YOSO-RBC and runner integration remain a
separate Algorithm-5 transport layer and are intentionally not implemented here.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import string
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from beaver.a2dpss import (
    A2DPSSContext,
    A2DPSSContribution,
    A2DPSSHolderState,
    A2DPSSPublicResult,
    A2DPSSSuccessorResult,
    a2dpss_contribution_digest,
    finalize_a2_validated_items,
    make_a2dpss_contribution,
    reconstruct_a2_validated_items,
    select_canonical_threshold,
    validate_a2_contribution,
)
from beaver.dag_tob import Justifier, PoolEntry, Qid
from beaver.g1_backend import FIELD_ORDER, G1Point
from beaver.operation_share import OperationAdapter, OperationShareProof
from beaver.pvss import (
    LEGACY_TRANSCRIPT_VERSION,
    SUPPORTED_TRANSCRIPT_VERSIONS,
    TRANSCRIPT_VERSION,
    PVSSTranscript,
)


OCDPSS_CONTEXT_VERSION = 1
OCDPSS_BUNDLE_VERSION = 1
OCDPSS_PUBLIC_RESULT_VERSION = 1
OCDPSS_SUCCESSOR_RESULT_VERSION = 1
OCDPSS_QID_KIND = "ocdpss"

_CONTEXT_KIND = "atcaas-ocdpss-context"
_BUNDLE_KIND = "atcaas-ocdpss-bundle"
_CONTEXT_DIGEST_DOMAIN = b"ATCaaS/OCDPSS/context/v1\x00"
_BUNDLE_DIGEST_DOMAIN = b"ATCaaS/OCDPSS/bundle/v1\x00"
_PUBLIC_RESULT_KIND = "atcaas-ocdpss-public-result"
_SUCCESSOR_RESULT_KIND = "atcaas-ocdpss-successor-result"


def _canonical_json(value: Any) -> bytes:
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


def _decode_json_object(data: bytes | bytearray | memoryview | str, *, name: str) -> dict:
    try:
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed {name} JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must encode a JSON object")
    return value


def _typed_canonical(value: Any) -> Any:
    """Return an unambiguous JSON representation for digesting public data."""

    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ["bytes", bytes(value).hex()]
    if isinstance(value, Mapping):
        entries = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical context mapping keys must be strings")
            entries.append([key, _typed_canonical(item)])
        entries.sort(key=lambda pair: pair[0])
        return ["map", entries]
    if isinstance(value, (list, tuple)):
        return ["list", [_typed_canonical(item) for item in value]]
    raise TypeError(f"unsupported canonical context value: {type(value).__name__}")


def canonical_context_digest(context: Mapping[str, Any] | "OCDPSSContext") -> str:
    """Hash a context statement independently of mapping insertion order.

    The derived ``context_digest`` field itself is omitted.  All other fields,
    including the format version, are domain-separated and type-tagged.
    """

    if isinstance(context, OCDPSSContext):
        material = context.digest_material()
    elif isinstance(context, Mapping):
        material = {
            key: value
            for key, value in context.items()
            if key != "context_digest"
        }
    else:
        raise TypeError("context must be an OCDPSSContext or mapping")
    encoded = _canonical_json(_typed_canonical(material))
    return hashlib.sha256(_CONTEXT_DIGEST_DOMAIN + encoded).hexdigest()


def _freeze_public_value(value: Any) -> Any:
    """Copy JSON-like public parameters into an immutable representation."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        copied = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("public-parameter mapping keys must be strings")
            copied[key] = _freeze_public_value(item)
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_public_value(item) for item in value)
    raise TypeError(
        "public data must contain only null, bool, int, str, mappings, and sequences"
    )


def _thaw_public_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_public_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_public_value(item) for item in value]
    return value


def _strict_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _text_sequence(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(_nonempty_text(item, name=f"{name} item") for item in value)
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _normalize_int_mapping(
    value: Mapping[Any, Any], *, name: str, value_minimum: int
) -> Mapping[int, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized = {}
    for raw_key, raw_value in value.items():
        if isinstance(raw_key, bool):
            raise TypeError(f"{name} keys must be integer node ids")
        if isinstance(raw_key, int):
            key = raw_key
        elif isinstance(raw_key, str) and (
            raw_key == "0" or (raw_key.isdigit() and not raw_key.startswith("0"))
        ):
            key = int(raw_key)
        else:
            raise TypeError(f"{name} keys must be canonical integer node ids")
        if key < 0:
            raise ValueError(f"{name} node ids must be non-negative")
        if key in normalized:
            raise ValueError(f"{name} contains duplicate normalized node id {key}")
        normalized[key] = _strict_int(
            raw_value,
            name=f"{name}[{key}]",
            minimum=value_minimum,
        )
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return MappingProxyType(dict(sorted(normalized.items())))


def _operation_input(value: bytes | bytearray | memoryview | str) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError("operation_input must be bytes-like or str")


def _context_digest(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("context_digest must be a hexadecimal string")
    if len(value) != 64 or value != value.lower() or any(
        character not in string.hexdigits.lower() for character in value
    ):
        raise ValueError("context_digest must be 64 lowercase hexadecimal characters")
    return value


@dataclass(frozen=True)
class OCDPSSContext:
    """Complete public statement for one Algorithm-5 handover instance."""

    epoch: int
    sid: str
    source_committee: str
    successor_committee: str
    transfer_seq: int
    g_hex: str
    h_hex: str
    successor_rpk_list: Sequence[str]
    service_t: int
    old_commitment_vector: Sequence[str]
    holder_eval_points: Mapping[int, int]
    successor_role_indices: Mapping[int, int]
    operation_id: str
    operation_input: bytes | str
    operation_public_parameters: Mapping[str, Any]
    key_base: str = ""
    version: int = OCDPSS_CONTEXT_VERSION
    context_digest: str = field(init=False)

    def __post_init__(self) -> None:
        version = _strict_int(self.version, name="version")
        if version != OCDPSS_CONTEXT_VERSION:
            raise ValueError("unsupported OCDPSS context version")
        epoch = _strict_int(self.epoch, name="epoch")
        transfer_seq = _strict_int(self.transfer_seq, name="transfer_seq")
        service_t = _strict_int(self.service_t, name="service_t")
        sid = _nonempty_text(self.sid, name="sid")
        source = _nonempty_text(self.source_committee, name="source_committee")
        successor = _nonempty_text(
            self.successor_committee, name="successor_committee"
        )
        operation_id = _nonempty_text(self.operation_id, name="operation_id")
        g_hex = _nonempty_text(self.g_hex, name="g_hex")
        h_hex = _nonempty_text(self.h_hex, name="h_hex")
        if g_hex == h_hex:
            raise ValueError("g_hex and h_hex must be distinct")
        key_base = _nonempty_text(self.key_base or g_hex, name="key_base")
        rpk_list = _text_sequence(
            self.successor_rpk_list, name="successor_rpk_list"
        )
        if len(rpk_list) < service_t + 1:
            raise ValueError(
                "successor_rpk_list must contain at least service_t+1 keys"
            )
        if len(set(rpk_list)) != len(rpk_list):
            raise ValueError("successor_rpk_list entries must be distinct")
        old_vector = _text_sequence(
            self.old_commitment_vector, name="old_commitment_vector"
        )
        holder_points = _normalize_int_mapping(
            self.holder_eval_points,
            name="holder_eval_points",
            value_minimum=1,
        )
        if len(set(holder_points.values())) != len(holder_points):
            raise ValueError("holder eval points must be distinct")
        if max(holder_points.values()) >= len(old_vector):
            raise ValueError("old_commitment_vector does not cover holder eval points")
        successor_indices = _normalize_int_mapping(
            self.successor_role_indices,
            name="successor_role_indices",
            value_minimum=0,
        )
        if len(set(successor_indices.values())) != len(successor_indices):
            raise ValueError("successor role indices must be distinct")
        if max(successor_indices.values()) >= len(rpk_list):
            raise ValueError("successor role index is outside successor_rpk_list")
        if not isinstance(self.operation_public_parameters, Mapping):
            raise TypeError("operation_public_parameters must be a mapping")
        operation_parameters = _freeze_public_value(
            self.operation_public_parameters
        )

        object.__setattr__(self, "version", version)
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "sid", sid)
        object.__setattr__(self, "source_committee", source)
        object.__setattr__(self, "successor_committee", successor)
        object.__setattr__(self, "transfer_seq", transfer_seq)
        object.__setattr__(self, "g_hex", g_hex)
        object.__setattr__(self, "h_hex", h_hex)
        object.__setattr__(self, "key_base", key_base)
        object.__setattr__(self, "successor_rpk_list", rpk_list)
        object.__setattr__(self, "service_t", service_t)
        object.__setattr__(self, "old_commitment_vector", old_vector)
        object.__setattr__(self, "holder_eval_points", holder_points)
        object.__setattr__(self, "successor_role_indices", successor_indices)
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "operation_input", _operation_input(self.operation_input))
        object.__setattr__(
            self, "operation_public_parameters", operation_parameters
        )
        object.__setattr__(self, "context_digest", canonical_context_digest(self))

    def digest_material(self) -> dict[str, Any]:
        return {
            "kind": _CONTEXT_KIND,
            "version": self.version,
            "epoch": self.epoch,
            "sid": self.sid,
            "source_committee": self.source_committee,
            "successor_committee": self.successor_committee,
            "transfer_seq": self.transfer_seq,
            "g_hex": self.g_hex,
            "h_hex": self.h_hex,
            "key_base": self.key_base,
            "successor_rpk_list": list(self.successor_rpk_list),
            "service_t": self.service_t,
            "old_commitment_vector": list(self.old_commitment_vector),
            "holder_eval_points": {
                str(node_id): eval_point
                for node_id, eval_point in self.holder_eval_points.items()
            },
            "successor_role_indices": {
                str(node_id): role_index
                for node_id, role_index in self.successor_role_indices.items()
            },
            "operation_id": self.operation_id,
            "operation_input": self.operation_input,
            "operation_public_parameters": _thaw_public_value(
                self.operation_public_parameters
            ),
        }

    def to_obj(self) -> dict[str, Any]:
        material = self.digest_material()
        material["operation_input"] = self.operation_input.hex()
        material["operation_input_encoding"] = "hex"
        material["context_digest"] = self.context_digest
        return material

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "OCDPSSContext":
        if not isinstance(value, Mapping):
            raise ValueError("OCDPSS context must be a mapping")
        expected_fields = {
            "kind",
            "version",
            "epoch",
            "sid",
            "source_committee",
            "successor_committee",
            "transfer_seq",
            "g_hex",
            "h_hex",
            "key_base",
            "successor_rpk_list",
            "service_t",
            "old_commitment_vector",
            "holder_eval_points",
            "successor_role_indices",
            "operation_id",
            "operation_input",
            "operation_input_encoding",
            "operation_public_parameters",
            "context_digest",
        }
        missing = expected_fields - set(value)
        extra = set(value) - expected_fields
        if missing:
            raise ValueError(f"OCDPSS context is missing fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"OCDPSS context has unknown fields: {sorted(extra)}")
        if value["kind"] != _CONTEXT_KIND:
            raise ValueError("expected an OCDPSS context object")
        if value["version"] != OCDPSS_CONTEXT_VERSION:
            raise ValueError("unsupported OCDPSS context version")
        if value["operation_input_encoding"] != "hex":
            raise ValueError("unsupported OCDPSS operation_input encoding")
        try:
            operation_input = bytes.fromhex(value["operation_input"])
        except (TypeError, ValueError) as exc:
            raise ValueError("operation_input must be canonical hexadecimal") from exc
        if operation_input.hex() != value["operation_input"]:
            raise ValueError("operation_input must be lowercase canonical hexadecimal")
        expected_digest = _context_digest(value["context_digest"])
        context = cls(
            version=value["version"],
            epoch=value["epoch"],
            sid=value["sid"],
            source_committee=value["source_committee"],
            successor_committee=value["successor_committee"],
            transfer_seq=value["transfer_seq"],
            g_hex=value["g_hex"],
            h_hex=value["h_hex"],
            key_base=value["key_base"],
            successor_rpk_list=value["successor_rpk_list"],
            service_t=value["service_t"],
            old_commitment_vector=value["old_commitment_vector"],
            holder_eval_points=value["holder_eval_points"],
            successor_role_indices=value["successor_role_indices"],
            operation_id=value["operation_id"],
            operation_input=operation_input,
            operation_public_parameters=value["operation_public_parameters"],
        )
        if not hmac.compare_digest(context.context_digest, expected_digest):
            raise ValueError("OCDPSS context_digest does not match public context")
        return context

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray | memoryview | str
    ) -> "OCDPSSContext":
        return cls.from_obj(_decode_json_object(data, name="OCDPSS context"))

    def build_qid(self) -> Qid:
        return build_ocdpss_qid(self)

    def encode_qid(self) -> str:
        return self.build_qid().encode()


def build_ocdpss_qid(context: OCDPSSContext) -> Qid:
    """Build a TOB qid that commits to the complete OCDPSS public context."""

    if not isinstance(context, OCDPSSContext):
        raise TypeError("context must be an OCDPSSContext")
    return Qid.build(
        OCDPSS_QID_KIND,
        epoch=context.epoch,
        sid=context.sid,
        source_committee=context.source_committee,
        successor_committee=context.successor_committee,
        transfer_seq=context.transfer_seq,
        operation_id=context.operation_id,
        context_digest=context.context_digest,
    )


def is_ocdpss_qid(qid: Qid | str) -> bool:
    if isinstance(qid, str):
        qid = Qid.decode(qid)
    return isinstance(qid, Qid) and qid.kind == OCDPSS_QID_KIND


def ocdpss_qid_matches_context(qid: Qid | str, context: OCDPSSContext) -> bool:
    try:
        if isinstance(qid, str):
            qid = Qid.decode(qid)
        return qid == build_ocdpss_qid(context)
    except (TypeError, ValueError):
        return False


def _transcript_from_obj(value: Any) -> PVSSTranscript:
    if not isinstance(value, dict):
        raise ValueError("bundle transcript must be an object")
    version = value.get("version")
    expected = {
        "version",
        "dealer_id",
        "dealer_eval_point",
        "commitment_vector",
        "enc_results",
        "pi_ve",
    }
    if version == TRANSCRIPT_VERSION:
        expected.add("derived_c_v_vector")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing:
        raise ValueError(f"bundle transcript is missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"bundle transcript has unknown fields: {sorted(extra)}")
    if version not in SUPPORTED_TRANSCRIPT_VERSIONS:
        raise ValueError("unsupported PVSS transcript version in OCDPSS bundle")
    if (
        version == LEGACY_TRANSCRIPT_VERSION
        and "derived_c_v_vector" in value
    ):
        raise ValueError(
            "legacy PVSS transcript cannot carry derived_c_v_vector"
        )
    _strict_int(value["dealer_id"], name="transcript dealer_id")
    _strict_int(
        value["dealer_eval_point"],
        name="transcript dealer_eval_point",
        minimum=1,
    )
    try:
        return PVSSTranscript.from_bytes(_canonical_json(value))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("malformed PVSS transcript in OCDPSS bundle") from exc


@dataclass(frozen=True)
class OCDPSSBundle:
    """One Algorithm-5 holder output ``(transcript, y_i, rho_i)``."""

    holder_id: int
    holder_eval_point: int
    context_digest: str
    transcript: PVSSTranscript
    operation_share: Mapping[str, Any]
    operation_proof: Mapping[str, Any]
    version: int = OCDPSS_BUNDLE_VERSION

    def __post_init__(self) -> None:
        version = _strict_int(self.version, name="version")
        if version != OCDPSS_BUNDLE_VERSION:
            raise ValueError("unsupported OCDPSS bundle version")
        holder_id = _strict_int(self.holder_id, name="holder_id")
        holder_eval_point = _strict_int(
            self.holder_eval_point, name="holder_eval_point", minimum=1
        )
        digest = _context_digest(self.context_digest)
        if not isinstance(self.transcript, PVSSTranscript):
            raise TypeError("transcript must be a PVSSTranscript")
        _strict_int(self.transcript.dealer_id, name="transcript dealer_id")
        _strict_int(
            self.transcript.dealer_eval_point,
            name="transcript dealer_eval_point",
            minimum=1,
        )
        if self.transcript.dealer_id != holder_id:
            raise ValueError("bundle holder_id does not match transcript dealer_id")
        if self.transcript.dealer_eval_point != holder_eval_point:
            raise ValueError(
                "bundle holder_eval_point does not match transcript dealer_eval_point"
            )
        if not isinstance(self.operation_share, Mapping) or not self.operation_share:
            raise ValueError("bundle operation_share must be a non-empty mapping")
        if not isinstance(self.operation_proof, Mapping) or not self.operation_proof:
            raise ValueError("bundle operation_proof must be a non-empty mapping")

        object.__setattr__(self, "version", version)
        object.__setattr__(self, "holder_id", holder_id)
        object.__setattr__(self, "holder_eval_point", holder_eval_point)
        object.__setattr__(self, "context_digest", digest)
        object.__setattr__(
            self, "operation_share", _freeze_public_value(self.operation_share)
        )
        object.__setattr__(
            self, "operation_proof", _freeze_public_value(self.operation_proof)
        )

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": _BUNDLE_KIND,
            "version": self.version,
            "holder_id": self.holder_id,
            "holder_eval_point": self.holder_eval_point,
            "context_digest": self.context_digest,
            "transcript": self.transcript.to_obj(),
            "operation_share": _thaw_public_value(self.operation_share),
            "operation_proof": _thaw_public_value(self.operation_proof),
        }

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_obj())

    @property
    def a2_contribution(self) -> Optional[A2DPSSContribution]:
        """Return the local A²DPSS component when it is available.

        The component is deliberately not a v1 wire field.  A bundle decoded
        in another process can reconstruct it from the registered projected
        A²DPSS context.
        """

        return getattr(self, "_a2_contribution", None)

    def _attach_a2_contribution(
        self,
        contribution: A2DPSSContribution,
    ) -> None:
        if not isinstance(contribution, A2DPSSContribution):
            raise TypeError(
                "contribution must be an A2DPSSContribution"
            )
        if (
            contribution.holder_id != self.holder_id
            or contribution.holder_eval_point
            != self.holder_eval_point
            or contribution.transcript != self.transcript
        ):
            raise ValueError(
                "A2DPSS contribution does not match OCDPSS bundle"
            )
        existing = self.a2_contribution
        if existing is not None and existing != contribution:
            raise ValueError(
                "conflicting A2DPSS contribution attachment"
            )
        object.__setattr__(
            self,
            "_a2_contribution",
            contribution,
        )

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "OCDPSSBundle":
        if not isinstance(value, Mapping):
            raise ValueError("OCDPSS bundle must be a mapping")
        expected_fields = {
            "kind",
            "version",
            "holder_id",
            "holder_eval_point",
            "context_digest",
            "transcript",
            "operation_share",
            "operation_proof",
        }
        missing = expected_fields - set(value)
        extra = set(value) - expected_fields
        if missing:
            raise ValueError(f"OCDPSS bundle is missing fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"OCDPSS bundle has unknown fields: {sorted(extra)}")
        if value["kind"] != _BUNDLE_KIND:
            raise ValueError("expected an OCDPSS bundle object")
        if value["version"] != OCDPSS_BUNDLE_VERSION:
            raise ValueError("unsupported OCDPSS bundle version")
        transcript = _transcript_from_obj(value["transcript"])
        return cls(
            version=value["version"],
            holder_id=value["holder_id"],
            holder_eval_point=value["holder_eval_point"],
            context_digest=value["context_digest"],
            transcript=transcript,
            operation_share=value["operation_share"],
            operation_proof=value["operation_proof"],
        )

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray | memoryview | str
    ) -> "OCDPSSBundle":
        return cls.from_obj(_decode_json_object(data, name="OCDPSS bundle"))


def _normalize_pool_entry(value: Any) -> Optional[PoolEntry]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    mid, payload = value
    if not isinstance(mid, str) or not mid:
        return None
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        return None
    return mid, bytes(payload)


def _stable_pool_entry(value: PoolEntry) -> tuple[str, str]:
    return value[0], value[1].hex()


def _a2_context_from_ocdpss(context: OCDPSSContext) -> A2DPSSContext:
    """Project the operation-coupled context onto its A²DPSS statement."""

    if not isinstance(context, OCDPSSContext):
        raise TypeError("context must be an OCDPSSContext")
    return A2DPSSContext(
        epoch=context.epoch,
        sid=context.sid,
        source_committee=context.source_committee,
        successor_committee=context.successor_committee,
        transfer_seq=context.transfer_seq,
        g_hex=context.g_hex,
        h_hex=context.h_hex,
        key_base=context.key_base,
        successor_rpk_list=context.successor_rpk_list,
        service_t=context.service_t,
        old_commitment_vector=context.old_commitment_vector,
        holder_eval_points=context.holder_eval_points,
        successor_role_indices=context.successor_role_indices,
    )


def _a2_contribution_from_bundle(
    a2_context: A2DPSSContext,
    bundle: OCDPSSBundle,
) -> A2DPSSContribution:
    """Project a flat OCDPSS v1 bundle without changing its wire encoding."""

    attached = bundle.a2_contribution
    if attached is not None:
        if attached.context_digest != a2_context.context_digest:
            raise ValueError(
                "attached A2DPSS contribution context mismatch"
            )
        return attached
    contribution = A2DPSSContribution(
        holder_id=bundle.holder_id,
        holder_eval_point=bundle.holder_eval_point,
        context_digest=a2_context.context_digest,
        transcript=bundle.transcript,
    )
    bundle._attach_a2_contribution(contribution)
    return contribution


class OCDPSSJustifierManager(Justifier):
    """Self-contained Algorithm-5 full-bundle predicate.

    Each registered context has the exact operation adapter needed to verify
    its operation shares.  Selection and verification depend only on the qid,
    the candidate entries, and this immutable public registration.
    """

    def __init__(
        self,
        registrations: Optional[
            Sequence[Tuple[OCDPSSContext, OperationAdapter]]
        ] = None,
        validation_cache_limit: int = 4096,
    ) -> None:
        validation_cache_limit = _strict_int(
            validation_cache_limit,
            name="validation_cache_limit",
            minimum=1,
        )
        self.contexts: Dict[str, OCDPSSContext] = {}
        self.operation_adapters: Dict[str, OperationAdapter] = {}
        self.a2_contexts: Dict[str, A2DPSSContext] = {}
        self.validation_cache_limit = validation_cache_limit
        self._validation_cache: Dict[tuple[str, int, str, str], bool] = {}
        self._validation_metrics = {
            stage: {
                "invocations": 0,
                "elapsed_sec": 0.0,
                "cpu_seconds": 0.0,
            }
            for stage in ("schema_identity", "pvss", "anchor", "operation_proof")
        }
        self._validation_cache_hits = 0
        for context, operation_adapter in registrations or ():
            self.register_context(context, operation_adapter)

    def _record_validation_metric(
        self, stage: str, started: float, cpu_started: float
    ) -> None:
        metric = self._validation_metrics[stage]
        metric["invocations"] += 1
        metric["elapsed_sec"] += time.perf_counter() - started
        metric["cpu_seconds"] += time.process_time() - cpu_started

    def _record_base_validation_elapsed(
        self,
        stage: str,
        elapsed: float,
        cpu_seconds: float,
    ) -> None:
        metric = self._validation_metrics[stage]
        metric["invocations"] += 1
        metric["elapsed_sec"] += float(elapsed)
        metric["cpu_seconds"] += float(cpu_seconds)

    def validation_metrics_snapshot(self) -> dict:
        return {
            "cache_hits": self._validation_cache_hits,
            **{
                stage: {
                    "invocations": int(metric["invocations"]),
                    "elapsed_sec": float(metric["elapsed_sec"]),
                    "cpu_seconds": float(metric["cpu_seconds"]),
                }
                for stage, metric in self._validation_metrics.items()
            },
        }

    def register_context(
        self,
        context: OCDPSSContext,
        operation_adapter: OperationAdapter,
    ) -> None:
        if not isinstance(context, OCDPSSContext):
            raise TypeError("context must be an OCDPSSContext")
        _validate_operation_adapter(context, operation_adapter)
        existing = self.contexts.get(context.context_digest)
        if existing is not None and existing != context:
            raise ValueError("conflicting OCDPSS context for context_digest")
        a2_context = _a2_context_from_ocdpss(context)
        self._drop_cached_context(context.context_digest)
        self.contexts[context.context_digest] = context
        self.operation_adapters[context.context_digest] = operation_adapter
        self.a2_contexts[context.context_digest] = a2_context

    def unregister_context(self, context_or_digest: OCDPSSContext | str) -> None:
        digest = (
            context_or_digest.context_digest
            if isinstance(context_or_digest, OCDPSSContext)
            else _context_digest(context_or_digest)
        )
        self.contexts.pop(digest, None)
        self.operation_adapters.pop(digest, None)
        self.a2_contexts.pop(digest, None)
        self._drop_cached_context(digest)

    def _a2_context_for(self, context: OCDPSSContext) -> A2DPSSContext:
        registered = self.contexts.get(context.context_digest)
        if registered == context:
            try:
                return self.a2_contexts[context.context_digest]
            except KeyError:
                pass
        return _a2_context_from_ocdpss(context)

    def _drop_cached_context(self, context_digest: str) -> None:
        for key in tuple(self._validation_cache):
            if key[0] == context_digest:
                self._validation_cache.pop(key, None)

    def _cache_validation(
        self, key: tuple[str, int, str, str], valid: bool
    ) -> None:
        if key not in self._validation_cache and (
            len(self._validation_cache) >= self.validation_cache_limit
        ):
            self._validation_cache.pop(next(iter(self._validation_cache)))
        self._validation_cache[key] = bool(valid)

    def _registration(
        self, qid: Qid | str
    ) -> tuple[OCDPSSContext, OperationAdapter]:
        if isinstance(qid, str):
            qid = Qid.decode(qid)
        if not isinstance(qid, Qid) or qid.kind != OCDPSS_QID_KIND:
            raise ValueError("expected an OCDPSS qid")
        fields = dict(qid.fields)
        digest = _context_digest(fields.get("context_digest"))
        try:
            context = self.contexts[digest]
            operation_adapter = self.operation_adapters[digest]
        except KeyError as exc:
            raise KeyError(f"unknown OCDPSS context: {digest}") from exc
        if not ocdpss_qid_matches_context(qid, context):
            raise ValueError("OCDPSS qid fields do not match registered context")
        return context, operation_adapter

    def validate_bundle(
        self,
        context: OCDPSSContext,
        operation_adapter: OperationAdapter,
        mid: str,
        payload: bytes,
    ) -> Optional[OCDPSSBundle]:
        """Return the decoded bundle only if every Algorithm-5 check passes."""

        cache_key = (
            context.context_digest,
            id(operation_adapter),
            str(mid),
            hashlib.sha256(bytes(payload)).hexdigest()
            if isinstance(payload, (bytes, bytearray, memoryview))
            else "malformed",
        )
        if cache_key in self._validation_cache:
            self._validation_cache_hits += 1
            if not self._validation_cache[cache_key]:
                return None
            try:
                return OCDPSSBundle.from_bytes(payload)
            except Exception:
                return None
        try:
            bundle = OCDPSSBundle.from_bytes(payload)
            if not hmac.compare_digest(
                bundle.context_digest, context.context_digest
            ):
                self._cache_validation(cache_key, False)
                return None
            if mid != f"node_{bundle.holder_id}":
                self._cache_validation(cache_key, False)
                return None
            a2_context = self._a2_context_for(context)
            a2_contribution = _a2_contribution_from_bundle(
                a2_context, bundle
            )
            if not validate_a2_contribution(
                a2_context,
                a2_contribution,
                stage_observer=self._record_base_validation_elapsed,
            ):
                self._cache_validation(cache_key, False)
                return None
            if set(bundle.operation_share) != {"point"}:
                self._cache_validation(cache_key, False)
                return None
            operation_share = bundle.operation_share["point"]
            if not isinstance(operation_share, str) or not operation_share:
                self._cache_validation(cache_key, False)
                return None
            expected_eval_point = a2_contribution.holder_eval_point
            commitment_i = context.old_commitment_vector[expected_eval_point]
            started = time.perf_counter()
            cpu_started = time.process_time()
            if not operation_adapter.verify_share(
                context.operation_input,
                commitment_i=commitment_i,
                operation_share_i=operation_share,
                proof=bundle.operation_proof,
            ):
                self._cache_validation(cache_key, False)
                return None
            self._record_validation_metric(
                "operation_proof", started, cpu_started
            )
            self._cache_validation(cache_key, True)
            return bundle
        except Exception:
            self._cache_validation(cache_key, False)
            return None

    def select_justified_subset(
        self,
        qid: Qid,
        entries: Sequence[PoolEntry],
    ) -> Optional[List[PoolEntry]]:
        try:
            context, operation_adapter = self._registration(qid)
        except (KeyError, TypeError, ValueError):
            return None

        normalized_entries = []
        for entry in entries:
            normalized = _normalize_pool_entry(entry)
            if normalized is not None:
                normalized_entries.append(normalized)

        a2_context = self._a2_context_for(context)
        validated_items = []
        for mid, payload in sorted(normalized_entries, key=_stable_pool_entry):
            bundle = self.validate_bundle(
                context, operation_adapter, mid, payload
            )
            if bundle is None:
                continue
            a2_contribution = _a2_contribution_from_bundle(
                a2_context, bundle
            )
            validated_items.append(
                ((mid, payload), a2_contribution)
            )
            selected = select_canonical_threshold(
                validated_items, context.service_t + 1
            )
            if selected is not None:
                return selected
        return None

    def verify_qid_set(
        self,
        qid: Qid,
        entries: Sequence[PoolEntry],
    ) -> bool:
        try:
            context, operation_adapter = self._registration(qid)
        except (KeyError, TypeError, ValueError):
            return False
        if len(entries) != context.service_t + 1:
            return False

        normalized_entries = []
        for entry in entries:
            normalized = _normalize_pool_entry(entry)
            if normalized is None:
                return False
            normalized_entries.append(normalized)

        a2_context = self._a2_context_for(context)
        validated_items = []
        for mid, payload in sorted(normalized_entries, key=_stable_pool_entry):
            bundle = self.validate_bundle(
                context, operation_adapter, mid, payload
            )
            if bundle is None:
                return False
            validated_items.append(
                (
                    (mid, payload),
                    _a2_contribution_from_bundle(a2_context, bundle),
                )
            )
        return (
            select_canonical_threshold(
                validated_items, context.service_t + 1
            )
            is not None
        )


def ocdpss_bundle_digest(payload: bytes | bytearray | memoryview) -> str:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("bundle payload must be bytes-like")
    return hashlib.sha256(_BUNDLE_DIGEST_DOMAIN + bytes(payload)).hexdigest()


def _canonical_ocdpss_qid(value: Qid | str) -> str:
    encoded = value.encode() if isinstance(value, Qid) else str(value)
    decoded = Qid.decode(encoded)
    if decoded.kind != OCDPSS_QID_KIND or decoded.encode() != encoded:
        raise ValueError("qid must be a canonical OCDPSS qid")
    return encoded


def _int_tuple(
    value: Sequence[int], *, name: str, minimum: int
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    normalized = tuple(
        _strict_int(item, name=f"{name} item", minimum=minimum)
        for item in value
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} entries must be distinct")
    return normalized


def _digest_tuple(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    normalized = tuple(_context_digest(item) for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} entries must be distinct")
    return normalized


def _normalized_point_tuple(
    value: Sequence[str], *, name: str
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    try:
        return tuple(G1Point.from_hex(item).to_hex() for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains a malformed G1 point") from exc


@dataclass(frozen=True)
class OCDPSSPublicResult:
    """Versioned public output derived by every observer from one fixed T."""

    qid: str
    context_digest: str
    accepted_holder_ids: Sequence[int]
    accepted_eval_points: Sequence[int]
    accepted_bundle_digests: Sequence[str]
    operation_output: Any
    successor_commitment_vector: Sequence[str]
    version: int = OCDPSS_PUBLIC_RESULT_VERSION

    def __post_init__(self) -> None:
        version = _strict_int(self.version, name="version")
        if version != OCDPSS_PUBLIC_RESULT_VERSION:
            raise ValueError("unsupported OCDPSS public-result version")
        qid = _canonical_ocdpss_qid(self.qid)
        context_digest = _context_digest(self.context_digest)
        holder_ids = _int_tuple(
            self.accepted_holder_ids,
            name="accepted_holder_ids",
            minimum=0,
        )
        eval_points = _int_tuple(
            self.accepted_eval_points,
            name="accepted_eval_points",
            minimum=1,
        )
        bundle_digests = _digest_tuple(
            self.accepted_bundle_digests,
            name="accepted_bundle_digests",
        )
        if not (len(holder_ids) == len(eval_points) == len(bundle_digests)):
            raise ValueError("accepted T fields must have the same length")
        commitments = _normalized_point_tuple(
            self.successor_commitment_vector,
            name="successor_commitment_vector",
        )
        operation_output = _freeze_public_value(self.operation_output)

        object.__setattr__(self, "version", version)
        object.__setattr__(self, "qid", qid)
        object.__setattr__(self, "context_digest", context_digest)
        object.__setattr__(self, "accepted_holder_ids", holder_ids)
        object.__setattr__(self, "accepted_eval_points", eval_points)
        object.__setattr__(self, "accepted_bundle_digests", bundle_digests)
        object.__setattr__(self, "operation_output", operation_output)
        object.__setattr__(self, "successor_commitment_vector", commitments)

    @property
    def a2_result(self) -> Optional[A2DPSSPublicResult]:
        """Return the local A²DPSS composition view when available.

        OCDPSS public-result wire v1 predates the explicit A²DPSS layer and
        does not carry enough context to reconstruct this view after standalone
        deserialization.  Results produced by ``OCDPSSPublicFinalizer`` attach
        it locally without changing the serialized fields.
        """

        return getattr(self, "_a2_result", None)

    def _attach_a2_result(
        self,
        result: A2DPSSPublicResult,
    ) -> None:
        if not isinstance(result, A2DPSSPublicResult):
            raise TypeError("result must be an A2DPSSPublicResult")
        if result.accepted_holder_ids != self.accepted_holder_ids:
            raise ValueError("A2DPSS/OCDPSS accepted holder ids differ")
        if result.accepted_eval_points != self.accepted_eval_points:
            raise ValueError("A2DPSS/OCDPSS accepted eval points differ")
        if (
            result.successor_commitment_vector
            != self.successor_commitment_vector
        ):
            raise ValueError(
                "A2DPSS/OCDPSS successor commitment vectors differ"
            )
        existing = self.a2_result
        if existing is not None and existing != result:
            raise ValueError("conflicting attached A2DPSS public result")
        object.__setattr__(self, "_a2_result", result)

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": _PUBLIC_RESULT_KIND,
            "version": self.version,
            "qid": self.qid,
            "context_digest": self.context_digest,
            "accepted_holder_ids": list(self.accepted_holder_ids),
            "accepted_eval_points": list(self.accepted_eval_points),
            "accepted_bundle_digests": list(self.accepted_bundle_digests),
            "operation_output": _thaw_public_value(self.operation_output),
            "successor_commitment_vector": list(
                self.successor_commitment_vector
            ),
        }

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "OCDPSSPublicResult":
        if not isinstance(value, Mapping):
            raise ValueError("OCDPSS public result must be a mapping")
        expected = {
            "kind",
            "version",
            "qid",
            "context_digest",
            "accepted_holder_ids",
            "accepted_eval_points",
            "accepted_bundle_digests",
            "operation_output",
            "successor_commitment_vector",
        }
        missing = expected - set(value)
        extra = set(value) - expected
        if missing:
            raise ValueError(f"OCDPSS public result is missing fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"OCDPSS public result has unknown fields: {sorted(extra)}")
        if value["kind"] != _PUBLIC_RESULT_KIND:
            raise ValueError("expected an OCDPSS public-result object")
        return cls(
            version=value["version"],
            qid=value["qid"],
            context_digest=value["context_digest"],
            accepted_holder_ids=value["accepted_holder_ids"],
            accepted_eval_points=value["accepted_eval_points"],
            accepted_bundle_digests=value["accepted_bundle_digests"],
            operation_output=value["operation_output"],
            successor_commitment_vector=value["successor_commitment_vector"],
        )

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray | memoryview | str
    ) -> "OCDPSSPublicResult":
        return cls.from_obj(
            _decode_json_object(data, name="OCDPSS public result")
        )


class OCDPSSPublicFinalizer:
    """Derive the common operation output and successor commitments."""

    def __init__(self, justifier: OCDPSSJustifierManager) -> None:
        if not isinstance(justifier, OCDPSSJustifierManager):
            raise TypeError("justifier must be an OCDPSSJustifierManager")
        self.justifier = justifier
        self.results: Dict[str, OCDPSSPublicResult] = {}
        self._phase_metrics = {
            "combine": {
                "invocations": 0,
                "elapsed_sec": 0.0,
                "cpu_seconds": 0.0,
            },
            "commitment_interpolation": {
                "invocations": 0,
                "elapsed_sec": 0.0,
                "cpu_seconds": 0.0,
            },
        }

    def _record_phase(
        self, phase: str, started: float, cpu_started: float
    ) -> None:
        metric = self._phase_metrics[phase]
        metric["invocations"] += 1
        metric["elapsed_sec"] += time.perf_counter() - started
        metric["cpu_seconds"] += time.process_time() - cpu_started

    def metrics_snapshot(self) -> dict:
        return {
            phase: {
                "invocations": int(metric["invocations"]),
                "elapsed_sec": float(metric["elapsed_sec"]),
                "cpu_seconds": float(metric["cpu_seconds"]),
            }
            for phase, metric in self._phase_metrics.items()
        }

    def on_deliver(
        self,
        qid: Qid | str,
        entries: Sequence[PoolEntry],
    ) -> Optional[OCDPSSPublicResult]:
        try:
            context, operation_adapter = self.justifier._registration(qid)
            qid_text = context.encode_qid()
        except (KeyError, TypeError, ValueError):
            return None

        normalized_entries = []
        for entry in entries:
            normalized = _normalize_pool_entry(entry)
            if normalized is None:
                return None
            normalized_entries.append(normalized)
        ordered_entries = sorted(normalized_entries, key=_stable_pool_entry)
        bundle_digests = tuple(
            ocdpss_bundle_digest(payload) for _mid, payload in ordered_entries
        )

        existing = self.results.get(qid_text)
        if existing is not None:
            if existing.accepted_bundle_digests != bundle_digests:
                raise ValueError("conflicting T for completed OCDPSS qid")
            return existing

        if not self.justifier.verify_qid_set(
            context.build_qid(), ordered_entries
        ):
            return None

        bundles = []
        for mid, payload in ordered_entries:
            bundle = self.justifier.validate_bundle(
                context, operation_adapter, mid, payload
            )
            if bundle is None:
                return None
            bundles.append(bundle)

        a2_context = self.justifier._a2_context_for(context)
        validated_a2_items = [
            (
                entry,
                _a2_contribution_from_bundle(a2_context, bundle),
            )
            for entry, bundle in zip(ordered_entries, bundles)
        ]
        started = time.perf_counter()
        cpu_started = time.process_time()
        a2_result = finalize_a2_validated_items(
            a2_context, validated_a2_items
        )
        self._record_phase("commitment_interpolation", started, cpu_started)

        holder_ids = a2_result.accepted_holder_ids
        eval_points = a2_result.accepted_eval_points
        started = time.perf_counter()
        cpu_started = time.process_time()
        operation_output = operation_adapter.combine(
            context.operation_input,
            eval_points=eval_points,
            operation_shares=[
                bundle.operation_share["point"] for bundle in bundles
            ],
        )
        self._record_phase("combine", started, cpu_started)
        result = OCDPSSPublicResult(
            qid=qid_text,
            context_digest=context.context_digest,
            accepted_holder_ids=holder_ids,
            accepted_eval_points=eval_points,
            accepted_bundle_digests=bundle_digests,
            operation_output=operation_output,
            successor_commitment_vector=(
                a2_result.successor_commitment_vector
            ),
        )
        result._attach_a2_result(a2_result)
        self.results[qid_text] = result
        return result

    def on_verified_deliver(
        self,
        qid: Qid | str,
        entries: Sequence[PoolEntry],
    ) -> Optional[OCDPSSPublicResult]:
        """Finalize an exact T already admitted by the formal DAG predicate.

        The DAG admission boundary has already executed every Algorithm-5
        proof.  This path deliberately does not call ``verify_qid_set`` or
        ``validate_bundle`` again: it rechecks canonical wire/context/role
        identities and exact canonical-T shape before combining the already
        verified operation and resharing shares.
        """

        try:
            context, operation_adapter = self.justifier._registration(qid)
            qid_text = context.encode_qid()
        except (KeyError, TypeError, ValueError):
            return None
        if len(entries) != context.service_t + 1:
            return None

        normalized_entries = []
        for entry in entries:
            normalized = _normalize_pool_entry(entry)
            if normalized is None:
                return None
            normalized_entries.append(normalized)
        ordered_entries = sorted(normalized_entries, key=_stable_pool_entry)
        if len({(mid, bytes(payload)) for mid, payload in ordered_entries}) != len(
            ordered_entries
        ):
            return None
        bundle_digests = tuple(
            ocdpss_bundle_digest(payload) for _mid, payload in ordered_entries
        )
        existing = self.results.get(qid_text)
        if existing is not None:
            if existing.accepted_bundle_digests != bundle_digests:
                raise ValueError("conflicting T for completed OCDPSS qid")
            return existing

        a2_context = self.justifier._a2_context_for(context)
        bundles = []
        validated_a2_items = []
        identities = set()
        try:
            for entry in ordered_entries:
                mid, payload = entry
                bundle = OCDPSSBundle.from_bytes(payload)
                if (
                    bundle.context_digest != context.context_digest
                    or mid != f"node_{bundle.holder_id}"
                    or context.holder_eval_points.get(bundle.holder_id)
                    != bundle.holder_eval_point
                ):
                    return None
                identity = (bundle.holder_id, bundle.holder_eval_point)
                if identity in identities:
                    return None
                identities.add(identity)
                contribution = _a2_contribution_from_bundle(a2_context, bundle)
                bundles.append(bundle)
                validated_a2_items.append((entry, contribution))
            if select_canonical_threshold(
                validated_a2_items, context.service_t + 1
            ) != ordered_entries:
                return None
        except Exception:
            return None

        started = time.perf_counter()
        cpu_started = time.process_time()
        a2_result = finalize_a2_validated_items(
            a2_context, validated_a2_items
        )
        self._record_phase("commitment_interpolation", started, cpu_started)
        started = time.perf_counter()
        cpu_started = time.process_time()
        operation_output = operation_adapter.combine(
            context.operation_input,
            eval_points=a2_result.accepted_eval_points,
            operation_shares=[
                bundle.operation_share["point"] for bundle in bundles
            ],
        )
        self._record_phase("combine", started, cpu_started)
        result = OCDPSSPublicResult(
            qid=qid_text,
            context_digest=context.context_digest,
            accepted_holder_ids=a2_result.accepted_holder_ids,
            accepted_eval_points=a2_result.accepted_eval_points,
            accepted_bundle_digests=bundle_digests,
            operation_output=operation_output,
            successor_commitment_vector=(
                a2_result.successor_commitment_vector
            ),
        )
        result._attach_a2_result(a2_result)
        self.results[qid_text] = result
        return result


@dataclass(frozen=True)
class OCDPSSSuccessorResult:
    """Versioned private output for one successor role."""

    qid: str
    context_digest: str
    my_node_id: int
    my_role_idx: int
    share_s: int
    share_s_hat: int
    accepted_eval_points: Sequence[int]
    accepted_bundle_digests: Sequence[str]
    version: int = OCDPSS_SUCCESSOR_RESULT_VERSION

    def __post_init__(self) -> None:
        version = _strict_int(self.version, name="version")
        if version != OCDPSS_SUCCESSOR_RESULT_VERSION:
            raise ValueError("unsupported OCDPSS successor-result version")
        eval_points = _int_tuple(
            self.accepted_eval_points,
            name="accepted_eval_points",
            minimum=1,
        )
        bundle_digests = _digest_tuple(
            self.accepted_bundle_digests,
            name="accepted_bundle_digests",
        )
        if len(eval_points) != len(bundle_digests):
            raise ValueError("successor result T fields must have the same length")
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "qid", _canonical_ocdpss_qid(self.qid))
        object.__setattr__(self, "context_digest", _context_digest(self.context_digest))
        object.__setattr__(
            self, "my_node_id", _strict_int(self.my_node_id, name="my_node_id")
        )
        object.__setattr__(
            self, "my_role_idx", _strict_int(self.my_role_idx, name="my_role_idx")
        )
        object.__setattr__(self, "share_s", _canonical_scalar(self.share_s, name="share_s"))
        object.__setattr__(
            self,
            "share_s_hat",
            _canonical_scalar(self.share_s_hat, name="share_s_hat"),
        )
        object.__setattr__(self, "accepted_eval_points", eval_points)
        object.__setattr__(self, "accepted_bundle_digests", bundle_digests)

    @property
    def a2_result(self) -> Optional[A2DPSSSuccessorResult]:
        """Return the local A²DPSS private-result composition view."""

        return getattr(self, "_a2_result", None)

    def _attach_a2_result(
        self,
        result: A2DPSSSuccessorResult,
    ) -> None:
        if not isinstance(result, A2DPSSSuccessorResult):
            raise TypeError("result must be an A2DPSSSuccessorResult")
        if result.my_node_id != self.my_node_id:
            raise ValueError("A2DPSS/OCDPSS successor node ids differ")
        if result.my_role_idx != self.my_role_idx:
            raise ValueError("A2DPSS/OCDPSS successor role indices differ")
        if (
            result.share_s != self.share_s
            or result.share_s_hat != self.share_s_hat
        ):
            raise ValueError("A2DPSS/OCDPSS successor shares differ")
        if result.accepted_eval_points != self.accepted_eval_points:
            raise ValueError("A2DPSS/OCDPSS accepted eval points differ")
        existing = self.a2_result
        if existing is not None and existing != result:
            raise ValueError("conflicting attached A2DPSS successor result")
        object.__setattr__(self, "_a2_result", result)

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": _SUCCESSOR_RESULT_KIND,
            "version": self.version,
            "qid": self.qid,
            "context_digest": self.context_digest,
            "my_node_id": self.my_node_id,
            "my_role_idx": self.my_role_idx,
            "share_s": str(self.share_s),
            "share_s_hat": str(self.share_s_hat),
            "accepted_eval_points": list(self.accepted_eval_points),
            "accepted_bundle_digests": list(self.accepted_bundle_digests),
        }

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "OCDPSSSuccessorResult":
        if not isinstance(value, Mapping):
            raise ValueError("OCDPSS successor result must be a mapping")
        expected = {
            "kind",
            "version",
            "qid",
            "context_digest",
            "my_node_id",
            "my_role_idx",
            "share_s",
            "share_s_hat",
            "accepted_eval_points",
            "accepted_bundle_digests",
        }
        missing = expected - set(value)
        extra = set(value) - expected
        if missing:
            raise ValueError(f"OCDPSS successor result is missing fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"OCDPSS successor result has unknown fields: {sorted(extra)}")
        if value["kind"] != _SUCCESSOR_RESULT_KIND:
            raise ValueError("expected an OCDPSS successor-result object")
        return cls(
            version=value["version"],
            qid=value["qid"],
            context_digest=value["context_digest"],
            my_node_id=value["my_node_id"],
            my_role_idx=value["my_role_idx"],
            share_s=int(value["share_s"]),
            share_s_hat=int(value["share_s_hat"]),
            accepted_eval_points=value["accepted_eval_points"],
            accepted_bundle_digests=value["accepted_bundle_digests"],
        )

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray | memoryview | str
    ) -> "OCDPSSSuccessorResult":
        return cls.from_obj(
            _decode_json_object(data, name="OCDPSS successor result")
        )

    def to_holder_state(self) -> "OCDPSSHolderState":
        return OCDPSSHolderState(self.share_s, self.share_s_hat)


class OCDPSSSuccessorReconstructionManager:
    """Reconstruct exactly the private successor share selected by public T."""

    def __init__(self) -> None:
        self.registrations: Dict[str, dict] = {}
        self.results: Dict[str, OCDPSSSuccessorResult] = {}
        self._consumed_results: Set[str] = set()
        self._phase_metrics = {
            stage: {
                "invocations": 0,
                "elapsed_sec": 0.0,
                "cpu_seconds": 0.0,
            }
            for stage in (
                "commitment_recompute",
                "pvss_decrypt",
                "share_interpolation",
                "commitment_opening_check",
            )
        }

    def _record_phase_elapsed(
        self,
        phase: str,
        elapsed: float,
        cpu_seconds: float,
    ) -> None:
        metric = self._phase_metrics[phase]
        metric["invocations"] += 1
        metric["elapsed_sec"] += float(elapsed)
        metric["cpu_seconds"] += float(cpu_seconds)

    def metrics_snapshot(self) -> dict:
        return {
            phase: {
                "invocations": int(metric["invocations"]),
                "elapsed_sec": float(metric["elapsed_sec"]),
                "cpu_seconds": float(metric["cpu_seconds"]),
            }
            for phase, metric in self._phase_metrics.items()
        }

    def register_context(
        self,
        context: OCDPSSContext,
        *,
        my_node_id: int,
        my_rsk: Optional[int] = None,
        on_new_share=None,
    ) -> None:
        if not isinstance(context, OCDPSSContext):
            raise TypeError("context must be an OCDPSSContext")
        my_node_id = _strict_int(my_node_id, name="my_node_id")
        role_idx = context.successor_role_indices.get(my_node_id)
        if role_idx is None:
            if my_rsk is not None:
                raise ValueError("non-successor observer must not register an rsk")
            normalized_rsk = None
        else:
            if my_rsk is None:
                raise ValueError("successor role requires my_rsk")
            normalized_rsk = _canonical_scalar(my_rsk, name="my_rsk")
            expected_rpk = G1Point.from_hex(
                context.successor_rpk_list[role_idx]
            )
            actual_rpk = G1Point.from_hex(context.key_base) * normalized_rsk
            if actual_rpk != expected_rpk:
                raise ValueError("my_rsk does not match successor role public key")
        if on_new_share is not None and not callable(on_new_share):
            raise TypeError("on_new_share must be callable")
        registration = {
            "context": context,
            "my_node_id": my_node_id,
            "my_role_idx": role_idx,
            "my_rsk": normalized_rsk,
            "on_new_share": on_new_share,
        }
        existing = self.registrations.get(context.context_digest)
        if existing is not None:
            same_registration = (
                existing["context"] == context
                and existing["my_node_id"] == my_node_id
                and existing["my_role_idx"] == role_idx
                and existing["my_rsk"] == normalized_rsk
                and existing["on_new_share"] is on_new_share
            )
            if not same_registration:
                raise ValueError("conflicting successor registration for context")
            return
        self.registrations[context.context_digest] = registration

    def unregister_context(self, context_or_digest: OCDPSSContext | str) -> None:
        digest = (
            context_or_digest.context_digest
            if isinstance(context_or_digest, OCDPSSContext)
            else _context_digest(context_or_digest)
        )
        self.registrations.pop(digest, None)
        self.results.pop(digest, None)

    def consume_result_to_holder_state(
        self,
        context_or_digest: OCDPSSContext | str,
    ) -> "OCDPSSHolderState":
        """Move one reconstructed share into a one-shot holder container.

        This is the O7 ownership-transfer boundary.  The manager drops its
        retained private result before returning, so a later handover cannot
        obtain a second holder state for the same context.
        """

        digest = (
            context_or_digest.context_digest
            if isinstance(context_or_digest, OCDPSSContext)
            else _context_digest(context_or_digest)
        )
        if digest in self._consumed_results:
            raise RuntimeError("successor result already consumed for next handover")
        try:
            result = self.results.pop(digest)
        except KeyError as exc:
            raise RuntimeError("no reconstructed successor result to consume") from exc
        self._consumed_results.add(digest)
        return result.to_holder_state()

    def reconstruct(
        self,
        public_result: OCDPSSPublicResult,
        entries: Sequence[PoolEntry],
    ) -> Optional[OCDPSSSuccessorResult]:
        if not isinstance(public_result, OCDPSSPublicResult):
            raise TypeError("public_result must be an OCDPSSPublicResult")
        local = self.registrations.get(public_result.context_digest)
        if local is None:
            return None
        if public_result.context_digest in self._consumed_results:
            raise RuntimeError("successor result was consumed for next handover")
        context = local["context"]
        if public_result.qid != context.encode_qid():
            raise ValueError("public result qid does not match registered context")
        if len(public_result.accepted_eval_points) != context.service_t + 1:
            raise ValueError("public result does not contain exactly service_t+1 entries")
        if local["my_role_idx"] is None:
            return None

        normalized_entries = []
        for entry in entries:
            normalized = _normalize_pool_entry(entry)
            if normalized is None:
                raise ValueError("successor entries contain a malformed PoolEntry")
            normalized_entries.append(normalized)
        if len(normalized_entries) != len(public_result.accepted_bundle_digests):
            raise ValueError("successor entries must contain exactly public result T")

        entries_by_digest = {}
        for mid, payload in normalized_entries:
            digest = ocdpss_bundle_digest(payload)
            if digest in entries_by_digest:
                raise ValueError("successor entries contain a duplicate bundle digest")
            entries_by_digest[digest] = (mid, payload)
        if set(entries_by_digest) != set(public_result.accepted_bundle_digests):
            raise ValueError("successor entries do not match public result T")

        existing = self.results.get(context.context_digest)
        if existing is not None:
            if (
                existing.accepted_bundle_digests
                != public_result.accepted_bundle_digests
            ):
                raise ValueError("conflicting T for completed successor reconstruction")
            return existing

        bundles = []
        for index, digest in enumerate(public_result.accepted_bundle_digests):
            mid, payload = entries_by_digest[digest]
            bundle = OCDPSSBundle.from_bytes(payload)
            if bundle.context_digest != context.context_digest:
                raise ValueError("successor bundle context digest mismatch")
            if bundle.holder_id != public_result.accepted_holder_ids[index]:
                raise ValueError("successor bundle holder does not match public T")
            if bundle.holder_eval_point != public_result.accepted_eval_points[index]:
                raise ValueError("successor bundle eval point does not match public T")
            if mid != f"node_{bundle.holder_id}":
                raise ValueError("successor bundle mid does not match holder")
            bundles.append(bundle)

        a2_context = _a2_context_from_ocdpss(context)
        a2_contributions = [
            _a2_contribution_from_bundle(a2_context, bundle)
            for bundle in bundles
        ]
        a2_public_result = A2DPSSPublicResult(
            qid=a2_context.encode_qid(),
            context_digest=a2_context.context_digest,
            accepted_holder_ids=public_result.accepted_holder_ids,
            accepted_eval_points=public_result.accepted_eval_points,
            accepted_contribution_digests=tuple(
                a2dpss_contribution_digest(
                    contribution.to_bytes()
                )
                for contribution in a2_contributions
            ),
            successor_commitment_vector=(
                public_result.successor_commitment_vector
            ),
        )
        if (
            public_result.a2_result is not None
            and public_result.a2_result != a2_public_result
        ):
            raise ValueError(
                "attached A2DPSS public result does not match OCDPSS T"
            )
        validated_a2_items = [
            (entry, contribution)
            for entry, contribution in zip(
                [
                    entries_by_digest[digest]
                    for digest in public_result.accepted_bundle_digests
                ],
                a2_contributions,
            )
        ]
        a2_successor_result = reconstruct_a2_validated_items(
            a2_context,
            a2_public_result,
            validated_a2_items,
            my_node_id=local["my_node_id"],
            my_role_idx=local["my_role_idx"],
            my_rsk=local["my_rsk"],
            stage_observer=self._record_phase_elapsed,
        )
        if public_result.a2_result is None:
            public_result._attach_a2_result(a2_public_result)

        result = OCDPSSSuccessorResult(
            qid=public_result.qid,
            context_digest=context.context_digest,
            my_node_id=local["my_node_id"],
            my_role_idx=local["my_role_idx"],
            share_s=a2_successor_result.share_s,
            share_s_hat=a2_successor_result.share_s_hat,
            accepted_eval_points=public_result.accepted_eval_points,
            accepted_bundle_digests=public_result.accepted_bundle_digests,
        )
        result._attach_a2_result(a2_successor_result)
        self.results[context.context_digest] = result
        callback = local["on_new_share"]
        if callback is not None:
            callback(result.share_s, result.share_s_hat)
        return result


def _canonical_scalar(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer scalar")
    if not 0 <= value < FIELD_ORDER:
        raise ValueError(f"{name} must be a canonical field scalar")
    return value


def _holder_eval_point(context: OCDPSSContext, holder_id: int) -> int:
    try:
        return context.holder_eval_points[holder_id]
    except KeyError as exc:
        raise ValueError(f"holder_id {holder_id} is not in the source committee") from exc


def _validate_operation_adapter(
    context: OCDPSSContext, operation_adapter: OperationAdapter
) -> None:
    for method_name in ("public_context", "partial_eval", "verify_share", "combine"):
        if not callable(getattr(operation_adapter, method_name, None)):
            raise TypeError(f"operation_adapter is missing {method_name}()")
    public_context = operation_adapter.public_context(context.operation_input)
    if not isinstance(public_context, Mapping):
        raise TypeError("operation_adapter.public_context() must return a mapping")
    expected = _typed_canonical(
        _thaw_public_value(context.operation_public_parameters)
    )
    actual = _typed_canonical(public_context)
    if actual != expected:
        raise ValueError(
            "operation_adapter public context does not match OCDPSS context"
        )


def _operation_share_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, str) and value:
        return {"point": value}
    if isinstance(value, Mapping) and value:
        return _thaw_public_value(value)
    raise TypeError("operation adapter must return a point string or non-empty mapping")


def _operation_proof_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, OperationShareProof):
        return value.to_dict()
    if isinstance(value, Mapping) and value:
        return _thaw_public_value(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        encoded = to_dict()
        if isinstance(encoded, Mapping) and encoded:
            return _thaw_public_value(encoded)
    raise TypeError("operation adapter must return a serializable non-empty proof")


@dataclass
class OCDPSSHolderState(A2DPSSHolderState):
    """Operation-coupled one-shot state with A²DPSS ownership semantics."""


@dataclass
class OCDPSSHolderMetrics:
    operation_seconds: float = 0.0
    operation_cpu_seconds: float = 0.0
    pvss_seconds: float = 0.0
    pvss_cpu_seconds: float = 0.0
    serialization_seconds: float = 0.0
    serialization_cpu_seconds: float = 0.0
    broadcast_seconds: float = 0.0
    broadcast_cpu_seconds: float = 0.0
    bundle_bytes: int = 0


@dataclass(frozen=True)
class OCDPSSHolderActionResult:
    qid: str
    mid: str
    bundle: OCDPSSBundle
    bundle_bytes: bytes
    metrics: OCDPSSHolderMetrics


def make_ocdpss_bundle(
    *,
    context: OCDPSSContext,
    holder_id: int,
    share_s: int,
    share_s_hat: int,
    operation_adapter: OperationAdapter,
    metrics: OCDPSSHolderMetrics | None = None,
) -> OCDPSSBundle:
    """Locally execute Algorithm 5 Lines 1--2 and construct one full bundle.

    This is a deterministic-boundary crypto builder, not the protocol action:
    it does not enforce one-shot state and never broadcasts.  The protocol
    entry point is :func:`run_ocdpss_handover_role` below.
    """

    if not isinstance(context, OCDPSSContext):
        raise TypeError("context must be an OCDPSSContext")
    holder_id = _strict_int(holder_id, name="holder_id")
    share_s = _canonical_scalar(share_s, name="share_s")
    share_s_hat = _canonical_scalar(share_s_hat, name="share_s_hat")
    holder_eval_point = _holder_eval_point(context, holder_id)
    _validate_operation_adapter(context, operation_adapter)
    timings = metrics if metrics is not None else OCDPSSHolderMetrics()
    if not isinstance(timings, OCDPSSHolderMetrics):
        raise TypeError("metrics must be an OCDPSSHolderMetrics")

    started = time.perf_counter()
    cpu_started = time.process_time()
    a2_contribution = make_a2dpss_contribution(
        context=_a2_context_from_ocdpss(context),
        holder_id=holder_id,
        share_s=share_s,
        share_s_hat=share_s_hat,
    )
    timings.pvss_seconds = time.perf_counter() - started
    timings.pvss_cpu_seconds = time.process_time() - cpu_started

    commitment_i = context.old_commitment_vector[holder_eval_point]
    started = time.perf_counter()
    cpu_started = time.process_time()
    operation_share, operation_proof = operation_adapter.partial_eval(
        context.operation_input,
        commitment_i=commitment_i,
        s_i=share_s,
        s_hat_i=share_s_hat,
    )
    timings.operation_seconds = time.perf_counter() - started
    timings.operation_cpu_seconds = time.process_time() - cpu_started

    bundle = OCDPSSBundle(
        holder_id=holder_id,
        holder_eval_point=holder_eval_point,
        context_digest=context.context_digest,
        transcript=a2_contribution.transcript,
        operation_share=_operation_share_obj(operation_share),
        operation_proof=_operation_proof_obj(operation_proof),
    )
    bundle._attach_a2_contribution(a2_contribution)
    return bundle


async def run_ocdpss_handover_role(
    *,
    context: OCDPSSContext,
    holder_id: int,
    share_s: int,
    share_s_hat: int,
    operation_adapter: OperationAdapter,
    broadcast_bundle,
    holder_state: OCDPSSHolderState,
) -> OCDPSSHolderActionResult:
    """Execute Algorithm 5's one-shot holder action with injected transport.

    ``make_ocdpss_bundle`` composes one A²DPSS contribution with
    ``operation_adapter.partial_eval`` over the same consumed share pair.
    This function owns the single ``broadcast_bundle`` submission, but the
    callback is transport-neutral until S6 connects it to YOSO-RBC.
    """

    if not callable(broadcast_bundle):
        raise TypeError("broadcast_bundle must be callable")
    if not isinstance(holder_state, OCDPSSHolderState):
        raise TypeError("holder_state must be an OCDPSSHolderState")
    share_s = _canonical_scalar(share_s, name="share_s")
    share_s_hat = _canonical_scalar(share_s_hat, name="share_s_hat")
    holder_id = _strict_int(holder_id, name="holder_id")

    # Public checks happen before consuming the role.  Once the share pair is
    # released, every later failure remains fail-closed: retrying could create
    # a second randomized proof/transcript for the same one-shot holder role.
    _holder_eval_point(context, holder_id)
    _validate_operation_adapter(context, operation_adapter)
    consumed_s, consumed_s_hat = holder_state.consume()
    if (consumed_s, consumed_s_hat) != (share_s, share_s_hat):
        raise ValueError("holder_state share pair does not match holder arguments")

    metrics = OCDPSSHolderMetrics()
    bundle = make_ocdpss_bundle(
        context=context,
        holder_id=holder_id,
        share_s=consumed_s,
        share_s_hat=consumed_s_hat,
        operation_adapter=operation_adapter,
        metrics=metrics,
    )
    # Drop the remaining local references before the first network await.  As
    # in the paper this is logical erasure; Python cannot guarantee physical
    # zeroization of immutable integer objects.
    del consumed_s, consumed_s_hat, share_s, share_s_hat
    started = time.perf_counter()
    cpu_started = time.process_time()
    bundle_bytes = bundle.to_bytes()
    metrics.serialization_seconds = time.perf_counter() - started
    metrics.serialization_cpu_seconds = time.process_time() - cpu_started
    metrics.bundle_bytes = len(bundle_bytes)

    qid = context.encode_qid()
    mid = f"node_{holder_id}"
    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        pending = broadcast_bundle(qid, mid, bundle_bytes)
        if inspect.isawaitable(pending):
            await pending
    finally:
        metrics.broadcast_seconds = time.perf_counter() - started
        metrics.broadcast_cpu_seconds = time.process_time() - cpu_started

    return OCDPSSHolderActionResult(
        qid=qid,
        mid=mid,
        bundle=bundle,
        bundle_bytes=bundle_bytes,
        metrics=metrics,
    )


__all__ = [
    "OCDPSS_BUNDLE_VERSION",
    "OCDPSS_CONTEXT_VERSION",
    "OCDPSS_PUBLIC_RESULT_VERSION",
    "OCDPSS_QID_KIND",
    "OCDPSS_SUCCESSOR_RESULT_VERSION",
    "OCDPSSBundle",
    "OCDPSSContext",
    "OCDPSSHolderActionResult",
    "OCDPSSHolderMetrics",
    "OCDPSSHolderState",
    "OCDPSSJustifierManager",
    "OCDPSSPublicFinalizer",
    "OCDPSSPublicResult",
    "OCDPSSSuccessorReconstructionManager",
    "OCDPSSSuccessorResult",
    "build_ocdpss_qid",
    "canonical_context_digest",
    "is_ocdpss_qid",
    "make_ocdpss_bundle",
    "ocdpss_bundle_digest",
    "ocdpss_qid_matches_context",
    "run_ocdpss_handover_role",
]
