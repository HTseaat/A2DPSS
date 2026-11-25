"""Canonical data model for Algorithm-4 A²DPSS.

This module defines the immutable public context, holder contribution,
one-shot local holder action, dedicated TOB qid namespace, base validation
predicate, deterministic threshold selection, public finalization, and
successor reconstruction.  Network transport belongs to a later layer.

A²DPSS is operation-independent.  Operation shares and their proofs are owned
by the Algorithm-5 OCDPSS extension and must not be added to these wire types.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import string
import threading
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from beaver.dag_tob import Justifier, PoolEntry, Qid
from beaver.g1_backend import FIELD_ORDER, G1Point
from beaver.pedersen_shamir import pedersen_commit
from beaver.pvss import (
    LEGACY_TRANSCRIPT_VERSION,
    SUPPORTED_TRANSCRIPT_VERSIONS,
    TRANSCRIPT_VERSION,
    PVSSTranscript,
    interpolate_commitment_vector,
    pvss_deal,
    pvss_deliver,
    pvss_interpolate,
    pvss_verify,
    pvss_verify_anchor,
)


A2DPSS_CONTEXT_VERSION = 1
A2DPSS_CONTRIBUTION_VERSION = 1
A2DPSS_PUBLIC_RESULT_VERSION = 1
A2DPSS_SUCCESSOR_RESULT_VERSION = 1
A2DPSS_QID_KIND = "a2dpss"

_CONTEXT_KIND = "atcaas-a2dpss-context"
_CONTRIBUTION_KIND = "atcaas-a2dpss-contribution"
_PUBLIC_RESULT_KIND = "atcaas-a2dpss-public-result"
_SUCCESSOR_RESULT_KIND = "atcaas-a2dpss-successor-result"
_CONTEXT_DIGEST_DOMAIN = b"ATCaaS/A2DPSS/context/v1\x00"
_CONTRIBUTION_DIGEST_DOMAIN = b"ATCaaS/A2DPSS/contribution/v1\x00"


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


def _decode_json_object(
    data: bytes | bytearray | memoryview | str,
    *,
    name: str,
) -> dict:
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
    """Return an unambiguous JSON value for public-context hashing."""

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
    normalized = tuple(
        _nonempty_text(item, name=f"{name} item") for item in value
    )
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _normalize_int_mapping(
    value: Mapping[Any, Any],
    *,
    name: str,
    value_minimum: int,
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
            raw_key == "0"
            or (raw_key.isdigit() and not raw_key.startswith("0"))
        ):
            key = int(raw_key)
        else:
            raise TypeError(f"{name} keys must be canonical integer node ids")
        if key < 0:
            raise ValueError(f"{name} node ids must be non-negative")
        if key in normalized:
            raise ValueError(
                f"{name} contains duplicate normalized node id {key}"
            )
        normalized[key] = _strict_int(
            raw_value,
            name=f"{name}[{key}]",
            minimum=value_minimum,
        )
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return MappingProxyType(dict(sorted(normalized.items())))


def _context_digest(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("context_digest must be a hexadecimal string")
    if (
        len(value) != 64
        or value != value.lower()
        or any(character not in string.hexdigits.lower() for character in value)
    ):
        raise ValueError(
            "context_digest must be 64 lowercase hexadecimal characters"
        )
    return value


def canonical_a2dpss_context_digest(
    context: Mapping[str, Any] | "A2DPSSContext",
) -> str:
    """Hash all public inputs needed to identify one A²DPSS transfer."""

    if isinstance(context, A2DPSSContext):
        material = context.digest_material()
    elif isinstance(context, Mapping):
        material = {
            key: value
            for key, value in context.items()
            if key != "context_digest"
        }
    else:
        raise TypeError("context must be an A2DPSSContext or mapping")
    encoded = _canonical_json(_typed_canonical(material))
    return hashlib.sha256(_CONTEXT_DIGEST_DOMAIN + encoded).hexdigest()


@dataclass(frozen=True)
class A2DPSSContext:
    """Complete operation-independent public statement for one handover."""

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
    key_base: str = ""
    version: int = A2DPSS_CONTEXT_VERSION
    context_digest: str = field(init=False)

    def __post_init__(self) -> None:
        version = _strict_int(self.version, name="version")
        if version != A2DPSS_CONTEXT_VERSION:
            raise ValueError("unsupported A2DPSS context version")
        epoch = _strict_int(self.epoch, name="epoch")
        transfer_seq = _strict_int(self.transfer_seq, name="transfer_seq")
        service_t = _strict_int(self.service_t, name="service_t")
        sid = _nonempty_text(self.sid, name="sid")
        source = _nonempty_text(
            self.source_committee, name="source_committee"
        )
        successor = _nonempty_text(
            self.successor_committee, name="successor_committee"
        )
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
        if len(holder_points) < service_t + 1:
            raise ValueError(
                "holder_eval_points must contain at least service_t+1 holders"
            )
        if len(set(holder_points.values())) != len(holder_points):
            raise ValueError("holder eval points must be distinct")
        if max(holder_points.values()) >= len(old_vector):
            raise ValueError(
                "old_commitment_vector does not cover holder eval points"
            )
        successor_indices = _normalize_int_mapping(
            self.successor_role_indices,
            name="successor_role_indices",
            value_minimum=0,
        )
        if len(set(successor_indices.values())) != len(successor_indices):
            raise ValueError("successor role indices must be distinct")
        if max(successor_indices.values()) >= len(rpk_list):
            raise ValueError(
                "successor role index is outside successor_rpk_list"
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
        object.__setattr__(
            self, "successor_role_indices", successor_indices
        )
        object.__setattr__(
            self,
            "context_digest",
            canonical_a2dpss_context_digest(self),
        )

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
        }

    def to_obj(self) -> dict[str, Any]:
        material = self.digest_material()
        material["context_digest"] = self.context_digest
        return material

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "A2DPSSContext":
        if not isinstance(value, Mapping):
            raise ValueError("A2DPSS context must be a mapping")
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
            "context_digest",
        }
        missing = expected_fields - set(value)
        extra = set(value) - expected_fields
        if missing:
            raise ValueError(
                f"A2DPSS context is missing fields: {sorted(missing)}"
            )
        if extra:
            raise ValueError(
                f"A2DPSS context has unknown fields: {sorted(extra)}"
            )
        if value["kind"] != _CONTEXT_KIND:
            raise ValueError("expected an A2DPSS context object")
        if value["version"] != A2DPSS_CONTEXT_VERSION:
            raise ValueError("unsupported A2DPSS context version")
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
        )
        if not hmac.compare_digest(
            context.context_digest, expected_digest
        ):
            raise ValueError(
                "A2DPSS context_digest does not match public context"
            )
        return context

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview | str,
    ) -> "A2DPSSContext":
        return cls.from_obj(
            _decode_json_object(data, name="A2DPSS context")
        )

    def build_qid(self) -> Qid:
        return build_a2dpss_qid(self)

    def encode_qid(self) -> str:
        return self.build_qid().encode()


def build_a2dpss_qid(context: A2DPSSContext) -> Qid:
    """Build a qid committed to the complete A²DPSS public context."""

    if not isinstance(context, A2DPSSContext):
        raise TypeError("context must be an A2DPSSContext")
    return Qid.build(
        A2DPSS_QID_KIND,
        epoch=context.epoch,
        sid=context.sid,
        source_committee=context.source_committee,
        successor_committee=context.successor_committee,
        transfer_seq=context.transfer_seq,
        context_digest=context.context_digest,
    )


def is_a2dpss_qid(qid: Qid | str) -> bool:
    try:
        if isinstance(qid, str):
            qid = Qid.decode(qid)
        return isinstance(qid, Qid) and qid.kind == A2DPSS_QID_KIND
    except (TypeError, ValueError):
        return False


def a2dpss_qid_matches_context(
    qid: Qid | str,
    context: A2DPSSContext,
) -> bool:
    try:
        if isinstance(qid, str):
            qid = Qid.decode(qid)
        return qid == build_a2dpss_qid(context)
    except (TypeError, ValueError):
        return False


def _transcript_from_obj(value: Any) -> PVSSTranscript:
    if not isinstance(value, dict):
        raise ValueError("contribution transcript must be an object")
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
        raise ValueError(
            f"contribution transcript is missing fields: {sorted(missing)}"
        )
    if extra:
        raise ValueError(
            f"contribution transcript has unknown fields: {sorted(extra)}"
        )
    if version not in SUPPORTED_TRANSCRIPT_VERSIONS:
        raise ValueError(
            "unsupported PVSS transcript version in A2DPSS contribution"
        )
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
        raise ValueError(
            "malformed PVSS transcript in A2DPSS contribution"
        ) from exc


@dataclass(frozen=True)
class A2DPSSContribution:
    """One holder's operation-independent Algorithm-4 contribution."""

    holder_id: int
    holder_eval_point: int
    context_digest: str
    transcript: PVSSTranscript
    version: int = A2DPSS_CONTRIBUTION_VERSION

    def __post_init__(self) -> None:
        version = _strict_int(self.version, name="version")
        if version != A2DPSS_CONTRIBUTION_VERSION:
            raise ValueError("unsupported A2DPSS contribution version")
        holder_id = _strict_int(self.holder_id, name="holder_id")
        holder_eval_point = _strict_int(
            self.holder_eval_point,
            name="holder_eval_point",
            minimum=1,
        )
        digest = _context_digest(self.context_digest)
        if not isinstance(self.transcript, PVSSTranscript):
            raise TypeError("transcript must be a PVSSTranscript")
        transcript = _transcript_from_obj(self.transcript.to_obj())
        if transcript.dealer_id != holder_id:
            raise ValueError(
                "contribution holder_id does not match transcript dealer_id"
            )
        if transcript.dealer_eval_point != holder_eval_point:
            raise ValueError(
                "contribution holder_eval_point does not match "
                "transcript dealer_eval_point"
            )

        object.__setattr__(self, "version", version)
        object.__setattr__(self, "holder_id", holder_id)
        object.__setattr__(
            self, "holder_eval_point", holder_eval_point
        )
        object.__setattr__(self, "context_digest", digest)
        object.__setattr__(self, "transcript", transcript)

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": _CONTRIBUTION_KIND,
            "version": self.version,
            "holder_id": self.holder_id,
            "holder_eval_point": self.holder_eval_point,
            "context_digest": self.context_digest,
            "transcript": self.transcript.to_obj(),
        }

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_obj())

    @classmethod
    def from_obj(
        cls,
        value: Mapping[str, Any],
    ) -> "A2DPSSContribution":
        if not isinstance(value, Mapping):
            raise ValueError("A2DPSS contribution must be a mapping")
        expected_fields = {
            "kind",
            "version",
            "holder_id",
            "holder_eval_point",
            "context_digest",
            "transcript",
        }
        missing = expected_fields - set(value)
        extra = set(value) - expected_fields
        if missing:
            raise ValueError(
                f"A2DPSS contribution is missing fields: {sorted(missing)}"
            )
        if extra:
            raise ValueError(
                f"A2DPSS contribution has unknown fields: {sorted(extra)}"
            )
        if value["kind"] != _CONTRIBUTION_KIND:
            raise ValueError("expected an A2DPSS contribution object")
        if value["version"] != A2DPSS_CONTRIBUTION_VERSION:
            raise ValueError("unsupported A2DPSS contribution version")
        transcript = _transcript_from_obj(value["transcript"])
        return cls(
            version=value["version"],
            holder_id=value["holder_id"],
            holder_eval_point=value["holder_eval_point"],
            context_digest=value["context_digest"],
            transcript=transcript,
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview | str,
    ) -> "A2DPSSContribution":
        return cls.from_obj(
            _decode_json_object(data, name="A2DPSS contribution")
        )


A2DPSSValidationStageObserver = Callable[[str, float, float], None]
A2DPSSValidatedItem = Tuple[PoolEntry, A2DPSSContribution]


def _observe_validation_stage(
    observer: Optional[A2DPSSValidationStageObserver],
    stage: str,
    wall_started: float,
    cpu_started: float,
) -> None:
    if observer is not None:
        try:
            observer(
                stage,
                time.perf_counter() - wall_started,
                time.process_time() - cpu_started,
            )
        except Exception:
            # Observability must never influence the protocol predicate.
            pass


def validate_a2_contribution(
    context: A2DPSSContext,
    contribution: A2DPSSContribution,
    *,
    stage_observer: Optional[A2DPSSValidationStageObserver] = None,
) -> bool:
    """Validate the operation-independent security predicate for one holder.

    The optional observer exposes only validation stage timings.  It cannot
    alter validity and deliberately carries no operation callback.
    """

    try:
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        if not isinstance(context, A2DPSSContext):
            return False
        if not isinstance(contribution, A2DPSSContribution):
            return False
        if not hmac.compare_digest(
            contribution.context_digest, context.context_digest
        ):
            return False
        holder_id = contribution.holder_id
        if holder_id not in context.holder_eval_points:
            return False
        expected_eval_point = context.holder_eval_points[holder_id]
        if contribution.holder_eval_point != expected_eval_point:
            return False
        if contribution.transcript.dealer_id != holder_id:
            return False
        if (
            contribution.transcript.dealer_eval_point
            != expected_eval_point
        ):
            return False
        _observe_validation_stage(
            stage_observer,
            "schema_identity",
            wall_started,
            cpu_started,
        )

        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        if not pvss_verify(
            context.g_hex,
            context.h_hex,
            context.successor_rpk_list,
            contribution.transcript,
            context.service_t,
            key_base_hex=context.key_base,
        ):
            return False
        _observe_validation_stage(
            stage_observer,
            "pvss",
            wall_started,
            cpu_started,
        )

        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        if not pvss_verify_anchor(
            contribution.transcript,
            context.old_commitment_vector,
        ):
            return False
        _observe_validation_stage(
            stage_observer,
            "anchor",
            wall_started,
            cpu_started,
        )
        return True
    except Exception:
        return False


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


def select_canonical_threshold(
    validated_items: Sequence[A2DPSSValidatedItem],
    threshold: int,
) -> Optional[List[PoolEntry]]:
    """Select the first stable threshold of unique validated holders.

    ``validated_items`` must already have passed the complete predicate of the
    calling protocol.  For A²DPSS that is base validation; for OCDPSS it also
    includes operation-share verification.
    """

    threshold = _strict_int(threshold, name="threshold", minimum=1)
    normalized: List[A2DPSSValidatedItem] = []
    for value in validated_items:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise TypeError(
                "validated_items must contain (PoolEntry, contribution) pairs"
            )
        raw_entry, contribution = value
        entry = _normalize_pool_entry(raw_entry)
        if entry is None:
            raise ValueError("validated item contains a malformed PoolEntry")
        if not isinstance(contribution, A2DPSSContribution):
            raise TypeError(
                "validated item contribution must be an A2DPSSContribution"
            )
        normalized.append((entry, contribution))

    selected: List[PoolEntry] = []
    seen_holder_ids: Set[int] = set()
    seen_eval_points: Set[int] = set()
    for entry, contribution in sorted(
        normalized, key=lambda item: _stable_pool_entry(item[0])
    ):
        if (
            contribution.holder_id in seen_holder_ids
            or contribution.holder_eval_point in seen_eval_points
        ):
            continue
        selected.append(entry)
        seen_holder_ids.add(contribution.holder_id)
        seen_eval_points.add(contribution.holder_eval_point)
        if len(selected) == threshold:
            return selected
    return None


class A2DPSSJustifierManager(Justifier):
    """Self-contained Algorithm-4 predicate for A²DPSS contributions."""

    def __init__(
        self,
        contexts: Optional[Sequence[A2DPSSContext]] = None,
        validation_cache_limit: int = 4096,
    ) -> None:
        validation_cache_limit = _strict_int(
            validation_cache_limit,
            name="validation_cache_limit",
            minimum=1,
        )
        self.contexts: Dict[str, A2DPSSContext] = {}
        self.validation_cache_limit = validation_cache_limit
        self._validation_cache: Dict[tuple[str, str, str], bool] = {}
        self._validation_metrics = {
            stage: {
                "invocations": 0,
                "elapsed_sec": 0.0,
                "cpu_seconds": 0.0,
            }
            for stage in ("schema_identity", "pvss", "anchor")
        }
        self._validation_cache_hits = 0
        for context in contexts or ():
            self.register_context(context)

    def _record_validation_elapsed(
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

    def register_context(self, context: A2DPSSContext) -> None:
        if not isinstance(context, A2DPSSContext):
            raise TypeError("context must be an A2DPSSContext")
        existing = self.contexts.get(context.context_digest)
        if existing is not None and existing != context:
            raise ValueError(
                "conflicting A2DPSS context for context_digest"
            )
        self._drop_cached_context(context.context_digest)
        self.contexts[context.context_digest] = context

    def unregister_context(
        self,
        context_or_digest: A2DPSSContext | str,
    ) -> None:
        digest = (
            context_or_digest.context_digest
            if isinstance(context_or_digest, A2DPSSContext)
            else _context_digest(context_or_digest)
        )
        self.contexts.pop(digest, None)
        self._drop_cached_context(digest)

    def _drop_cached_context(self, context_digest: str) -> None:
        for key in tuple(self._validation_cache):
            if key[0] == context_digest:
                self._validation_cache.pop(key, None)

    def _cache_validation(
        self,
        key: tuple[str, str, str],
        valid: bool,
    ) -> None:
        if key not in self._validation_cache and (
            len(self._validation_cache) >= self.validation_cache_limit
        ):
            self._validation_cache.pop(next(iter(self._validation_cache)))
        self._validation_cache[key] = bool(valid)

    def _registration(self, qid: Qid | str) -> A2DPSSContext:
        if isinstance(qid, str):
            qid = Qid.decode(qid)
        if not isinstance(qid, Qid) or qid.kind != A2DPSS_QID_KIND:
            raise ValueError("expected an A2DPSS qid")
        fields = dict(qid.fields)
        digest = _context_digest(fields.get("context_digest"))
        try:
            context = self.contexts[digest]
        except KeyError as exc:
            raise KeyError(
                f"unknown A2DPSS context: {digest}"
            ) from exc
        if not a2dpss_qid_matches_context(qid, context):
            raise ValueError(
                "A2DPSS qid fields do not match registered context"
            )
        return context

    def validate_contribution(
        self,
        context: A2DPSSContext,
        mid: str,
        payload: bytes,
    ) -> Optional[A2DPSSContribution]:
        """Cached validation for non-block auxiliary paths.

        Formal block predicates deliberately bypass this cache through
        :meth:`_validate_contribution_uncached`.  A cache hit here must never
        allow one candidate block to inherit another block's validation work.
        """

        if not isinstance(context, A2DPSSContext):
            return None
        cache_key = (
            context.context_digest,
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
                return A2DPSSContribution.from_bytes(payload)
            except Exception:
                return None
        contribution = self._validate_contribution_uncached(
            context,
            mid,
            payload,
        )
        self._cache_validation(cache_key, contribution is not None)
        return contribution

    def _validate_contribution_uncached(
        self,
        context: A2DPSSContext,
        mid: str,
        payload: bytes,
    ) -> Optional[A2DPSSContribution]:
        """Decode bytes and execute every contribution check from scratch."""

        if not isinstance(context, A2DPSSContext):
            return None
        try:
            contribution = A2DPSSContribution.from_bytes(payload)
            if mid != f"node_{contribution.holder_id}":
                return None
            if not validate_a2_contribution(
                context,
                contribution,
                stage_observer=self._record_validation_elapsed,
            ):
                return None
            return contribution
        except Exception:
            return None

    def select_justified_subset(
        self,
        qid: Qid,
        entries: Sequence[PoolEntry],
    ) -> Optional[List[PoolEntry]]:
        try:
            context = self._registration(qid)
        except (KeyError, TypeError, ValueError):
            return None

        normalized_entries = []
        for entry in entries:
            normalized = _normalize_pool_entry(entry)
            if normalized is not None:
                normalized_entries.append(normalized)

        validated_items: List[A2DPSSValidatedItem] = []
        for mid, payload in sorted(
            normalized_entries, key=_stable_pool_entry
        ):
            contribution = self._validate_contribution_uncached(
                context, mid, payload
            )
            if contribution is not None:
                validated_items.append(
                    ((mid, payload), contribution)
                )
                selected = select_canonical_threshold(
                    validated_items, context.service_t + 1
                )
                if selected is not None:
                    return selected
        return None

    def select_preverified_subset(
        self,
        qid: Qid | str,
        entries: Sequence[PoolEntry],
    ) -> Optional[List[PoolEntry]]:
        """Select canonical ``T`` from an already-verified candidate.

        The caller must supply the exact immutable entry bytes from one or
        more successful ``verify_candidate_set`` invocations (or equivalent
        receiver-side validation).  This method deliberately does not repeat
        PVSS or anchor verification.  It does repeat all cheap bindings needed
        to catch a wiring error at the trusted protocol/finalizer boundary:
        entry shape, qid/context, sender identity, holder/evaluation-point
        uniqueness, threshold size, and deterministic ordering.

        Untrusted callers must continue to use ``select_justified_subset`` or
        ``verify_candidate_set`` first.
        """

        try:
            context = self._registration(qid)
            if len(entries) < context.service_t + 1:
                return None
        except (KeyError, TypeError, ValueError):
            return None

        validated_items: List[A2DPSSValidatedItem] = []
        seen_holder_ids: Set[int] = set()
        seen_eval_points: Set[int] = set()
        try:
            for raw_entry in entries:
                entry = _normalize_pool_entry(raw_entry)
                if entry is None:
                    return None
                mid, payload = entry
                contribution = A2DPSSContribution.from_bytes(payload)
                if (
                    mid != f"node_{contribution.holder_id}"
                    or not hmac.compare_digest(
                        contribution.context_digest,
                        context.context_digest,
                    )
                    or context.holder_eval_points.get(
                        contribution.holder_id
                    )
                    != contribution.holder_eval_point
                    or contribution.holder_id in seen_holder_ids
                    or contribution.holder_eval_point in seen_eval_points
                ):
                    return None
                seen_holder_ids.add(contribution.holder_id)
                seen_eval_points.add(contribution.holder_eval_point)
                validated_items.append((entry, contribution))
        except (TypeError, ValueError):
            return None

        return select_canonical_threshold(
            validated_items,
            context.service_t + 1,
        )

    def verify_candidate_set(
        self,
        qid: Qid | str,
        entries: Sequence[PoolEntry],
    ) -> bool:
        """Validate an all-valid A²DPSS candidate superset for ACS.

        Unlike ``verify_qid_set``, an ACS candidate may contain more than the
        reconstruction threshold.  Every supplied entry must nevertheless be
        well formed, valid for the registered context, and identify a distinct
        holder and evaluation point.
        """

        try:
            context = self._registration(qid)
            if len(entries) < context.service_t + 1:
                return False
        except (KeyError, TypeError, ValueError):
            return False

        normalized_entries: List[PoolEntry] = []
        try:
            for entry in entries:
                normalized = _normalize_pool_entry(entry)
                if normalized is None:
                    return False
                normalized_entries.append(normalized)
        except TypeError:
            return False

        validated_items: List[A2DPSSValidatedItem] = []
        seen_holder_ids: Set[int] = set()
        seen_eval_points: Set[int] = set()
        for mid, payload in sorted(
            normalized_entries, key=_stable_pool_entry
        ):
            contribution = self._validate_contribution_uncached(
                context, mid, payload
            )
            if contribution is None:
                return False
            if (
                contribution.holder_id in seen_holder_ids
                or contribution.holder_eval_point in seen_eval_points
            ):
                return False
            seen_holder_ids.add(contribution.holder_id)
            seen_eval_points.add(contribution.holder_eval_point)
            validated_items.append(((mid, payload), contribution))

        return (
            select_canonical_threshold(
                validated_items, context.service_t + 1
            )
            is not None
        )

    def verify_qid_set(
        self,
        qid: Qid,
        entries: Sequence[PoolEntry],
    ) -> bool:
        try:
            context = self._registration(qid)
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

        validated_items: List[A2DPSSValidatedItem] = []
        for mid, payload in sorted(
            normalized_entries, key=_stable_pool_entry
        ):
            contribution = self._validate_contribution_uncached(
                context, mid, payload
            )
            if contribution is None:
                return False
            validated_items.append(((mid, payload), contribution))
        return (
            select_canonical_threshold(
                validated_items, context.service_t + 1
            )
            is not None
        )


def a2dpss_contribution_digest(
    payload: bytes | bytearray | memoryview,
) -> str:
    """Return the domain-separated digest of canonical contribution bytes."""

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("contribution payload must be bytes-like")
    return hashlib.sha256(
        _CONTRIBUTION_DIGEST_DOMAIN + bytes(payload)
    ).hexdigest()


def _canonical_a2dpss_qid(value: Qid | str) -> str:
    encoded = value.encode() if isinstance(value, Qid) else str(value)
    decoded = Qid.decode(encoded)
    if decoded.kind != A2DPSS_QID_KIND or decoded.encode() != encoded:
        raise ValueError("qid must be a canonical A2DPSS qid")
    return encoded


def _int_tuple(
    value: Sequence[int],
    *,
    name: str,
    minimum: int,
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


def _digest_tuple(
    value: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    normalized = tuple(_context_digest(item) for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} entries must be distinct")
    return normalized


def _normalized_point_tuple(
    value: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty sequence")
    try:
        return tuple(G1Point.from_hex(item).to_hex() for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains a malformed G1 point") from exc


@dataclass(frozen=True)
class A2DPSSPublicResult:
    """Versioned public output derived by every observer from one fixed T."""

    qid: str
    context_digest: str
    accepted_holder_ids: Sequence[int]
    accepted_eval_points: Sequence[int]
    accepted_contribution_digests: Sequence[str]
    successor_commitment_vector: Sequence[str]
    version: int = A2DPSS_PUBLIC_RESULT_VERSION

    def __post_init__(self) -> None:
        version = _strict_int(self.version, name="version")
        if version != A2DPSS_PUBLIC_RESULT_VERSION:
            raise ValueError("unsupported A2DPSS public-result version")
        qid = _canonical_a2dpss_qid(self.qid)
        context_digest = _context_digest(self.context_digest)
        qid_digest = dict(Qid.decode(qid).fields).get("context_digest")
        if qid_digest != context_digest:
            raise ValueError(
                "A2DPSS public-result qid and context_digest do not match"
            )
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
        contribution_digests = _digest_tuple(
            self.accepted_contribution_digests,
            name="accepted_contribution_digests",
        )
        if not (
            len(holder_ids)
            == len(eval_points)
            == len(contribution_digests)
        ):
            raise ValueError("accepted T fields must have the same length")
        commitments = _normalized_point_tuple(
            self.successor_commitment_vector,
            name="successor_commitment_vector",
        )

        object.__setattr__(self, "version", version)
        object.__setattr__(self, "qid", qid)
        object.__setattr__(self, "context_digest", context_digest)
        object.__setattr__(self, "accepted_holder_ids", holder_ids)
        object.__setattr__(self, "accepted_eval_points", eval_points)
        object.__setattr__(
            self,
            "accepted_contribution_digests",
            contribution_digests,
        )
        object.__setattr__(
            self, "successor_commitment_vector", commitments
        )

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": _PUBLIC_RESULT_KIND,
            "version": self.version,
            "qid": self.qid,
            "context_digest": self.context_digest,
            "accepted_holder_ids": list(self.accepted_holder_ids),
            "accepted_eval_points": list(self.accepted_eval_points),
            "accepted_contribution_digests": list(
                self.accepted_contribution_digests
            ),
            "successor_commitment_vector": list(
                self.successor_commitment_vector
            ),
        }

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_obj())

    @classmethod
    def from_obj(
        cls,
        value: Mapping[str, Any],
    ) -> "A2DPSSPublicResult":
        if not isinstance(value, Mapping):
            raise ValueError("A2DPSS public result must be a mapping")
        expected = {
            "kind",
            "version",
            "qid",
            "context_digest",
            "accepted_holder_ids",
            "accepted_eval_points",
            "accepted_contribution_digests",
            "successor_commitment_vector",
        }
        missing = expected - set(value)
        extra = set(value) - expected
        if missing:
            raise ValueError(
                f"A2DPSS public result is missing fields: {sorted(missing)}"
            )
        if extra:
            raise ValueError(
                f"A2DPSS public result has unknown fields: {sorted(extra)}"
            )
        if value["kind"] != _PUBLIC_RESULT_KIND:
            raise ValueError("expected an A2DPSS public-result object")
        if value["version"] != A2DPSS_PUBLIC_RESULT_VERSION:
            raise ValueError("unsupported A2DPSS public-result version")
        return cls(
            version=value["version"],
            qid=value["qid"],
            context_digest=value["context_digest"],
            accepted_holder_ids=value["accepted_holder_ids"],
            accepted_eval_points=value["accepted_eval_points"],
            accepted_contribution_digests=value[
                "accepted_contribution_digests"
            ],
            successor_commitment_vector=value[
                "successor_commitment_vector"
            ],
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview | str,
    ) -> "A2DPSSPublicResult":
        return cls.from_obj(
            _decode_json_object(data, name="A2DPSS public result")
        )


def finalize_a2_validated_items(
    context: A2DPSSContext,
    validated_items: Sequence[A2DPSSValidatedItem],
) -> A2DPSSPublicResult:
    """Finalize one exact, already-validated T into public commitments.

    Both A²DPSS and OCDPSS call this function.  The caller must first apply
    its complete justifier; this function enforces exact threshold size,
    canonical ordering, and holder/evaluation-point uniqueness before
    interpolating the commitment vectors.
    """

    if not isinstance(context, A2DPSSContext):
        raise TypeError("context must be an A2DPSSContext")
    threshold = context.service_t + 1
    if len(validated_items) != threshold:
        raise ValueError(
            "validated_items must contain exactly service_t+1 entries"
        )

    normalized: List[A2DPSSValidatedItem] = []
    for value in validated_items:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise TypeError(
                "validated_items must contain (PoolEntry, contribution) pairs"
            )
        raw_entry, contribution = value
        entry = _normalize_pool_entry(raw_entry)
        if entry is None:
            raise ValueError("validated item contains a malformed PoolEntry")
        if not isinstance(contribution, A2DPSSContribution):
            raise TypeError(
                "validated item contribution must be an A2DPSSContribution"
            )
        if not hmac.compare_digest(
            contribution.context_digest, context.context_digest
        ):
            raise ValueError(
                "validated contribution context_digest mismatch"
            )
        expected_eval_point = context.holder_eval_points.get(
            contribution.holder_id
        )
        if expected_eval_point != contribution.holder_eval_point:
            raise ValueError(
                "validated contribution identity does not match context"
            )
        normalized.append((entry, contribution))

    selected = select_canonical_threshold(normalized, threshold)
    ordered = sorted(
        normalized, key=lambda item: _stable_pool_entry(item[0])
    )
    ordered_entries = [entry for entry, _contribution in ordered]
    if selected is None or selected != ordered_entries:
        raise ValueError(
            "validated_items do not form one canonical exact threshold T"
        )

    contributions = [
        contribution for _entry, contribution in ordered
    ]
    eval_points = tuple(
        contribution.holder_eval_point
        for contribution in contributions
    )
    successor_commitments = interpolate_commitment_vector(
        eval_points,
        [
            contribution.transcript.commitment_vector
            for contribution in contributions
        ],
        context.service_t,
    )
    if len(successor_commitments) != len(
        context.successor_rpk_list
    ) + 1:
        raise ValueError(
            "interpolated commitment vector has unexpected length"
        )
    return A2DPSSPublicResult(
        qid=context.encode_qid(),
        context_digest=context.context_digest,
        accepted_holder_ids=tuple(
            contribution.holder_id for contribution in contributions
        ),
        accepted_eval_points=eval_points,
        accepted_contribution_digests=tuple(
            a2dpss_contribution_digest(contribution.to_bytes())
            for contribution in contributions
        ),
        successor_commitment_vector=successor_commitments,
    )


class A2DPSSPublicFinalizer:
    """Derive the public T record and vc' for each delivered A²DPSS qid."""

    def __init__(self, justifier: A2DPSSJustifierManager) -> None:
        if not isinstance(justifier, A2DPSSJustifierManager):
            raise TypeError(
                "justifier must be an A2DPSSJustifierManager"
            )
        self.justifier = justifier
        self.results: Dict[str, A2DPSSPublicResult] = {}
        self._phase_metrics = {
            "total": {
                "invocations": 0,
                "elapsed_sec": 0.0,
                "cpu_seconds": 0.0,
            },
            "commitment_interpolation": {
                "invocations": 0,
                "elapsed_sec": 0.0,
                "cpu_seconds": 0.0,
            }
        }

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
    ) -> Optional[A2DPSSPublicResult]:
        """Validate an untrusted delivery before deriving its public result.

        This entry point is intentionally self-contained and remains suitable
        for analyzers and callers that did not obtain ``entries`` directly
        from a successful TOB predicate invocation.  Integrated TOB delivery
        paths should use :meth:`on_verified_deliver` so that the justifier is
        not evaluated a second time after ``t_deliver``.
        """
        return self._finalize_delivery(qid, entries, verify_entries=True)

    def on_verified_deliver(
        self,
        qid: Qid | str,
        entries: Sequence[PoolEntry],
    ) -> Optional[A2DPSSPublicResult]:
        """Finalize the exact set already accepted by the TOB justifier.

        The caller must pass the same immutable entry bytes for which its
        local TOB instance just returned ``verify_qid_set(...)=True``.  This
        method repeats inexpensive schema, identity, threshold, uniqueness,
        and canonical-order checks, but deliberately does not repeat PVSS or
        anchor verification.  Algorithm 4 places those cryptographic checks
        in the justifier that guards ``t_deliver``.
        """
        return self._finalize_delivery(qid, entries, verify_entries=False)

    def _finalize_delivery(
        self,
        qid: Qid | str,
        entries: Sequence[PoolEntry],
        *,
        verify_entries: bool,
    ) -> Optional[A2DPSSPublicResult]:
        total_wall_started = time.perf_counter()
        total_cpu_started = time.process_time()
        try:
            context = self.justifier._registration(qid)
            qid_text = context.encode_qid()
        except (KeyError, TypeError, ValueError):
            return None

        normalized_entries = []
        for entry in entries:
            normalized = _normalize_pool_entry(entry)
            if normalized is None:
                return None
            normalized_entries.append(normalized)
        ordered_entries = sorted(
            normalized_entries, key=_stable_pool_entry
        )

        decoded_items: List[A2DPSSValidatedItem] = []
        try:
            for mid, payload in ordered_entries:
                contribution = A2DPSSContribution.from_bytes(payload)
                decoded_items.append(
                    ((mid, payload), contribution)
                )
        except (TypeError, ValueError):
            return None
        candidate_digests = tuple(
            a2dpss_contribution_digest(
                contribution.to_bytes()
            )
            for _entry, contribution in decoded_items
        )

        existing = self.results.get(qid_text)
        if existing is not None:
            if (
                existing.accepted_contribution_digests
                != candidate_digests
            ):
                raise ValueError(
                    "conflicting T for completed A2DPSS qid"
                )
            return existing

        if verify_entries:
            if not self.justifier.verify_qid_set(
                context.build_qid(), ordered_entries
            ):
                return None

            validated_items: List[A2DPSSValidatedItem] = []
            for mid, payload in ordered_entries:
                contribution = self.justifier.validate_contribution(
                    context, mid, payload
                )
                if contribution is None:
                    return None
                validated_items.append(
                    ((mid, payload), contribution)
                )
        else:
            # These bindings are cheap to repeat and keep a wiring mistake at
            # the trusted TOB/finalizer boundary from producing a public
            # result.  The expensive PVSS and anchor checks have already run
            # in the justifier immediately before t_deliver.
            for (mid, _payload), contribution in decoded_items:
                if mid != f"node_{contribution.holder_id}":
                    return None
            validated_items = decoded_items

        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        result = finalize_a2_validated_items(
            context, validated_items
        )
        metric = self._phase_metrics["commitment_interpolation"]
        metric["invocations"] += 1
        metric["elapsed_sec"] += time.perf_counter() - wall_started
        metric["cpu_seconds"] += time.process_time() - cpu_started
        self.results[qid_text] = result
        total = self._phase_metrics["total"]
        total["invocations"] += 1
        total["elapsed_sec"] += time.perf_counter() - total_wall_started
        total["cpu_seconds"] += time.process_time() - total_cpu_started
        return result


def _canonical_scalar(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer scalar")
    if not 0 <= value < FIELD_ORDER:
        raise ValueError(f"{name} must be a canonical field scalar")
    return value


def _scalar_from_wire(value: Any, *, name: str) -> int:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a canonical decimal string")
    if not value or not value.isdigit():
        raise ValueError(f"{name} must be a canonical decimal string")
    scalar = int(value)
    if str(scalar) != value:
        raise ValueError(f"{name} must be a canonical decimal string")
    return _canonical_scalar(scalar, name=name)


@dataclass(frozen=True)
class A2DPSSSuccessorResult:
    """Versioned private output for one A²DPSS successor role."""

    qid: str
    context_digest: str
    my_node_id: int
    my_role_idx: int
    share_s: int
    share_s_hat: int
    accepted_eval_points: Sequence[int]
    accepted_contribution_digests: Sequence[str]
    version: int = A2DPSS_SUCCESSOR_RESULT_VERSION

    def __post_init__(self) -> None:
        version = _strict_int(self.version, name="version")
        if version != A2DPSS_SUCCESSOR_RESULT_VERSION:
            raise ValueError("unsupported A2DPSS successor-result version")
        qid = _canonical_a2dpss_qid(self.qid)
        context_digest = _context_digest(self.context_digest)
        qid_digest = dict(Qid.decode(qid).fields).get("context_digest")
        if qid_digest != context_digest:
            raise ValueError(
                "A2DPSS successor-result qid and context_digest do not match"
            )
        eval_points = _int_tuple(
            self.accepted_eval_points,
            name="accepted_eval_points",
            minimum=1,
        )
        contribution_digests = _digest_tuple(
            self.accepted_contribution_digests,
            name="accepted_contribution_digests",
        )
        if len(eval_points) != len(contribution_digests):
            raise ValueError(
                "successor result T fields must have the same length"
            )

        object.__setattr__(self, "version", version)
        object.__setattr__(self, "qid", qid)
        object.__setattr__(self, "context_digest", context_digest)
        object.__setattr__(
            self,
            "my_node_id",
            _strict_int(self.my_node_id, name="my_node_id"),
        )
        object.__setattr__(
            self,
            "my_role_idx",
            _strict_int(self.my_role_idx, name="my_role_idx"),
        )
        object.__setattr__(
            self,
            "share_s",
            _canonical_scalar(self.share_s, name="share_s"),
        )
        object.__setattr__(
            self,
            "share_s_hat",
            _canonical_scalar(self.share_s_hat, name="share_s_hat"),
        )
        object.__setattr__(self, "accepted_eval_points", eval_points)
        object.__setattr__(
            self,
            "accepted_contribution_digests",
            contribution_digests,
        )

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
            "accepted_contribution_digests": list(
                self.accepted_contribution_digests
            ),
        }

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_obj())

    @classmethod
    def from_obj(
        cls,
        value: Mapping[str, Any],
    ) -> "A2DPSSSuccessorResult":
        if not isinstance(value, Mapping):
            raise ValueError("A2DPSS successor result must be a mapping")
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
            "accepted_contribution_digests",
        }
        missing = expected - set(value)
        extra = set(value) - expected
        if missing:
            raise ValueError(
                f"A2DPSS successor result is missing fields: {sorted(missing)}"
            )
        if extra:
            raise ValueError(
                f"A2DPSS successor result has unknown fields: {sorted(extra)}"
            )
        if value["kind"] != _SUCCESSOR_RESULT_KIND:
            raise ValueError("expected an A2DPSS successor-result object")
        if value["version"] != A2DPSS_SUCCESSOR_RESULT_VERSION:
            raise ValueError("unsupported A2DPSS successor-result version")
        return cls(
            version=value["version"],
            qid=value["qid"],
            context_digest=value["context_digest"],
            my_node_id=value["my_node_id"],
            my_role_idx=value["my_role_idx"],
            share_s=_scalar_from_wire(value["share_s"], name="share_s"),
            share_s_hat=_scalar_from_wire(
                value["share_s_hat"], name="share_s_hat"
            ),
            accepted_eval_points=value["accepted_eval_points"],
            accepted_contribution_digests=value[
                "accepted_contribution_digests"
            ],
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview | str,
    ) -> "A2DPSSSuccessorResult":
        return cls.from_obj(
            _decode_json_object(data, name="A2DPSS successor result")
        )

    def to_holder_state(self) -> "A2DPSSHolderState":
        """Move this successor share pair into the next holder abstraction."""

        return A2DPSSHolderState(self.share_s, self.share_s_hat)


A2DPSSReconstructionStageObserver = Callable[[str, float, float], None]


def _observe_reconstruction_stage(
    observer: Optional[A2DPSSReconstructionStageObserver],
    stage: str,
    wall_started: float,
    cpu_started: float,
) -> None:
    if observer is not None:
        try:
            observer(
                stage,
                time.perf_counter() - wall_started,
                time.process_time() - cpu_started,
            )
        except Exception:
            # Observability must never influence reconstruction validity.
            pass


def _point_vectors_equal(
    left: Sequence[str],
    right: Sequence[str],
) -> bool:
    if len(left) != len(right):
        return False
    try:
        return all(
            G1Point.from_hex(a) == G1Point.from_hex(b)
            for a, b in zip(left, right)
        )
    except (TypeError, ValueError):
        return False


def reconstruct_a2_validated_items(
    context: A2DPSSContext,
    public_result: A2DPSSPublicResult,
    validated_items: Sequence[A2DPSSValidatedItem],
    *,
    my_node_id: int,
    my_role_idx: int,
    my_rsk: int,
    stage_observer: Optional[
        A2DPSSReconstructionStageObserver
    ] = None,
) -> A2DPSSSuccessorResult:
    """Reconstruct exactly the successor share committed by public T."""

    if not isinstance(context, A2DPSSContext):
        raise TypeError("context must be an A2DPSSContext")
    if not isinstance(public_result, A2DPSSPublicResult):
        raise TypeError("public_result must be an A2DPSSPublicResult")
    my_node_id = _strict_int(my_node_id, name="my_node_id")
    my_role_idx = _strict_int(my_role_idx, name="my_role_idx")
    my_rsk = _canonical_scalar(my_rsk, name="my_rsk")
    if public_result.qid != context.encode_qid():
        raise ValueError("public result qid does not match context")
    if public_result.context_digest != context.context_digest:
        raise ValueError("public result context_digest does not match context")
    if len(public_result.accepted_eval_points) != context.service_t + 1:
        raise ValueError(
            "public result does not contain exactly service_t+1 entries"
        )
    if context.successor_role_indices.get(my_node_id) != my_role_idx:
        raise ValueError("successor node/role does not match context")
    if not 0 <= my_role_idx < len(context.successor_rpk_list):
        raise ValueError("successor role index is out of range")

    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    recomputed_public = finalize_a2_validated_items(
        context, validated_items
    )
    _observe_reconstruction_stage(
        stage_observer,
        "commitment_recompute",
        wall_started,
        cpu_started,
    )
    if (
        recomputed_public.accepted_holder_ids
        != public_result.accepted_holder_ids
        or recomputed_public.accepted_eval_points
        != public_result.accepted_eval_points
        or recomputed_public.accepted_contribution_digests
        != public_result.accepted_contribution_digests
    ):
        raise ValueError("validated entries do not match public result T")
    if not _point_vectors_equal(
        recomputed_public.successor_commitment_vector,
        public_result.successor_commitment_vector,
    ):
        raise ValueError(
            "public successor commitment vector does not match T"
        )

    ordered = sorted(
        validated_items, key=lambda item: _stable_pool_entry(item[0])
    )
    contributions = [
        contribution for _entry, contribution in ordered
    ]
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    share_pairs = [
        pvss_deliver(
            context.g_hex,
            context.h_hex,
            contribution.transcript,
            my_rsk,
            my_role_idx,
        )
        for contribution in contributions
    ]
    _observe_reconstruction_stage(
        stage_observer,
        "pvss_decrypt",
        wall_started,
        cpu_started,
    )

    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    share_s, share_s_hat = pvss_interpolate(
        public_result.accepted_eval_points,
        share_pairs,
        context.service_t,
    )
    _observe_reconstruction_stage(
        stage_observer,
        "share_interpolation",
        wall_started,
        cpu_started,
    )

    successor_eval_point = my_role_idx + 1
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    reconstructed_commitment = pedersen_commit(
        context.g_hex,
        context.h_hex,
        share_s,
        share_s_hat,
    )
    expected_commitment = public_result.successor_commitment_vector[
        successor_eval_point
    ]
    if G1Point.from_hex(reconstructed_commitment) != G1Point.from_hex(
        expected_commitment
    ):
        raise ValueError(
            "reconstructed successor share does not open vc' entry"
        )
    _observe_reconstruction_stage(
        stage_observer,
        "commitment_opening_check",
        wall_started,
        cpu_started,
    )

    return A2DPSSSuccessorResult(
        qid=public_result.qid,
        context_digest=context.context_digest,
        my_node_id=my_node_id,
        my_role_idx=my_role_idx,
        share_s=share_s,
        share_s_hat=share_s_hat,
        accepted_eval_points=public_result.accepted_eval_points,
        accepted_contribution_digests=(
            public_result.accepted_contribution_digests
        ),
    )


class A2DPSSSuccessorReconstructionManager:
    """Recover one private successor share from a fixed public A²DPSS T."""

    def __init__(self) -> None:
        self.registrations: Dict[str, dict] = {}
        self.results: Dict[str, A2DPSSSuccessorResult] = {}
        self._consumed_results: Set[str] = set()
        self._phase_metrics = {
            stage: {
                "invocations": 0,
                "elapsed_sec": 0.0,
                "cpu_seconds": 0.0,
            }
            for stage in (
                "total",
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
        context: A2DPSSContext,
        *,
        my_node_id: int,
        my_rsk: Optional[int] = None,
        on_new_share=None,
    ) -> None:
        if not isinstance(context, A2DPSSContext):
            raise TypeError("context must be an A2DPSSContext")
        my_node_id = _strict_int(my_node_id, name="my_node_id")
        role_idx = context.successor_role_indices.get(my_node_id)
        if role_idx is None:
            if my_rsk is not None:
                raise ValueError(
                    "non-successor observer must not register an rsk"
                )
            normalized_rsk = None
        else:
            if my_rsk is None:
                raise ValueError("successor role requires my_rsk")
            normalized_rsk = _canonical_scalar(my_rsk, name="my_rsk")
            expected_rpk = G1Point.from_hex(
                context.successor_rpk_list[role_idx]
            )
            actual_rpk = (
                G1Point.from_hex(context.key_base) * normalized_rsk
            )
            if actual_rpk != expected_rpk:
                raise ValueError(
                    "my_rsk does not match successor role public key"
                )
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
                raise ValueError(
                    "conflicting successor registration for context"
                )
            return
        self.registrations[context.context_digest] = registration

    def unregister_context(
        self,
        context_or_digest: A2DPSSContext | str,
    ) -> None:
        digest = (
            context_or_digest.context_digest
            if isinstance(context_or_digest, A2DPSSContext)
            else _context_digest(context_or_digest)
        )
        self.registrations.pop(digest, None)
        self.results.pop(digest, None)

    def consume_result_to_holder_state(
        self,
        context_or_digest: A2DPSSContext | str,
    ) -> "A2DPSSHolderState":
        """Transfer one reconstructed share into a one-shot holder state.

        The result is removed from this manager before the holder state is
        returned.  This is logical ownership transfer only: Python cannot
        guarantee physical zeroization of immutable integers or interpreter
        managed memory.
        """

        digest = (
            context_or_digest.context_digest
            if isinstance(context_or_digest, A2DPSSContext)
            else _context_digest(context_or_digest)
        )
        if digest in self._consumed_results:
            raise RuntimeError(
                "successor result already consumed for next handover"
            )
        try:
            result = self.results.pop(digest)
        except KeyError as exc:
            raise RuntimeError(
                "no reconstructed successor result to consume"
            ) from exc
        self._consumed_results.add(digest)
        return result.to_holder_state()

    def _resolve_public_items(
        self,
        context: A2DPSSContext,
        public_result: A2DPSSPublicResult,
        entries: Sequence[PoolEntry],
    ) -> List[A2DPSSValidatedItem]:
        normalized_entries = []
        for entry in entries:
            normalized = _normalize_pool_entry(entry)
            if normalized is None:
                raise ValueError(
                    "successor entries contain a malformed PoolEntry"
                )
            normalized_entries.append(normalized)
        if len(normalized_entries) != len(
            public_result.accepted_contribution_digests
        ):
            raise ValueError(
                "successor entries must contain exactly public result T"
            )

        entries_by_digest = {}
        for mid, payload in normalized_entries:
            try:
                contribution = A2DPSSContribution.from_bytes(payload)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "successor entry is not an A2DPSS contribution"
                ) from exc
            digest = a2dpss_contribution_digest(
                contribution.to_bytes()
            )
            if digest in entries_by_digest:
                raise ValueError(
                    "successor entries contain a duplicate contribution digest"
                )
            entries_by_digest[digest] = (
                (mid, payload),
                contribution,
            )
        if set(entries_by_digest) != set(
            public_result.accepted_contribution_digests
        ):
            raise ValueError(
                "successor entries do not match public result T"
            )

        validated_items = []
        for index, digest in enumerate(
            public_result.accepted_contribution_digests
        ):
            entry, contribution = entries_by_digest[digest]
            if (
                contribution.context_digest
                != context.context_digest
            ):
                raise ValueError(
                    "successor contribution context digest mismatch"
                )
            if (
                contribution.holder_id
                != public_result.accepted_holder_ids[index]
            ):
                raise ValueError(
                    "successor contribution holder does not match public T"
                )
            if (
                contribution.holder_eval_point
                != public_result.accepted_eval_points[index]
            ):
                raise ValueError(
                    "successor contribution eval point does not match public T"
                )
            if entry[0] != f"node_{contribution.holder_id}":
                raise ValueError(
                    "successor contribution mid does not match holder"
                )
            validated_items.append((entry, contribution))
        return validated_items

    def reconstruct(
        self,
        public_result: A2DPSSPublicResult,
        entries: Sequence[PoolEntry],
    ) -> Optional[A2DPSSSuccessorResult]:
        if not isinstance(public_result, A2DPSSPublicResult):
            raise TypeError(
                "public_result must be an A2DPSSPublicResult"
            )
        local = self.registrations.get(public_result.context_digest)
        if local is None:
            return None
        context = local["context"]
        if public_result.qid != context.encode_qid():
            raise ValueError(
                "public result qid does not match registered context"
            )
        if len(public_result.accepted_eval_points) != (
            context.service_t + 1
        ):
            raise ValueError(
                "public result does not contain exactly service_t+1 entries"
            )
        if local["my_role_idx"] is None:
            return None

        total_wall_started = time.perf_counter()
        total_cpu_started = time.process_time()
        validated_items = self._resolve_public_items(
            context, public_result, entries
        )
        existing = self.results.get(context.context_digest)
        if existing is not None:
            if (
                existing.accepted_contribution_digests
                != public_result.accepted_contribution_digests
            ):
                raise ValueError(
                    "conflicting T for completed successor reconstruction"
                )
            return existing

        result = reconstruct_a2_validated_items(
            context,
            public_result,
            validated_items,
            my_node_id=local["my_node_id"],
            my_role_idx=local["my_role_idx"],
            my_rsk=local["my_rsk"],
            stage_observer=self._record_phase_elapsed,
        )
        self.results[context.context_digest] = result
        callback = local["on_new_share"]
        if callback is not None:
            callback(result.share_s, result.share_s_hat)
        self._record_phase_elapsed(
            "total",
            time.perf_counter() - total_wall_started,
            time.process_time() - total_cpu_started,
        )
        return result


def _holder_eval_point(
    context: A2DPSSContext,
    holder_id: int,
) -> int:
    try:
        return context.holder_eval_points[holder_id]
    except KeyError as exc:
        raise ValueError(
            f"holder_id {holder_id} is not in the source committee"
        ) from exc


@dataclass
class A2DPSSHolderState:
    """Logical one-shot container for one holder's private share pair."""

    share_s: int
    share_s_hat: int
    consumed: bool = field(default=False, init=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.share_s = _canonical_scalar(
            self.share_s,
            name="share_s",
        )
        self.share_s_hat = _canonical_scalar(
            self.share_s_hat,
            name="share_s_hat",
        )

    def consume(self) -> tuple[int, int]:
        """Return the share pair once, then logically erase local ownership."""

        with self._lock:
            if self.consumed:
                raise RuntimeError(
                    "A2DPSS holder state already consumed (one-shot)"
                )
            share_pair = (self.share_s, self.share_s_hat)
            self.share_s = None
            self.share_s_hat = None
            self.consumed = True
            return share_pair


@dataclass
class A2DPSSHolderMetrics:
    pvss_seconds: float = 0.0
    pvss_cpu_seconds: float = 0.0
    serialization_seconds: float = 0.0
    broadcast_seconds: float = 0.0
    contribution_bytes: int = 0


@dataclass(frozen=True)
class A2DPSSHolderActionResult:
    qid: str
    mid: str
    contribution: A2DPSSContribution
    contribution_bytes: bytes
    metrics: A2DPSSHolderMetrics


def make_a2dpss_contribution(
    *,
    context: A2DPSSContext,
    holder_id: int,
    share_s: int,
    share_s_hat: int,
    metrics: A2DPSSHolderMetrics | None = None,
) -> A2DPSSContribution:
    """Construct one operation-independent holder contribution locally."""

    if not isinstance(context, A2DPSSContext):
        raise TypeError("context must be an A2DPSSContext")
    holder_id = _strict_int(holder_id, name="holder_id")
    share_s = _canonical_scalar(share_s, name="share_s")
    share_s_hat = _canonical_scalar(
        share_s_hat,
        name="share_s_hat",
    )
    holder_eval_point = _holder_eval_point(context, holder_id)
    timings = metrics if metrics is not None else A2DPSSHolderMetrics()
    if not isinstance(timings, A2DPSSHolderMetrics):
        raise TypeError("metrics must be an A2DPSSHolderMetrics")

    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    transcript = pvss_deal(
        context.g_hex,
        context.h_hex,
        context.successor_rpk_list,
        share_s,
        share_s_hat,
        context.service_t,
        dealer_id=holder_id,
        dealer_eval_point=holder_eval_point,
        key_base_hex=context.key_base,
    )
    timings.pvss_seconds = time.perf_counter() - wall_started
    timings.pvss_cpu_seconds = time.process_time() - cpu_started
    return A2DPSSContribution(
        holder_id=holder_id,
        holder_eval_point=holder_eval_point,
        context_digest=context.context_digest,
        transcript=transcript,
    )


async def run_a2dpss_handover_role(
    *,
    context: A2DPSSContext,
    holder_id: int,
    share_s: int,
    share_s_hat: int,
    broadcast_contribution,
    holder_state: A2DPSSHolderState,
) -> A2DPSSHolderActionResult:
    """Execute one A²DPSS holder role with an injected transport callback."""

    if not callable(broadcast_contribution):
        raise TypeError("broadcast_contribution must be callable")
    if not isinstance(holder_state, A2DPSSHolderState):
        raise TypeError(
            "holder_state must be an A2DPSSHolderState"
        )
    if not isinstance(context, A2DPSSContext):
        raise TypeError("context must be an A2DPSSContext")
    holder_id = _strict_int(holder_id, name="holder_id")
    share_s = _canonical_scalar(share_s, name="share_s")
    share_s_hat = _canonical_scalar(
        share_s_hat,
        name="share_s_hat",
    )

    # Reject public/configuration errors before consuming the private role.
    # Every failure after consume remains fail-closed because a retry could
    # create a second randomized transcript for the same one-shot role.
    _holder_eval_point(context, holder_id)
    consumed_s, consumed_s_hat = holder_state.consume()
    if (consumed_s, consumed_s_hat) != (share_s, share_s_hat):
        raise ValueError(
            "holder_state share pair does not match holder arguments"
        )

    metrics = A2DPSSHolderMetrics()
    contribution = make_a2dpss_contribution(
        context=context,
        holder_id=holder_id,
        share_s=consumed_s,
        share_s_hat=consumed_s_hat,
        metrics=metrics,
    )
    # Logical erasure only: Python cannot physically zero immutable integers.
    del consumed_s, consumed_s_hat, share_s, share_s_hat
    started = time.perf_counter()
    contribution_bytes = contribution.to_bytes()
    metrics.serialization_seconds = time.perf_counter() - started
    metrics.contribution_bytes = len(contribution_bytes)

    qid = context.encode_qid()
    mid = f"node_{holder_id}"
    started = time.perf_counter()
    try:
        pending = broadcast_contribution(
            qid,
            mid,
            contribution_bytes,
        )
        if inspect.isawaitable(pending):
            await pending
    finally:
        metrics.broadcast_seconds = time.perf_counter() - started

    return A2DPSSHolderActionResult(
        qid=qid,
        mid=mid,
        contribution=contribution,
        contribution_bytes=contribution_bytes,
        metrics=metrics,
    )
