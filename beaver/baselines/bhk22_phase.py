"""Step 7 ordering artifacts, phase contexts, and controller for BHK+22."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from beaver.baselines.bhk22_context import (
    BHK22PublicContext,
    canonical_json,
    digest_hex,
    nonempty_text,
    strict_int,
)
from beaver.baselines.bhk22_contract import (
    BHK22_QID_KIND,
    COMPLETION_MODES,
    DEFAULT_COMPLETION_MODE,
    PATH_ONE_ACCUSATION,
    PATH_OPTIMISTIC,
    PHASE_DEAL,
    PHASE_FINALIZE,
    PHASE_RESPONSE,
    PHASE_VERIFY,
    ROLE_DEALER,
    ROLE_RESPONDER,
    ROLE_SHAREHOLDER,
    ROLE_VERIFIER,
    path_phases,
)
from beaver.baselines.bhk22_messages import (
    BHK22DealerBatch,
    BHK22PublicFinalizeOutput,
    BHK22ResponderBatch,
    BHK22VerifierBatch,
)
from beaver.dag_tob import Justifier, PoolEntry, Qid


ARTIFACT_VERSION = 1
ARTIFACT_DIGEST_DOMAIN = b"ATCaaS/BHK22/ordered-artifact/v1\x00"
NO_PUBLIC_STATE = "-"

PHASE_ROLE_TYPES = {
    PHASE_DEAL: ROLE_DEALER,
    PHASE_VERIFY: ROLE_VERIFIER,
    PHASE_RESPONSE: ROLE_RESPONDER,
    PHASE_FINALIZE: ROLE_SHAREHOLDER,
}
PHASE_MESSAGE_TYPES = {
    PHASE_DEAL: BHK22DealerBatch,
    PHASE_VERIFY: BHK22VerifierBatch,
    PHASE_RESPONSE: BHK22ResponderBatch,
    PHASE_FINALIZE: BHK22PublicFinalizeOutput,
}
PHASE_ARTIFACT_KINDS = {
    PHASE_DEAL: "atcaas-bhk22-deal-ordered-artifact",
    PHASE_VERIFY: "atcaas-bhk22-verify-ordered-artifact",
    PHASE_RESPONSE: "atcaas-bhk22-response-ordered-artifact",
    PHASE_FINALIZE: "atcaas-bhk22-finalize-ordered-artifact",
}


def bhk22_role_mid(role_type: str, role_id: int, network_sender_id: int) -> str:
    if role_type not in PHASE_ROLE_TYPES.values():
        raise ValueError("unsupported BHK22 role type")
    role = strict_int(role_id, name="role id")
    sender = strict_int(network_sender_id, name="network sender id")
    return f"node_{sender}:bhk22:{role_type}:{role}"


def _canonical_ids(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an integer sequence")
    result = tuple(strict_int(value, name=f"{name} entry") for value in values)
    if not result or result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be non-empty, sorted, and unique")
    return result


@dataclass(frozen=True)
class BHK22PhaseContext:
    context: BHK22PublicContext
    path_mode: str
    phase: str
    sequence: int
    expected_role_ids: Sequence[int]
    completion_mode: str = DEFAULT_COMPLETION_MODE
    parent_artifact_digests: Sequence[str] = ()
    public_state_digest: str | None = None
    qid_override: str | None = field(default=None, repr=False)
    qid: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.context, BHK22PublicContext):
            raise TypeError("phase context requires BHK22PublicContext")
        path_mode = str(self.path_mode)
        completion_mode = str(self.completion_mode)
        if completion_mode not in COMPLETION_MODES:
            raise ValueError("unsupported BHK22 completion mode")
        phases = path_phases(path_mode, completion_mode)
        phase = str(self.phase)
        sequence = strict_int(self.sequence, name="phase sequence")
        if sequence >= len(phases) or phases[sequence] != phase:
            raise ValueError("phase/sequence does not match path")
        expected = _canonical_ids(self.expected_role_ids, name="expected role ids")
        if any(role >= self.context.parameters.n_r for role in expected):
            raise ValueError("expected role id is outside n_r")
        parents = tuple(
            digest_hex(value, name="parent artifact digest")
            for value in self.parent_artifact_digests
        )
        if len(set(parents)) != len(parents):
            raise ValueError("parent artifact digests must be distinct")
        if sequence == 0 and parents:
            raise ValueError("deal phase cannot have a parent artifact")
        if sequence > 0 and not parents:
            raise ValueError("non-deal phase requires parent artifacts")
        public_state = self.public_state_digest
        if public_state is not None:
            public_state = digest_hex(public_state, name="public state digest")
        if sequence > 0 and public_state is None:
            raise ValueError("non-deal phase requires a public-state digest")
        if self.qid_override is None:
            qid = Qid.build(
                BHK22_QID_KIND,
                root_context_digest=self.context.root_context_digest,
                trial_id=self.context.trial_id,
                trial_epoch=self.context.trial_epoch,
                transfer_seq=self.context.transfer_seq,
                path_mode=path_mode,
                completion_mode=completion_mode,
                phase=phase,
                sequence=sequence,
                parent_artifact_digests=",".join(parents) or "-",
                public_state_digest=public_state or NO_PUBLIC_STATE,
            ).encode()
        else:
            qid = nonempty_text(self.qid_override, name="phase qid override")
            if Qid.decode(qid).encode() != qid:
                raise ValueError("phase qid override must be canonical")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "path_mode", path_mode)
        object.__setattr__(self, "completion_mode", completion_mode)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "expected_role_ids", expected)
        object.__setattr__(self, "parent_artifact_digests", parents)
        object.__setattr__(self, "public_state_digest", public_state)
        object.__setattr__(self, "qid", qid)

    @property
    def role_type(self) -> str:
        return PHASE_ROLE_TYPES[self.phase]


@dataclass(frozen=True)
class BHK22DeliveryMetadata:
    sequence: int
    ordered_round: int
    ordered_source: int
    committed_wave: int

    def __post_init__(self) -> None:
        for name in (
            "sequence",
            "ordered_round",
            "ordered_source",
            "committed_wave",
        ):
            object.__setattr__(self, name, strict_int(getattr(self, name), name=name))

    def to_obj(self) -> dict[str, int]:
        return {
            "sequence": self.sequence,
            "ordered_round": self.ordered_round,
            "ordered_source": self.ordered_source,
            "committed_wave": self.committed_wave,
        }


@dataclass(frozen=True)
class BHK22OrderedEntryEvidence:
    role_id: int
    network_sender_id: int
    mid: str
    message_digest: str
    payload_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "role_id", strict_int(self.role_id, name="role id"))
        object.__setattr__(
            self,
            "network_sender_id",
            strict_int(self.network_sender_id, name="network sender id"),
        )
        object.__setattr__(self, "mid", nonempty_text(self.mid, name="mid"))
        object.__setattr__(
            self,
            "message_digest",
            digest_hex(self.message_digest, name="message digest"),
        )
        object.__setattr__(
            self,
            "payload_bytes",
            strict_int(self.payload_bytes, name="payload bytes", minimum=1),
        )

    def to_obj(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "network_sender_id": self.network_sender_id,
            "mid": self.mid,
            "message_digest": self.message_digest,
            "payload_bytes": self.payload_bytes,
        }


@dataclass(frozen=True)
class BHK22OrderedArtifact:
    phase_context: BHK22PhaseContext
    entries: Sequence[BHK22OrderedEntryEvidence]
    delivery: BHK22DeliveryMetadata
    artifact_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.phase_context, BHK22PhaseContext):
            raise TypeError("artifact requires BHK22PhaseContext")
        if not isinstance(self.delivery, BHK22DeliveryMetadata):
            raise TypeError("artifact requires delivery metadata")
        entries = tuple(self.entries)
        if not all(isinstance(item, BHK22OrderedEntryEvidence) for item in entries):
            raise TypeError("artifact entries have the wrong type")
        if tuple(item.role_id for item in entries) != self.phase_context.expected_role_ids:
            raise ValueError("artifact must contain exact expected roles in role order")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(
            self,
            "artifact_digest",
            hashlib.sha256(
                ARTIFACT_DIGEST_DOMAIN + canonical_json(self.digest_material())
            ).hexdigest(),
        )

    @property
    def phase(self) -> str:
        return self.phase_context.phase

    @property
    def kind(self) -> str:
        return PHASE_ARTIFACT_KINDS[self.phase]

    def digest_material(self) -> dict[str, Any]:
        return {
            "kind": PHASE_ARTIFACT_KINDS[self.phase_context.phase],
            "version": ARTIFACT_VERSION,
            "root_context_digest": self.phase_context.context.root_context_digest,
            "phase_qid": self.phase_context.qid,
            "phase": self.phase_context.phase,
            "role_type": self.phase_context.role_type,
            "parent_artifact_digests": list(
                self.phase_context.parent_artifact_digests
            ),
            "public_state_digest": self.phase_context.public_state_digest,
            "entries": [item.to_obj() for item in self.entries],
            "delivery": self.delivery.to_obj(),
        }

    def to_obj(self) -> dict[str, Any]:
        result = self.digest_material()
        result["artifact_digest"] = self.artifact_digest
        return result

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())


class BHK22DealOrderedArtifact(BHK22OrderedArtifact):
    pass


class BHK22VerifyOrderedArtifact(BHK22OrderedArtifact):
    pass


class BHK22ResponseOrderedArtifact(BHK22OrderedArtifact):
    pass


class BHK22FinalizeOrderedArtifact(BHK22OrderedArtifact):
    pass


PHASE_ARTIFACT_TYPES = {
    PHASE_DEAL: BHK22DealOrderedArtifact,
    PHASE_VERIFY: BHK22VerifyOrderedArtifact,
    PHASE_RESPONSE: BHK22ResponseOrderedArtifact,
    PHASE_FINALIZE: BHK22FinalizeOrderedArtifact,
}


def decode_structural_entry(
    phase_context: BHK22PhaseContext,
    entry: PoolEntry,
    *,
    message_types: Mapping[str, type] | None = None,
) -> tuple[PoolEntry, Any]:
    if (
        not isinstance(entry, (tuple, list))
        or len(entry) != 2
        or not isinstance(entry[0], str)
        or not isinstance(entry[1], (bytes, bytearray, memoryview))
    ):
        raise TypeError("pool entry must contain (mid, payload bytes)")
    mid, payload = str(entry[0]), bytes(entry[1])
    codecs = PHASE_MESSAGE_TYPES if message_types is None else message_types
    message = codecs[phase_context.phase].from_bytes(payload)
    context = phase_context.context
    expected_sender = context.committees[phase_context.role_type].role_to_node.get(
        message.role_id
    )
    if (
        message.root_context_digest != context.root_context_digest
        or message.phase_qid != phase_context.qid
        or message.n_r != context.parameters.n_r
        or message.role_id not in phase_context.expected_role_ids
        or message.network_sender_id != expected_sender
        or mid
        != bhk22_role_mid(
            phase_context.role_type,
            message.role_id,
            message.network_sender_id,
        )
    ):
        raise ValueError("BHK22 structural entry binding mismatch")
    return (mid, payload), message


def build_ordered_artifact(
    phase_context: BHK22PhaseContext,
    entries: Sequence[PoolEntry],
    delivery: BHK22DeliveryMetadata,
    *,
    message_types: Mapping[str, type] | None = None,
) -> BHK22OrderedArtifact:
    by_role: dict[int, tuple[PoolEntry, Any]] = {}
    for entry in entries:
        normalized, message = decode_structural_entry(
            phase_context, entry, message_types=message_types
        )
        if message.role_id in by_role:
            raise ValueError("duplicate role in delivered BHK22 phase")
        by_role[message.role_id] = (normalized, message)
    if tuple(sorted(by_role)) != phase_context.expected_role_ids:
        raise ValueError("delivered phase does not contain exact expected roles")
    evidence = tuple(
        BHK22OrderedEntryEvidence(
            role_id=role,
            network_sender_id=by_role[role][1].network_sender_id,
            mid=by_role[role][0][0],
            message_digest=by_role[role][1].message_digest,
            payload_bytes=len(by_role[role][0][1]),
        )
        for role in sorted(by_role)
    )
    return PHASE_ARTIFACT_TYPES[phase_context.phase](
        phase_context=phase_context,
        entries=evidence,
        delivery=delivery,
    )


class BHK22PhaseOrderingJustifier(Justifier):
    """Ordering/readiness predicate; deliberately executes no BHK proof."""

    def __init__(self, message_types: Mapping[str, type] | None = None) -> None:
        codecs = dict(PHASE_MESSAGE_TYPES if message_types is None else message_types)
        if set(codecs) != set(PHASE_MESSAGE_TYPES):
            raise ValueError("BHK22 phase codecs must cover every phase")
        self.message_types = MappingProxyType(codecs)
        self.contexts: dict[str, BHK22PhaseContext] = {}
        self._metrics = {
            stage: {"invocations": 0, "elapsed_sec": 0.0, "cpu_seconds": 0.0}
            for stage in ("select", "committed_verify")
        }

    def register_context(self, context: BHK22PhaseContext) -> None:
        if not isinstance(context, BHK22PhaseContext):
            raise TypeError("expected BHK22PhaseContext")
        existing = self.contexts.get(context.qid)
        if existing is not None and existing != context:
            raise ValueError("conflicting BHK22 phase qid registration")
        self.contexts[context.qid] = context

    def unregister_context(self, context_or_qid: BHK22PhaseContext | str) -> None:
        qid = context_or_qid.qid if isinstance(context_or_qid, BHK22PhaseContext) else str(context_or_qid)
        self.contexts.pop(qid, None)

    def is_registered(self, context_or_qid: BHK22PhaseContext | str) -> bool:
        qid = context_or_qid.qid if isinstance(context_or_qid, BHK22PhaseContext) else str(context_or_qid)
        return qid in self.contexts

    def _registration(self, qid: Qid) -> BHK22PhaseContext:
        if not isinstance(qid, Qid):
            raise ValueError("expected a typed BHK22 phase qid")
        raw = qid.encode()
        context = self.contexts[raw]
        if Qid.decode(context.qid) != qid:
            raise ValueError("BHK22 qid does not match registered phase")
        return context

    def _record(self, stage: str, wall: float, cpu: float) -> None:
        metric = self._metrics[stage]
        metric["invocations"] += 1
        metric["elapsed_sec"] += time.perf_counter() - wall
        metric["cpu_seconds"] += time.process_time() - cpu

    def metrics_snapshot(self) -> dict[str, dict[str, float | int]]:
        return {stage: dict(values) for stage, values in self._metrics.items()}

    def select_justified_subset(
        self, qid: Qid, entries: Sequence[PoolEntry]
    ) -> list[PoolEntry] | None:
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            phase_context = self._registration(qid)
            by_role: dict[int, PoolEntry] = {}
            for entry in entries:
                try:
                    normalized, message = decode_structural_entry(
                        phase_context, entry, message_types=self.message_types
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if message.role_id in by_role:
                    return None
                by_role[message.role_id] = normalized
            if tuple(sorted(by_role)) != phase_context.expected_role_ids:
                return None
            return [by_role[role] for role in phase_context.expected_role_ids]
        except (KeyError, TypeError, ValueError):
            return None
        finally:
            self._record("select", wall, cpu)

    def verify_qid_set(self, qid: Qid, entries: Sequence[PoolEntry]) -> bool:
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            phase_context = self._registration(qid)
            if len(entries) != len(phase_context.expected_role_ids):
                return False
            roles = []
            for entry in entries:
                _normalized, message = decode_structural_entry(
                    phase_context, entry, message_types=self.message_types
                )
                roles.append(message.role_id)
            return tuple(sorted(roles)) == phase_context.expected_role_ids and len(set(roles)) == len(roles)
        except (KeyError, TypeError, ValueError):
            return False
        finally:
            self._record("committed_verify", wall, cpu)


class BHK22PhaseController:
    """Fail-closed dynamic qid registration and artifact lifecycle."""

    def __init__(
        self,
        *,
        context: BHK22PublicContext,
        path_mode: str,
        expected_active_roles: Mapping[str, Sequence[int]],
        justifier: BHK22PhaseOrderingJustifier,
        completion_mode: str = DEFAULT_COMPLETION_MODE,
        qid_builder: Callable[[str], str] | None = None,
    ) -> None:
        if not isinstance(context, BHK22PublicContext):
            raise TypeError("controller requires BHK22PublicContext")
        if not isinstance(justifier, BHK22PhaseOrderingJustifier):
            raise TypeError("controller requires BHK22 ordering justifier")
        completion_mode = str(completion_mode)
        path_phases(path_mode, completion_mode)
        roles = {
            role_type: _canonical_ids(values, name=f"active {role_type} roles")
            for role_type, values in expected_active_roles.items()
        }
        if set(roles) != set(PHASE_ROLE_TYPES.values()):
            raise ValueError("active roles must cover all four role types")
        if any(values != tuple(range(context.parameters.n_r)) for values in roles.values()):
            raise ValueError("first BHK22 profile requires every role active")
        self.context = context
        self.path_mode = str(path_mode)
        self.completion_mode = completion_mode
        self.expected_active_roles = MappingProxyType(roles)
        self.justifier = justifier
        if qid_builder is not None and not callable(qid_builder):
            raise TypeError("qid_builder must be callable or null")
        self.qid_builder = qid_builder
        self.state = "INIT"
        self.current: BHK22PhaseContext | None = None
        self.phase_contexts: dict[str, BHK22PhaseContext] = {}
        self.artifacts: dict[str, BHK22OrderedArtifact] = {}

    def _register(
        self,
        phase: str,
        *,
        parents: Sequence[str],
        public_state_digest: str | None,
    ) -> BHK22PhaseContext:
        if self.current is not None and self.current.phase == phase:
            raise RuntimeError("BHK22 phase already registered")
        sequence = path_phases(self.path_mode, self.completion_mode).index(phase)
        phase_context = BHK22PhaseContext(
            context=self.context,
            path_mode=self.path_mode,
            phase=phase,
            sequence=sequence,
            expected_role_ids=self.expected_active_roles[PHASE_ROLE_TYPES[phase]],
            completion_mode=self.completion_mode,
            parent_artifact_digests=parents,
            public_state_digest=public_state_digest,
            qid_override=(self.qid_builder(phase) if self.qid_builder else None),
        )
        self.justifier.register_context(phase_context)
        self.phase_contexts[phase] = phase_context
        self.current = phase_context
        self.state = f"{phase.upper()}_REGISTERED"
        return phase_context

    def start(self) -> BHK22PhaseContext:
        if self.state != "INIT":
            raise RuntimeError("BHK22 controller was already started")
        return self._register(PHASE_DEAL, parents=(), public_state_digest=None)

    def is_registered(self, context_or_qid: BHK22PhaseContext | str) -> bool:
        qid = (
            context_or_qid.qid
            if isinstance(context_or_qid, BHK22PhaseContext)
            else str(context_or_qid)
        )
        controller_owned = any(
            phase_context.qid == qid
            for phase_context in self.phase_contexts.values()
        )
        return controller_owned and self.justifier.is_registered(qid)

    def on_delivery(
        self,
        qid: str,
        entries: Sequence[PoolEntry],
        delivery: BHK22DeliveryMetadata,
    ) -> BHK22OrderedArtifact:
        current = self.current
        if current is None or qid != current.qid:
            raise ValueError("delivery is not for the currently registered BHK22 phase")
        if self.state != f"{current.phase.upper()}_REGISTERED":
            raise RuntimeError("BHK22 phase is not awaiting delivery")
        artifact = build_ordered_artifact(
            current,
            entries,
            delivery,
            message_types=self.justifier.message_types,
        )
        self.artifacts[current.phase] = artifact
        phases = path_phases(self.path_mode, self.completion_mode)
        self.state = (
            "FINALIZED"
            if current.phase == phases[-1]
            else f"{current.phase.upper()}_DELIVERED"
        )
        return artifact

    def register_next(
        self,
        *,
        public_state_digest: str,
        response_required: bool | None = None,
    ) -> BHK22PhaseContext:
        current = self.current
        if current is None or current.phase not in self.artifacts:
            raise RuntimeError("current BHK22 phase has not been delivered")
        if self.state == "FINALIZED":
            raise RuntimeError("finalize is the last BHK22 phase")
        digest_hex(public_state_digest, name="derived public-state digest")
        if current.phase == PHASE_DEAL:
            if response_required is not None:
                raise ValueError("deal transition does not choose response branch")
            next_phase = PHASE_VERIFY
        elif current.phase == PHASE_VERIFY:
            if not isinstance(response_required, bool):
                raise TypeError("verify transition requires response_required")
            expected_response = self.path_mode == PATH_ONE_ACCUSATION
            if response_required != expected_response:
                raise ValueError("response decision conflicts with configured path")
            next_phase = PHASE_RESPONSE if response_required else PHASE_FINALIZE
        elif current.phase == PHASE_RESPONSE:
            if response_required is not None:
                raise ValueError("response transition does not choose a branch")
            next_phase = PHASE_FINALIZE
        else:
            raise RuntimeError("unsupported BHK22 transition")
        parents = tuple(
            self.artifacts[phase].artifact_digest
            for phase in path_phases(self.path_mode, self.completion_mode)
            if phase in self.artifacts
        )
        return self._register(
            next_phase,
            parents=parents,
            public_state_digest=public_state_digest,
        )

    def retire(self) -> None:
        for phase_context in tuple(self.phase_contexts.values()):
            self.justifier.unregister_context(phase_context)
        self.current = None
        self.state = "RETIRED"

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "path_mode": self.path_mode,
            "completion_mode": self.completion_mode,
            "current_qid": self.current.qid if self.current is not None else None,
            "registered_phases": tuple(self.phase_contexts),
            "artifact_digests": {
                phase: artifact.artifact_digest
                for phase, artifact in self.artifacts.items()
            },
        }


__all__ = [
    "BHK22DealOrderedArtifact",
    "BHK22DeliveryMetadata",
    "BHK22FinalizeOrderedArtifact",
    "BHK22OrderedArtifact",
    "BHK22OrderedEntryEvidence",
    "BHK22PhaseContext",
    "BHK22PhaseController",
    "BHK22PhaseOrderingJustifier",
    "BHK22ResponseOrderedArtifact",
    "BHK22VerifyOrderedArtifact",
    "bhk22_role_mid",
    "build_ordered_artifact",
    "decode_structural_entry",
]
