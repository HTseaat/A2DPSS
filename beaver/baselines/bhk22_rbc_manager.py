"""Step 8 phase-parameterized YOSO-RBC transport for BHK+22."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from beaver.baselines.bhk22_context import (
    bhk22_context_committee_sortition_label,
    canonical_json,
    strict_int,
)
from beaver.baselines.bhk22_contract import PHASES
from beaver.baselines.bhk22_phase import (
    BHK22PhaseContext,
    BHK22PhaseController,
    PHASE_MESSAGE_TYPES,
    bhk22_role_mid,
    decode_structural_entry,
)
from beaver.dag_tob import DagTOB
from beaver.yosorbc import YosoRBC
from beaver.vrf import (
    generator_hex_from_srs,
    vrf_backend_available,
    vrf_prove_sortition,
    vrf_verify_sortition,
)


BHK22_PHASE_RBC_VERSION = 2
BHK22_PHASE_RBC_KIND = "atcaas-bhk22-phase-role-rbc"


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate BHK22 RBC payload field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _qid_tag(phase_context: BHK22PhaseContext) -> str:
    return hashlib.sha256(phase_context.qid.encode("utf-8")).hexdigest()[:16]


def bhk22_phase_rbc_prefix(
    epoch: str,
    phase_context: BHK22PhaseContext,
    role_id: int,
    network_sender_id: int,
) -> str:
    return (
        f"BHK22:{epoch}:PHASE_RBC:{phase_context.context.root_context_digest[:16]}:"
        f"{_qid_tag(phase_context)}:{phase_context.phase}:{phase_context.role_type}:"
        f"{int(role_id)}:node_{int(network_sender_id)}"
    )


def bhk22_phase_echo_label(
    epoch: str,
    phase_context: BHK22PhaseContext,
    role_id: int,
    network_sender_id: int,
) -> bytes:
    return (
        f"ATCAAS:RBC:{epoch}:BHK22:{phase_context.context.root_context_digest[:16]}:"
        f"{_qid_tag(phase_context)}:{phase_context.phase}:{phase_context.role_type}:"
        f"{int(role_id)}:node_{int(network_sender_id)}:ECHO"
    ).encode("utf-8")


def bhk22_phase_ready_label(
    epoch: str,
    phase_context: BHK22PhaseContext,
    role_id: int,
    network_sender_id: int,
) -> bytes:
    return (
        f"ATCAAS:RBC:{epoch}:BHK22:{phase_context.context.root_context_digest[:16]}:"
        f"{_qid_tag(phase_context)}:{phase_context.phase}:{phase_context.role_type}:"
        f"{int(role_id)}:node_{int(network_sender_id)}:READY"
    ).encode("utf-8")


def _selected_component_bytes(value: Any, predicate) -> int:
    total = 0
    if isinstance(value, Mapping):
        for key, item in value.items():
            if predicate(str(key)):
                total += len(canonical_json(item))
            else:
                total += _selected_component_bytes(item, predicate)
    elif isinstance(value, (list, tuple)):
        total += sum(_selected_component_bytes(item, predicate) for item in value)
    return total


@dataclass(frozen=True)
class BHK22PhasePayloadSize:
    application_bytes: int
    ciphertext_bytes: int
    proof_bytes: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                strict_int(getattr(self, name), name=name),
            )

    def to_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


def classify_bhk22_phase_payload(
    phase_context: BHK22PhaseContext,
    batch_bytes: bytes,
    *,
    message_types: Mapping[str, type] | None = None,
) -> BHK22PhasePayloadSize:
    raw = bytes(batch_bytes)
    expected_sender = phase_context.context.committees[
        phase_context.role_type
    ].role_to_node
    # Decode through the same structural boundary, but synthesize the expected
    # mid from the wire header. This does not execute any proof verification.
    codecs = PHASE_MESSAGE_TYPES if message_types is None else message_types
    message_type = codecs[phase_context.phase]
    message = message_type.from_bytes(raw)
    mid = bhk22_role_mid(
        phase_context.role_type,
        message.role_id,
        expected_sender[message.role_id],
    )
    _entry, message = decode_structural_entry(
        phase_context, (mid, raw), message_types=codecs
    )
    obj = message.to_obj()
    ciphertext_bytes = _selected_component_bytes(
        obj,
        lambda key: "digest" not in key
        and (
            key.startswith("encrypted_")
            or key.endswith("ciphertext")
            or key.endswith("ciphertexts")
        ),
    )
    proof_bytes = _selected_component_bytes(
        obj, lambda key: key.endswith("proof") or key.endswith("proofs")
    )
    return BHK22PhasePayloadSize(
        application_bytes=len(raw),
        ciphertext_bytes=ciphertext_bytes,
        proof_bytes=proof_bytes,
    )


@dataclass
class BHK22PhaseRBCMetrics:
    outbound_messages: int = 0
    delivered_messages: int = 0
    rejected_payloads: int = 0
    duplicate_outputs: int = 0
    equivocations: int = 0
    outbound_rbc_envelope_bytes: int = 0
    delivered_rbc_envelope_bytes: int = 0
    outbound_application_bytes: int = 0
    delivered_application_bytes: int = 0
    outbound_ciphertext_bytes: int = 0
    delivered_ciphertext_bytes: int = 0
    outbound_proof_bytes: int = 0
    delivered_proof_bytes: int = 0
    outbound_role_vrf_proof_bytes: int = 0
    delivered_role_vrf_proof_bytes: int = 0
    role_vrf_proof_generation_seconds: float = 0.0
    role_vrf_proof_generation_cpu_seconds: float = 0.0
    role_vrf_proof_verification_seconds: float = 0.0
    role_vrf_proof_verification_cpu_seconds: float = 0.0
    rejected_role_vrf_proofs: int = 0

    def record_outbound(
        self, envelope_bytes: int, sizes: BHK22PhasePayloadSize
    ) -> None:
        self.outbound_messages += 1
        self.outbound_rbc_envelope_bytes += int(envelope_bytes)
        self.outbound_application_bytes += sizes.application_bytes
        self.outbound_ciphertext_bytes += sizes.ciphertext_bytes
        self.outbound_proof_bytes += sizes.proof_bytes

    def record_delivered(
        self, envelope_bytes: int, sizes: BHK22PhasePayloadSize
    ) -> None:
        self.delivered_messages += 1
        self.delivered_rbc_envelope_bytes += int(envelope_bytes)
        self.delivered_application_bytes += sizes.application_bytes
        self.delivered_ciphertext_bytes += sizes.ciphertext_bytes
        self.delivered_proof_bytes += sizes.proof_bytes

    def snapshot(self) -> dict[str, int | float]:
        return {
            name: (
                float(getattr(self, name))
                if name.endswith("_seconds")
                else int(getattr(self, name))
            )
            for name in self.__dataclass_fields__
        }


@dataclass
class _BHK22PhaseRBCInstance:
    phase_context: BHK22PhaseContext
    role_id: int
    network_sender_id: int
    rbc: YosoRBC
    reactor_task: asyncio.Task
    output_task: asyncio.Task
    sender_task: Optional[asyncio.Task] = None
    outbound_digest: Optional[str] = None
    delivered: bool = False
    stopped: bool = False


class BHK22PhaseRBCManager:
    """Run one retained YOSO-RBC instance per active BHK role and phase."""

    def __init__(
        self,
        *,
        dag: DagTOB,
        controller: BHK22PhaseController,
        epoch: str,
        my_id: int,
        M: int,
        t: int,
        threshold_fraction: float,
        expected_committee_size: int,
        committee_m: int,
        quorum_h: int,
        send,
        shared_subscribe,
        crypto,
        vrf_sk,
        vrf_pks,
        payload_limits: Mapping[str, int],
        sender_delay: float = 0.1,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not isinstance(controller, BHK22PhaseController):
            raise TypeError("RBC manager requires BHK22PhaseController")
        limits = {
            str(phase): strict_int(limit, name=f"{phase} payload limit", minimum=1)
            for phase, limit in payload_limits.items()
        }
        required_phases = set(PHASES)
        if set(limits) != required_phases:
            raise ValueError("payload limits must cover all BHK22 phases")
        if isinstance(sender_delay, bool) or not isinstance(sender_delay, (int, float)) or sender_delay < 0:
            raise ValueError("sender_delay must be non-negative")
        self.dag = dag
        self.controller = controller
        self.message_types = controller.justifier.message_types
        self.epoch = str(epoch)
        self.my_id = strict_int(my_id, name="my_id")
        self.M = strict_int(M, name="M", minimum=1)
        self.t = strict_int(t, name="t")
        self.threshold_fraction = float(threshold_fraction)
        self.expected_committee_size = strict_int(
            expected_committee_size, name="expected_committee_size", minimum=1
        )
        self.committee_m = strict_int(committee_m, name="committee_m", minimum=1)
        self.quorum_h = strict_int(quorum_h, name="quorum_h", minimum=1)
        self.send = send
        self.shared_subscribe = shared_subscribe
        self.crypto = crypto
        self.vrf_sk = vrf_sk
        self.vrf_pks = vrf_pks
        self.payload_limits = limits
        self.sender_delay = float(sender_delay)
        self.log = logger or logging.getLogger(f"bhk22_rbc_manager.{self.my_id}")
        self.instances: dict[tuple[str, int], _BHK22PhaseRBCInstance] = {}
        self.delivered_payload_digests: dict[tuple[str, int], str] = {}
        self.metrics: dict[str, BHK22PhaseRBCMetrics] = {
            phase: BHK22PhaseRBCMetrics() for phase in required_phases
        }
        self._stopping = False
        self._prevalidated_envelope_digests: set[
            tuple[str, int, str]
        ] = set()

    def _role_vrf_label(self, phase_context: BHK22PhaseContext) -> bytes:
        return bhk22_context_committee_sortition_label(
            phase_context.context, phase_context.role_type
        )

    def _make_role_vrf_proof(
        self, phase_context: BHK22PhaseContext, sender: int
    ) -> Mapping[str, Any] | None:
        committee = phase_context.context.committees[phase_context.role_type]
        if not committee.requires_vrf_proof:
            return None
        if sender != self.my_id:
            raise ValueError("only the selected BHK22 role owner may prove eligibility")
        if not vrf_backend_available():
            raise RuntimeError("BHK22 role sortition requires the VRF backend")
        _pbk, _pvk, _pks, _sk, srs = self.crypto
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        selected, proof = vrf_prove_sortition(
            label=self._role_vrf_label(phase_context),
            vrf_sk=self.vrf_sk,
            g_hex=generator_hex_from_srs(srs),
            threshold_fraction=committee.n_r / committee.M,
        )
        metric = self.metrics[phase_context.phase]
        metric.role_vrf_proof_generation_seconds += (
            time.perf_counter() - wall_started
        )
        metric.role_vrf_proof_generation_cpu_seconds += (
            time.process_time() - cpu_started
        )
        if not selected or not isinstance(proof, Mapping) or not proof:
            raise RuntimeError("assigned BHK22 role did not satisfy its VRF sortition")
        return dict(proof)

    def _verify_role_vrf_proof(
        self,
        phase_context: BHK22PhaseContext,
        sender: int,
        proof: Any,
    ) -> bool:
        committee = phase_context.context.committees[phase_context.role_type]
        if not committee.requires_vrf_proof:
            return proof is None
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        valid = False
        try:
            if not vrf_backend_available() or not isinstance(proof, Mapping) or not proof:
                return False
            _pbk, _pvk, _pks, _sk, srs = self.crypto
            valid = vrf_verify_sortition(
                label=self._role_vrf_label(phase_context),
                proof=dict(proof),
                sender_id=sender,
                vrf_pks=self.vrf_pks,
                g_hex=generator_hex_from_srs(srs),
                threshold_fraction=committee.n_r / committee.M,
            )
            return valid
        finally:
            metric = self.metrics[phase_context.phase]
            metric.role_vrf_proof_verification_seconds += (
                time.perf_counter() - wall_started
            )
            metric.role_vrf_proof_verification_cpu_seconds += (
                time.process_time() - cpu_started
            )
            if not valid:
                metric.rejected_role_vrf_proofs += 1

    @staticmethod
    def _key(phase_context: BHK22PhaseContext, role_id: int) -> tuple[str, int]:
        return phase_context.qid, int(role_id)

    def _validate_phase(self, phase_context: BHK22PhaseContext) -> None:
        if not isinstance(phase_context, BHK22PhaseContext):
            raise TypeError("expected BHK22PhaseContext")
        if phase_context.context.root_context_digest != self.controller.context.root_context_digest:
            raise ValueError("RBC phase belongs to another BHK22 root context")
        if not self.controller.is_registered(phase_context):
            raise ValueError("BHK22 phase qid is not registered")

    def ensure_instance(self, phase_context: BHK22PhaseContext, role_id: int) -> None:
        if self._stopping:
            raise RuntimeError("BHK22 RBC manager is stopping")
        self._validate_phase(phase_context)
        role_id = strict_int(role_id, name="role id")
        if role_id not in phase_context.expected_role_ids:
            raise ValueError("role is not active in this BHK22 phase")
        key = self._key(phase_context, role_id)
        if key in self.instances:
            return
        sender = phase_context.context.committees[
            phase_context.role_type
        ].role_to_node[role_id]
        pbk, pvk, pks, sk, srs = self.crypto
        rbc = YosoRBC(
            pks,
            sk,
            pbk,
            pvk,
            self.expected_committee_size,
            self.t,
            srs,
            self.my_id,
            self.send,
            None,
            prefix=bhk22_phase_rbc_prefix(
                self.epoch, phase_context, role_id, sender
            ),
            shared_subscribe=self.shared_subscribe,
            threshold_fraction=self.threshold_fraction,
            network_size=self.M,
            vrf_sk=self.vrf_sk,
            vrf_pks=self.vrf_pks,
            expected_committee_size=self.expected_committee_size,
            committee_m=self.committee_m,
            quorum_h=self.quorum_h,
            ready_amplify_threshold=(self.quorum_h + 1) // 2 + 1,
            vrf_echo_label=bhk22_phase_echo_label(
                self.epoch, phase_context, role_id, sender
            ),
            vrf_ready_label=bhk22_phase_ready_label(
                self.epoch, phase_context, role_id, sender
            ),
            designated_sender_id=sender,
            payload_validator=lambda payload, pc=phase_context, rid=role_id: (
                self._prevalidate_rbc_payload(pc, rid, payload)
            ),
        )
        reactor_task = asyncio.create_task(rbc.reactor())
        output_task = asyncio.create_task(
            self._watch_output(phase_context, role_id, rbc)
        )
        self.instances[key] = _BHK22PhaseRBCInstance(
            phase_context=phase_context,
            role_id=role_id,
            network_sender_id=sender,
            rbc=rbc,
            reactor_task=reactor_task,
            output_task=output_task,
        )

    def ensure_phase_instances(self, phase_context: BHK22PhaseContext) -> None:
        self._validate_phase(phase_context)
        for role_id in phase_context.expected_role_ids:
            self.ensure_instance(phase_context, role_id)

    async def broadcast_role_batch(
        self,
        phase_context: BHK22PhaseContext,
        *,
        role_id: int,
        batch_bytes: bytes,
    ) -> None:
        self._validate_phase(phase_context)
        role_id = strict_int(role_id, name="role id")
        raw = bytes(batch_bytes)
        if len(raw) > self.payload_limits[phase_context.phase]:
            raise ValueError("BHK22 phase payload exceeds configured limit")
        sender = phase_context.context.committees[
            phase_context.role_type
        ].role_to_node.get(role_id)
        if sender != self.my_id:
            raise ValueError("only the node assigned to this BHK22 role may send")
        mid = bhk22_role_mid(phase_context.role_type, role_id, sender)
        _entry, message = decode_structural_entry(
            phase_context, (mid, raw), message_types=self.message_types
        )
        if message.role_id != role_id:
            raise ValueError("batch role does not match RBC role")
        self.ensure_instance(phase_context, role_id)
        instance = self.instances[self._key(phase_context, role_id)]
        digest = hashlib.sha256(raw).hexdigest()
        if instance.sender_task is not None:
            if instance.outbound_digest != digest:
                self.metrics[phase_context.phase].equivocations += 1
                raise ValueError("BHK22 role attempted conflicting broadcasts")
            return
        role_vrf_proof = self._make_role_vrf_proof(phase_context, sender)
        envelope = self._encode_payload(
            self.epoch,
            phase_context,
            role_id,
            sender,
            raw,
            role_vrf_proof=role_vrf_proof,
        )
        sizes = classify_bhk22_phase_payload(
            phase_context, raw, message_types=self.message_types
        )
        instance.outbound_digest = digest
        self.metrics[phase_context.phase].record_outbound(len(envelope), sizes)
        if role_vrf_proof is not None:
            self.metrics[
                phase_context.phase
            ].outbound_role_vrf_proof_bytes += len(canonical_json(role_vrf_proof))

        async def _send() -> None:
            if self.sender_delay:
                await asyncio.sleep(self.sender_delay)
            await instance.rbc.sender(envelope)

        instance.sender_task = asyncio.create_task(_send())

    @staticmethod
    def _encode_payload(
        epoch: str,
        phase_context: BHK22PhaseContext,
        role_id: int,
        network_sender_id: int,
        batch_bytes: bytes,
        *,
        role_vrf_proof: Mapping[str, Any] | None = None,
    ) -> bytes:
        raw = bytes(batch_bytes)
        digest = hashlib.sha256(raw).hexdigest()
        mid = bhk22_role_mid(
            phase_context.role_type, role_id, network_sender_id
        )
        return canonical_json(
            {
                "kind": BHK22_PHASE_RBC_KIND,
                "version": BHK22_PHASE_RBC_VERSION,
                "epoch": str(epoch),
                "root_context_digest": phase_context.context.root_context_digest,
                "phase_qid": phase_context.qid,
                "phase": phase_context.phase,
                "role_type": phase_context.role_type,
                "role_id": int(role_id),
                "network_sender_id": int(network_sender_id),
                "mid": mid,
                "batch_digest": digest,
                "batch_hex": raw.hex(),
                "role_vrf_proof": (
                    dict(role_vrf_proof) if role_vrf_proof is not None else None
                ),
            }
        )

    def _decode_payload(
        self,
        payload: bytes,
        *,
        phase_context: BHK22PhaseContext,
        role_id: int,
        verify_role_vrf: bool = True,
    ) -> tuple[str, bytes, Any]:
        try:
            data = json.loads(
                bytes(payload).decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("malformed BHK22 RBC payload") from exc
        expected_fields = {
            "kind",
            "version",
            "epoch",
            "root_context_digest",
            "phase_qid",
            "phase",
            "role_type",
            "role_id",
            "network_sender_id",
            "mid",
            "batch_digest",
            "batch_hex",
            "role_vrf_proof",
        }
        if not isinstance(data, dict) or set(data) != expected_fields:
            raise ValueError(
                f"BHK22 RBC payload fields do not match version "
                f"{BHK22_PHASE_RBC_VERSION}"
            )
        if bytes(payload) != canonical_json(data):
            raise ValueError("BHK22 RBC payload is not canonical JSON")
        sender = phase_context.context.committees[
            phase_context.role_type
        ].role_to_node[role_id]
        mid = bhk22_role_mid(phase_context.role_type, role_id, sender)
        if (
            data["kind"] != BHK22_PHASE_RBC_KIND
            or isinstance(data["version"], bool)
            or data["version"] != BHK22_PHASE_RBC_VERSION
            or data["epoch"] != self.epoch
            or data["root_context_digest"]
            != phase_context.context.root_context_digest
            or data["phase_qid"] != phase_context.qid
            or data["phase"] != phase_context.phase
            or data["role_type"] != phase_context.role_type
            or isinstance(data["role_id"], bool)
            or data["role_id"] != role_id
            or isinstance(data["network_sender_id"], bool)
            or data["network_sender_id"] != sender
            or data["mid"] != mid
        ):
            raise ValueError("BHK22 RBC envelope identity mismatch")
        encoded = data["batch_hex"]
        if not isinstance(encoded, str):
            raise ValueError("BHK22 RBC batch_hex must be a string")
        try:
            raw = bytes.fromhex(encoded)
        except ValueError as exc:
            raise ValueError("BHK22 RBC batch_hex is malformed") from exc
        if encoded != raw.hex():
            raise ValueError("BHK22 RBC batch_hex must be canonical lowercase hex")
        if len(raw) > self.payload_limits[phase_context.phase]:
            raise ValueError("BHK22 RBC payload exceeds configured limit")
        if data["batch_digest"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("BHK22 RBC batch digest mismatch")
        if verify_role_vrf and not self._verify_role_vrf_proof(
            phase_context, sender, data["role_vrf_proof"]
        ):
            raise ValueError("BHK22 RBC role VRF proof is invalid")
        _entry, message = decode_structural_entry(
            phase_context, (mid, raw), message_types=self.message_types
        )
        return mid, raw, message

    def _prevalidate_rbc_payload(
        self,
        phase_context: BHK22PhaseContext,
        role_id: int,
        payload: bytes,
    ) -> bool:
        try:
            self._decode_payload(
                payload,
                phase_context=phase_context,
                role_id=role_id,
                verify_role_vrf=True,
            )
        except Exception as exc:
            self.metrics[phase_context.phase].rejected_payloads += 1
            self.log.warning(
                "[BHK22:RBC] pre-admission rejected phase=%s role=%d: %s",
                phase_context.phase,
                role_id,
                exc,
            )
            return False
        digest = hashlib.sha256(bytes(payload)).hexdigest()
        self._prevalidated_envelope_digests.add(
            (phase_context.qid, int(role_id), digest)
        )
        return True

    def admit_output(
        self,
        phase_context: BHK22PhaseContext,
        role_id: int,
        envelope: bytes,
    ) -> bool:
        self._validate_phase(phase_context)
        key = self._key(phase_context, role_id)
        envelope_digest = hashlib.sha256(bytes(envelope)).hexdigest()
        prevalidated = (
            phase_context.qid,
            int(role_id),
            envelope_digest,
        ) in self._prevalidated_envelope_digests
        mid, raw, _message = self._decode_payload(
            envelope,
            phase_context=phase_context,
            role_id=role_id,
            verify_role_vrf=not prevalidated,
        )
        digest = hashlib.sha256(raw).hexdigest()
        previous = self.delivered_payload_digests.get(key)
        if previous is not None:
            if previous == digest:
                self.metrics[phase_context.phase].duplicate_outputs += 1
                return False
            self.metrics[phase_context.phase].equivocations += 1
            raise ValueError("BHK22 RBC output equivocation")
        sizes = classify_bhk22_phase_payload(
            phase_context, raw, message_types=self.message_types
        )
        self.dag.deliver_message(phase_context.qid, mid, raw)
        self.delivered_payload_digests[key] = digest
        self.metrics[phase_context.phase].record_delivered(len(envelope), sizes)
        role_vrf_proof = json.loads(bytes(envelope).decode("utf-8"))[
            "role_vrf_proof"
        ]
        if role_vrf_proof is not None:
            self.metrics[
                phase_context.phase
            ].delivered_role_vrf_proof_bytes += len(
                canonical_json(role_vrf_proof)
            )
        return True

    async def _watch_output(
        self,
        phase_context: BHK22PhaseContext,
        role_id: int,
        rbc: YosoRBC,
    ) -> None:
        key = self._key(phase_context, role_id)
        try:
            envelope, _ = await rbc.output_queue.get()
            self.admit_output(phase_context, role_id, envelope)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.metrics[phase_context.phase].rejected_payloads += 1
            self.log.warning(
                "[BHK22:RBC] rejected phase=%s role=%d payload: %s",
                phase_context.phase,
                role_id,
                exc,
            )
        finally:
            instance = self.instances.get(key)
            if instance is not None:
                instance.delivered = key in self.delivered_payload_digests

    async def _stop_instance(self, key: tuple[str, int]) -> None:
        instance = self.instances.pop(key, None)
        if instance is None or instance.stopped:
            return
        instance.stopped = True
        tasks = [
            task
            for task in (
                instance.sender_task,
                instance.output_task,
                instance.reactor_task,
            )
            if task is not None
        ]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self, timeout: Optional[float] = None) -> bool:
        if self._stopping:
            return self.active_count() == 0
        self._stopping = True
        stopping = asyncio.gather(
            *(self._stop_instance(key) for key in tuple(self.instances)),
            return_exceptions=True,
        )
        if timeout is None:
            await stopping
            return True
        try:
            await asyncio.wait_for(stopping, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def active_count(self) -> int:
        return len(self.instances)

    def active_count_snapshot(self) -> dict[str, Any]:
        by_phase = {phase: 0 for phase in self.metrics}
        for instance in self.instances.values():
            by_phase[instance.phase_context.phase] += 1
        return {"total": self.active_count(), "by_phase": by_phase}

    def metrics_snapshot(self) -> dict[str, dict[str, int | float]]:
        return {
            phase: self.metrics[phase].snapshot() for phase in sorted(self.metrics)
        }


__all__ = [
    "BHK22_PHASE_RBC_KIND",
    "BHK22_PHASE_RBC_VERSION",
    "BHK22PhasePayloadSize",
    "BHK22PhaseRBCManager",
    "BHK22PhaseRBCMetrics",
    "bhk22_phase_echo_label",
    "bhk22_phase_rbc_prefix",
    "bhk22_phase_ready_label",
    "classify_bhk22_phase_payload",
]
