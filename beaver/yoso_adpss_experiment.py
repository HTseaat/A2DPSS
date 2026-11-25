"""Common lifecycle and metrics envelope for YOSO ADPSS DAG arms."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from beaver.a2dpss import (
    A2DPSSContext, A2DPSSHolderState, A2DPSSJustifierManager,
    A2DPSSPublicFinalizer, A2DPSSSuccessorReconstructionManager,
    run_a2dpss_handover_role,
)
from beaver.a2dpss_rbc_manager import classify_a2dpss_contribution
from beaver.baselines.bhk22_contract import BHK22_QID_KIND
from beaver.baselines.bhk22_dag_runner import (
    BHK22_DAG_NODE_METRICS_SCHEMA,
    BHK22OptimisticDAGRuntime,
)
from beaver.baselines.ru22_dag_runner import RU22DAGHandoverRuntime
from beaver.baselines.ru22_rbc_manager import classify_ru22_batch


NODE_METRICS_SCHEMA = "yoso-adpss-dag-node-metrics/v1"
TRIAL_METRICS_SCHEMA = "yoso-adpss-dag-trial-summary/v1"


@dataclass
class YOSOADPSSTimeline:
    _origin: float | None = None
    values: dict[str, float | None] = field(default_factory=lambda: {
        "t0": None, "t_first_role_message_sent": None,
        "t_first_pool_ready": None, "t_public_finalized": None,
        "t_successor_done": None, "frozen_at": None,
    })
    frozen: bool = False

    def start(self) -> None:
        if self._origin is not None:
            raise RuntimeError("timeline already started")
        self._origin = time.perf_counter()
        self.values["t0"] = 0.0

    def mark(self, name: str) -> None:
        if self.frozen:
            raise RuntimeError("metrics timeline is frozen")
        if self._origin is None:
            raise RuntimeError("timeline has not started")
        value = time.perf_counter() - self._origin
        existing = self.values.get(name)
        if existing is None:
            self.values[name] = value

    def freeze(self) -> None:
        if not self.frozen:
            self.mark("frozen_at")
            self.frozen = True

    def snapshot(self) -> dict[str, float | None]:
        result = dict(self.values)
        observed = [value for value in result.values() if value is not None]
        if observed != sorted(observed):
            raise ValueError("relative handover timeline is not monotonic")
        return result


@dataclass
class CPUMetric:
    cpu_seconds: float = 0.0
    invocations: int = 0


@dataclass
class YOSOADPSSCommunicationMetrics:
    application_payload_bytes: int = 0
    cryptographic_proof_bytes: int | None = None
    recipient_ciphertext_bytes: int | None = None
    committee_vrf_proof_bytes: int | None = None
    yoso_rbc_envelope_bytes: int | None = None
    dag_protocol_bytes: int | None = None
    common_coin_bytes: int | None = None
    total_network_bytes: int | None = None
    committed_application_block_count: int = 0
    dag_vertex_count: int | None = None
    dag_committed_wave_count: int | None = None

    def record_transport(self, family: str, byte_count: int) -> None:
        fields = {
            "yoso-rbc": "yoso_rbc_envelope_bytes",
            "dag": "dag_protocol_bytes",
            "common-coin": "common_coin_bytes",
            "total": "total_network_bytes",
        }
        try:
            field_name = fields[family]
        except KeyError as exc:
            raise ValueError("unknown transport message family") from exc
        if isinstance(byte_count, bool) or int(byte_count) < 0:
            raise ValueError("transport byte count must be non-negative")
        current = getattr(self, field_name)
        setattr(self, field_name, (current or 0) + int(byte_count))


class YOSOADPSSHandoverAdapter:
    protocol = ""

    def __init__(self, *, my_node_id: int, service_grace_seconds: float = 0.0):
        self.my_node_id = int(my_node_id)
        self.service_grace_seconds = float(service_grace_seconds)
        self.timeline = YOSOADPSSTimeline()
        self.cpu = {
            name: CPUMetric() for name in (
                "role_message_generation", "candidate_selection_predicate",
                "committed_order_verify", "public_finalization",
                "successor_reconstruction",
            )
        }
        self.communication = YOSOADPSSCommunicationMetrics()
        self.dag_runtime = None
        self.rbc_runtime = None
        self.failure: dict[str, str] | None = None
        self.retired = False

    def _measure(self, category: str, callback, *args, **kwargs):
        started = time.process_time()
        try:
            return callback(*args, **kwargs)
        finally:
            metric = self.cpu[category]
            metric.cpu_seconds += time.process_time() - started
            metric.invocations += 1

    def _record_failure(self, stage: str, exc: BaseException) -> None:
        self.failure = {
            "stage": stage, "type": type(exc).__name__, "message": str(exc)
        }

    def register_initial_contexts(self, dag_runtime, rbc_runtime):
        self.dag_runtime, self.rbc_runtime = dag_runtime, rbc_runtime

    def mark_first_pool_ready(self) -> None:
        self.timeline.mark("t_first_pool_ready")

    def expected_application_deliveries(self) -> int:
        raise NotImplementedError

    def current_qids(self) -> tuple[str, ...]:
        raise NotImplementedError

    def public_finalized(self) -> bool:
        raise NotImplementedError

    def local_successor_done(self) -> bool:
        raise NotImplementedError

    def _protocol_metrics(self) -> Mapping[str, Any]:
        return {}

    def collect_metrics(self) -> dict[str, Any]:
        return {
            "schema": NODE_METRICS_SCHEMA, "protocol": self.protocol,
            "node_id": self.my_node_id,
            "status": "failure" if self.failure else (
                "success" if self.public_finalized() and self.local_successor_done()
                else "incomplete"
            ),
            "failure": self.failure, "timeline": self.timeline.snapshot(),
            "cpu": {name: asdict(value) for name, value in self.cpu.items()},
            "communication": asdict(self.communication),
            "protocol_metrics": dict(self._protocol_metrics()),
        }

    async def retire_and_cleanup(self):
        self.timeline.freeze()
        if self.service_grace_seconds:
            await asyncio.sleep(self.service_grace_seconds)
        if self.rbc_runtime is not None and hasattr(self.rbc_runtime, "stop"):
            result = self.rbc_runtime.stop()
            if hasattr(result, "__await__"):
                await result
        self.retired = True


class A2DPSSAdapter(YOSOADPSSHandoverAdapter):
    protocol = "a2dpss"

    def __init__(self, *, context: A2DPSSContext, my_node_id: int,
                 holder_share: tuple[int, int] | None,
                 successor_rsk: int | None, service_grace_seconds: float = 0.0):
        super().__init__(my_node_id=my_node_id, service_grace_seconds=service_grace_seconds)
        self.context = context
        self.holder_share = holder_share
        self.holder_state = A2DPSSHolderState(*holder_share) if holder_share else None
        self.justifier = A2DPSSJustifierManager()
        self.finalizer = A2DPSSPublicFinalizer(self.justifier)
        self.reconstruction = A2DPSSSuccessorReconstructionManager()
        self.successor_rsk = successor_rsk
        self.public_result = None
        self.successor_result = None
        self.holder_result = None
        self.communication.cryptographic_proof_bytes = 0
        self.communication.recipient_ciphertext_bytes = 0

    def register_initial_contexts(self, dag_runtime, rbc_runtime):
        super().register_initial_contexts(dag_runtime, rbc_runtime)
        self.justifier.register_context(self.context)
        self.reconstruction.register_context(
            self.context, my_node_id=self.my_node_id, my_rsk=self.successor_rsk
        )
        if hasattr(rbc_runtime, "ensure_context"):
            rbc_runtime.ensure_context(self.context, self.context.holder_eval_points)

    async def start_at_t0(self):
        self.timeline.start()
        if self.holder_state is None:
            return None
        try:
            async def broadcast(_qid, _mid, payload):
                await self.rbc_runtime.broadcast_contribution(
                    self.context, holder_id=self.my_node_id, contribution_bytes=payload
                )
                self.timeline.mark("t_first_role_message_sent")
                self.communication.application_payload_bytes += len(payload)
                sizes = classify_a2dpss_contribution(payload)
                self.communication.cryptographic_proof_bytes += sizes.proof_bytes
                self.communication.recipient_ciphertext_bytes += sizes.ciphertext_bytes
            started = time.process_time()
            self.holder_result = await run_a2dpss_handover_role(
                context=self.context, holder_id=self.my_node_id,
                share_s=self.holder_share[0], share_s_hat=self.holder_share[1],
                broadcast_contribution=broadcast, holder_state=self.holder_state,
            )
            metric = self.cpu["role_message_generation"]
            metric.cpu_seconds += time.process_time() - started
            metric.invocations += 1
            return self.holder_result
        except Exception as exc:
            self._record_failure("start", exc); raise

    async def on_delivery(self, event):
        try:
            self.public_result = self._measure(
                "public_finalization", self.finalizer.on_verified_deliver,
                event.qid, event.entries,
            )
            if self.public_result is None:
                raise RuntimeError("A2DPSS public finalization rejected delivery")
            self.timeline.mark("t_public_finalized")
            self.successor_result = self._measure(
                "successor_reconstruction", self.reconstruction.reconstruct,
                self.public_result, event.entries,
            )
            if self.successor_rsk is not None:
                if self.successor_result is None:
                    raise RuntimeError("local A2DPSS successor did not reconstruct")
                self.timeline.mark("t_successor_done")
            self.communication.committed_application_block_count = 1
        except Exception as exc:
            self._record_failure("delivery", exc); raise

    def current_qids(self): return (self.context.encode_qid(),)
    def expected_application_deliveries(self): return 1
    def public_finalized(self): return self.public_result is not None
    def local_successor_done(self): return self.successor_rsk is None or self.successor_result is not None
    def _protocol_metrics(self):
        return {"justifier": self.justifier.metrics_snapshot(),
                "public_finalization": self.finalizer.metrics_snapshot(),
                "successor_reconstruction": self.reconstruction.metrics_snapshot()}


class RU22Adapter(YOSOADPSSHandoverAdapter):
    protocol = "ru22"
    def __init__(self, *, runtime: RU22DAGHandoverRuntime, service_grace_seconds: float = 0.0):
        super().__init__(my_node_id=runtime.my_node_id, service_grace_seconds=service_grace_seconds)
        self.runtime = runtime
        self.communication.cryptographic_proof_bytes = 0
        self.communication.recipient_ciphertext_bytes = 0
    def register_initial_contexts(self, dag_runtime, rbc_runtime):
        super().register_initial_contexts(dag_runtime, rbc_runtime)
        if hasattr(rbc_runtime, "ensure_context"):
            rbc_runtime.ensure_context(self.runtime.context, self.runtime.source_role_ids)
    async def start_at_t0(self):
        self.timeline.start(); self.runtime.start()
        started = time.process_time()
        try:
            result = await self.runtime.broadcast_local(self.rbc_runtime)
            if result:
                self.timeline.mark("t_first_role_message_sent")
                self.communication.application_payload_bytes += result.metrics.batch_bytes
                sizes = classify_ru22_batch(result.batch_bytes)
                self.communication.cryptographic_proof_bytes += sizes.proof_bytes
                self.communication.recipient_ciphertext_bytes += sizes.ciphertext_bytes
            return result
        except Exception as exc:
            self._record_failure("start", exc); raise
        finally:
            if self.runtime.is_source:
                m=self.cpu["role_message_generation"]; m.cpu_seconds += time.process_time()-started; m.invocations += 1
    async def on_delivery(self, event):
        try:
            public, private = self.runtime.handle_delivery(event)
            self.timeline.mark("t_public_finalized")
            if self.runtime.is_successor:
                if private is None: raise RuntimeError("local RU22 successor did not reconstruct")
                self.timeline.mark("t_successor_done")
            self.communication.committed_application_block_count = 1
            return public, private
        except Exception as exc:
            self._record_failure("delivery", exc); raise
    def current_qids(self): return (self.runtime.qid,)
    def expected_application_deliveries(self): return 1
    def public_finalized(self): return self.runtime.public_result is not None
    def local_successor_done(self): return not self.runtime.is_successor or self.runtime.successor_result is not None
    def _protocol_metrics(self): return self.runtime.summary()


class BHK22Adapter(YOSOADPSSHandoverAdapter):
    protocol = "bhk22"
    def __init__(self, *, runtime: BHK22OptimisticDAGRuntime, service_grace_seconds: float = 0.0):
        super().__init__(my_node_id=runtime.my_node_id, service_grace_seconds=service_grace_seconds)
        self.runtime = runtime
    def _register_phase(self, phase):
        if hasattr(self.rbc_runtime, "ensure_phase_instances"):
            self.rbc_runtime.ensure_phase_instances(phase)
    async def _send(self, outbound):
        if outbound is None: return
        await self.rbc_runtime.broadcast_role_batch(
            outbound.phase_context, role_id=outbound.role_id,
            batch_bytes=outbound.batch_bytes,
        )
        self.runtime.mark_sent(outbound.phase_context.phase)
        self.timeline.mark("t_first_role_message_sent")
        self.communication.application_payload_bytes += len(outbound.batch_bytes)
    def register_initial_contexts(self, dag_runtime, rbc_runtime):
        super().register_initial_contexts(dag_runtime, rbc_runtime)
        self._register_phase(self.runtime.prepare())
    async def start_at_t0(self):
        self.timeline.start(); started=time.process_time()
        try:
            outbound=self.runtime.start(); await self._send(outbound); return outbound
        except Exception as exc:
            self._record_failure("start", exc); raise
        finally:
            m=self.cpu["role_message_generation"]; m.cpu_seconds+=time.process_time()-started; m.invocations+=1
    async def on_delivery(self, event):
        try:
            outbound=self.runtime.handle_delivery(event, on_phase_registered=self._register_phase)
            await self._send(outbound)
            self.communication.committed_application_block_count += 1
            if self.runtime.public_commitment is not None:
                self.timeline.mark("t_public_finalized")
            if self.runtime.is_successor and self.runtime.successor_result is not None:
                self.timeline.mark("t_successor_done")
            return outbound
        except Exception as exc:
            self._record_failure("delivery", exc); raise
    def current_qids(self):
        return tuple(context.qid for context in self.runtime.controller.phase_contexts.values())
    def expected_application_deliveries(self):
        return 2 if self.runtime.manifest.path_mode == "optimistic" else 3
    def public_finalized(self): return self.runtime.public_commitment is not None
    def local_successor_done(self): return not self.runtime.is_successor or self.runtime.successor_result is not None
    def _protocol_metrics(self): return self.runtime.summary()
    async def retire_and_cleanup(self):
        await super().retire_and_cleanup(); self.runtime.retire()


def normalize_dag_node_summary(
    *,
    protocol: str,
    node_id: int,
    common_workload_digest: str,
    arm_manifest_digest: str,
    legacy_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Project an existing persistent-DAG result into the common envelope."""

    ciphertext_bytes = None
    if protocol == "a2dpss":
        private = legacy_summary.get("a2dpss_handover_metrics") or {}
        timeline_raw = private.get("timestamps_sec") or {}
        timeline = {
            "t0": timeline_raw.get("t0"),
            "t_first_role_message_sent": timeline_raw.get("t_contribution_sent"),
            "t_first_pool_ready": timeline_raw.get("t_pool_ready"),
            "t_public_finalized": timeline_raw.get("t_public_finalized"),
            "t_successor_done": timeline_raw.get("t_successor_done"),
            "frozen_at": timeline_raw.get("frozen_at"),
        }
        public_events = [
            event for event in (legacy_summary.get("a2dpss_events") or [])
            if event.get("event") == "public_finalized"
        ]
        public_material = (
            public_events[-1].get("successor_commitment_vector")
            if public_events else None
        )
        protocol_summary = {
            "public_result_digest": (
                hashlib.sha256(json.dumps(
                    public_material, sort_keys=True, separators=(",", ":")
                ).encode()).hexdigest()
                if public_material is not None else None
            ),
            "formal_cache_hits": 0,
            "detail": legacy_summary.get("a2dpss_computation_metrics"),
        }
        success = bool(private.get("frozen")) and timeline["t_public_finalized"] is not None
        if private.get("is_successor"):
            success = success and timeline["t_successor_done"] is not None
        application_bytes = int(
            (legacy_summary.get("a2dpss_rbc_metrics") or {}).get(
                "outbound_contribution_bytes", 0
            )
        )
        a2_rbc = legacy_summary.get("a2dpss_rbc_metrics") or {}
        proof_bytes = a2_rbc.get("outbound_proof_bytes")
        ciphertext_bytes = a2_rbc.get("outbound_ciphertext_bytes")
        protocol_summary["byte_breakdown"] = {
            "pvss_transcript_bytes": a2_rbc.get("outbound_pvss_bytes"),
            "pvss_proof_bytes": proof_bytes,
            "recipient_ciphertext_bytes": ciphertext_bytes,
        }
        cpu_metrics = legacy_summary.get("a2dpss_computation_metrics") or {}
    elif protocol == "ru22":
        private = legacy_summary.get("ru22_summary") or {}
        raw = private.get("timings") or {}
        timeline = {
            "t0": raw.get("t0", raw.get("t0_unix") and 0.0),
            "t_first_role_message_sent": raw.get("holder_broadcast_sec"),
            "t_first_pool_ready": raw.get("pool_ready_sec"),
            "t_public_finalized": raw.get("public_finalized_sec"),
            "t_successor_done": raw.get("successor_done_sec"),
            "frozen_at": raw.get("successor_done_sec") or raw.get("public_finalized_sec"),
        }
        protocol_summary = {
            "public_result_digest": private.get("public_result_digest"),
            "new_vps_id": private.get("new_vps_id"),
            "formal_cache_hits": private.get("formal_cache_hits"),
            "vps_shape": private.get("ru22_state_shape"),
            "detail": private.get("cpu_metrics"),
        }
        success = private.get("status") == "success"
        rbc = private.get("rbc_metrics") or {}
        application_bytes = int(rbc.get("outbound_batch_bytes", 0))
        ciphertext_bytes = rbc.get("outbound_ciphertext_bytes")
        proof_bytes = rbc.get("outbound_proof_bytes")
        protocol_summary["byte_breakdown"] = {
            "pvr_bytes": rbc.get("outbound_pvr_bytes"),
            "pvss_transcript_bytes": rbc.get("outbound_pvss_bytes"),
            "pvss_proof_bytes": rbc.get("outbound_pvss_proof_bytes"),
            "dleq_proof_bytes": rbc.get("outbound_dleq_bytes"),
            "recipient_ciphertext_bytes": ciphertext_bytes,
        }
        cpu_metrics = private.get("cpu_metrics") or {}
    elif protocol == "bhk22":
        private = legacy_summary.get("bhk22_summary") or {}
        raw = private.get("timeline") or {}
        generation_sent = [
            value for name, value in raw.items()
            if name.endswith("_sent") and value is not None
        ]
        pool_ready = [
            value for name, value in raw.items()
            if name.endswith("_pool_ready") and value is not None
        ]
        timeline = {
            "t0": raw.get("t0"),
            "t_first_role_message_sent": min(generation_sent) if generation_sent else None,
            "t_first_pool_ready": min(pool_ready) if pool_ready else None,
            "t_public_finalized": raw.get("t_public_finalized"),
            "t_successor_done": raw.get("t_successor_done"),
            "frozen_at": raw.get("frozen_at"),
        }
        protocol_summary = {
            "raw_metrics_schema": private.get("metrics_schema"),
            "public_result_digest": private.get("public_commitment_digest"),
            "public_commitment_kind": private.get("public_commitment_kind"),
            "path_mode": private.get("path_mode"),
            "completion_mode": private.get("completion_mode"),
            "phase_qids": private.get("phase_qids"),
            "committee_selection": private.get("committee_selection"),
            "role_metrics": private.get("role_metrics"),
            "application_verification_metrics": private.get(
                "application_verification_metrics"
            ),
            "rbc_metrics": private.get("rbc_metrics"),
            "formal_cache_hits": 0,
        }
        success = private.get("status") == "success"
        rbc_by_phase = private.get("rbc_metrics") or {}
        application_bytes = sum(
            int(values.get("outbound_application_bytes", 0))
            for values in rbc_by_phase.values()
        )
        proof_bytes = sum(int(v.get("outbound_proof_bytes", 0)) for v in rbc_by_phase.values())
        ciphertext_bytes = sum(int(v.get("outbound_ciphertext_bytes", 0)) for v in rbc_by_phase.values())
        committee_vrf_proof_bytes = sum(
            int(v.get("outbound_role_vrf_proof_bytes", 0))
            for v in rbc_by_phase.values()
        )
        cpu_metrics = {
            role: {
                "cpu_seconds": float(metric.get("cpu_seconds", 0.0)),
                "invocations": 1,
                "detail": metric,
            }
            for role, metric in (private.get("role_metrics") or {}).items()
        }
        cpu_metrics.update(
            {
                f"application_verification:{stage}": {
                    "cpu_seconds": float(metric.get("cpu_seconds", 0.0)),
                    "invocations": int(metric.get("invocations", 0)),
                    "detail": metric,
                }
                for stage, metric in (
                    private.get("application_verification_metrics") or {}
                ).items()
            }
        )
        cpu_metrics["committee_vrf_proof_generation"] = {
            "cpu_seconds": sum(
                float(v.get("role_vrf_proof_generation_cpu_seconds", 0.0))
                for v in rbc_by_phase.values()
            ),
            "invocations": sum(
                int(v.get("outbound_role_vrf_proof_bytes", 0) > 0)
                for v in rbc_by_phase.values()
            ),
        }
        cpu_metrics["committee_vrf_proof_verification"] = {
            "cpu_seconds": sum(
                float(v.get("role_vrf_proof_verification_cpu_seconds", 0.0))
                for v in rbc_by_phase.values()
            ),
            "invocations": sum(
                int(v.get("delivered_messages", 0))
                for v in rbc_by_phase.values()
                if int(v.get("delivered_role_vrf_proof_bytes", 0)) > 0
            ),
        }
    else:
        raise ValueError("unsupported YOSO ADPSS protocol")

    from beaver.yoso_adpss_distributed import valid_timeline
    monotonic = valid_timeline(timeline)
    cleanup = bool(legacy_summary.get("cleanup_completed")) and all(
        int(legacy_summary.get(field, 0)) == 0
        for field in (
            "active_rbc_instances_after_cleanup",
            "active_a2dpss_rbc_instances_after_cleanup",
            "active_ru22_rbc_instances_after_cleanup",
            "active_bhk22_rbc_instances_after_cleanup",
        )
    )
    transport = legacy_summary.get("transport_bytes_by_family") or {}
    communication = YOSOADPSSCommunicationMetrics(
        application_payload_bytes=application_bytes,
        cryptographic_proof_bytes=proof_bytes,
        recipient_ciphertext_bytes=ciphertext_bytes,
        committee_vrf_proof_bytes=(
            committee_vrf_proof_bytes if protocol == "bhk22" else None
        ),
        yoso_rbc_envelope_bytes=transport.get("yoso_rbc"),
        dag_protocol_bytes=transport.get("dag_protocol"),
        common_coin_bytes=transport.get("common_coin"),
        total_network_bytes=transport.get("total", legacy_summary.get("bytes_sent")),
        committed_application_block_count=int(legacy_summary.get("delivery_event_count", 0)),
        dag_vertex_count=len(legacy_summary.get("ordered_vertices") or []),
        dag_committed_wave_count=int(legacy_summary.get("committed_wave") or 0),
    )
    status = "success" if success and monotonic and cleanup else "failure"
    failure = None if status == "success" else {
        "stage": "normalize",
        "type": "ArmValidationFailure",
        "message": f"protocol_success={success}, timeline_monotonic={monotonic}, cleanup={cleanup}",
    }
    return {
        "schema": NODE_METRICS_SCHEMA,
        "protocol": protocol,
        "node_id": int(node_id),
        "common_workload_digest": common_workload_digest,
        "arm_manifest_digest": arm_manifest_digest,
        "status": status,
        "failure": failure,
        "timeline": timeline,
        "cpu": cpu_metrics,
        "communication": asdict(communication),
        "protocol_metrics": protocol_summary,
        "cleanup": {"completed": cleanup},
    }


__all__ = [
    "A2DPSSAdapter", "BHK22Adapter", "CPUMetric", "NODE_METRICS_SCHEMA",
    "BHK22_DAG_NODE_METRICS_SCHEMA",
    "RU22Adapter", "TRIAL_METRICS_SCHEMA", "YOSOADPSSCommunicationMetrics",
    "YOSOADPSSHandoverAdapter", "YOSOADPSSTimeline",
    "normalize_dag_node_summary",
]
