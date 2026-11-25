"""Real DAG/YOSO-RBC runtime for the RU22 threshold-operation arm."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Sequence

from beaver.baselines.ru22_dag_runner import RU22StructuralPoolState
from beaver.baselines.ru22_experiment import (
    FIXTURE_MODE_WORST_CASE_MULTI_VPS,
    RU22DAGExperimentManifest,
    RU22DAGLocalShard,
)
from beaver.baselines.ru22_operation import (
    RU22OperationContributionBatch,
    RU22OperationJustifierManager,
    RU22OperationPublicFinalizer,
    consume_ru22_operation_handover_role,
    reconstruct_ru22_operation_successor,
)
from beaver.dag_tob import DeliveryEvent, PoolEntry
from beaver.threshold_operation_contract import ThresholdOperationContext


class RU22OperationDAGRuntime:
    """One node's cold multi-VPS operation-coupled RU22 handover."""

    def __init__(
        self,
        *,
        manifest: RU22DAGExperimentManifest,
        local_shard: RU22DAGLocalShard,
        operation_context: ThresholdOperationContext,
        my_node_id: int,
    ) -> None:
        if not isinstance(manifest, RU22DAGExperimentManifest):
            raise TypeError("manifest has the wrong type")
        if not isinstance(local_shard, RU22DAGLocalShard):
            raise TypeError("local_shard has the wrong type")
        if not isinstance(operation_context, ThresholdOperationContext):
            raise TypeError("operation_context has the wrong type")
        if local_shard.node_id != int(my_node_id):
            raise ValueError("RU22 operation shard belongs to another node")
        local_shard.validate(manifest)
        self.manifest = manifest
        self.local_shard = local_shard
        self.operation_context = operation_context
        self.my_node_id = int(my_node_id)
        self.context = manifest.context
        self.justifier = RU22OperationJustifierManager()
        self.qid = self.justifier.register_context(
            self.context,
            operation_context,
            trial_id=manifest.trial_id,
            parent_vps_catalog=manifest.parent_vps_catalog,
        )
        self.finalizer = RU22OperationPublicFinalizer(self.justifier)
        self.holder_state = local_shard.holder_state(manifest)
        self.holder_result = None
        self.public_result = None
        self.successor_result = None
        self.pool_state = None
        self.delivery_event = None
        self._times: dict[str, float] = {}
        self._cpu_metrics = {
            "public_finalization": {"invocations": 0, "cpu_seconds": 0.0},
            "successor_reconstruction": {"invocations": 0, "cpu_seconds": 0.0},
        }

    @property
    def source_role_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.manifest.source_role_nodes))

    @property
    def requires_complete_source_pool(self) -> bool:
        return self.manifest.fixture_mode == FIXTURE_MODE_WORST_CASE_MULTI_VPS

    @property
    def is_source(self) -> bool:
        return self.local_shard.source is not None

    @property
    def is_successor(self) -> bool:
        return self.my_node_id in self.context.successor_role_indices

    @property
    def completed(self) -> bool:
        return self.public_result is not None and (
            not self.is_successor or self.successor_result is not None
        )

    def validate_network_parameters(
        self, *, M: int, t_a: int, n_r: int, t_r: int,
        epoch: str, coin_mode: str,
    ) -> None:
        if (
            int(M) != self.manifest.dag_m
            or int(t_a) != self.manifest.dag_t
            or int(n_r) != self.context.resharing_n
            or int(t_r) != self.context.resharing_threshold
        ):
            raise ValueError("RU22 operation manifest/DAG parameters mismatch")
        if str(epoch) != self.manifest.trial_epoch:
            raise ValueError("RU22 operation epoch mismatch")
        if str(coin_mode).strip().lower() != "threshold-bls":
            raise ValueError("RU22 operation arm requires threshold-bls")

    def start(self) -> None:
        if self._times:
            raise RuntimeError("RU22 operation runtime already started")
        self._times["t0"] = time.perf_counter()

    async def broadcast_local(self, rbc_manager):
        if "t0" not in self._times:
            raise RuntimeError("RU22 operation runtime is not started")
        if self.holder_state is None:
            return None
        if self.holder_result is not None:
            raise RuntimeError("RU22 operation source already spoke")
        result = consume_ru22_operation_handover_role(
            context=self.context,
            operation_context=self.operation_context,
            trial_id=self.manifest.trial_id,
            holder_state=self.holder_state,
        )
        await rbc_manager.broadcast_batch(
            self.context,
            holder_role_id=result.batch.holder_role_id,
            batch_bytes=result.batch_bytes,
        )
        self.holder_result = result
        self._times["holder_broadcast"] = time.perf_counter()
        return result

    def structural_pool_state(
        self, entries: Sequence[PoolEntry]
    ) -> RU22StructuralPoolState:
        parent_roles: dict[str, dict[int, int]] = defaultdict(dict)
        valid_batches = 0
        rejected = 0
        seen_roles = set()
        for entry in entries:
            try:
                mid, payload = entry
                batch = RU22OperationContributionBatch.from_bytes(payload)
                if (
                    batch.qid != self.qid
                    or batch.base_context_digest != self.context.context_digest
                    or batch.operation_context_digest
                    != self.operation_context.context_digest
                    or mid != f"node_{batch.holder_role_id}"
                    or batch.holder_role_id in seen_roles
                ):
                    raise ValueError("invalid structural operation batch")
            except (TypeError, ValueError):
                rejected += 1
                continue
            valid_batches += 1
            seen_roles.add(batch.holder_role_id)
            for pvr in batch.pvr_entries:
                base = pvr.base_pvr
                parent_roles[base.parent_vps_id][base.dealer_role_id] = (
                    base.dealer_eval_point
                )
        required = self.context.resharing_threshold + 1
        candidates = sorted(
            (
                parent,
                tuple(
                    role for role, _point in sorted(
                        roles.items(), key=lambda item: (item[1], item[0])
                    )
                ),
            )
            for parent, roles in parent_roles.items()
            if len(set(roles.values())) >= required
        )
        parent, roles = candidates[0] if candidates else (None, ())
        return RU22StructuralPoolState(
            ready=parent is not None,
            valid_batch_count=valid_batches,
            common_parent_vps_id=parent,
            common_parent_role_ids=roles,
            rejected_entry_count=rejected,
        )

    def mark_pool_ready(self, state: RU22StructuralPoolState) -> None:
        if not isinstance(state, RU22StructuralPoolState) or not state.ready:
            raise ValueError("RU22 operation pool is not ready")
        self.pool_state = state
        self._times["pool_ready"] = time.perf_counter()

    def handle_delivery(self, event: DeliveryEvent):
        if not isinstance(event, DeliveryEvent) or event.qid != self.qid:
            raise ValueError("delivery does not match RU22 operation qid")
        if self.public_result is not None:
            if event == self.delivery_event:
                return self.public_result, self.successor_result
            raise ValueError("conflicting RU22 operation delivery")
        self._times["ordered"] = time.perf_counter()
        cpu_started = time.process_time()
        public = self.finalizer.on_verified_deliver(
            event.qid, event.entries, ordered_vertex=event.ordered_vertex
        )
        self._cpu_metrics["public_finalization"]["invocations"] += 1
        self._cpu_metrics["public_finalization"]["cpu_seconds"] += (
            time.process_time() - cpu_started
        )
        if public is None:
            raise RuntimeError("verified RU22 operation delivery did not finalize")
        selection = self.justifier.verified_selection(event.qid, event.entries)
        if selection is None:
            raise RuntimeError("RU22 operation selection handoff is missing")
        private = None
        if self.is_successor:
            cpu_started = time.process_time()
            private = reconstruct_ru22_operation_successor(
                context=self.context,
                result=public,
                selection=selection,
                my_node_id=self.my_node_id,
                my_role_idx=self.context.successor_role_indices[self.my_node_id],
                my_rsk=self.local_shard.successor_recipient_secret_key,
            )
            self._cpu_metrics["successor_reconstruction"]["invocations"] += 1
            self._cpu_metrics["successor_reconstruction"]["cpu_seconds"] += (
                time.process_time() - cpu_started
            )
        self.public_result = public
        self.successor_result = private
        self.delivery_event = event
        self._times["public_finalized"] = time.perf_counter()
        if private is not None:
            self._times["successor_done"] = time.perf_counter()
        return public, private

    def retire(self) -> None:
        self.justifier._registrations.clear()
        self.justifier._base.contexts.clear()

    def summary(self, *, rbc_metrics=None) -> dict[str, Any]:
        t0 = self._times.get("t0")
        elapsed = lambda name: (
            self._times[name] - t0
            if t0 is not None and name in self._times else None
        )
        public = self.public_result
        metrics = self.holder_result.metrics if self.holder_result else None
        return {
            "protocol": "ru22",
            "application": "threshold-operation",
            "transport": "dag-tob",
            "metrics_schema": "ru22-operation-dag-node-metrics/v1",
            "trial_id": self.manifest.trial_id,
            "trial_epoch": self.manifest.trial_epoch,
            "node_id": self.my_node_id,
            "manifest_digest": self.manifest.digest(),
            "context_digest": self.context.context_digest,
            "operation_context_digest": self.operation_context.context_digest,
            "qid": self.qid,
            "fixture_mode": self.manifest.fixture_mode,
            "status": "success" if self.completed else "incomplete",
            "is_source": self.is_source,
            "is_successor": self.is_successor,
            "successor_completed": not self.is_successor or self.successor_result is not None,
            "committed_application_blocks": 1 if public else 0,
            "public_result_digest": public.result_digest if public else None,
            "operation_output": public.operation_output if public else None,
            "operation_output_digest": public.operation_output_digest if public else None,
            "public_successor_state_digest": public.new_vps.vps_id if public else None,
            "selected_parent_vps_id": public.parent_vps_id if public else None,
            "selected_roles": list(public.selected_role_ids) if public else [],
            "selected_base_pvr_digests": list(public.selected_base_pvr_digests) if public else [],
            "selected_operation_pvr_digests": list(public.selected_operation_pvr_digests) if public else [],
            "timings": {
                "t0": 0.0 if t0 is not None else None,
                "holder_broadcast_sec": elapsed("holder_broadcast"),
                "pool_ready_sec": elapsed("pool_ready"),
                "ordered_sec": elapsed("ordered"),
                "public_finalized_sec": elapsed("public_finalized"),
                "successor_done_sec": elapsed("successor_done"),
            },
            "proof_bytes": {
                "link": metrics.link_proof_bytes if metrics else 0,
                "operation": metrics.operation_proof_bytes if metrics else 0,
                "vk": metrics.operation_vk_bytes if metrics else 0,
                "operation_share": metrics.operation_share_bytes if metrics else 0,
            },
            "holder_metrics": asdict(metrics) if metrics else None,
            "formal": self.justifier.metrics_snapshot(),
            "application_cpu_metrics": {
                name: dict(metric)
                for name, metric in self._cpu_metrics.items()
            },
            "rbc_metrics": rbc_metrics,
        }


__all__ = ["RU22OperationDAGRuntime"]
