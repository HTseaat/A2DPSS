"""Runtime adapter that wires one RU22 handover into DAG-TOB."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from beaver.baselines.ru22_dpss import (
    QID_KIND,
    RU22ContributionBatch,
    RU22DPSSJustifierManager,
    RU22HolderActionResult,
    RU22PublicFinalizer,
    RU22SuccessorReconstructionManager,
    RU22SuccessorResult,
    RU22DeliveryResult,
    run_ru22_handover_role,
    validate_ru22_contribution_batch_shape,
)
from beaver.baselines.ru22_experiment import (
    FIXTURE_MODE_WORST_CASE_MULTI_VPS,
    RU22DAGExperimentManifest,
    RU22DAGLocalShard,
)
from beaver.dag_tob import DeliveryEvent, PoolEntry


@dataclass(frozen=True)
class RU22StructuralPoolState:
    ready: bool
    valid_batch_count: int
    common_parent_vps_id: str | None
    common_parent_role_ids: tuple[int, ...]
    rejected_entry_count: int

    def to_obj(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "valid_batch_count": self.valid_batch_count,
            "common_parent_vps_id": self.common_parent_vps_id,
            "common_parent_role_ids": list(
                self.common_parent_role_ids
            ),
            "rejected_entry_count": self.rejected_entry_count,
        }


class RU22DAGHandoverRuntime:
    """Node-local state for exactly one fixture-backed RU22 handover."""

    def __init__(
        self,
        *,
        manifest: RU22DAGExperimentManifest,
        local_shard: RU22DAGLocalShard,
        my_node_id: int,
    ) -> None:
        if not isinstance(manifest, RU22DAGExperimentManifest):
            raise TypeError("manifest must be an RU22DAGExperimentManifest")
        if not isinstance(local_shard, RU22DAGLocalShard):
            raise TypeError("local_shard must be an RU22DAGLocalShard")
        my_node_id = int(my_node_id)
        if local_shard.node_id != my_node_id:
            raise ValueError("RU22 local shard belongs to another node")
        local_shard.validate(manifest)

        self.manifest = manifest
        self.local_shard = local_shard
        self.my_node_id = my_node_id
        self.context = manifest.context
        self.justifier = RU22DPSSJustifierManager()
        self.justifier.register_context(
            self.context, manifest.parent_vps_catalog
        )
        self.finalizer = RU22PublicFinalizer(self.justifier)
        self.reconstruction = RU22SuccessorReconstructionManager(
            self.justifier
        )
        self.reconstruction.register_context(
            self.context,
            my_node_id=my_node_id,
            my_rsk=local_shard.successor_recipient_secret_key,
        )
        self.holder_state = local_shard.holder_state(manifest)
        self.holder_result: RU22HolderActionResult | None = None
        self.public_result: RU22DeliveryResult | None = None
        self.successor_result: RU22SuccessorResult | None = None
        self.pool_state: RU22StructuralPoolState | None = None
        self.delivery_event: DeliveryEvent | None = None
        self.events: list[dict[str, Any]] = []
        self._times: dict[str, float] = {}
        self._cpu: dict[str, float] = {}

    @property
    def qid(self) -> str:
        return self.context.encode_qid()

    @property
    def source_role_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.manifest.source_role_nodes))

    @property
    def requires_complete_source_pool(self) -> bool:
        return (
            self.manifest.fixture_mode
            == FIXTURE_MODE_WORST_CASE_MULTI_VPS
        )

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
        self,
        *,
        M: int,
        t_a: int,
        n_r: int,
        t_r: int,
        epoch: str,
        coin_mode: str,
    ) -> None:
        expected = (
            int(M) == self.manifest.dag_m
            and int(t_a) == self.manifest.dag_t
            and int(n_r) == self.context.resharing_n
            and int(t_r) == self.context.resharing_threshold
        )
        if not expected:
            raise ValueError(
                "RU22 manifest parameters do not match the DAG process"
            )
        if str(epoch) != self.manifest.trial_epoch:
            raise ValueError("RU22 trial epoch does not match the DAG run")
        if str(coin_mode).strip().lower() != "threshold-bls":
            raise ValueError(
                "RU22 Step-4 network runs require threshold-bls coin"
            )

    def start(self) -> None:
        if "t0" in self._times:
            raise RuntimeError("RU22 handover runtime already started")
        self._times["t0"] = time.perf_counter()
        self._cpu["t0"] = time.process_time()
        self.events.append({"event": "released", "qid": self.qid})

    async def broadcast_local(self, rbc_manager) -> RU22HolderActionResult | None:
        if "t0" not in self._times:
            raise RuntimeError("RU22 handover has not been released")
        if self.holder_state is None:
            return None
        if self.holder_result is not None:
            raise RuntimeError("RU22 source role already broadcast")
        source = self.local_shard.source
        assert source is not None

        async def _broadcast(qid: str, mid: str, payload: bytes) -> None:
            if qid != self.qid or mid != f"node_{source.holder_role_id}":
                raise ValueError("RU22 holder emitted an unexpected identity")
            await rbc_manager.broadcast_batch(
                self.context,
                holder_role_id=source.holder_role_id,
                batch_bytes=payload,
            )

        result = await run_ru22_handover_role(
            context=self.context,
            holder_state=self.holder_state,
            broadcast_batch=_broadcast,
        )
        self.holder_result = result
        self._times["holder_broadcast"] = time.perf_counter()
        self._cpu["holder_broadcast"] = time.process_time()
        self.events.append(
            {
                "event": "holder_broadcast",
                "qid": result.qid,
                "mid": result.mid,
                "pvr_count": result.metrics.pvr_count,
                "pvr_bytes": result.metrics.pvr_bytes,
                "batch_bytes": result.metrics.batch_bytes,
            }
        )
        return result

    def structural_pool_state(
        self, entries: Sequence[PoolEntry]
    ) -> RU22StructuralPoolState:
        parent_roles: dict[str, dict[int, int]] = defaultdict(dict)
        valid_batches = 0
        rejected = 0
        seen_roles: set[int] = set()
        for entry in entries:
            try:
                mid, payload = entry
                batch = RU22ContributionBatch.from_bytes(payload)
                validate_ru22_contribution_batch_shape(
                    self.context, batch
                )
                if mid != f"node_{batch.holder_role_id}":
                    raise ValueError("mid does not bind holder role")
                if batch.holder_role_id in seen_roles:
                    raise ValueError("duplicate holder role")
            except (TypeError, ValueError):
                rejected += 1
                continue
            valid_batches += 1
            seen_roles.add(batch.holder_role_id)
            for pvr in batch.pvr_entries:
                parent_roles[pvr.parent_vps_id][
                    pvr.dealer_role_id
                ] = pvr.dealer_eval_point

        required = self.context.resharing_threshold + 1
        candidates = [
            (
                parent_id,
                tuple(
                    role
                    for role, _point in sorted(
                        roles.items(), key=lambda item: (item[1], item[0])
                    )
                ),
            )
            for parent_id, roles in parent_roles.items()
            if len(set(roles.values())) >= required
        ]
        candidates.sort(key=lambda item: item[0])
        parent_id, roles = candidates[0] if candidates else (None, ())
        return RU22StructuralPoolState(
            ready=parent_id is not None,
            valid_batch_count=valid_batches,
            common_parent_vps_id=parent_id,
            common_parent_role_ids=roles,
            rejected_entry_count=rejected,
        )

    def mark_pool_ready(
        self, state: RU22StructuralPoolState
    ) -> None:
        if not isinstance(state, RU22StructuralPoolState) or not state.ready:
            raise ValueError("RU22 pool is not structurally ready")
        self.pool_state = state
        self._times["pool_ready"] = time.perf_counter()
        self._cpu["pool_ready"] = time.process_time()
        self.events.append(
            {"event": "pool_ready", **state.to_obj()}
        )

    def handle_delivery(
        self, event: DeliveryEvent
    ) -> tuple[RU22DeliveryResult, RU22SuccessorResult | None]:
        if not isinstance(event, DeliveryEvent) or event.qid != self.qid:
            raise ValueError("delivery event does not match RU22 runtime")
        if self.public_result is not None:
            if self.delivery_event == event:
                return self.public_result, self.successor_result
            raise ValueError("conflicting RU22 delivery event")
        self._times["ordered"] = time.perf_counter()
        self._cpu["ordered"] = time.process_time()
        public = self.finalizer.on_delivery_event(event)
        if public is None:
            raise RuntimeError(
                "committed RU22 delivery lacks verified-selection handoff"
            )
        self.public_result = public
        self.delivery_event = event
        self._times["public_finalized"] = time.perf_counter()
        self._cpu["public_finalized"] = time.process_time()
        private = self.reconstruction.reconstruct(public, event.entries)
        self.successor_result = private
        if private is not None:
            self._times["successor_done"] = time.perf_counter()
            self._cpu["successor_done"] = time.process_time()
        self.events.append(
            {
                "event": "public_finalized",
                "public_result_digest": public.delivery_result_digest,
                "new_vps_id": public.new_vps_id,
                "private_reconstructed": private is not None,
            }
        )
        return public, private

    def summary(self, *, rbc_metrics=None) -> dict[str, Any]:
        t0 = self._times.get("t0")

        def _latency(stage: str) -> float | None:
            value = self._times.get(stage)
            return value - t0 if value is not None and t0 is not None else None

        public = self.public_result
        private = self.successor_result
        event = self.delivery_event
        holder_metrics = (
            self.holder_result.metrics if self.holder_result else None
        )
        holder_generation = None
        if holder_metrics is not None:
            generation_stages = (
                "old_share_binding",
                "pvss",
                "dleq",
                "serialization",
            )
            holder_generation = {
                "invocations": 1,
                "pvr_count": holder_metrics.pvr_count,
                "pvr_bytes": holder_metrics.pvr_bytes,
                "batch_bytes": holder_metrics.batch_bytes,
                "elapsed_sec": sum(
                    float(getattr(holder_metrics, f"{stage}_seconds"))
                    for stage in generation_stages
                ),
                "cpu_seconds": sum(
                    float(
                        getattr(holder_metrics, f"{stage}_cpu_seconds")
                    )
                    for stage in generation_stages
                ),
                "stages": {
                    stage: {
                        "elapsed_sec": float(
                            getattr(holder_metrics, f"{stage}_seconds")
                        ),
                        "cpu_seconds": float(
                            getattr(
                                holder_metrics,
                                f"{stage}_cpu_seconds",
                            )
                        ),
                    }
                    for stage in generation_stages
                },
                "broadcast": {
                    "elapsed_sec": holder_metrics.broadcast_seconds,
                    "cpu_seconds": holder_metrics.broadcast_cpu_seconds,
                },
            }
        success = public is not None and (
            not self.is_successor or private is not None
        )
        return {
            "protocol": "ru22",
            "transport": "dag-tob",
            "metrics_schema": "ru22-dag-node-metrics/v1",
            "trial_id": self.manifest.trial_id,
            "trial_epoch": self.manifest.trial_epoch,
            "node_id": self.my_node_id,
            "manifest_digest": self.manifest.digest(),
            "context_digest": self.context.context_digest,
            "qid": self.qid,
            "fixture_mode": self.manifest.fixture_mode,
            "status": "success" if success else "incomplete",
            "failure_reason": None,
            "is_source": self.is_source,
            "is_successor": self.is_successor,
            "public_result_digest": (
                public.delivery_result_digest if public else None
            ),
            "new_vps_id": public.new_vps_id if public else None,
            "selected_parent_vps_id": (
                public.parent_vps_id if public else None
            ),
            "selected_mids": list(public.selected_mids) if public else [],
            "selected_roles": (
                list(public.selected_role_ids) if public else []
            ),
            "selected_evaluation_points": (
                list(public.selected_eval_points) if public else []
            ),
            "selected_pvr_digests": (
                list(public.selected_pvr_digests) if public else []
            ),
            "ordered_vertex": (
                {
                    "round": event.ordered_vertex[0],
                    "source": event.ordered_vertex[1],
                }
                if event
                else None
            ),
            "committed_wave": event.committed_wave if event else None,
            "successor_completed": private is not None,
            "pool_state": (
                self.pool_state.to_obj() if self.pool_state else None
            ),
            "timings": {
                "t0": 0.0 if t0 is not None else None,
                "holder_broadcast_sec": _latency("holder_broadcast"),
                "pool_ready_sec": _latency("pool_ready"),
                "ordered_sec": _latency("ordered"),
                "public_finalized_sec": _latency("public_finalized"),
                "successor_done_sec": _latency("successor_done"),
            },
            "cpu_metrics": {
                "holder_generation": holder_generation,
                "formal": self.justifier.formal_metrics_snapshot(),
                "validation": self.justifier.validation_metrics_snapshot(),
                "public_finalization": self.finalizer.metrics_snapshot(),
                "successor_reconstruction": (
                    self.reconstruction.metrics_snapshot()
                ),
            },
            "rbc_metrics": rbc_metrics,
            "ru22_state_shape": {
                "parent_vps_count": len(
                    self.manifest.parent_vps_catalog
                ),
                "local_vps_count": (
                    len(self.local_shard.source.local_vps_shares)
                    if self.local_shard.source
                    else 0
                ),
                "distinct_vps_count": len(
                    {
                        vps_id
                        for view in self.manifest.source_vps_views.values()
                        for vps_id in view
                    }
                ),
                "local_view_sizes": {
                    str(role): len(view)
                    for role, view in sorted(
                        self.manifest.source_vps_views.items()
                    )
                },
                "view_intersection_size": len(
                    set.intersection(
                        *(
                            set(view)
                            for view in self.manifest.source_vps_views.values()
                        )
                    )
                ),
                "common_parent_vps_id": (
                    self.manifest.common_parent_vps_id
                ),
                "source_role_count": len(self.source_role_ids),
                "successor_count": len(
                    self.manifest.expected_successor_nodes
                ),
            },
            "formal_validation_mode": "uncached-per-invocation",
            "formal_cache_hits": 0,
            "auxiliary_cache_hits": int(
                self.justifier.validation_metrics_snapshot()[
                    "auxiliary_cache_hits"
                ]
            ),
            "events": list(self.events),
        }


__all__ = [
    "QID_KIND",
    "RU22DAGHandoverRuntime",
    "RU22StructuralPoolState",
]
