"""Shared, protocol-neutral workload trace and R6 lifecycle metrics.

The protocol replay trace fixes ideal-coin and VRF-role inputs.  This module
separately fixes application arrivals so DAG-TOB and the ACS baseline can
replay the same qids, message identifiers, payloads, and logical triggers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from beaver.dag_tob import Qid


START = "start"
INITIAL_PROPOSAL_FIXED = "initial_proposal_fixed"
_VALID_TRIGGERS = {START, INITIAL_PROPOSAL_FIXED}
A2DPSS_HANDOVER_METRICS_SCHEMA = "a2dpss-handover-metrics-v1"
A2DPSS_COMPUTATION_METRICS_SCHEMA = "a2dpss-computation-metrics-v1"


class A2DPSSHandoverMetrics:
    """Protocol-neutral monotonic timeline for one A²DPSS handover.

    Only offsets from the local ``perf_counter`` origin are serialized.  This
    keeps DAG-TOB and ACS-TOB summaries comparable without pretending that
    absolute monotonic clocks from different processes share an epoch.
    """

    _ORDER = (
        "t0",
        "t_contribution_sent",
        "t_pool_ready",
        "t_ordered",
        "t_public_finalized",
        "t_successor_done",
    )

    def __init__(
        self,
        *,
        protocol: str,
        node: int,
        qid: str,
        is_holder: bool,
        is_successor: bool,
    ) -> None:
        if protocol not in {"dag-tob", "acs-tob", "async-algorand"}:
            raise ValueError("unknown A2DPSS agreement protocol")
        if not isinstance(qid, str) or not qid:
            raise ValueError("qid must be a non-empty encoded qid")
        self.protocol = protocol
        self.node = int(node)
        self.qid = qid
        self.is_holder = bool(is_holder)
        self.is_successor = bool(is_successor)
        self._origin = None
        self._timestamps = {name: None for name in self._ORDER}
        self._frozen_at = None

    def start(self, when: Optional[float] = None) -> None:
        if self._origin is not None:
            raise RuntimeError("handover metrics already started")
        self._origin = time.perf_counter() if when is None else float(when)
        self._timestamps["t0"] = 0.0

    def _mark(self, name: str, when: Optional[float] = None) -> None:
        if self._origin is None:
            raise RuntimeError("handover metrics have not started")
        if self._frozen_at is not None:
            raise RuntimeError("handover metrics are frozen")
        if name not in self._timestamps or name == "t0":
            raise ValueError(f"unsupported handover timestamp {name!r}")
        if self._timestamps[name] is not None:
            return
        absolute = time.perf_counter() if when is None else float(when)
        relative = absolute - self._origin
        if relative < 0:
            raise ValueError("handover timestamp precedes t0")
        prior = [
            self._timestamps[item]
            for item in self._ORDER[: self._ORDER.index(name)]
            if self._timestamps[item] is not None
        ]
        if prior and relative < max(prior):
            raise ValueError(f"handover timestamp {name} is not monotonic")
        self._timestamps[name] = relative

    def mark_contribution_sent(self, when: Optional[float] = None) -> None:
        if not self.is_holder:
            raise RuntimeError("non-holder cannot mark contribution sent")
        self._mark("t_contribution_sent", when)

    def mark_pool_ready(self, when: Optional[float] = None) -> None:
        self._mark("t_pool_ready", when)

    def mark_ordered(self, when: Optional[float] = None) -> None:
        if self._timestamps["t_pool_ready"] is None:
            raise RuntimeError("cannot mark ordered before pool ready")
        self._mark("t_ordered", when)

    def mark_public_finalized(self, when: Optional[float] = None) -> None:
        if self._timestamps["t_ordered"] is None:
            raise RuntimeError("cannot finalize before TOB order")
        self._mark("t_public_finalized", when)
        if not self.is_successor:
            self.freeze()

    def mark_successor_done(self, when: Optional[float] = None) -> None:
        if not self.is_successor:
            raise RuntimeError("non-successor cannot mark successor done")
        if self._timestamps["t_public_finalized"] is None:
            raise RuntimeError("cannot reconstruct before public finalization")
        self._mark("t_successor_done", when)
        self.freeze()

    def freeze(self, when: Optional[float] = None) -> None:
        if self._origin is None:
            raise RuntimeError("handover metrics have not started")
        if self._frozen_at is not None:
            return
        completion = (
            self._timestamps["t_successor_done"]
            if self.is_successor
            else self._timestamps["t_public_finalized"]
        )
        if completion is None:
            raise RuntimeError("cannot freeze before local logical completion")
        if when is None:
            self._frozen_at = completion
            return
        relative = float(when) - self._origin
        if relative < completion:
            raise ValueError("freeze timestamp precedes logical completion")
        self._frozen_at = relative

    def snapshot(self) -> dict:
        timestamps = dict(self._timestamps)
        contribution_sent = timestamps["t_contribution_sent"]
        pool_ready = timestamps["t_pool_ready"]
        ordered = timestamps["t_ordered"]
        public_finalized = timestamps["t_public_finalized"]
        successor_done = timestamps["t_successor_done"]
        return {
            "schema": A2DPSS_HANDOVER_METRICS_SCHEMA,
            "protocol": self.protocol,
            "node": self.node,
            "qid": self.qid,
            "is_holder": self.is_holder,
            "is_successor": self.is_successor,
            "clock": "perf_counter-relative",
            "timestamps_sec": {
                **timestamps,
                "frozen_at": self._frozen_at,
            },
            "durations_sec": {
                "contribution_generation": contribution_sent,
                "pool_ready_latency": pool_ready,
                "ordering_latency": (
                    ordered - pool_ready
                    if ordered is not None and pool_ready is not None
                    else None
                ),
                "public_finalize_latency": (
                    public_finalized - ordered
                    if public_finalized is not None and ordered is not None
                    else None
                ),
                "successor_latency": (
                    successor_done - public_finalized
                    if successor_done is not None
                    and public_finalized is not None
                    else None
                ),
                "handover_latency": successor_done,
                "local_completion_latency": (
                    successor_done if self.is_successor else public_finalized
                ),
            },
            "frozen": self._frozen_at is not None,
        }


def validate_a2dpss_handover_metrics(
    snapshot: dict,
    *,
    protocol: str,
    node: int,
    qid: str,
    is_successor: bool,
) -> list:
    """Return invariant failures for one serialized handover timeline."""

    failures = []
    if not isinstance(snapshot, dict):
        return ["handover metrics missing"]
    expected = {
        "schema": A2DPSS_HANDOVER_METRICS_SCHEMA,
        "protocol": protocol,
        "node": int(node),
        "qid": qid,
        "is_successor": bool(is_successor),
        "clock": "perf_counter-relative",
    }
    for field, value in expected.items():
        if snapshot.get(field) != value:
            failures.append(f"{field} mismatch")
    timestamps = snapshot.get("timestamps_sec") or {}
    t0 = timestamps.get("t0")
    pool_ready = timestamps.get("t_pool_ready")
    ordered = timestamps.get("t_ordered")
    finalized = timestamps.get("t_public_finalized")
    successor_done = timestamps.get("t_successor_done")
    if t0 != 0.0:
        failures.append("t0 must be zero-relative")
    required = [pool_ready, ordered, finalized]
    if any(value is None for value in required):
        failures.append("required handover timestamp missing")
    elif not 0.0 <= pool_ready <= ordered <= finalized:
        failures.append("handover timestamps are not monotonic")
    if is_successor:
        if successor_done is None:
            failures.append("successor completion missing")
        elif finalized is not None and successor_done < finalized:
            failures.append("successor completion precedes finalization")
    elif successor_done is not None:
        failures.append("non-successor has private completion")
    if snapshot.get("frozen") is not True:
        failures.append("handover metrics are not frozen")
    expected_completion = successor_done if is_successor else finalized
    if timestamps.get("frozen_at") != expected_completion:
        failures.append("metrics did not freeze at logical completion")
    durations = snapshot.get("durations_sec") or {}
    if durations.get("handover_latency") != (
        successor_done if is_successor else None
    ):
        failures.append("handover latency mismatch")
    expected_durations = {
        "pool_ready_latency": pool_ready,
        "ordering_latency": (
            ordered - pool_ready
            if ordered is not None and pool_ready is not None
            else None
        ),
        "public_finalize_latency": (
            finalized - ordered
            if finalized is not None and ordered is not None
            else None
        ),
        "successor_latency": (
            successor_done - finalized
            if successor_done is not None and finalized is not None
            else None
        ),
        "local_completion_latency": expected_completion,
    }
    for field, expected_value in expected_durations.items():
        actual = durations.get(field)
        if expected_value is None:
            if actual is not None:
                failures.append(f"{field} mismatch")
        elif actual is None or abs(float(actual) - expected_value) > 1e-9:
            failures.append(f"{field} mismatch")
    if snapshot.get("is_holder") is True:
        sent = timestamps.get("t_contribution_sent")
        if sent is None or (pool_ready is not None and sent > pool_ready):
            failures.append("holder contribution timestamp invalid")
        elif durations.get("contribution_generation") != sent:
            failures.append("contribution generation duration mismatch")
    elif timestamps.get("t_contribution_sent") is not None:
        failures.append("non-holder has contribution timestamp")
    elif durations.get("contribution_generation") is not None:
        failures.append("non-holder has contribution duration")
    return failures


def _computation_metric(
    value: Optional[dict] = None,
    *,
    invocations: int = 0,
    cpu_seconds: float = 0.0,
) -> dict:
    source = dict(value or {})
    source["invocations"] = int(source.get("invocations", invocations))
    source["cpu_seconds"] = float(
        source.get("cpu_seconds", cpu_seconds)
    )
    if "records" in source:
        source["records"] = [dict(record) for record in source["records"]]
    return source


def build_a2dpss_computation_metrics(
    *,
    protocol: str,
    node: int,
    proposer_predicate: Optional[dict],
    receiver_predicate: Optional[dict],
    candidate_formation: Optional[dict],
    pvss_generation: Optional[dict],
    pvss_verification: Optional[dict],
    public_finalization: Optional[dict],
    successor_reconstruction: Optional[dict],
) -> dict:
    """Build the shared non-common-coin A²DPSS CPU accounting schema."""

    if protocol not in {"dag-tob", "acs-tob"}:
        raise ValueError("protocol must be dag-tob or acs-tob")
    proposer = _computation_metric(proposer_predicate)
    receiver = _computation_metric(receiver_predicate)
    predicate_total = {
        "invocations": proposer["invocations"] + receiver["invocations"],
        "cpu_seconds": (
            proposer["cpu_seconds"] + receiver["cpu_seconds"]
        ),
    }
    auxiliary = _computation_metric(candidate_formation)
    auxiliary["formal_predicate"] = False
    return {
        "schema": A2DPSS_COMPUTATION_METRICS_SCHEMA,
        "protocol": protocol,
        "node": int(node),
        "clock": "process_time",
        "predicate": {
            "roles": {
                "proposer": proposer,
                "receiver_or_committed_order": receiver,
            },
            "total": predicate_total,
        },
        "auxiliary": {
            "candidate_formation": auxiliary,
        },
        "resharing": {
            "pvss_generation": _computation_metric(pvss_generation),
            "pvss_verification": _computation_metric(pvss_verification),
            "public_finalization": _computation_metric(
                public_finalization
            ),
            "successor_reconstruction": _computation_metric(
                successor_reconstruction
            ),
        },
        "accounting_note": (
            "resharing detail can overlap full predicate CPU; common-coin "
            "cryptography and communication are excluded"
        ),
        "excluded": ["threshold_bls_common_coin"],
    }


def validate_a2dpss_computation_metrics(
    snapshot: dict,
    *,
    protocol: str,
    node: int,
) -> list[str]:
    """Return invariant failures for one serialized CPU summary."""

    failures: list[str] = []
    if not isinstance(snapshot, dict):
        return ["computation metrics missing"]
    expected = {
        "schema": A2DPSS_COMPUTATION_METRICS_SCHEMA,
        "protocol": protocol,
        "node": int(node),
        "clock": "process_time",
    }
    for field, value in expected.items():
        if snapshot.get(field) != value:
            failures.append(f"{field} mismatch")

    roles = (snapshot.get("predicate") or {}).get("roles") or {}
    if set(roles) != {"proposer", "receiver_or_committed_order"}:
        failures.append("predicate role set mismatch")
    role_invocations = 0
    role_cpu = 0.0
    for role in ("proposer", "receiver_or_committed_order"):
        metric = roles.get(role) or {}
        invocations = int(metric.get("invocations", -1))
        cpu_seconds = float(metric.get("cpu_seconds", -1.0))
        records = metric.get("records", [])
        if invocations < 0 or cpu_seconds < 0.0:
            failures.append(f"{role} predicate metric is negative")
            continue
        if records and invocations != len(records):
            failures.append(f"{role} predicate invocation/record mismatch")
        if any(float(record.get("cpu_seconds", -1.0)) < 0.0 for record in records):
            failures.append(f"{role} predicate record has negative CPU")
        if records and abs(
            cpu_seconds
            - sum(float(record["cpu_seconds"]) for record in records)
        ) > 1e-9:
            failures.append(f"{role} predicate CPU/record mismatch")
        role_invocations += invocations
        role_cpu += cpu_seconds

    total = (snapshot.get("predicate") or {}).get("total") or {}
    if int(total.get("invocations", -1)) != role_invocations:
        failures.append("predicate invocation total mismatch")
    if abs(float(total.get("cpu_seconds", -1.0)) - role_cpu) > 1e-9:
        failures.append("predicate CPU total mismatch")

    candidate = (
        (snapshot.get("auxiliary") or {}).get("candidate_formation")
        or {}
    )
    if (
        candidate.get("formal_predicate") is not False
        or int(candidate.get("invocations", -1)) < 0
        or float(candidate.get("cpu_seconds", -1.0)) < 0.0
    ):
        failures.append("candidate formation accounting mismatch")

    resharing = snapshot.get("resharing") or {}
    required = {
        "pvss_generation",
        "pvss_verification",
        "public_finalization",
        "successor_reconstruction",
    }
    if set(resharing) != required:
        failures.append("resharing metric set mismatch")
    for name in required:
        metric = resharing.get(name) or {}
        if (
            int(metric.get("invocations", -1)) < 0
            or float(metric.get("cpu_seconds", -1.0)) < 0.0
        ):
            failures.append(f"{name} metric is negative")
    if "common_coin" in resharing:
        failures.append("common coin entered resharing computation metrics")
    if snapshot.get("excluded") != ["threshold_bls_common_coin"]:
        failures.append("common-coin exclusion marker mismatch")
    return failures


def aggregate_a2dpss_computation_metrics(
    snapshots: Dict[int, dict],
) -> dict:
    """Sum validated per-node computation metrics for one trial."""

    roles = {
        role: {"invocations": 0, "cpu_seconds": 0.0}
        for role in ("proposer", "receiver_or_committed_order")
    }
    resharing = {
        name: {"invocations": 0, "cpu_seconds": 0.0}
        for name in (
            "pvss_generation",
            "pvss_verification",
            "public_finalization",
            "successor_reconstruction",
        )
    }
    candidate = {"invocations": 0, "cpu_seconds": 0.0}
    for snapshot in snapshots.values():
        node_roles = snapshot["predicate"]["roles"]
        for role, total in roles.items():
            metric = node_roles[role]
            total["invocations"] += int(metric["invocations"])
            total["cpu_seconds"] += float(metric["cpu_seconds"])
        metric = snapshot["auxiliary"]["candidate_formation"]
        candidate["invocations"] += int(metric["invocations"])
        candidate["cpu_seconds"] += float(metric["cpu_seconds"])
        for name, total in resharing.items():
            metric = snapshot["resharing"][name]
            total["invocations"] += int(metric["invocations"])
            total["cpu_seconds"] += float(metric["cpu_seconds"])
    return {
        "nodes": sorted(snapshots),
        "predicate": {
            "roles": roles,
            "total": {
                "invocations": sum(
                    metric["invocations"] for metric in roles.values()
                ),
                "cpu_seconds": sum(
                    metric["cpu_seconds"] for metric in roles.values()
                ),
            },
        },
        "auxiliary": {"candidate_formation": candidate},
        "resharing": resharing,
        "excluded": ["threshold_bls_common_coin"],
    }


@dataclass(frozen=True)
class WorkloadBatch:
    batch_id: str
    trigger: str
    instances: Tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict) -> "WorkloadBatch":
        batch = cls(
            batch_id=str(data["batch_id"]),
            trigger=str(data["trigger"]),
            instances=tuple(str(item) for item in data.get("instances", [])),
        )
        if batch.trigger not in _VALID_TRIGGERS:
            raise ValueError(f"unsupported workload trigger {batch.trigger!r}")
        if not batch.instances:
            raise ValueError(f"workload batch {batch.batch_id!r} is empty")
        if len(set(batch.instances)) != len(batch.instances):
            raise ValueError(f"workload batch {batch.batch_id!r} has duplicate instances")
        return batch

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "trigger": self.trigger,
            "instances": list(self.instances),
        }


@dataclass(frozen=True)
class ExperimentWorkloadTrace:
    trace_id: str
    messages_per_node: int
    batches: Tuple[WorkloadBatch, ...]

    @classmethod
    def dynamic_default(
        cls,
        *,
        messages_per_node: int = 1,
        trace_id: str = "atcaas-pipeline-dynamic-v1",
    ) -> "ExperimentWorkloadTrace":
        return cls.from_dict(
            {
                "trace_id": trace_id,
                "messages_per_node": messages_per_node,
                "batches": [
                    {
                        "batch_id": "early",
                        "trigger": START,
                        "instances": ["early-0"],
                    },
                    {
                        "batch_id": "late",
                        "trigger": INITIAL_PROPOSAL_FIXED,
                        "instances": ["late-0"],
                    },
                ],
            }
        )

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentWorkloadTrace":
        trace = cls(
            trace_id=str(data["trace_id"]),
            messages_per_node=int(data.get("messages_per_node", 1)),
            batches=tuple(
                WorkloadBatch.from_dict(item) for item in data.get("batches", [])
            ),
        )
        if not trace.trace_id:
            raise ValueError("workload trace_id cannot be empty")
        if trace.messages_per_node <= 0:
            raise ValueError("messages_per_node must be positive")
        if not trace.batches:
            raise ValueError("workload trace must contain at least one batch")
        batch_ids = [batch.batch_id for batch in trace.batches]
        instances = [instance for batch in trace.batches for instance in batch.instances]
        if len(set(batch_ids)) != len(batch_ids):
            raise ValueError("workload trace has duplicate batch ids")
        if len(set(instances)) != len(instances):
            raise ValueError("workload trace has duplicate qid instances")
        return trace

    @classmethod
    def from_file(cls, path: str) -> "ExperimentWorkloadTrace":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "messages_per_node": self.messages_per_node,
            "batches": [batch.to_dict() for batch in self.batches],
        }

    def qid(self, instance: str) -> str:
        self.batch_for_instance(instance)
        return Qid.build(
            "demo",
            service="atcaas-r6-workload",
            trace=self.trace_id,
            instance=instance,
        ).encode()

    def mid(self, instance: str, sender: int, index: int) -> str:
        self.batch_for_instance(instance)
        return f"{self.trace_id}:{instance}:node:{int(sender)}:msg:{int(index)}"

    def payload(self, instance: str, sender: int, index: int) -> bytes:
        batch = self.batch_for_instance(instance)
        return json.dumps(
            {
                "kind": "demo",
                "qid": self.qid(instance),
                "trace_id": self.trace_id,
                "batch_id": batch.batch_id,
                "instance": instance,
                "sender": int(sender),
                "index": int(index),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def batches_for_trigger(self, trigger: str) -> Tuple[WorkloadBatch, ...]:
        if trigger not in _VALID_TRIGGERS:
            raise ValueError(f"unsupported workload trigger {trigger!r}")
        return tuple(batch for batch in self.batches if batch.trigger == trigger)

    def batch_for_instance(self, instance: str) -> WorkloadBatch:
        for batch in self.batches:
            if instance in batch.instances:
                return batch
        raise KeyError(f"workload trace has no instance {instance!r}")

    def qids(self) -> Tuple[str, ...]:
        return tuple(self.qid(instance) for batch in self.batches for instance in batch.instances)


def load_dynamic_workload(path: Optional[str], messages_per_node: int = 1):
    if path:
        return ExperimentWorkloadTrace.from_file(path), "recorded"
    return (
        ExperimentWorkloadTrace.dynamic_default(
            messages_per_node=messages_per_node,
        ),
        "builtin",
    )


class QueryLifecycleMetrics:
    """First-observation timestamps for each logical qid at one observer."""

    def __init__(self, trace: ExperimentWorkloadTrace):
        self.trace = trace
        self._records: Dict[str, dict] = {}
        for batch in trace.batches:
            for instance in batch.instances:
                qid = trace.qid(instance)
                self._records[qid] = {
                    "qid": qid,
                    "instance": instance,
                    "batch_id": batch.batch_id,
                    "trigger": batch.trigger,
                    "proposal_boundary_time": None,
                    "proposal_boundary_unit": None,
                    "request_release_time": None,
                    "pool_ready_time": None,
                    # Compatibility alias.  From this revision onward ready_time
                    # means J(qid,S)=true, never merely request injection.
                    "ready_time": None,
                    "first_inclusion_time": None,
                    "first_inclusion_unit": None,
                    "delivery_time": None,
                    "delivery_unit": None,
                }

    def mark_proposal_boundary(self, qid: str, when: float, unit) -> None:
        record = self._record(qid)
        if record["proposal_boundary_time"] is None:
            record["proposal_boundary_time"] = float(when)
            record["proposal_boundary_unit"] = unit

    def mark_released(self, qid: str, when: float) -> None:
        self._mark_first(qid, "request_release_time", float(when))

    def mark_pool_ready(self, qid: str, when: float) -> None:
        value = float(when)
        self._mark_first(qid, "pool_ready_time", value)
        self._mark_first(qid, "ready_time", value)

    def mark_ready(self, qid: str, when: float) -> None:
        """Compatibility alias for callers that already proved J(qid,S)."""

        self.mark_pool_ready(qid, when)

    def mark_included(self, qid: str, when: float, unit) -> None:
        record = self._record(qid)
        if record["first_inclusion_time"] is None:
            record["first_inclusion_time"] = float(when)
            record["first_inclusion_unit"] = unit

    def mark_delivered(self, qid: str, when: float, unit) -> None:
        record = self._record(qid)
        if record["delivery_time"] is None:
            record["delivery_time"] = float(when)
            record["delivery_unit"] = unit

    def _record(self, qid: str) -> dict:
        if qid not in self._records:
            raise KeyError(f"qid is absent from workload trace: {qid}")
        return self._records[qid]

    def _mark_first(self, qid: str, field: str, value) -> None:
        record = self._record(qid)
        if record[field] is None:
            record[field] = value

    def to_list(self) -> list:
        out = []
        for record in self._records.values():
            item = dict(record)
            released = item["request_release_time"]
            ready = item["pool_ready_time"]
            included = item["first_inclusion_time"]
            delivered = item["delivery_time"]
            item["release_to_pool_ready_sec"] = (
                ready - released
                if released is not None and ready is not None
                else None
            )
            item["ready_to_inclusion_sec"] = (
                included - ready if ready is not None and included is not None else None
            )
            item["ready_to_delivery_sec"] = (
                delivered - ready if ready is not None and delivered is not None else None
            )
            item["inclusion_to_delivery_sec"] = (
                delivered - included
                if included is not None and delivered is not None
                else None
            )
            item["release_to_delivery_sec"] = (
                delivered - released
                if released is not None and delivered is not None
                else None
            )
            out.append(item)
        return out

    def summary(self) -> dict:
        records = self.to_list()
        released = [
            item["request_release_time"]
            for item in records
            if item["request_release_time"] is not None
        ]
        ready = [
            item["pool_ready_time"]
            for item in records
            if item["pool_ready_time"] is not None
        ]
        delivered = [
            item["delivery_time"] for item in records if item["delivery_time"] is not None
        ]
        batches: Dict[str, dict] = {}
        for item in records:
            aggregate = batches.setdefault(
                item["batch_id"],
                {
                    "request_release_time": None,
                    "pool_ready_time": None,
                    "delivery_time": None,
                },
            )
            if item["request_release_time"] is not None:
                aggregate["request_release_time"] = (
                    item["request_release_time"]
                    if aggregate["request_release_time"] is None
                    else min(
                        aggregate["request_release_time"],
                        item["request_release_time"],
                    )
                )
            if item["pool_ready_time"] is not None:
                aggregate["pool_ready_time"] = (
                    item["pool_ready_time"]
                    if aggregate["pool_ready_time"] is None
                    else min(
                        aggregate["pool_ready_time"],
                        item["pool_ready_time"],
                    )
                )
            if item["delivery_time"] is not None:
                aggregate["delivery_time"] = (
                    item["delivery_time"]
                    if aggregate["delivery_time"] is None
                    else max(aggregate["delivery_time"], item["delivery_time"])
                )
        batch_ids = [batch.batch_id for batch in self.trace.batches]
        pipelining_overlap = None
        if len(batch_ids) >= 2:
            first_delivery = batches[batch_ids[0]]["delivery_time"]
            second_ready = batches[batch_ids[1]]["pool_ready_time"]
            if first_delivery is not None and second_ready is not None:
                pipelining_overlap = max(0.0, first_delivery - second_ready)
        return {
            "queries": records,
            "makespan_sec": (
                max(delivered) - min(released)
                if released and delivered
                else None
            ),
            "ready_makespan_sec": (
                max(delivered) - min(ready) if ready and delivered else None
            ),
            "pipelining_overlap_sec": pipelining_overlap,
            "later_batch_ready_before_first_delivery": (
                pipelining_overlap is not None and pipelining_overlap > 0
            ),
        }
