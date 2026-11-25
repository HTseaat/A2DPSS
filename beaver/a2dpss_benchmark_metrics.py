"""Non-overlapping application CPU accounting for the three-backend entry.

These are application work counters, not the CPU time of the entire process.
Missing historical counters remain unknown; component timings cannot recover
the CPU of an uninstrumented enclosing operation.
"""
from __future__ import annotations

import math


CPU_SCHEMA = "a2dpss-three-backend-cpu/v1"
CPU_STAGES = (
    "pvss_generation", "candidate_formation", "auxiliary_validation",
    "formal_predicate", "public_finalization", "successor_reconstruction",
)


def _metric(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("CPU stage must be an object")
    if "cpu_seconds" not in value or "invocations" not in value:
        return None
    cpu, calls = value["cpu_seconds"], value["invocations"]
    if (type(cpu) not in (int, float) or not math.isfinite(cpu) or cpu < 0
            or type(calls) is not int or calls < 0):
        raise ValueError("CPU stage requires finite nonnegative CPU and integer invocations")
    return {"cpu_seconds": float(cpu), "invocations": calls}


def _sum_metrics(values):
    if any(value is None for value in values):
        return None
    return {"cpu_seconds": sum(value["cpu_seconds"] for value in values),
            "invocations": sum(value["invocations"] for value in values)}


def _summary(stages):
    missing = [name for name in CPU_STAGES if stages[name] is None]
    return dict(schema=CPU_SCHEMA, clock="process_time",
        available=len(missing) < len(CPU_STAGES), complete=not missing,
        stages=stages, missing_stages=missing,
        total_cpu_seconds=None if missing else sum(v["cpu_seconds"] for v in stages.values()),
        accounting="sum disjoint application stages; PVSS verification detail is inside full J",
        excluded=["offline_setup", "paired_fixture_generation", "generic_transport",
                  "consensus_control_and_coin_crypto", "serialization_outside_measured_stages"])


def normalize_cpu(raw, protocol, node):
    """Extract measured stages, preserving unavailable historical measurements."""
    stages = {name: None for name in CPU_STAGES}
    if protocol in ("dag", "acs"):
        source = raw.get("a2dpss_computation_metrics")
        if source is not None:
            if (not isinstance(source, dict)
                    or source.get("schema") != "a2dpss-computation-metrics-v1"
                    or source.get("protocol") != {"dag": "dag-tob", "acs": "acs-tob"}[protocol]
                    or source.get("node") != node or source.get("clock") != "process_time"):
                raise ValueError("A2DPSS CPU source identity/schema mismatch")
            predicate = source.get("predicate") or {}
            roles = predicate.get("roles") or {}
            formal = _sum_metrics([_metric(roles.get(name))
                for name in ("proposer", "receiver_or_committed_order")])
            reported = _metric(predicate.get("total"))
            if formal is not None and reported is not None and (
                    formal["invocations"] != reported["invocations"] or not math.isclose(
                        formal["cpu_seconds"], reported["cpu_seconds"], abs_tol=1e-9)):
                raise ValueError("formal predicate CPU total differs from role counters")
            stages["formal_predicate"] = formal if reported is not None else None
            stages["candidate_formation"] = _metric(
                (source.get("auxiliary") or {}).get("candidate_formation"))
            resharing = source.get("resharing") or {}
            for name in ("pvss_generation", "public_finalization", "successor_reconstruction"):
                stages[name] = _metric(resharing.get(name))
            # Neither existing runner has Algorand's separate receive-side
            # contribution prevalidation. Their verification is inside J.
            stages["auxiliary_validation"] = {"cpu_seconds": 0.0, "invocations": 0}
    elif protocol == "algorand":
        source = raw.get("a2dpss_benchmark_cpu_stages")
        if source is not None:
            if not isinstance(source, dict) or set(source) != set(CPU_STAGES):
                raise ValueError("Algorand application CPU stage set mismatch")
            stages = {name: _metric(source[name]) for name in CPU_STAGES}
        else:
            # Old Algorand results retained these components, but neither full
            # J nor holder generation CPU. Never infer those from wall time.
            metrics = raw.get("metrics") or {}
            if "a2_receive_validation_cpu_seconds" in metrics:
                stages["auxiliary_validation"] = _metric({
                    "cpu_seconds": metrics["a2_receive_validation_cpu_seconds"],
                    "invocations": metrics.get("a2_receive_validation_calls")})
            for name, key in (("public_finalization", "a2dpss_finalization_metrics"),
                              ("successor_reconstruction", "a2dpss_reconstruction_metrics")):
                stages[name] = _metric((raw.get(key) or {}).get("total"))
    else:
        raise ValueError("unknown benchmark protocol")
    return _summary(stages)


def aggregate_cpu(records, expected_nodes):
    """Sum only complete node sets; one missing counter prevents a total."""
    nodes = [record["node"] for record in records]
    complete_nodes = len(nodes) == expected_nodes and set(nodes) == set(range(expected_nodes))
    stages = {}
    for name in CPU_STAGES:
        values = [record["adpss_cpu"]["stages"][name] for record in records]
        stages[name] = _sum_metrics(values) if complete_nodes else None
    result = _summary(stages)
    result.update(available=any(record["adpss_cpu"]["available"] for record in records),
                  participants=len(nodes), expected_participants=expected_nodes,
                  incomplete_nodes=[record["node"] for record in records
                                    if not record["adpss_cpu"]["complete"]])
    return result
