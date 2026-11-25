"""Collect paired pure-A2DPSS handovers using local elapsed times only.

The proposal-boundary window is an experimental condition, separate from
protocol correctness. A successful run with an invalid window remains visible.
"""

from __future__ import annotations

import math
from pathlib import Path
import statistics

from beaver.a2dpss_benchmark import read
from beaver.async_algorand.model import digest
from beaver.experiment_trace import INITIAL_PROPOSAL_FIXED, START


SCHEMA = "thesis-ch4-a2dpss-pipeline/v1"
CPU_STAGES = (
    "ready_check", "candidate_formation", "formal_predicate",
    "public_finalization", "successor_reconstruction",
)
LATENCIES = (
    "release_to_pool_ready_sec", "ready_to_inclusion_sec",
    "ready_to_delivery_sec", "ready_to_completion_sec",
    "release_to_completion_sec",
)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is missing or non-numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _stats(values):
    values = list(values)
    if not values:
        return None
    return {
        "count": len(values), "mean": statistics.fmean(values),
        "median": statistics.median(values), "min": min(values),
        "max": max(values),
    }


def normalize_result(manifest, protocol: str, node: int, raw: dict) -> dict:
    from beaver.a2dpss_pipeline_workload import A2DPSSPairedWorkload

    fixture = A2DPSSPairedWorkload.from_file(
        str(Path(manifest["bundle_root"]) / "public/fixture.json")
    )
    expected_qids = set(fixture.qids())
    expected = {
        "workload_trace": manifest["workload"],
        "a2dpss_pipeline_fixture_digest": manifest["fixture_digest"],
        "a2dpss_pipeline": True,
        "coin_mode": "threshold-bls",
        "coin_public_key_digest": manifest["coin_public_key_digest"],
        "role_trace_digest": manifest["role_trace_digest"],
        "cleanup_completed": True,
    }
    if protocol not in ("dag", "acs"):
        raise ValueError("unknown pipeline protocol")
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"pipeline {key} mismatch")
    backend = raw.get("vrf_backend", {})
    if backend.get("required") is not True or any(
        backend.get("implementations", {}).get(name) is not True
        for name in ("beaver.vrf", "beaver.yosorbc")
    ):
        raise ValueError("native VRF preflight missing")

    records = raw.get("query_lifecycle", {}).get("queries", [])
    events = raw.get("a2dpss_pipeline_events", [])
    for name, items in (("lifecycle", records), ("completion", events)):
        if len(items) != len(expected_qids) or {x.get("qid") for x in items} != expected_qids:
            raise ValueError(f"{name} must contain each expected qid exactly once")
    by_qid = {event["qid"]: event for event in events}
    early = [item for item in records if item["trigger"] == START]
    late = [item for item in records if item["trigger"] == INITIAL_PROPOSAL_FIXED]
    if len(early) != manifest["initial_query_count"] or len(late) != manifest["late_query_count"]:
        raise ValueError("early/late partition mismatch")
    early_delivery = min(_number(x.get("delivery_time"), "early delivery") for x in early)
    early_release = min(_number(x.get("request_release_time"), "early release") for x in early)
    queries, public_results, window_issues = [], [], []
    for item in records:
        qid = item["qid"]
        context = fixture.context_for_qid(qid)
        event = by_qid[qid]
        is_successor = node in context.successor_role_indices
        if event.get("event") != "public_finalized" or event.get("context_digest") != context.context_digest:
            raise ValueError("public result context mismatch")
        if event.get("private_reconstructed") is not is_successor:
            raise ValueError("successor reconstruction missing or role mismatch")
        if "operation_output" in event or "accepted_bundle_digests" in event:
            raise ValueError("operation-coupled output in a pure A2DPSS experiment")
        contribution_digests = event.get("accepted_contribution_digests", [])
        commitments = event.get("successor_commitment_vector", [])
        if len(contribution_digests) != context.service_t + 1 or not commitments:
            raise ValueError("incomplete A2DPSS public result")
        release = _number(item.get("request_release_time"), "release")
        ready = _number(item.get("pool_ready_time"), "ready")
        included = _number(item.get("first_inclusion_time"), "inclusion")
        ordered = _number(item.get("delivery_time"), "ordering delivery")
        public = _number(event.get("t_public"), "public finalization")
        if release > ready or ordered > public:
            raise ValueError(f"non-monotonic local timeline for {item['instance']}")
        if not ready <= included <= ordered:
            window_issues.append(f"{item['instance']}: readiness/inclusion/ordering window not established")
        successor = event.get("t_successor")
        if is_successor:
            successor = _number(successor, "successor completion")
            if successor < public:
                raise ValueError("successor completion precedes public finalization")
        elif successor is not None:
            raise ValueError("non-successor has private completion timestamp")
        is_late = item["trigger"] == INITIAL_PROPOSAL_FIXED
        boundary = item.get("proposal_boundary_time")
        if is_late:
            boundary = _number(boundary, "proposal boundary")
            if boundary > release:
                window_issues.append(f"{item['instance']}: late release precedes proposal boundary")
            if ready >= early_delivery:
                window_issues.append(f"{item['instance']}: late readiness does not precede early delivery")
        queries.append({
            **item,
            "t_public": public, "t_successor": successor,
            "is_successor": is_successor,
            "release_to_pool_ready_sec": ready - release,
            "ready_to_inclusion_sec": included - ready,
            "ready_to_delivery_sec": ordered - ready,
            "ready_to_completion_sec": successor - ready if is_successor else None,
            "release_to_completion_sec": successor - release if is_successor else None,
            "ready_before_early_delivery": ready < early_delivery if is_late else None,
            "included_before_early_delivery": included < early_delivery if is_late else None,
        })
        public_results.append({key: event[key] for key in (
            "qid", "context_digest", "accepted_contribution_digests",
            "successor_commitment_vector",
        )})

    cpu = raw.get("a2dpss_pipeline_cpu", {})
    if cpu.get("clock") != "process_time":
        raise ValueError("pipeline CPU clock must be process_time")
    stages = cpu.get("stages", {})
    cpu_total = sum(_number(stages.get(stage, {}).get("cpu_seconds"), stage) for stage in CPU_STAGES)
    if not math.isclose(cpu_total, _number(cpu.get("total_cpu_seconds"), "CPU total"), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("non-overlapping CPU stage sum mismatch")
    wire = raw.get("benchmark_protocol_wire_bytes")
    if type(wire) is not int or wire < 0:
        raise ValueError("protocol wire bytes missing")
    gates = raw.get("a2dpss_pipeline_gates", [])
    if not gates:
        raise ValueError("controlled readiness gate timing missing")
    for gate in gates:
        entered, exited = (_number(gate.get(key), f"gate {key}") for key in ("enter", "exit"))
        if exited < entered or not math.isclose(exited - entered, _number(gate.get("duration"), "gate duration"), abs_tol=1e-9):
            raise ValueError("gate timing mismatch")

    def completion(batch):
        selected = [q for q in queries if batch is None or q["trigger"] == batch]
        if not selected or any(q["t_successor"] is None for q in selected):
            return None
        return max(q["t_successor"] for q in selected) - early_release

    return {
        "schema": SCHEMA, "status": "ok", "protocol": protocol, "node": node,
        "trial_id": manifest["trial_id"], "bundle_digest": manifest["bundle_digest"],
        "fixture_digest": manifest["fixture_digest"], "clock": "perf_counter-relative",
        "queries": queries,
        "public_results": sorted(public_results, key=lambda x: x["qid"]),
        "late_window_valid": not window_issues,
        **({"experimental_condition_failures": window_issues} if window_issues else {}),
        "early_completion_sec": completion(START), "batch_completion_sec": completion(None),
        "online_adpss_cpu_seconds": cpu_total, "cpu_stages": stages,
        "protocol_wire_bytes": wire,
        "protocol_wire_bytes_by_category": raw.get("benchmark_protocol_wire_bytes_by_category", {}),
        "gate_wait_seconds": sum(gate["duration"] for gate in gates),
        "ordered_units": int(raw.get("committed_wave", 0)) if protocol == "dag" else len(raw.get("ledger", [])),
        "raw": raw,
    }


def collect(directory, results, protocol: str) -> dict:
    from beaver.a2dpss_pipeline_benchmark import load_bundle

    directory = Path(directory).resolve()
    manifest = {**load_bundle(directory), "bundle_root": str(directory)}
    records, failures = [], []
    for node in range(manifest["parameters"]["M"]):
        try:
            stored = read(Path(results) / protocol / f"node-{node}.json")
            for key, value in (("node", node), ("protocol", protocol), ("bundle_digest", manifest["bundle_digest"])):
                if stored.get(key) != value:
                    raise ValueError(f"result {key} mismatch")
            if stored.get("status") != "ok":
                raise ValueError(f"{stored.get('status')}: {stored.get('reason', '')}")
            checked = normalize_result(manifest, protocol, node, stored["raw"])
            if checked != {k: v for k, v in stored.items() if k != "machine"}:
                raise ValueError("stored derived metrics differ from raw events")
            records.append(checked)
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(f"node {node}: {error}")
    if records and len({digest(r["public_results"]) for r in records}) != 1:
        failures.append("observers disagree on A2DPSS public results")

    # First take each request's slowest relevant local duration. Never subtract
    # timestamps recorded on different machines, or call M*Q independent trials.
    grouped = {}
    for record in records:
        for query in record["queries"]:
            grouped.setdefault(query["qid"], []).append(query)
    per_qid = []
    for qid, group in sorted(grouped.items()):
        per_qid.append({
            "qid": qid, "instance": group[0]["instance"], "trigger": group[0]["trigger"],
            **{metric: max((q[metric] for q in group if q[metric] is not None), default=None) for metric in LATENCIES},
        })
    late = [q for q in per_qid if q["trigger"] == INITIAL_PROPOSAL_FIXED]
    late_observations = [q for r in records for q in r["queries"] if q["trigger"] == INITIAL_PROPOSAL_FIXED]
    window = not failures and bool(late_observations) and all(r["late_window_valid"] for r in records)
    total_cpu = sum(r["online_adpss_cpu_seconds"] for r in records) if not failures else None
    total_wire = sum(r["protocol_wire_bytes"] for r in records) if not failures else None
    count = manifest["query_count"]
    category_names = {name for r in records for name in r["protocol_wire_bytes_by_category"]}
    return {
        "schema": SCHEMA, "status": "failed" if failures else "ok", "failures": failures,
        "protocol": protocol, "trial_id": manifest["trial_id"],
        "bundle_digest": manifest["bundle_digest"], "fixture_digest": manifest["fixture_digest"],
        "parameters": manifest["parameters"], "query_count": count,
        "initial_query_count": manifest["initial_query_count"], "late_query_count": manifest["late_query_count"],
        "participants": len(records), "late_window_valid": window,
        "experimental_condition": "established" if window else "not-established",
        "experimental_condition_failures": [
            f"node {r['node']}: {reason}"
            for r in records for reason in r.get("experimental_condition_failures", [])
        ],
        "measurement_scope": manifest.get("measurement_scope"),
        "aggregation": "per-qid maximum of local observer/successor durations; request distribution within one trial",
        "late_latency_sec": {metric: _stats(q[metric] for q in late if q[metric] is not None) for metric in LATENCIES} if not failures else None,
        "early_completion_sec": max((r["early_completion_sec"] for r in records if r["early_completion_sec"] is not None), default=None) if not failures else None,
        "batch_completion_sec": max((r["batch_completion_sec"] for r in records if r["batch_completion_sec"] is not None), default=None) if not failures else None,
        "online_adpss_cpu_seconds": total_cpu,
        "online_adpss_cpu_seconds_per_handover": total_cpu / count if total_cpu is not None else None,
        "cpu_stage_seconds": {stage: sum(r["cpu_stages"][stage]["cpu_seconds"] for r in records) for stage in CPU_STAGES} if not failures else None,
        "protocol_wire_bytes": total_wire,
        "protocol_wire_bytes_per_handover": total_wire / count if total_wire is not None else None,
        "protocol_wire_bytes_by_category": {name: sum(r["protocol_wire_bytes_by_category"].get(name, 0) for r in records) for name in sorted(category_names)} if not failures else None,
        "offline_fixture_generation_seconds": manifest.get("fixture_generation_seconds"),
        "offline_fixture_generation_cpu_seconds": manifest.get("fixture_generation_cpu_seconds"),
        "offline_holder_metrics": manifest.get("holder_generation"),
        "late_inclusion_before_early_delivery_fraction": sum(q["included_before_early_delivery"] for q in late_observations) / len(late_observations) if late_observations else None,
        "late_inclusion_observations": len(late_observations),
        "gate_wait_seconds_per_node": _stats(r["gate_wait_seconds"] for r in records),
        "public_results_digest": digest(records[0]["public_results"]) if records else None,
        "per_qid": per_qid,
        "per_node": [{k: v for k, v in r.items() if k != "raw"} for r in records],
    }


def compare(directory, dag_results, acs_results) -> dict:
    dag = collect(directory, dag_results, "dag")
    acs = collect(directory, acs_results, "acs")
    failures = [f"{name}: {arm['failures']}" for name, arm in (("dag", dag), ("acs", acs)) if arm["status"] != "ok"]
    paired = bool(dag["public_results_digest"]) and dag["public_results_digest"] == acs["public_results_digest"]
    window = dag["late_window_valid"] and acs["late_window_valid"]
    comparable = not failures and paired and window
    dag_fraction = dag["late_inclusion_before_early_delivery_fraction"]
    acs_fraction = acs["late_inclusion_before_early_delivery_fraction"]
    ratios = {}
    if comparable:
        for metric in ("batch_completion_sec", "online_adpss_cpu_seconds_per_handover", "protocol_wire_bytes_per_handover"):
            denominator = dag[metric]
            ratios[metric] = acs[metric] / denominator if denominator else None
        d = dag["late_latency_sec"]["ready_to_completion_sec"]["median"]
        a = acs["late_latency_sec"]["ready_to_completion_sec"]["median"]
        ratios["late_ready_to_completion_median_sec"] = a / d if d else None
    return {
        "schema": "a2dpss-pipeline-comparison/v1",
        "status": "failed" if failures else "ok", "failures": failures,
        "comparison_valid": comparable, "late_window_valid": window,
        "paired_public_results_equal": paired,
        "pairing_note": None if paired else "Different public outputs require checking selected legal subsets; this alone is not a cross-protocol safety failure.",
        "pipeline_admission_advantage_observed": dag_fraction > acs_fraction and acs_fraction == 0.0 if comparable else None,
        "acs_over_dag_ratios": ratios,
        "dag": {k: v for k, v in dag.items() if k != "per_node"},
        "acs": {k: v for k, v in acs.items() if k != "per_node"},
    }
