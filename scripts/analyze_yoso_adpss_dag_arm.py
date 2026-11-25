"""Analyze a single independent ADPSS run, including DAG-only observers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from beaver.yoso_adpss_distributed import valid_timeline
from beaver.yoso_adpss_experiment import (
    BHK22_DAG_NODE_METRICS_SCHEMA,
    NODE_METRICS_SCHEMA,
)
from beaver.yoso_adpss_workload import load_yoso_adpss_public_fixture, verify_same_initial_workload
from scripts.analyze_yoso_adpss_dag_trial import _cpu_leaf_total, _walk_private


def cpu_breakdown(protocol, metrics):
    # Formal predicate CPU includes PVSS/DLEQ validation. Do not sum both the
    # enclosing operation and its diagnostics, or roles and their total again.
    if protocol == "a2dpss":
        resharing = metrics.get("resharing") or {}
        stages = {k: resharing.get(k) for k in (
            "pvss_generation", "public_finalization", "successor_reconstruction")}
        stages["formal_predicate"] = (metrics.get("predicate") or {}).get("total")
    elif protocol == "ru22":
        stages = {"holder_generation": metrics.get("holder_generation"),
                  "formal_predicate": (metrics.get("formal") or {}).get("roles"),
                  "public_finalization": metrics.get("public_finalization"),
                  "successor_reconstruction": (metrics.get("successor_reconstruction") or {}).get("total")}
    else:
        stages = metrics
    return {name: _cpu_leaf_total(value) for name, value in stages.items()}


def analyze_arm(fixture, protocol, nodes, *, experiment_digest=None):
    manifest, arms = load_yoso_adpss_public_fixture(Path(fixture))
    if protocol not in arms or not verify_same_initial_workload(manifest, arms):
        raise ValueError("invalid protocol/common workload")
    if protocol == "ru22" and any(n.get("schema") in ("ru22-native-node-metrics/v2", "ru22-native-node-metrics/v3") for n in nodes):
        from scripts.analyze_ru22_native import analyze_native
        result = analyze_native(fixture, nodes, experiment_digest=experiment_digest)
        if (result["common_workload_digest"] != manifest.digest()
                or result["arm_manifest_digest"] != arms["ru22"].digest()):
            raise ValueError("native RU22 does not match the comparison workload")
        result["parameters"] = manifest.parameters.to_config_obj()
        from beaver.baselines.ru22_experiment import RU22DAGExperimentManifest
        base = RU22DAGExperimentManifest.from_obj(arms["ru22"].protocol_manifest)
        result["vps_shape"] = "multi" if any(len(v) > 1 for v in base.source_vps_views.values()) else "single"
        return result
    M = manifest.parameters.M
    if len(nodes) != M or {n.get("node_id") for n in nodes} != set(range(M)):
        raise ValueError("missing or duplicate node summaries")
    successors = set(manifest.successor_role_to_node.values())
    blocks = 1
    bhk_committees = None
    bhk_committee_summary = None
    if protocol == "bhk22":
        path = arms[protocol].protocol_manifest["path_mode"]
        if path not in ("optimistic", "one-accusation"):
            raise ValueError("unknown BHK path")
        blocks = 2 if path == "optimistic" else 3
        bhk_committees = (
            (arms[protocol].protocol_manifest.get("context") or {}).get(
                "committees"
            )
            or {}
        )
        if set(bhk_committees) != {
            "dealer", "verifier", "responder", "shareholder"
        }:
            raise ValueError("BHK manifest has an incomplete committee set")
        bhk_committee_summary = {}
        for role, expected_mode in (
            ("dealer", "externally-established"),
            ("verifier", "vrf-bernoulli-exact-size"),
            ("responder", "vrf-bernoulli-exact-size"),
            ("shareholder", "externally-established"),
        ):
            committee = bhk_committees.get(role) or {}
            attempt = committee.get("selection_attempt")
            role_to_node = committee.get("role_to_node") or {}
            if (
                committee.get("version") != 2
                or committee.get("selection_mode") != expected_mode
                or isinstance(attempt, bool)
                or not isinstance(attempt, int)
                or attempt < 0
                or (expected_mode == "externally-established" and attempt != 0)
                or set(role_to_node) != {
                    str(role_id) for role_id in range(manifest.parameters.n_r)
                }
                or len(set(role_to_node.values())) != manifest.parameters.n_r
            ):
                raise ValueError(f"invalid BHK {role} committee selection")
            bhk_committee_summary[role] = {
                "selection_mode": expected_mode,
                "selection_attempt": attempt,
                "role_to_node": role_to_node,
            }
    result_digests = set()
    latencies = []
    cpu = 0.0
    cpu_stages = {}
    totals = {k: 0 for k in ("application_payload_bytes", "total_network_bytes",
                             "yoso_rbc_envelope_bytes", "dag_protocol_bytes", "common_coin_bytes")}
    for node in nodes:
        if (node.get("schema") != NODE_METRICS_SCHEMA or node.get("protocol") != protocol
                or node.get("status") != "success" or node.get("failure") is not None):
            raise ValueError(f"node {node.get('node_id')} did not succeed")
        if (node.get("common_workload_digest") != manifest.digest()
                or node.get("arm_manifest_digest") != arms[protocol].digest()):
            raise ValueError("node workload/arm digest mismatch")
        if experiment_digest is not None and node.get("experiment_digest") != experiment_digest:
            raise ValueError("node experiment digest mismatch")
        if _walk_private(node):
            raise ValueError("private material in public summary")
        if not (node.get("cleanup") or {}).get("completed"):
            raise ValueError("node cleanup incomplete")
        timeline = node.get("timeline") or {}
        if not valid_timeline(timeline):
            raise ValueError("invalid node timeline")
        done = timeline.get("t_successor_done")
        if node["node_id"] in successors:
            if done is None:
                raise ValueError("expected successor did not reconstruct")
            latencies.append(done)
        elif done is not None:
            raise ValueError("DAG-only observer unexpectedly produced a successor result")
        metric = node.get("protocol_metrics") or {}
        if metric.get("formal_cache_hits") != 0:
            raise ValueError("formal predicate cache hits must be zero")
        result_digests.add(metric.get("public_result_digest"))
        if protocol == "bhk22" and (
            metric.get("raw_metrics_schema") != BHK22_DAG_NODE_METRICS_SCHEMA
            or metric.get("committee_selection") != bhk_committee_summary
            or metric.get("rbc_metrics") is None
            or metric.get("application_verification_metrics") is None
            or metric.get("role_metrics") is None
            or metric.get("completion_mode") != "encrypted-carry-forward"
            or metric.get("public_commitment_kind") != "encrypted"
            or metric.get("path_mode") != path
        ):
            raise ValueError("BHK completion/path mismatch")
        communication = node.get("communication") or {}
        if communication.get("committed_application_block_count") != blocks:
            raise ValueError("application block count mismatch")
        for key in totals:
            value = communication.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"missing/invalid communication metric: {key}")
            totals[key] += value
        if communication["total_network_bytes"] != sum(communication[k] for k in (
                "yoso_rbc_envelope_bytes", "dag_protocol_bytes", "common_coin_bytes")):
            raise ValueError("network byte partition mismatch")
        if protocol == "bhk22":
            rbc_by_phase = metric["rbc_metrics"]
            proof_total = sum(
                int((phase_metrics or {}).get("outbound_role_vrf_proof_bytes", 0))
                for phase_metrics in rbc_by_phase.values()
            )
            if communication.get("committee_vrf_proof_bytes") != proof_total:
                raise ValueError("BHK committee VRF proof byte total mismatch")
            active_vrf_phases = {"verify": "verifier"}
            if path == "one-accusation":
                active_vrf_phases["response"] = "responder"
            for phase, role in active_vrf_phases.items():
                phase_metrics = rbc_by_phase.get(phase) or {}
                selected_nodes = set(
                    bhk_committees[role]["role_to_node"].values()
                )
                locally_selected = node["node_id"] in selected_nodes
                outbound_messages = phase_metrics.get("outbound_messages")
                outbound_proof_bytes = phase_metrics.get(
                    "outbound_role_vrf_proof_bytes"
                )
                if (
                    outbound_messages != int(locally_selected)
                    or isinstance(outbound_proof_bytes, bool)
                    or not isinstance(outbound_proof_bytes, int)
                    or (outbound_proof_bytes > 0) != locally_selected
                    or phase_metrics.get("delivered_messages")
                    != manifest.parameters.n_r
                    or not isinstance(
                        phase_metrics.get("delivered_role_vrf_proof_bytes"), int
                    )
                    or phase_metrics["delivered_role_vrf_proof_bytes"] <= 0
                    or phase_metrics.get("rejected_role_vrf_proofs") != 0
                ):
                    raise ValueError(
                        f"incomplete BHK {phase} committee VRF evidence"
                    )
        breakdown = cpu_breakdown(protocol, node.get("cpu") or {})
        node_cpu = sum(breakdown.values())
        if not math.isfinite(node_cpu):
            raise ValueError("non-finite CPU metric")
        cpu += node_cpu  # DAG-only BHK roles legitimately contribute zero here.
        for name, value in breakdown.items():
            cpu_stages[name] = cpu_stages.get(name, 0.0) + value
    if len(result_digests) != 1 or None in result_digests or not latencies:
        raise ValueError("missing/disagreeing public result or successor set")
    for key in (
        "cryptographic_proof_bytes",
        "recipient_ciphertext_bytes",
        "committee_vrf_proof_bytes",
    ):
        values = [n["communication"].get(key) for n in nodes]
        totals[key] = sum(values) if all(isinstance(v, int) and v >= 0 for v in values) else None
    result = dict(schema="yoso-adpss-dag-arm-summary/v1", status="success",
                  protocol=protocol, node_count=M, successor_nodes=sorted(successors),
                  common_workload_digest=manifest.digest(), arm_manifest_digest=arms[protocol].digest(),
                  public_result_digest=next(iter(result_digests)), parameters=manifest.parameters.to_config_obj(),
                  successor_max_latency_sec=max(latencies), cpu_seconds=cpu,
                  cpu_stage_seconds=cpu_stages,
                  cpu_accounting="non-overlapping measured ADPSS stages; excludes DAG and coin CPU",
                  analyzer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  application_block_count=blocks, communication=totals,
                  experiment_digest=experiment_digest)
    comparison_stages = {name: value for name, value in cpu_stages.items()
                         if not name.startswith("committee_vrf_")}
    result.update(comparison_cpu_seconds=sum(comparison_stages.values()),
                  comparison_cpu_stage_seconds=comparison_stages,
                  comparison_cpu_accounting="measured ADPSS computation; separately reported sortition, DAG, coin and transport costs excluded")
    if protocol == "bhk22":
        result.update(path_mode=path, completion_mode="encrypted-carry-forward")
    if protocol == "ru22":
        result["vps_shape"] = nodes[0]["protocol_metrics"].get("vps_shape")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--protocol", choices=("a2dpss", "ru22", "bhk22"), required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    try:
        nodes = [json.loads(p.read_text()) for p in sorted(args.results.glob("node.*.json"))]
        result = analyze_arm(args.fixture, args.protocol, nodes)
    except Exception as exc:
        print(json.dumps({"status": "failure", "reason": str(exc)}))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
