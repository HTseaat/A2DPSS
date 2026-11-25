#!/usr/bin/env python3
"""Fail-closed analyzer for one A2DPSS/RU22/BHK22 DAG triplet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from beaver.yoso_adpss_experiment import NODE_METRICS_SCHEMA, TRIAL_METRICS_SCHEMA
from beaver.yoso_adpss_workload import ARM_ORDER, load_yoso_adpss_public_fixture, verify_same_initial_workload


TRIPLET_SCHEMA = "yoso-adpss-dag-triplet/v1"
_PRIVATE_KEYS = {
    "share", "share_s", "share_s_hat", "blind", "recipient_secret_key",
    "private_key", "private_binding_salt", "witness", "source_share",
    "public_value_secret_key",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _walk_private(value: Any, path: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in _PRIVATE_KEYS or lowered.endswith("_private_key"):
                found.append(f"{path}/{key}")
            found.extend(_walk_private(child, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_private(child, f"{path}/{index}"))
    return found


def _cpu_leaf_total(value: Any) -> float:
    if isinstance(value, dict):
        if "cpu_seconds" in value and "invocations" in value:
            cpu = float(value["cpu_seconds"]); invocations = int(value["invocations"])
            if cpu < 0 or invocations < 0:
                raise ValueError("CPU metric is negative")
            return cpu
        return sum(_cpu_leaf_total(child) for child in value.values())
    if isinstance(value, list):
        return sum(_cpu_leaf_total(child) for child in value)
    return 0.0


def analyze(*, fixture: Path, run_dir: Path, expected_nodes: int | None = None) -> dict[str, Any]:
    manifest, arms = load_yoso_adpss_public_fixture(fixture)
    expected_nodes = manifest.parameters.M if expected_nodes is None else int(expected_nodes)
    if expected_nodes != manifest.parameters.M:
        raise ValueError("expected node count does not match common workload")
    if not verify_same_initial_workload(manifest, arms):
        raise ValueError("arm manifests do not bind the same initial workload")
    triplet_path = run_dir / "triplet-manifest.json"
    if not triplet_path.is_file():
        raise ValueError("triplet manifest is missing")
    triplet = json.loads(triplet_path.read_text())
    if triplet.get("schema") != TRIPLET_SCHEMA:
        raise ValueError("unsupported triplet manifest schema")
    if triplet.get("common_workload_digest") != manifest.digest():
        raise ValueError("triplet/common workload digest mismatch")
    order = tuple(triplet.get("arm_order") or ())
    if set(order) != set(ARM_ORDER) or len(order) != len(ARM_ORDER):
        raise ValueError("triplet arm order is not a permutation")
    expected_arm_digests = {name: arms[name].digest() for name in ARM_ORDER}
    if triplet.get("arm_manifest_digests") != expected_arm_digests:
        raise ValueError("triplet arm digest binding mismatch")
    if triplet.get("ru22_mode", "legacy-dag") not in ("native", "legacy-dag"):
        raise ValueError("unsupported triplet RU22 mode")
    if triplet.get("ru22_mode") == "native":
        from scripts.analyze_yoso_adpss_dag_arm import analyze_arm
        summaries = {}
        for protocol in ARM_ORDER:
            nodes = [json.loads((run_dir / "arms" / protocol / f"node.{i}.json").read_text())
                     for i in range(expected_nodes)]
            if any(_walk_private(node) for node in nodes):
                raise ValueError("node public summary contains private fields")
            summaries[protocol] = analyze_arm(fixture, protocol, nodes)
        if summaries["ru22"].get("protocol_mode") != "native":
            raise ValueError("native triplet contains historical RU22 results")
        material = dict(schema=TRIAL_METRICS_SCHEMA, status="success", ru22_mode="native",
            common_workload_digest=manifest.digest(), arm_order=list(order), arms=summaries)
        return dict(material, summary_digest=hashlib.sha256(_canonical(material)).hexdigest())

    nodes_by_arm = {}
    arm_summaries = {}
    block_expectation = {"a2dpss": 1, "ru22": 1}
    for protocol in ARM_ORDER:
        nodes = []
        for node_id in range(expected_nodes):
            path = run_dir / "arms" / protocol / f"node.{node_id}.json"
            if not path.is_file():
                raise ValueError(f"missing {protocol} node {node_id} summary")
            value = json.loads(path.read_text())
            if value.get("schema") != NODE_METRICS_SCHEMA:
                raise ValueError("node metrics schema mismatch")
            if value.get("protocol") != protocol or value.get("node_id") != node_id:
                raise ValueError("node summary identity mismatch")
            if value.get("common_workload_digest") != manifest.digest():
                raise ValueError("node/common workload digest mismatch")
            if value.get("arm_manifest_digest") != arms[protocol].digest():
                raise ValueError("node/arm manifest digest mismatch")
            if value.get("status") != "success" or value.get("failure") is not None:
                raise ValueError(f"{protocol} node {node_id} did not succeed")
            timeline = value.get("timeline") or {}
            observed = [item for item in timeline.values() if item is not None]
            if observed != sorted(observed) or timeline.get("frozen_at") is None:
                raise ValueError("node timeline is not monotonic and frozen")
            if not (value.get("cleanup") or {}).get("completed"):
                raise ValueError("node cleanup is incomplete")
            if _walk_private(value):
                raise ValueError("node public summary contains private fields")
            raw_log = run_dir / "arms" / protocol / "logs" / f"node.{node_id}.log"
            if raw_log.is_file():
                encoded_log = raw_log.read_text(errors="replace").lower()
                forbidden_tokens = (
                    '"share_s"', '"share_s_hat"', '"recipient_secret_key"',
                    '"coin_private_key"', '"private_binding_salt"',
                    '"public_value_secret_key"',
                )
                if any(token in encoded_log for token in forbidden_tokens):
                    raise ValueError("node public log contains private material")
            formal_hits = (value.get("protocol_metrics") or {}).get("formal_cache_hits")
            if formal_hits != 0:
                raise ValueError("formal predicate cache hits must be zero")
            if _cpu_leaf_total(value.get("cpu") or {}) <= 0.0:
                raise ValueError("node CPU invocation records are missing")
            nodes.append(value)
        nodes_by_arm[protocol] = nodes
        if protocol == "bhk22":
            path_mode = nodes[0]["protocol_metrics"].get("path_mode")
            expected_blocks = 2 if path_mode == "optimistic" else 3
            if nodes[0]["protocol_metrics"].get("completion_mode") != "encrypted-carry-forward":
                raise ValueError("BHK22 triplet arm must use encrypted carry-forward")
            if any(node["protocol_metrics"].get("public_commitment_kind") != "encrypted" for node in nodes):
                raise ValueError("BHK22 public output must remain encrypted")
        else:
            expected_blocks = block_expectation[protocol]
        if any(node["communication"]["committed_application_block_count"] != expected_blocks for node in nodes):
            raise ValueError(f"{protocol} committed application block count mismatch")
        public_digests = {node["protocol_metrics"].get("public_result_digest") for node in nodes}
        if len(public_digests) != 1 or None in public_digests:
            raise ValueError(f"{protocol} observers disagree on public result")
        successor_times = [node["timeline"].get("t_successor_done") for node in nodes]
        if any(value is None for value in successor_times):
            raise ValueError(f"{protocol} successor completion is missing")
        arm_summaries[protocol] = {
            "status": "success",
            "node_count": len(nodes),
            "public_result_digest": next(iter(public_digests)),
            "successor_max_latency_sec": max(successor_times),
            "cpu_seconds": sum(_cpu_leaf_total(node.get("cpu") or {}) for node in nodes),
            "application_payload_bytes": sum(node["communication"]["application_payload_bytes"] for node in nodes),
            "cryptographic_proof_bytes": (
                sum(node["communication"]["cryptographic_proof_bytes"] for node in nodes)
                if all(
                    isinstance(node["communication"].get("cryptographic_proof_bytes"), int)
                    for node in nodes
                )
                else None
            ),
            "recipient_ciphertext_bytes": (
                sum(node["communication"]["recipient_ciphertext_bytes"] for node in nodes)
                if all(
                    isinstance(node["communication"].get("recipient_ciphertext_bytes"), int)
                    for node in nodes
                )
                else None
            ),
            "total_network_bytes": sum(node["communication"]["total_network_bytes"] for node in nodes),
            "application_block_count": expected_blocks,
        }
        if protocol == "ru22":
            arm_summaries[protocol]["vps_shape"] = nodes[0]["protocol_metrics"].get("vps_shape")
        if protocol == "bhk22":
            arm_summaries[protocol]["path_mode"] = path_mode

    material = {
        "schema": TRIAL_METRICS_SCHEMA,
        "status": "success",
        "common_workload_digest": manifest.digest(),
        "arm_order": list(order),
        "arms": arm_summaries,
    }
    return {**material, "summary_digest": hashlib.sha256(_canonical(material)).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(fixture=args.fixture, run_dir=args.run_dir)
    except Exception as exc:
        result = {"schema": TRIAL_METRICS_SCHEMA, "status": "failure", "failure_reason": str(exc)}
        code = 1
    else:
        code = 0
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
