#!/usr/bin/env python3
"""Validate/load one Step-2 arm; network execution is integrated in Step 4."""
import argparse
import json
from pathlib import Path

from beaver.a2dpss import A2DPSSContext
from beaver.baselines.bhk22_dag_runner import BHK22OptimisticDAGRuntime
from beaver.baselines.bhk22_experiment import BHK22DAGExperimentManifest, BHK22NodePrivateShard
from beaver.baselines.ru22_dag_runner import RU22DAGHandoverRuntime
from beaver.baselines.ru22_experiment import RU22DAGExperimentManifest, RU22DAGLocalShard
from beaver.yoso_adpss_experiment import A2DPSSAdapter, BHK22Adapter, RU22Adapter
from beaver.yoso_adpss_workload import (
    YOSOADPSSNodePrivateShard, load_yoso_adpss_public_fixture,
    verify_same_initial_workload,
)


def build_adapter_from_fixture(root: Path, arm_name: str, node_id: int):
    manifest, arms = load_yoso_adpss_public_fixture(root)
    shard = YOSOADPSSNodePrivateShard.from_obj(
        json.loads((root / f"node.{node_id}.private.json").read_text())
    )
    if not verify_same_initial_workload(manifest, arms):
        raise ValueError("three arm manifests do not bind the same workload")
    arm = arms[arm_name]
    if shard.arm_manifest_digests[arm_name] != arm.digest():
        raise ValueError("node shard does not bind the selected arm")
    if arm_name == "a2dpss":
        context = A2DPSSContext.from_obj(arm.protocol_manifest["context"])
        holder = shard.a2dpss["holder"]
        successor = shard.a2dpss["successor"]
        return A2DPSSAdapter(
            context=context, my_node_id=node_id,
            holder_share=(int(holder["share_s"]), int(holder["share_s_hat"])) if holder else None,
            successor_rsk=int(successor["recipient_secret_key"]) if successor else None,
        )
    if arm_name == "ru22":
        protocol_manifest = RU22DAGExperimentManifest.from_obj(arm.protocol_manifest)
        local = RU22DAGLocalShard.from_obj(shard.ru22, protocol_manifest)
        return RU22Adapter(runtime=RU22DAGHandoverRuntime(
            manifest=protocol_manifest, local_shard=local, my_node_id=node_id
        ))
    protocol_manifest = BHK22DAGExperimentManifest.from_obj(arm.protocol_manifest)
    local = BHK22NodePrivateShard.from_obj(shard.bhk22, protocol_manifest)
    return BHK22Adapter(runtime=BHK22OptimisticDAGRuntime(
        manifest=protocol_manifest, local_shard=local, my_node_id=node_id
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--arm", choices=("a2dpss", "ru22", "bhk22"), required=True)
    parser.add_argument("--node", type=int, required=True)
    args = parser.parse_args()
    root = Path(args.fixture)
    manifest, arms = load_yoso_adpss_public_fixture(root)
    shard = YOSOADPSSNodePrivateShard.from_obj(
        json.loads((root / f"node.{args.node}.private.json").read_text())
    )
    valid = verify_same_initial_workload(manifest, arms)
    valid = valid and shard.common_workload_digest == manifest.digest()
    valid = valid and shard.arm_manifest_digests[args.arm] == arms[args.arm].digest()
    adapter = build_adapter_from_fixture(root, args.arm, args.node) if valid else None
    result = {
        "status": "ready" if valid else "failure", "arm": args.arm,
        "node_id": args.node, "workload_digest": manifest.digest(),
        "arm_manifest_digest": arms[args.arm].digest(),
        "expected_application_deliveries": (
            adapter.expected_application_deliveries() if adapter else None
        ),
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
