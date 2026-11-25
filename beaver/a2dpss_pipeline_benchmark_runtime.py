"""One-node runtime for the paired A2DPSS DAG/ACS pipeline benchmark."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import platform
import sys
import time

from beaver.a2dpss_benchmark import read, write
from beaver.a2dpss_benchmark_runtime import (
    ExperimentBarrier,
    require_native_vrf_backends,
)
from beaver.async_algorand.model import digest
from beaver.config import HbmpcConfig
from beaver.a2dpss_pipeline_workload import A2DPSSPairedWorkload
from beaver.a2dpss_pipeline_benchmark import load_bundle, normalize_result


def protocol_wire_bytes_by_category(barrier: ExperimentBarrier, epoch: str) -> dict:
    """Disjoint actual send-byte buckets, classified by the outer namespace."""
    totals = {key: 0 for key in ("resharing_records", "ordering", "common_coin", "other")}
    for path, count in barrier.runner.node_communicator.bytes_by_tag_path.items():
        namespace = str(path).strip("/").split("/", 1)[0]
        if namespace == "benchmark":
            continue
        if namespace == "ATCAAS_COIN_SHARE_V1":
            category = "common_coin"
        elif namespace.startswith((f"MSGRBC:{epoch}:", f"AYTOB:{epoch}:MSG:", "A2DPSS:")):
            category = "resharing_records"
        elif namespace.startswith((f"DAGTOB:{epoch}:", f"AYTOB:{epoch}:ACS:")):
            category = "ordering"
        else:
            category = "other"
        if int(count) < 0:
            raise ValueError("transport byte count must be nonnegative")
        totals[category] += int(count)
    return totals


def _load_hbmpc_config(path: Path) -> None:
    previous = sys.argv
    try:
        sys.argv = [previous[0], "-d", "-f", str(path), "-time", "0"]
        HbmpcConfig.load_config()
    finally:
        sys.argv = previous


async def _run_existing(
    directory: Path,
    manifest: dict,
    node: int,
    config_path: Path,
    fixture: A2DPSSPairedWorkload,
    barrier: ExperimentBarrier,
    protocol: str,
) -> dict:
    _load_hbmpc_config(config_path)
    parameters = manifest["parameters"]
    common = dict(
        peers=HbmpcConfig.peers,
        n=HbmpcConfig.N,
        t=HbmpcConfig.t,
        my_id=node,
        start_time=0,
        M=parameters["M"],
        epoch=manifest["trial_id"],
        expected_committee_size=parameters["n_a"],
        committee_m=parameters["m_a"],
        quorum_h=parameters["quorum_h"],
        max_runtime=manifest["timeout"],
        stop_grace_seconds=3,
        multi_qid_smoke=True,
        a2dpss_pipeline=True,
        coin_mode_requested="threshold-bls",
        coin_timeout=manifest["coin_timeout"],
        experiment_barrier=barrier,
        **{
            key: parameters[key]
            for key in (
                "n_r",
                "t_r",
                "n_a",
                "t_a",
                "m_a",
                "coin_c",
                "coin_t",
                "coin_k",
            )
        },
    )
    if protocol == "dag":
        from scripts.run_dag_tob_phase3 import _run

        return await _run(
            **common,
            rounds=manifest["rounds"],
            target_deliveries=manifest["query_count"],
            lookahead_rounds=manifest["lookahead_rounds"],
        )
    if protocol == "acs":
        from scripts.run_async_yoso_tob import _run

        return await _run(
            **common,
            messages_per_node=fixture.messages_per_node,
            enable_ledger=True,
            acs_activation_mode="vrf",
            acs_collect_all=True,
            acs_collect_threshold=parameters["quorum_h"],
            acs_max_rounds=manifest["acs_max_rounds"],
            acs_stage_timeout=manifest["acs_stage_timeout"],
            acs_output_service_grace=0.0,
        )
    raise ValueError(f"unsupported pipeline protocol {protocol!r}")


async def worker(directory, output, protocol: str, node: int) -> dict:
    directory = Path(directory).resolve()
    output = Path(output).resolve()
    manifest = load_bundle(directory)
    if protocol not in ("dag", "acs"):
        raise ValueError("pipeline protocol must be dag or acs")
    if not 0 <= node < manifest["parameters"]["M"]:
        raise ValueError("node id is outside the prepared population")
    config_path = directory / f"nodes/{node}/config.json"
    result_path = output / protocol / f"node-{node}.json"
    config = read(config_path)
    if (
        config.get("my_id") != node
        or digest(config) != manifest["private_config_digests"][str(node)]
    ):
        raise ValueError("private node configuration does not match the bundle")
    fixture = A2DPSSPairedWorkload.from_file(
        str(directory / "public/fixture.json")
    )
    if fixture.digest() != manifest["fixture_digest"]:
        raise ValueError("A2DPSS fixture does not match the manifest")
    if (directory / f"nodes/{node}/{protocol}.used").exists():
        raise FileExistsError("node/protocol was already used; use a fresh trial bundle")
    if result_path.exists():
        raise FileExistsError("result already exists; use a fresh trial bundle")

    vrf_backend = require_native_vrf_backends()
    write(
        directory / f"nodes/{node}/{protocol}.used",
        {"trial_id": manifest["trial_id"]},
        private=True,
    )
    for key in list(os.environ):
        if key.startswith(("ATCAAS_", "DAG_TOB_", "ASYNC_YOSO_TOB_")):
            del os.environ[key]
    os.environ.update(
        ATCAAS_TRIAL_EPOCH=manifest["trial_id"],
        ATCAAS_ROLE_TRACE=str(directory / "public/role-trace.json"),
        ATCAAS_A2DPSS_PIPELINE_FIXTURE=str(directory / "public/fixture.json"),
        ATCAAS_COMMITTEE_SELECTION_MODE=manifest["selection_mode"],
        ATCAAS_FIXED_COMMITTEE_SEED=manifest["committee_seed"],
        DAG_TOB_DPSS_POOL_WAIT_SECONDS=str(manifest["timeout"]),
        DAG_TOB_A2DPSS_POOL_WAIT_SECONDS=str(manifest["timeout"]),
    )
    barrier = ExperimentBarrier(manifest, protocol)
    started = time.perf_counter()
    try:
        raw = await _run_existing(
            directory, manifest, node, config_path, fixture, barrier, protocol
        )
        raw["vrf_backend"] = vrf_backend
        raw["lookup_table_preload_seconds"] = HbmpcConfig.lookup_table_preload_seconds
        raw["benchmark_protocol_wire_bytes"] = barrier.wire_bytes()
        raw["benchmark_protocol_wire_bytes_by_category"] = protocol_wire_bytes_by_category(
            barrier, manifest["trial_id"]
        )
        if sum(raw["benchmark_protocol_wire_bytes_by_category"].values()) != raw["benchmark_protocol_wire_bytes"]:
            raise ValueError("protocol network categories do not sum to actual sends")
        raw["pipeline_clock"] = {
            "source": "time.perf_counter", "monotonic": True,
            "scope": "local-process-relative",
            "cross_machine_subtraction": False,
        }
        raw["benchmark_barrier_seconds"] = dict(barrier.elapsed)
        checked_manifest = {**manifest, "bundle_root": str(directory)}
        result = normalize_result(checked_manifest, protocol, node, raw)
    except Exception as error:
        result = {
            "schema": manifest["schema"],
            "status": (
                "timeout" if isinstance(error, asyncio.TimeoutError) else "failed"
            ),
            "protocol": protocol,
            "node": node,
            "bundle_digest": manifest["bundle_digest"],
            "trial_id": manifest["trial_id"],
            "reason": f"{type(error).__name__}: {error}",
            "process_elapsed_seconds": time.perf_counter() - started,
        }
        write(result_path, result)
        raise
    result["machine"] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
    }
    write(result_path, result)
    logging.warning("[PIPELINE-BENCHMARK] node=%s protocol=%s complete", node, protocol)
    return result


__all__ = ["worker"]
