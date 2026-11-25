"""Run one node of one common-fixture YOSO ADPSS arm on persistent DAG-TOB."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from beaver.baselines.bhk22_contract import path_phases
from beaver.config import HbmpcConfig
from beaver.protocol_trace import CommitteeRoleTraceArtifact
from beaver.yoso_adpss_experiment import normalize_dag_node_summary
from beaver.yoso_adpss_workload import (
    YOSOADPSSNodePrivateShard,
    load_yoso_adpss_public_fixture,
)
from scripts.run_dag_tob_phase3 import _env_float, _env_int, _node_id_set, _run
from scripts.run_yoso_adpss_dag_arm import build_adapter_from_fixture


def _consume(name: str, *, required: bool = True) -> str | None:
    values = []
    index = 1
    while index < len(sys.argv):
        argument = sys.argv[index]
        if argument == name:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{name} requires a value")
            values.append(sys.argv[index + 1]); del sys.argv[index:index + 2]; continue
        if argument.startswith(name + "="):
            values.append(argument.split("=", 1)[1]); del sys.argv[index]; continue
        index += 1
    if len(values) > 1:
        raise ValueError(f"{name} may be specified only once")
    if required and not values:
        raise ValueError(f"{name} is required")
    return values[0] if values else None


def main() -> None:
    fixture_path = Path(_consume("--fixture"))
    arm_name = _consume("--arm")
    settings_path = _consume("--experiment-settings", required=False)
    if arm_name not in {"a2dpss", "ru22", "bhk22"}:
        raise ValueError("--arm must be a2dpss, ru22, or bhk22")
    HbmpcConfig.load_config()
    node_id = HbmpcConfig.my_id
    adapter = build_adapter_from_fixture(fixture_path, arm_name, node_id)
    manifest, arms = load_yoso_adpss_public_fixture(fixture_path)
    private_shard = YOSOADPSSNodePrivateShard.from_obj(json.loads(
        (fixture_path / f"node.{node_id}.private.json").read_text()
    ))
    coin_private = private_shard.threshold_bls
    if coin_private is not None and (
        HbmpcConfig.extras.get("coin_private_key") != coin_private["private_key"]
        or HbmpcConfig.extras.get("coin_private_share_index")
        != coin_private["private_share_index"]
    ):
        raise ValueError("node config threshold-BLS share does not match common shard")
    trace_path = os.environ.get("ATCAAS_ROLE_TRACE")
    if not trace_path:
        raise ValueError("formal YOSO ADPSS arm requires ATCAAS_ROLE_TRACE")
    trace = CommitteeRoleTraceArtifact.from_file(trace_path)
    if trace.digest() != manifest.role_trace_digest:
        raise ValueError("role trace does not match common workload")
    if os.environ.get("DAG_TOB_COIN_MODE", "threshold-bls") != "threshold-bls":
        raise ValueError("formal YOSO ADPSS arm requires threshold-bls")
    epoch = manifest.trial_epoch
    parameters = manifest.parameters
    barrier = None
    if settings_path:
        from beaver.yoso_adpss_distributed import DistributedBarrier, digest
        settings = json.loads(Path(settings_path).read_text())
        material = {k: v for k, v in settings.items() if k != "digest"}
        if settings.get("digest") != digest(material):
            raise ValueError("experiment settings digest mismatch")
        if settings["common_workload_digest"] != manifest.digest():
            raise ValueError("experiment settings do not match common workload")
        if settings["parameters"] != parameters.to_config_obj():
            raise ValueError("experiment parameters do not match workload")
        barrier = DistributedBarrier(settings, arm_name)
    coin_silent_nodes = _node_id_set(
        os.environ.get("DAG_TOB_COIN_SILENT_NODES", ""),
        M=parameters.M, name="DAG_TOB_COIN_SILENT_NODES",
    )
    kwargs = dict(
        peers=HbmpcConfig.peers, n=HbmpcConfig.N, t=HbmpcConfig.t,
        my_id=node_id, start_time=HbmpcConfig.time, M=parameters.M,
        expected_committee_size=parameters.n_a, committee_m=parameters.m_a,
        quorum_h=parameters.quorum_h, epoch=epoch,
        rounds=_env_int("DAG_TOB_MAX_ROUNDS", 32),
        max_runtime=_env_float("DAG_TOB_MAX_RUNTIME", 720.0),
        lookahead_rounds=_env_int("DAG_TOB_LOOKAHEAD_ROUNDS", 4),
        retention_rounds=_env_int("DAG_TOB_RETENTION_ROUNDS", 12),
        stop_grace_seconds=_env_float("DAG_TOB_STOP_GRACE_SECONDS", 5.0),
        coin_mode_requested="threshold-bls",
        coin_timeout=_env_float("DAG_TOB_COIN_TIMEOUT", 30.0),
        coin_silent_nodes=coin_silent_nodes,
        n_r=parameters.n_r, t_r=parameters.t_r,
        n_a=parameters.n_a, t_a=parameters.t_a, m_a=parameters.m_a,
        coin_c=parameters.coin_c, coin_t=parameters.coin_t,
        coin_k=parameters.coin_k,
    )
    if barrier is not None:
        kwargs.update(experiment_barrier=barrier, start_time=0,
                      rounds=settings["rounds"], max_runtime=settings["timeout"],
                      coin_timeout=settings["coin_timeout"])
    if arm_name == "a2dpss":
        expected_context = adapter.context
        # _run builds this same live context from the node-local source share;
        # no frozen contribution fixture is supplied.
        kwargs.update(a2dpss_smoke=True, target_deliveries=1)
    elif arm_name == "ru22":
        expected_context = adapter.runtime.context
        kwargs.update(
            ru22_smoke=True, target_deliveries=1,
            ru22_manifest=adapter.runtime.manifest,
            ru22_local_shard=adapter.runtime.local_shard,
        )
    else:
        expected_context = adapter.runtime.context
        kwargs.update(
            bhk22_smoke=True,
            target_deliveries=len(path_phases(
                adapter.runtime.manifest.path_mode,
                adapter.runtime.manifest.completion_mode,
            )),
            bhk22_manifest=adapter.runtime.manifest,
            bhk22_local_shard=adapter.runtime.local_shard,
        )
    logging.info(
        "[YOSO-ADPSS:ARM] protocol=%s node=%d workload=%s context=%s",
        arm_name, node_id, manifest.digest(),
        getattr(expected_context, "context_digest", getattr(expected_context, "root_context_digest", None)),
    )
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    try:
        legacy = loop.run_until_complete(_run(**kwargs))
    finally:
        loop.close()
    legacy["lookup_table_preload_seconds"] = (
        HbmpcConfig.lookup_table_preload_seconds
    )
    if arm_name == "a2dpss" and (
        legacy.get("a2dpss_context_digest") != adapter.context.context_digest
    ):
        raise RuntimeError("live A2DPSS context does not match common arm manifest")
    unified = normalize_dag_node_summary(
        protocol=arm_name, node_id=node_id,
        common_workload_digest=manifest.digest(),
        arm_manifest_digest=arms[arm_name].digest(),
        legacy_summary=legacy,
    )
    if barrier is not None:
        transport = barrier.transport
        if transport is None:
            raise RuntimeError("global completion was not reached")
        for field, source in (("yoso_rbc_envelope_bytes", "yoso_rbc"),
                              ("dag_protocol_bytes", "dag_protocol"),
                              ("common_coin_bytes", "common_coin"),
                              ("total_network_bytes", "total")):
            unified["communication"][field] = transport[source]
        unified["experiment_digest"] = settings["digest"]
        unified["barrier_seconds"] = barrier.elapsed
        unified["communication_endpoint"] = "global-completion-excluding-control"
    result = {**legacy, "yoso_adpss_metrics": unified}
    print(json.dumps(result, sort_keys=True))
    if unified["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
