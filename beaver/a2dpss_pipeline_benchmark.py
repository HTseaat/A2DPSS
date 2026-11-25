"""Portable paired DAG/ACS benchmark for A2DPSS query pipelining."""

from __future__ import annotations

import hashlib
import math
import os
import errno
from pathlib import Path
import re
import shutil
import tempfile
import time

from beaver.a2dpss_benchmark import parse_peers, read, write
from beaver.atcaas_params import ATCaaSProtocolParameters
from beaver.experiment_trace import (
    INITIAL_PROPOSAL_FIXED,
    START,
    ExperimentWorkloadTrace,
)
from beaver.a2dpss_pipeline_workload import (
    INDEPENDENT_SERVICE_STATES,
    A2DPSSPairedWorkload,
    build_paired_fixture,
)
from beaver.async_algorand.model import digest


PROTOCOLS = ("dag", "acs")
SCHEMA = "thesis-ch4-a2dpss-pipeline/v1"


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive(value, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def load_bundle(directory) -> dict:
    directory = Path(directory)
    manifest = read(directory / "public/manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported A2DPSS pipeline bundle")
    expected = manifest.get("bundle_digest")
    if digest({k: v for k, v in manifest.items() if k != "bundle_digest"}) != expected:
        raise ValueError("pipeline bundle digest mismatch")
    allowed = {
        "public/role-trace.json",
        "public/workload.json",
        "public/fixture.json",
    }
    if set(manifest.get("artifacts", {})) != allowed:
        raise ValueError("pipeline bundle artifact set mismatch")
    for relative, checksum in manifest["artifacts"].items():
        if _file_digest(directory / relative) != checksum:
            raise ValueError(f"pipeline artifact digest mismatch: {relative}")
    fixture = A2DPSSPairedWorkload.from_file(
        str(directory / "public/fixture.json")
    )
    if (
        fixture.digest() != manifest["fixture_digest"]
        or fixture.service_state_mode != INDEPENDENT_SERVICE_STATES
    ):
        raise ValueError("pipeline fixture identity/state mode mismatch")
    return manifest


def prepare(args) -> dict:
    from scripts.generate_coin_keys import validate_coin_key_directory
    from scripts.generate_dag_role_trace import selected_roles_for_round
    from beaver.a2dpss_experiment import g_h_from_config_extra
    from beaver.protocol_trace import CommitteeRoleTraceArtifact

    source = Path(args.config_dir).resolve()
    first = read(source / "local.0.json")
    parameters = ATCaaSProtocolParameters.from_mapping(first)
    for name in (
        "timeout",
        "startup_timeout",
        "coin_timeout",
        "acs_stage_timeout",
    ):
        _positive(getattr(args, name), name=name)
    if args.initial_queries < 1 or args.late_queries < 1:
        raise ValueError("initial-queries and late-queries must be positive")
    if args.rounds < 8:
        raise ValueError("rounds must be at least 8 for a two-window DAG run")
    if args.acs_max_rounds < 1 or args.lookahead_rounds < 2:
        raise ValueError("acs-max-rounds must be positive and lookahead-rounds >= 2")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.trial_id):
        raise ValueError("trial-id must use 1-80 letters, digits, underscores or hyphens")
    peers = parse_peers(
        read(args.peers) if args.peers else first["peers"], parameters.M
    )
    validate_coin_key_directory(config_dir=source, network_size=parameters.M)
    configs = []
    for node in range(parameters.M):
        config = read(source / f"local.{node}.json")
        if (
            config.get("my_id") != node
            or ATCaaSProtocolParameters.from_mapping(config) != parameters
        ):
            raise ValueError("node identity or protocol parameters disagree")
        if any(
            config["extra"].get(field) != first["extra"].get(field)
            for field in ("SRS", "vrf_pks", "chunk_lookup_tables")
        ):
            raise ValueError("SRS, VRF public keys or lookup tables disagree")
        config["peers"] = peers
        config["extra"]["run_id"] = args.trial_id
        configs.append(config)

    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    (directory / "nodes").mkdir(mode=0o700)
    workload = ExperimentWorkloadTrace.from_dict(
        {
            "trace_id": f"a2dpss-pipeline-{args.trial_id}",
            "messages_per_node": 1,
            "batches": [
                {
                    "batch_id": "early",
                    "trigger": START,
                    "instances": [
                        f"early-{index:03d}"
                        for index in range(args.initial_queries)
                    ],
                },
                {
                    "batch_id": "late",
                    "trigger": INITIAL_PROPOSAL_FIXED,
                    "instances": [
                        f"late-{index:03d}"
                        for index in range(args.late_queries)
                    ],
                },
            ],
        }
    )
    write(directory / "public/workload.json", workload.to_dict())

    g_hex, _h_hex = g_h_from_config_extra(first["extra"])
    selection_count = max(args.rounds, args.acs_max_rounds) + 2
    vrf_sks = [config["extra"]["vrf_sk"] for config in configs]
    roles = {
        index: selected_roles_for_round(
            epoch=args.trial_id,
            parent_round=4 * (index - 1),
            M=parameters.M,
            threshold_fraction=parameters.n_a / parameters.M,
            vrf_sks=vrf_sks,
            vrf_pks=first["extra"]["vrf_pks"],
            g_hex=g_hex,
            selection_mode=args.selection_mode,
            committee_size=parameters.n_a,
            fixed_committee_seed=args.committee_seed,
        )
        for index in range(1, selection_count + 1)
    }
    write(
        directory / "public/role-trace.json",
        CommitteeRoleTraceArtifact(roles).to_dict(),
    )
    role_trace_digest = CommitteeRoleTraceArtifact.from_file(
        directory / "public/role-trace.json"
    ).digest()

    generation_started = time.perf_counter()
    generation_cpu_started = time.process_time()
    with tempfile.TemporaryDirectory(prefix="a2dpss-pipeline-") as temporary:
        temporary_root = Path(temporary)
        for node, config in enumerate(configs):
            write(temporary_root / f"local.{node}.json", config, private=True)
        fixture = build_paired_fixture(
            config_path=str(temporary_root / "local.0.json"),
            workload_trace_path=str(directory / "public/workload.json"),
            service_state_mode=INDEPENDENT_SERVICE_STATES,
        )
    fixture_generation_seconds = time.perf_counter() - generation_started
    fixture_generation_cpu_seconds = time.process_time() - generation_cpu_started
    write(directory / "public/fixture.json", fixture.to_fixture_obj())
    lookup_artifacts = {}
    for table in (first["extra"].get("chunk_lookup_tables") or {}).get("tables", {}).values():
        relative = Path(table["artifact_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("chunk lookup artifact path is unsafe")
        source_artifact = source / relative
        if not source_artifact.is_file():
            raise ValueError(f"missing chunk lookup artifact: {relative}")
        checksum = _file_digest(source_artifact)
        if checksum != table["artifact_sha256"]:
            raise ValueError(f"chunk lookup artifact checksum mismatch: {relative}")
        lookup_artifacts[relative.as_posix()] = checksum
    for node, config in enumerate(configs):
        write(directory / f"nodes/{node}/config.json", config, private=True)
        for relative in lookup_artifacts:
            destination = directory / f"nodes/{node}" / relative
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.link(source / relative, destination)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copy2(source / relative, destination)

    artifacts = {
        relative: _file_digest(directory / relative)
        for relative in (
            "public/role-trace.json",
            "public/workload.json",
            "public/fixture.json",
        )
    }
    context = fixture.queries[0].context
    manifest = {
        "schema": SCHEMA,
        "experiment": "thesis-ch4-a2dpss-pipeline",
        "protocol": "a2dpss",
        "algorithm": 4,
        "arrival_mode": "controlled-proposal-boundary",
        "contribution_generation": "offline-pregenerated",
        "clock": "local-monotonic",
        "lookup_artifacts": lookup_artifacts,
        "trial_id": args.trial_id,
        "parameters": parameters.to_config_obj(),
        "peers": peers,
        "selection_mode": args.selection_mode,
        "committee_seed": args.committee_seed,
        "role_trace_digest": role_trace_digest,
        "coin_public_key_digest": first["extra"]["coin_public_key_digest"],
        "workload": workload.to_dict(),
        "initial_query_count": args.initial_queries,
        "late_query_count": args.late_queries,
        "query_count": args.initial_queries + args.late_queries,
        "request_trigger": "after-current-proposal-fixed-before-first-delivery",
        "fixture_digest": fixture.digest(),
        "sharing_state_mode": fixture.service_state_mode,
        "source_nodes": sorted(context.holder_eval_points),
        "successor_nodes": sorted(context.successor_role_indices),
        "rounds": args.rounds,
        "lookahead_rounds": args.lookahead_rounds,
        "acs_max_rounds": args.acs_max_rounds,
        "timeout": args.timeout,
        "startup_timeout": args.startup_timeout,
        "coin_timeout": args.coin_timeout,
        "acs_stage_timeout": args.acs_stage_timeout,
        "artifacts": artifacts,
        "fixture_generation_seconds": fixture_generation_seconds,
        "fixture_generation_cpu_seconds": fixture_generation_cpu_seconds,
        "holder_generation": fixture.holder_metrics_summary(),
        "measurement_scope": (
            "pregenerated-contribution handover: dissemination, validation, ordering, "
            "threshold-BLS coin, public commitment update and successor recovery online; "
            "controlled readiness gate, local monotonic timestamps"
        ),
        "excluded_online_costs": [
            "A2DPSS holder contribution generation",
            "role-assignment setup",
            "lookup generation and startup authentication/mmap",
            "fixture/configuration deployment and startup barrier",
        ],
        "latency_definitions": {
            "admission": "t_first_inclusion - t_pool_ready",
            "delivery": "t_delivery - t_pool_ready",
            "completion_after_readiness": "t_successor - t_pool_ready",
            "completion_after_release": "t_successor - t_request_release",
            "batch": "max_node(max_batch_successor - min_early_release)",
        },
        "private_config_digests": {
            str(node): digest(config) for node, config in enumerate(configs)
        },
    }
    manifest["bundle_digest"] = digest(manifest)
    write(directory / "public/manifest.json", manifest)
    return manifest


def validate_nodes(nodes, manifest) -> None:
    if (
        not nodes
        or len(set(nodes)) != len(nodes)
        or any(
            type(node) is not int
            or not 0 <= node < manifest["parameters"]["M"]
            for node in nodes
        )
    ):
        raise ValueError("nodes must be distinct node ids in [0,M)")


def export_nodes(directory, output, nodes) -> Path:
    manifest = load_bundle(directory)
    validate_nodes(nodes, manifest)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    shutil.copytree(Path(directory) / "public", output / "public")
    (output / "nodes").mkdir(mode=0o700)
    for node in nodes:
        source = Path(directory) / f"nodes/{node}"
        if any(source.glob("*.used")):
            raise ValueError("cannot export a node after a protocol started")
        shutil.copytree(source, output / f"nodes/{node}")
    return output



from beaver.a2dpss_pipeline_results import normalize_result, collect, compare
