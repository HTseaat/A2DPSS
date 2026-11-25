from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import pickle
import sys
import time

from beaver.broadcast.crypto.boldyreva import TBLSPublicKey  # noqa:F401
from beaver.broadcast.crypto.boldyreva import TBLSPrivateKey  # noqa:F401
from beaver.config import HbmpcConfig
from beaver.a2dpss import (
    A2DPSSContext,
    A2DPSSHolderState,
    A2DPSSJustifierManager,
    A2DPSSPublicFinalizer,
    A2DPSSSuccessorReconstructionManager,
    a2dpss_contribution_digest,
    run_a2dpss_handover_role,
)
from beaver.a2dpss_experiment import A2DPSSPairedFixture
from beaver.a2dpss_rbc_manager import (
    A2DPSSContributionRBCManager,
)
from beaver.baselines.ru22_dag_runner import RU22DAGHandoverRuntime
from beaver.baselines.ru22_experiment import (
    RU22DAGExperimentManifest,
    RU22DAGLocalShard,
)
from beaver.baselines.ru22_rbc_manager import (
    RU22ContributionRBCManager,
)
from beaver.baselines.ru22_operation_dag_runner import RU22OperationDAGRuntime
from beaver.baselines.ru22_operation_rbc_manager import (
    RU22OperationContributionRBCManager,
)
from beaver.baselines.bhk22_contract import (
    BHK22_QID_KIND,
    PHASE_DEAL as BHK22_PHASE_DEAL,
    PHASE_FINALIZE as BHK22_PHASE_FINALIZE,
    PHASE_RESPONSE as BHK22_PHASE_RESPONSE,
    PHASE_VERIFY as BHK22_PHASE_VERIFY,
    path_phases as bhk22_path_phases,
)
from beaver.baselines.bhk22_dag_runner import BHK22DAGRuntime
from beaver.baselines.bhk22_experiment import (
    BHK22DAGExperimentManifest,
    BHK22NodePrivateShard,
)
from beaver.baselines.bhk22_rbc_manager import BHK22PhaseRBCManager
from beaver.baselines.bhk22_operation import BHK22OperationPublicContext
from beaver.baselines.bhk22_operation_dag_runner import BHK22OperationDAGRuntime
from beaver.a2dpss_lifecycle import A2DPSSHandoverLifecycle
from beaver.async_yoso_coin import (
    ObservedCoinOracle,
    ThresholdBLSCoinOracle,
    load_threshold_coin_key_material,
)
from beaver.dag_rbc_manager import DagRBCManager
from beaver.dag_tob import (
    DagTOB,
    IdealCoinLeaderOracle,
    JustifierRegistry,
    Qid,
    TobMessage,
    ThresholdCoinLeaderOracle,
    TypedWaitPredicate,
    Vertex,
)
from beaver.dpss_rbc_manager import DPSSTranscriptRBCManager
from beaver.experiment_trace import (
    A2DPSSHandoverMetrics,
    INITIAL_PROPOSAL_FIXED,
    START,
    QueryLifecycleMetrics,
    build_a2dpss_computation_metrics,
    load_dynamic_workload,
)
from beaver.message_rbc_manager import MessageRBCManager
from beaver.g1_backend import G1Point
from beaver.ocdpss import (
    OCDPSSContext,
    OCDPSSHolderState,
    OCDPSSJustifierManager,
    OCDPSSPublicFinalizer,
    OCDPSSSuccessorReconstructionManager,
    run_ocdpss_handover_role,
)
from beaver.ocdpss_rbc_manager import OCDPSSBundleRBCManager
from beaver.ocdpss_lifecycle import OCDPSSHandoverLifecycle
from beaver.ocdpss_experiment import OCDPSSPairedWorkload
from beaver.operation_share import ThresholdExponentiationOperation
from beaver.ipc import ProcessProgramRunner
from beaver.protocol_trace import (
    CommitteeRoleTraceArtifact,
    DeterministicIdealCoin,
    ProtocolReplayTrace,
    RecordedCommitteeRoleTrace,
)
from beaver.utils.misc import subscribe_recv
from beaver.sigma import hash_to_scalar
from beaver.vrf import (
    BERNOULLI_COMMITTEE_MODE,
    FIXED_SIZE_COMMITTEE_MODE,
    fixed_size_committee,
    generator_hex_from_srs,
    normalize_committee_selection_mode,
    vrf_prove_sortition,
    vrf_verify_sortition,
)
from beaver.handover_fixtures import build_handover_local_config


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _transport_family_totals(
    transport_by_tag: dict[str, int], *, total_bytes: int
) -> dict[str, int]:
    """Partition exact transport bytes by the outer protocol namespace.

    Trial ids and epochs are deliberately allowed to contain protocol names.
    Classifying the complete nested tag path would therefore misclassify a
    ``/DAGTOB:<epoch-containing-a2dpss>:...`` message as A2DPSS traffic.  The
    first non-empty tag component is the transport owner and is the only part
    used for the explicit common-coin and ADPSS-RBC buckets.  Everything else
    is DAG/runtime traffic, preserving an exact partition of ``total_bytes``.
    """

    common_coin = 0
    yoso_rbc = 0
    for tag, raw_size in transport_by_tag.items():
        size = int(raw_size)
        if size < 0:
            raise ValueError("transport byte count must be non-negative")
        namespace = str(tag).strip("/").split("/", 1)[0].upper()
        if any(
            token in namespace
            for token in ("COIN", "COMMON-COIN", "THRESHOLD_BLS")
        ):
            common_coin += size
        elif namespace.startswith(("A2DPSS:", "RU22:", "BHK22:")):
            yoso_rbc += size

    total = int(total_bytes)
    if total < common_coin + yoso_rbc:
        raise ValueError("classified transport bytes exceed total bytes")
    return {
        "yoso_rbc": yoso_rbc,
        "dag_protocol": total - common_coin - yoso_rbc,
        "common_coin": common_coin,
        "total": total,
    }


def _dag_coin_mode(raw: str, *, replay_enabled: bool) -> tuple[str, bool]:
    requested = str(raw).strip().lower()
    if requested not in {"ideal", "threshold-bls"}:
        raise ValueError(
            "DAG_TOB_COIN_MODE must be 'ideal' or 'threshold-bls'"
        )
    # Loading a replay fixture must never change the explicitly requested
    # common-coin implementation.  The compatibility flag is always false.
    return requested, False


def _node_id_set(raw: str, *, M: int, name: str) -> frozenset[int]:
    if not str(raw).strip():
        return frozenset()
    try:
        node_ids = frozenset(int(item.strip()) for item in str(raw).split(","))
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated node-id list") from exc
    invalid = sorted(node_id for node_id in node_ids if not 0 <= node_id < M)
    if invalid:
        raise ValueError(f"{name} contains out-of-range node ids: {invalid}")
    return node_ids


def _legacy_dpss_smoke_enabled() -> bool:
    """Return only the explicit transcript-only compatibility mode."""

    return _env_bool("DAG_TOB_LEGACY_DPSS_SMOKE", False)


def _deprecated_dpss_alias_enabled() -> bool:
    return _env_bool(
        "DAG_TOB_DPSS_SMOKE", False
    )


def _a2dpss_smoke_enabled() -> bool:
    """Read the formal Algorithm-4 flag and its deprecated old alias."""

    return _env_bool(
        "DAG_TOB_A2DPSS_SMOKE", False
    ) or _deprecated_dpss_alias_enabled()


_HANDOVER_MODE_ALIASES = {
    "none": "none",
    "a2dpss": "a2dpss",
    "algorithm-4": "a2dpss",
    "ocdpss": "ocdpss",
    "algorithm-5": "ocdpss",
    "legacy-dpss": "legacy-dpss",
}


def _normalize_handover_mode(value: str) -> str:
    try:
        return _HANDOVER_MODE_ALIASES[
            str(value).strip().lower()
        ]
    except KeyError as exc:
        raise ValueError(
            "handover mode must be one of "
            "none, a2dpss, ocdpss, legacy-dpss"
        ) from exc


def _consume_handover_mode_cli(argv=None) -> str | None:
    """Remove this runner's optional mode flag before HbmpcConfig parsing."""

    arguments = sys.argv if argv is None else argv
    found = []
    index = 1
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--handover-mode":
            if index + 1 >= len(arguments):
                raise ValueError(
                    "--handover-mode requires a value"
                )
            found.append(arguments[index + 1])
            del arguments[index : index + 2]
            continue
        if argument.startswith("--handover-mode="):
            found.append(argument.split("=", 1)[1])
            del arguments[index]
            continue
        index += 1
    if len(found) > 1:
        raise ValueError(
            "--handover-mode may be specified only once"
        )
    return (
        _normalize_handover_mode(found[0])
        if found
        else None
    )


def _install_cli_handover_mode() -> None:
    cli_mode = _consume_handover_mode_cli()
    if cli_mode is None:
        return
    environment_mode = os.environ.get("DAG_TOB_HANDOVER_MODE")
    if (
        environment_mode is not None
        and _normalize_handover_mode(environment_mode)
        != cli_mode
    ):
        raise ValueError(
            "--handover-mode conflicts with DAG_TOB_HANDOVER_MODE"
        )
    os.environ["DAG_TOB_HANDOVER_MODE"] = cli_mode


def _handover_mode_from_env() -> str:
    """Resolve one explicit runner mode and reject ambiguous combinations."""

    requested = []
    named_mode = os.environ.get("DAG_TOB_HANDOVER_MODE")
    if named_mode is not None:
        requested.append(
            _normalize_handover_mode(named_mode)
        )
    if _a2dpss_smoke_enabled():
        requested.append("a2dpss")
    if _env_bool("DAG_TOB_OCDPSS_SMOKE", False):
        requested.append("ocdpss")
    if _legacy_dpss_smoke_enabled():
        requested.append("legacy-dpss")
    active = {mode for mode in requested if mode != "none"}
    if "none" in requested and active:
        raise ValueError(
            "DAG_TOB_HANDOVER_MODE=none conflicts with a smoke flag"
        )
    if len(active) > 1:
        raise ValueError(
            "A2DPSS, OCDPSS, and legacy DPSS modes are mutually exclusive"
        )
    return next(iter(active), "none")


def _configure_logging(my_id: int) -> None:
    os.makedirs("log", exist_ok=True)
    logfile = f"log/dag-tob-phase3-{my_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:[%(filename)s:%(lineno)s]:[%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(logfile, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _load_crypto_from_config():
    pbk = pickle.loads(base64.b64decode(HbmpcConfig.extras["public_key"]))
    pvk = pickle.loads(base64.b64decode(HbmpcConfig.extras["private_key"]))

    pks = base64.b64decode(HbmpcConfig.extras["pks_acss"])
    sk = base64.b64decode(HbmpcConfig.extras["sk_acss"])
    serialized_srs = base64.b64decode(HbmpcConfig.extras["SRS"])
    deserialized_srs_kzg = json.loads(serialized_srs.decode("utf-8"))
    srs = {
        "Pk": json.dumps(deserialized_srs_kzg["Pk"]).encode("utf-8"),
        "Vk": json.dumps(deserialized_srs_kzg["Vk"]).encode("utf-8"),
    }
    return pbk, pvk, pks, sk, srs


def _g1_raw(point: dict) -> str:
    return (
        b"\x04"
        + int(point["X"]).to_bytes(48, byteorder="big")
        + int(point["Y"]).to_bytes(48, byteorder="big")
    ).hex()


def _gh_from_srs(srs: dict) -> tuple[str, str]:
    pk = json.loads(srs["Pk"].decode("utf-8"))
    return _g1_raw(pk["G1_g"][0]), _g1_raw(pk["G1_h"][0])


def _vertex_committee_label(epoch: str, parent_round: int) -> bytes:
    return f"ATCAAS:DAGTOB:{epoch}:VERTEX_COMMITTEE:{parent_round}".encode("utf-8")


def _highest_ready_round(dag: DagTOB) -> int:
    ready = [round_id for round_id in dag.DAG if dag.is_round_ready(round_id)]
    return max(ready) if ready else 0


def _incomplete_runtime_reason(
    *, highest_ready_round: int, max_rounds: int
) -> str:
    """Classify a fail-closed runtime limit without claiming protocol success."""

    if int(highest_ready_round) >= int(max_rounds):
        return "round_budget_exhausted"
    return "max_runtime_exceeded"


def _demo_qid(epoch: str, instance: str) -> str:
    return Qid.build(
        "demo",
        epoch=epoch,
        service="multi-qid-smoke",
        instance=instance,
    ).encode()


def _broadcast_demo_message(dag: DagTOB, qid: str, label: str, my_id: int) -> None:
    msg = TobMessage(
        kind="demo",
        sender=my_id,
        payload={"label": label, "node": my_id},
    )
    dag.t_broadcast(qid, f"{label}-node-{my_id}", msg.to_bytes())


def _inject_workload_batches(
    dag, workload, lifecycle, trigger, my_id, events, *, clock=None
):
    now = (clock or time.time)()
    for batch in workload.batches_for_trigger(trigger):
        for instance in batch.instances:
            qid = workload.qid(instance)
            lifecycle.mark_released(qid, now)
            should_broadcast = getattr(workload, "should_broadcast", None)
            if should_broadcast is None or should_broadcast(instance, my_id):
                for index in range(workload.messages_per_node):
                    dag.t_broadcast(
                        qid,
                        workload.mid(instance, my_id, index),
                        workload.payload(instance, my_id, index),
                    )
            events.append(
                {
                    "batch_id": batch.batch_id,
                    "trigger": batch.trigger,
                    "instance": instance,
                    "qid": qid,
                    "request_release_time": now,
                }
            )
    return now


async def _wait_for_workload_pool_ready(
    dag,
    workload,
    lifecycle,
    instances,
    *,
    timeout: float,
    clock=None,
    cpu_metrics=None,
):
    """Wait until the actual per-query predicate J accepts every instance."""

    now = clock or time.time
    pending = {str(instance) for instance in instances}
    deadline = now() + max(0.0, float(timeout))
    while pending and now() < deadline:
        for instance in tuple(sorted(pending)):
            qid = workload.qid(instance)
            entries = set(dag._messages_for_qid(qid))
            expected = (
                workload.expected_entry_count(instance)
                if hasattr(workload, "expected_entry_count")
                else workload.messages_per_node * dag.M
            )
            if len(entries) < expected:
                continue
            registry = getattr(dag.wait_predicate, "registry", None)
            before = (
                registry.metrics_snapshot()["select"]
                if cpu_metrics is not None and registry is not None
                else None
            )
            cpu_started = time.process_time() if cpu_metrics is not None else None
            try:
                block = dag.wait_predicate.select_block({qid: entries}, set()) or {}
            finally:
                if cpu_metrics is not None:
                    cpu_metrics["cpu_seconds"] += time.process_time() - cpu_started
                    cpu_metrics["invocations"] += 1
                    if before is not None:
                        after = registry.metrics_snapshot()["select"]
                        cpu_metrics["registry_cpu_seconds"] += (
                            after["cpu_seconds"] - before["cpu_seconds"]
                        )
                        cpu_metrics["registry_invocations"] += (
                            after["invocations"] - before["invocations"]
                        )
            if qid in block:
                lifecycle.mark_pool_ready(qid, now())
                pending.remove(instance)
        if pending:
            await asyncio.sleep(0.05)
    if pending:
        raise RuntimeError(
            "timed out waiting for J(qid,S)=true: "
            + ",".join(sorted(pending))
        )


async def _wait_for_pool_entries(
    dag: DagTOB,
    qid: str,
    min_entries: int,
    timeout: float,
) -> int:
    deadline = time.time() + max(0.0, timeout)
    while time.time() < deadline:
        pool_size = len(dag.Pool.get(qid, set()))
        if pool_size >= min_entries:
            return pool_size
        await asyncio.sleep(0.05)
    return len(dag.Pool.get(qid, set()))


async def _wait_for_ru22_structural_pool(
    dag: DagTOB,
    runtime: RU22DAGHandoverRuntime,
    timeout: float,
):
    """Wait for a structural common parent without verifying any proof."""

    deadline = time.time() + max(0.0, timeout)
    state = runtime.structural_pool_state(())
    while time.time() < deadline:
        state = runtime.structural_pool_state(
            dag._messages_for_qid(runtime.qid)
        )
        complete = (
            state.valid_batch_count >= len(runtime.source_role_ids)
        )
        if state.ready and (
            not runtime.requires_complete_source_pool or complete
        ):
            return state
        await asyncio.sleep(0.05)
    return state


async def _run(
    *,
    peers,
    n: int,
    t: int,
    my_id: int,
    start_time: int,
    M: int,
    expected_committee_size: int,
    committee_m: int,
    quorum_h: int,
    epoch: str,
    rounds: int = 12,
    target_deliveries: int = 1,
    max_runtime: float = 90.0,
    lookahead_rounds: int = 4,
    retention_rounds: int = 8,
    stop_grace_seconds: float = 10.0,
    completion_service_grace_seconds: float = 0.0,
    multi_qid_smoke: bool = False,
    unique_qid_per_node: bool = False,
    dpss_smoke: bool = False,
    a2dpss_smoke: bool = False,
    ocdpss_smoke: bool = False,
    ru22_smoke: bool = False,
    bhk22_smoke: bool = False,
    ocdpss_paired: bool = False,
    a2dpss_pipeline: bool = False,
    coin_mode_requested: str = "ideal",
    coin_timeout: float = 30.0,
    coin_silent_nodes=(),
    a2dpss_fixture: A2DPSSPairedFixture | None = None,
    ru22_manifest: RU22DAGExperimentManifest | None = None,
    ru22_local_shard: RU22DAGLocalShard | None = None,
    bhk22_manifest: BHK22DAGExperimentManifest | None = None,
    bhk22_local_shard: BHK22NodePrivateShard | None = None,
    ru22_operation_context=None,
    bhk22_operation_context: BHK22OperationPublicContext | None = None,
    bhk22_operation_dealers=None,
    ocdpss_context_override: OCDPSSContext | None = None,
    ocdpss_operation_adapter_override=None,
    ocdpss_holder_share_override=None,
    ocdpss_successor_rsk_override=None,
    n_r: int | None = None,
    t_r: int | None = None,
    n_a: int | None = None,
    t_a: int | None = None,
    m_a: int | None = None,
    coin_c: int | None = None,
    coin_t: int | None = None,
    coin_k: int | None = None,
    experiment_barrier=None,
):
    frozen_summary = None
    completion_service_started_at = None
    experiment_done_task = None
    pipeline_started_at = None
    pipeline_events = []
    pipeline_gates = []
    pipeline_cpu_stages = {
        stage: {"cpu_seconds": 0.0, "invocations": 0}
        for stage in (
            "ready_check", "candidate_formation", "formal_predicate",
            "public_finalization", "successor_reconstruction",
        )
    }
    pipeline_ready_cpu = {
        "cpu_seconds": 0.0, "invocations": 0,
        "registry_cpu_seconds": 0.0, "registry_invocations": 0,
    }

    def workload_clock():
        if not a2dpss_pipeline:
            return time.time()
        if pipeline_started_at is None:
            raise RuntimeError("A2DPSS pipeline event precedes the ready barrier")
        return time.perf_counter() - pipeline_started_at

    def pipeline_cpu_snapshot():
        registry_metrics = pipeline_registry.metrics_snapshot()
        stages = {name: dict(value) for name, value in pipeline_cpu_stages.items()}
        stages["ready_check"] = {
            key: pipeline_ready_cpu[key] for key in ("cpu_seconds", "invocations")
        }
        stages["candidate_formation"] = {
            "cpu_seconds": max(
                0.0,
                registry_metrics["select"]["cpu_seconds"]
                - pipeline_ready_cpu["registry_cpu_seconds"],
            ),
            "invocations": (
                registry_metrics["select"]["invocations"]
                - pipeline_ready_cpu["registry_invocations"]
            ),
        }
        stages["formal_predicate"] = {
            key: registry_metrics["verify"][key]
            for key in ("cpu_seconds", "invocations")
        }
        return {
            "clock": "process_time",
            "stages": stages,
            "total_cpu_seconds": sum(value["cpu_seconds"] for value in stages.values()),
        }
    n_a = expected_committee_size if n_a is None else int(n_a)
    m_a = committee_m if m_a is None else int(m_a)
    t_a = n_a - quorum_h if t_a is None else int(t_a)
    t_r = t if t_r is None else int(t_r)
    n_r = (
        min(M, max(t_r + 2, t_r + 1, 3))
        if n_r is None
        else int(n_r)
    )
    if expected_committee_size != n_a or committee_m != m_a:
        raise ValueError("DAG agreement aliases do not match n_a/m_a")
    if not 0 < quorum_h <= m_a <= M:
        raise ValueError(
            "DAG agreement parameters must satisfy "
            "0 < quorum_h <= m_a <= M"
        )
    if 3 * t_a >= n_a or quorum_h != n_a - t_a:
        raise ValueError(
            "DAG paper parameters require t_a < n_a/3 and "
            "quorum_h = n_a-t_a"
        )
    if 2 * quorum_h - m_a <= t_a:
        raise ValueError("DAG quorum intersection requires 2*h-m_a > t_a")
    if not 0 <= t_r or not 2 * t_r < n_r <= M:
        raise ValueError("DAG resharing parameters require t_r < n_r/2 <= M")
    if coin_c is not None:
        coin_c = int(coin_c)
        coin_t = int(coin_t) if coin_t is not None else None
        coin_k = int(coin_k) if coin_k is not None else None
        if (
            not 1 <= coin_c <= M
            or coin_t is None
            or coin_k is None
            or not 0 <= coin_t < coin_k <= coin_c - coin_t
        ):
            raise ValueError(
                "DAG coin parameters require "
                "0 <= coin_t < coin_k <= coin_c-coin_t and coin_c <= M"
            )
    elif coin_t is not None or coin_k is not None:
        raise ValueError("DAG coin_t/coin_k require coin_c")
    agreement_t = t_a
    agreement_t_impl = m_a - quorum_h
    if a2dpss_fixture is not None and not a2dpss_smoke:
        raise ValueError("paired A2DPSS fixture requires A2DPSS mode")
    if a2dpss_fixture is not None:
        if a2dpss_fixture.trial_epoch != epoch:
            raise ValueError("paired fixture trial_epoch does not match run epoch")
        if os.environ.get("ATCAAS_PROTOCOL_TRACE"):
            raise ValueError(
                "paired A2DPSS runs reject legacy protocol/coin replay"
            )
        if not os.environ.get("ATCAAS_ROLE_TRACE"):
            raise ValueError("paired A2DPSS runs require a role-only trace")
        if str(coin_mode_requested).strip().lower() != "threshold-bls":
            raise ValueError("paired A2DPSS runs require threshold-bls coin")
    if (ru22_manifest is None) != (ru22_local_shard is None):
        raise ValueError("RU22 manifest and local shard must be supplied together")
    if ru22_manifest is not None and not ru22_smoke:
        raise ValueError("RU22 fixture files require RU22 mode")
    if ru22_smoke and ru22_manifest is None:
        raise ValueError("RU22 mode requires manifest and local shard")
    if (bhk22_manifest is None) != (bhk22_local_shard is None):
        raise ValueError("BHK22 manifest and local shard must be supplied together")
    if bhk22_manifest is not None and not bhk22_smoke:
        raise ValueError("BHK22 fixture files require BHK22 mode")
    if bhk22_smoke and bhk22_manifest is None:
        raise ValueError("BHK22 mode requires manifest and local shard")
    if ru22_operation_context is not None and not ru22_smoke:
        raise ValueError("RU22 operation context requires RU22 mode")
    if bhk22_operation_context is not None and not bhk22_smoke:
        raise ValueError("BHK operation context requires BHK22 mode")
    if ocdpss_context_override is not None and not ocdpss_smoke:
        raise ValueError("OCDPSS context override requires OCDPSS mode")
    if dpss_smoke:
        # Keep the historical protocol out of formal Algorithm-4/5 imports.
        from beaver.legacy_yoso_dpss import (
            DPSSJustifierManager,
            DPSSReconstructionManager,
            make_resharing_transcript,
            run_resharing,
        )

        logging.getLogger(f"dag_tob.phase3.{my_id}").warning(
            "[LEGACY-DPSS] transcript-only compatibility smoke enabled; "
            "this is not formal Algorithm-4 A2DPSS and must not supply "
            "experiment data"
        )
    if a2dpss_smoke:
        logging.getLogger(f"dag_tob.phase3.{my_id}").info(
            "[ALGORITHM-4:A2DPSS] formal contribution handover mode enabled"
        )
    if ocdpss_smoke:
        logging.getLogger(f"dag_tob.phase3.{my_id}").info(
            "[ALGORITHM-5:OCDPSS] operation-coupled handover mode enabled"
        )
    if ru22_smoke:
        logging.getLogger(f"dag_tob.phase3.{my_id}").info(
            "[RU22] DAG-TOB handover mode enabled"
        )
    if bhk22_smoke:
        logging.getLogger(f"dag_tob.phase3.{my_id}").info(
            "[BHK22] optimistic DAG-TOB handover mode enabled"
        )
    active_handover_modes = sum(
        bool(value)
        for value in (
            dpss_smoke,
            a2dpss_smoke,
            ocdpss_smoke,
            ru22_smoke,
            bhk22_smoke,
        )
    )
    if active_handover_modes > 1:
        raise ValueError(
            "BHK22, RU22, A2DPSS, OCDPSS, and legacy DPSS modes are mutually exclusive"
        )
    if (a2dpss_smoke or ocdpss_smoke or ru22_smoke or bhk22_smoke) and (
        multi_qid_smoke or unique_qid_per_node
    ):
        raise ValueError(
            "formal handover smoke mode is mutually exclusive with demo workloads"
        )
    if ocdpss_paired and (
        not multi_qid_smoke
        or dpss_smoke
        or a2dpss_smoke
        or ocdpss_smoke
        or unique_qid_per_node
    ):
        raise ValueError(
            "OCDPSS paired mode requires multi-qid mode and excludes smoke/legacy modes"
        )
    if a2dpss_pipeline:
        if (
            not multi_qid_smoke or unique_qid_per_node or ocdpss_paired
            or active_handover_modes
        ):
            raise ValueError(
                "A2DPSS pipeline requires multi-qid mode and excludes other handover modes"
            )
        if (
            str(coin_mode_requested).strip().lower() != "threshold-bls"
            or os.environ.get("ATCAAS_PROTOCOL_TRACE")
            or not os.environ.get("ATCAAS_ROLE_TRACE")
        ):
            raise ValueError(
                "A2DPSS pipeline requires threshold-BLS coin and a role-only trace"
            )
        if experiment_barrier is None:
            raise ValueError("A2DPSS pipeline requires the global completion barrier")
    crypto = _load_crypto_from_config()
    _, _, _, _, srs = crypto
    g_hex = generator_hex_from_srs(srs)
    dpss_g_hex, dpss_h_hex = _gh_from_srs(srs)
    vrf_sk = HbmpcConfig.extras.get("vrf_sk")
    vrf_pks = HbmpcConfig.extras.get("vrf_pks")
    if vrf_sk is None or vrf_pks is None:
        raise ValueError("Generate node VRF keys with scripts.run_key_gen_atcaas before running")
    threshold_fraction = expected_committee_size / M
    committee_selection_mode = normalize_committee_selection_mode(
        os.environ.get(
            "ATCAAS_COMMITTEE_SELECTION_MODE",
            BERNOULLI_COMMITTEE_MODE,
        )
    )
    fixed_committee_seed = os.environ.get(
        "ATCAAS_FIXED_COMMITTEE_SEED",
        epoch,
    )

    def _fixed_roles(label: bytes) -> tuple[int, ...]:
        return fixed_size_committee(
            label=label,
            population_size=M,
            committee_size=expected_committee_size,
            seed=fixed_committee_seed,
        )

    def _local_vertex_sortition(label: bytes):
        if committee_selection_mode == FIXED_SIZE_COMMITTEE_MODE:
            if my_id not in _fixed_roles(label):
                return False, None
            # Membership comes from the exact-size public fixture.  A full
            # probability keeps the cryptographic VRF proof while avoiding a
            # second, conflicting Bernoulli selection decision.
            return vrf_prove_sortition(
                label=label,
                vrf_sk=vrf_sk,
                g_hex=g_hex,
                threshold_fraction=1.0,
            )
        return vrf_prove_sortition(
            label=label,
            vrf_sk=vrf_sk,
            g_hex=g_hex,
            threshold_fraction=threshold_fraction,
        )

    role_trace_path = os.environ.get("ATCAAS_ROLE_TRACE")
    replay_trace_path = os.environ.get("ATCAAS_PROTOCOL_TRACE")
    if role_trace_path and replay_trace_path:
        raise ValueError(
            "ATCAAS_ROLE_TRACE and legacy ATCAAS_PROTOCOL_TRACE are "
            "mutually exclusive"
        )
    role_trace_artifact = (
        CommitteeRoleTraceArtifact.from_file(role_trace_path)
        if role_trace_path
        else None
    )
    if role_trace_artifact is not None:
        role_trace_artifact.validate_population(M)
        for wave in range(1, max(1, (rounds + 3) // 4) + 1):
            leader_round = 4 * (wave - 1) + 1
            if leader_round > rounds:
                break
            label = _vertex_committee_label(epoch, leader_round - 1)
            recorded_roles = role_trace_artifact.committees.roles(wave)
            if committee_selection_mode == FIXED_SIZE_COMMITTEE_MODE:
                expected_roles = _fixed_roles(label)
                if recorded_roles != expected_roles:
                    raise ValueError(
                        "role-only trace does not match the fixed-size "
                        f"committee fixture: wave={wave} "
                        f"expected={list(expected_roles)} "
                        f"recorded={list(recorded_roles)}"
                    )
            else:
                selected, _proof = _local_vertex_sortition(label)
                recorded = my_id in recorded_roles
                if bool(selected) != recorded:
                    raise ValueError(
                        "role-only trace does not match the configured VRF "
                        f"selection: node={my_id} wave={wave} "
                        f"selected={bool(selected)} recorded={recorded}"
                    )
        replay_trace = None
        ideal_coin = DeterministicIdealCoin(epoch)
        leader_committees = role_trace_artifact.committees
        trace_source = "recorded"
        role_trace_format = "role-only"
        role_trace_digest = role_trace_artifact.digest()
    elif replay_trace_path:
        replay_trace = ProtocolReplayTrace.from_file(replay_trace_path)
        ideal_coin = replay_trace.coin
        leader_committees = replay_trace.committees
        legacy_roles = CommitteeRoleTraceArtifact(
            replay_trace.committees.roles_by_index
        )
        legacy_roles.validate_population(M)
        trace_source = "recorded"
        role_trace_format = "legacy-protocol-replay"
        role_trace_digest = legacy_roles.digest()
    else:
        replay_trace = None
        ideal_coin = DeterministicIdealCoin(epoch)
        if committee_selection_mode == FIXED_SIZE_COMMITTEE_MODE:
            leader_committees = RecordedCommitteeRoleTrace(
                {
                    wave: _fixed_roles(
                        _vertex_committee_label(
                            epoch,
                            4 * (wave - 1),
                        )
                    )
                    for wave in range(1, max(1, (rounds + 3) // 4) + 1)
                }
            )
            trace_source = "fixed-size-fixture"
            role_trace_format = "runtime-fixed-size-fixture"
        else:
            raise ValueError("Generate a role trace with scripts.generate_dag_role_trace for Bernoulli selection")
        generated_roles = CommitteeRoleTraceArtifact(
            leader_committees.roles_by_index
        )
        generated_roles.validate_population(M)
        role_trace_digest = generated_roles.digest()
    effective_coin_mode, coin_mode_forced_by_replay = _dag_coin_mode(
        coin_mode_requested,
        replay_enabled=replay_trace is not None,
    )
    coin_silent_nodes = frozenset(int(node_id) for node_id in coin_silent_nodes)
    invalid_silent_nodes = sorted(
        node_id for node_id in coin_silent_nodes if not 0 <= node_id < M
    )
    if invalid_silent_nodes:
        raise ValueError(
            "coin_silent_nodes contains out-of-range node ids: "
            f"{invalid_silent_nodes}"
        )
    if effective_coin_mode == "ideal" and coin_silent_nodes:
        if coin_mode_forced_by_replay:
            logging.warning(
                "[COMMON-COIN] protocol replay ignored threshold coin "
                "silent-node configuration"
            )
            coin_silent_nodes = frozenset()
        else:
            raise ValueError(
                "coin_silent_nodes is only meaningful in threshold-bls mode"
            )
    threshold_coin_keys = (
        load_threshold_coin_key_material(
            HbmpcConfig.extras,
            network_size=M,
            my_id=my_id,
        )
        if effective_coin_mode == "threshold-bls"
        else None
    )
    if threshold_coin_keys is not None:
        loaded_coin_c = len(threshold_coin_keys.config.committee)
        loaded_coin_k = threshold_coin_keys.config.threshold
        if coin_c is not None and coin_c != loaded_coin_c:
            raise ValueError("configured coin_c does not match loaded coin committee")
        if coin_k is not None and coin_k != loaded_coin_k:
            raise ValueError("configured coin_k does not match loaded coin threshold")
        coin_c = loaded_coin_c
        coin_k = loaded_coin_k
    if coin_mode_forced_by_replay:
        logging.warning(
            "[COMMON-COIN] protocol replay forced coin mode from "
            "threshold-bls to ideal"
        )
    leader_oracle = IdealCoinLeaderOracle(ideal_coin, leader_committees)
    deliver_events = []
    dpss_events = []
    a2dpss_events = []
    ocdpss_events = []
    broadcast_start_time = None
    broadcast_end_time = None
    first_deliver_time = None
    workload = None
    workload_source = None
    lifecycle = None
    if multi_qid_smoke:
        if a2dpss_pipeline:
            from beaver.a2dpss_pipeline_workload import A2DPSSPairedWorkload

            fixture_path = os.environ.get("ATCAAS_A2DPSS_PIPELINE_FIXTURE")
            if not fixture_path:
                raise ValueError("ATCAAS_A2DPSS_PIPELINE_FIXTURE is required")
            workload = A2DPSSPairedWorkload.from_file(fixture_path)
            workload_source = "recorded-a2dpss-pipeline-fixture"
            if target_deliveries != len(workload.qids()):
                raise ValueError("A2DPSS pipeline target must equal its fixture query count")
        elif ocdpss_paired:
            fixture_path = os.environ.get("ATCAAS_OCDPSS_FIXTURE")
            if not fixture_path:
                raise ValueError("ATCAAS_OCDPSS_FIXTURE is required in paired mode")
            workload = OCDPSSPairedWorkload.from_file(fixture_path)
            workload_source = "recorded-ocdpss-fixture"
        else:
            workload, workload_source = load_dynamic_workload(
                os.environ.get("ATCAAS_WORKLOAD_TRACE"),
                messages_per_node=_env_int("DAG_TOB_MESSAGES_PER_NODE", 1),
            )
        if workload.messages_per_node != 1:
            raise ValueError(
                "DAG message-RBC currently supports one message per sender/qid; "
                "R6 workload messages_per_node must be 1"
            )
        lifecycle = QueryLifecycleMetrics(workload)
    dpss_transcript_rbc = dpss_smoke and _env_bool(
        "DAG_TOB_DPSS_TRANSCRIPT_RBC",
        True,
    )
    dpss_smoke_inject_all = _env_bool("DAG_TOB_DPSS_SMOKE_INJECT_ALL", False)
    dpss_pool_wait_seconds = _env_float("DAG_TOB_DPSS_POOL_WAIT_SECONDS", 20.0)
    ocdpss_pool_wait_seconds = _env_float(
        "DAG_TOB_OCDPSS_POOL_WAIT_SECONDS",
        dpss_pool_wait_seconds,
    )
    ocdpss_hops = _env_int("DAG_TOB_OCDPSS_HOPS", 1)
    if ocdpss_hops < 1:
        raise ValueError("DAG_TOB_OCDPSS_HOPS must be positive")
    if not ocdpss_smoke and ocdpss_hops != 1:
        raise ValueError("DAG_TOB_OCDPSS_HOPS requires OCDPSS smoke mode")
    a2dpss_hops = _env_int("DAG_TOB_A2DPSS_HOPS", 1)
    if a2dpss_hops < 1:
        raise ValueError("DAG_TOB_A2DPSS_HOPS must be positive")
    if not a2dpss_smoke and a2dpss_hops != 1:
        raise ValueError("DAG_TOB_A2DPSS_HOPS requires A2DPSS smoke mode")

    def _verify_vertex(v: Vertex) -> bool:
        label = _vertex_committee_label(epoch, v.round - 1)
        if (
            committee_selection_mode == FIXED_SIZE_COMMITTEE_MODE
            and v.source not in _fixed_roles(label)
        ):
            return False
        proof_valid = vrf_verify_sortition(
            label=label,
            proof=v.proof,
            sender_id=v.source,
            vrf_pks=vrf_pks,
            g_hex=g_hex,
            threshold_fraction=(
                1.0
                if committee_selection_mode == FIXED_SIZE_COMMITTEE_MODE
                else threshold_fraction
            ),
        )
        if not proof_valid:
            return False
        if v.round % 4 == 1:
            wave = (v.round + 3) // 4
            return v.source in leader_committees.roles(wave)
        return True

    dpss_local = None
    dpss_reconstruction_manager = None
    a2dpss_context = None
    a2dpss_public_finalizer = None
    a2dpss_reconstruction_manager = None
    a2dpss_holder_state = None
    a2dpss_source_holder_ids = ()
    a2dpss_successor_rpk_list = ()
    a2dpss_successor_role_indices = {}
    a2dpss_my_successor_rsk = None
    a2dpss_handover_metrics = None
    a2dpss_lifecycle = None
    a2dpss_completed_public_results = {}
    ru22_runtime = None
    bhk22_runtime = None
    ocdpss_context = None
    ocdpss_adapter = None
    ocdpss_justifier_manager = None
    ocdpss_public_finalizer = None
    ocdpss_reconstruction_manager = None
    ocdpss_holder_state = None
    ocdpss_handover_metrics = None
    ocdpss_lifecycle = None
    ocdpss_completed_public_results = {}
    ocdpss_source_holder_ids = ()
    ocdpss_successor_rpk_list = ()
    ocdpss_successor_role_indices = {}
    ocdpss_my_successor_rsk = None
    ocdpss_am_successor = False
    ocdpss_am_holder = False
    paired_ocdpss_justifier = None
    paired_ocdpss_finalizer = None
    paired_ocdpss_reconstruction = None
    paired_ocdpss_events = []
    pipeline_registry = None
    pipeline_justifier = None
    pipeline_finalizer = None
    pipeline_reconstruction = None
    paired_public_finalize_seconds = 0.0
    paired_public_finalize_invocations = 0
    paired_reconstruction_seconds = 0.0
    paired_reconstruction_invocations = 0
    wait_predicate = None
    if dpss_smoke or a2dpss_smoke or ocdpss_smoke:
        dpss_local = build_handover_local_config(
            HbmpcConfig.extras,
            g_hex=dpss_g_hex,
            h_hex=dpss_h_hex,
            my_id=my_id,
            M=M,
            service_t=t_r,
            epoch=0,
        )
        if (
            len(dpss_local.source_committee) != n_r
            or len(dpss_local.successor_committee) != n_r
            or dpss_local.ctx.service_t != t_r
        ):
            raise ValueError(
                "resolved DPSS context does not match n_r/t_r parameters"
            )
    if bhk22_smoke:
        if bhk22_operation_context is not None:
            bhk22_runtime = BHK22OperationDAGRuntime(
                manifest=bhk22_manifest,
                local_shard=bhk22_local_shard,
                operation_context=bhk22_operation_context,
                operation_dealers=list(bhk22_operation_dealers or ()),
                my_node_id=my_id,
            )
        else:
            bhk22_runtime = BHK22DAGRuntime(
                manifest=bhk22_manifest,
                local_shard=bhk22_local_shard,
                my_node_id=my_id,
            )
        bhk22_runtime.validate_network_parameters(
            M=M,
            n_r=n_r,
            t_r=t_r,
            n_a=n_a,
            t_a=t_a,
            m_a=m_a,
            quorum_h=quorum_h,
            coin_c=coin_c,
            coin_t=coin_t,
            coin_k=coin_k,
            trial_epoch=epoch,
        )
        bhk22_runtime.validate_context_bindings(
            role_trace_digest=role_trace_digest,
            coin_config_extras=HbmpcConfig.extras or {},
        )
        bhk22_runtime.prepare()
        registry = JustifierRegistry()
        for phase_context in bhk22_runtime.controller.phase_contexts.values():
            registry.register(
                Qid.decode(phase_context.qid).kind,
                bhk22_runtime.justifier,
            )
        if bhk22_operation_context is not None:
            from beaver.threshold_operation_contract import BHK22_OPERATION_QID_KINDS
            for kind in BHK22_OPERATION_QID_KINDS.values():
                registry.register(kind, bhk22_runtime.justifier)
        wait_predicate = TypedWaitPredicate(registry)
    elif ru22_smoke:
        if ru22_operation_context is not None:
            ru22_runtime = RU22OperationDAGRuntime(
                manifest=ru22_manifest,
                local_shard=ru22_local_shard,
                operation_context=ru22_operation_context,
                my_node_id=my_id,
            )
        else:
            ru22_runtime = RU22DAGHandoverRuntime(
                manifest=ru22_manifest,
                local_shard=ru22_local_shard,
                my_node_id=my_id,
            )
        ru22_runtime.validate_network_parameters(
            M=M,
            t_a=t_a,
            n_r=n_r,
            t_r=t_r,
            epoch=epoch,
            coin_mode=coin_mode_requested,
        )
        registry = JustifierRegistry()
        registry.register(Qid.decode(ru22_runtime.qid).kind, ru22_runtime.justifier)
        wait_predicate = TypedWaitPredicate(registry)
    elif dpss_smoke:
        dpss_justifier_manager = DPSSJustifierManager([dpss_local.ctx])
        registry = JustifierRegistry()
        registry.register("dpss", dpss_justifier_manager)
        wait_predicate = TypedWaitPredicate(registry)
        dpss_reconstruction_manager = DPSSReconstructionManager()
        if dpss_local.am_successor:
            dpss_reconstruction_manager.register_context(
                dpss_local.ctx,
                my_node_id=my_id,
                my_rsk=dpss_local.my_successor_rsk,
                on_new_share=lambda s, sh: dpss_events.append(
                    {
                        "event": "new_share",
                        "share_s": str(s),
                        "share_s_hat": str(sh),
                    }
                ),
            )
    elif a2dpss_smoke:
        legacy_context = dpss_local.ctx
        if a2dpss_fixture is not None:
            fixture_context = a2dpss_fixture.context
            comparable_fields = (
                "epoch",
                "source_committee",
                "successor_committee",
                "transfer_seq",
                "g_hex",
                "h_hex",
                "key_base",
                "successor_rpk_list",
                "service_t",
                "old_commitment_vector",
                "holder_eval_points",
                "successor_role_indices",
            )
            if any(
                getattr(fixture_context, field) != getattr(legacy_context, field)
                for field in comparable_fields
            ):
                raise ValueError(
                    "paired fixture is incompatible with local handover config"
                )
            a2dpss_context = fixture_context
            a2dpss_source_holder_ids = (
                a2dpss_fixture.source_holder_ids()
            )
        else:
            a2dpss_source_holder_ids = tuple(
                dpss_local.source_committee
            )
        a2dpss_successor_rpk_list = tuple(
            legacy_context.successor_rpk_list
        )
        a2dpss_successor_role_indices = dict(
            legacy_context.successor_role_indices
        )
        a2dpss_my_successor_rsk = dpss_local.my_successor_rsk
        if a2dpss_context is None:
            a2dpss_context = A2DPSSContext(
                epoch=legacy_context.epoch,
                sid=str(legacy_context.sid).replace(
                    "dpss",
                    "a2dpss",
                ),
                source_committee=legacy_context.source_committee,
                successor_committee=legacy_context.successor_committee,
                transfer_seq=legacy_context.transfer_seq,
                g_hex=legacy_context.g_hex,
                h_hex=legacy_context.h_hex,
                key_base=legacy_context.key_base,
                successor_rpk_list=legacy_context.successor_rpk_list,
                service_t=legacy_context.service_t,
                old_commitment_vector=legacy_context.old_commitment_vector,
                holder_eval_points=legacy_context.holder_eval_points,
                successor_role_indices=(
                    legacy_context.successor_role_indices
                ),
            )
        a2dpss_justifier_manager = A2DPSSJustifierManager()
        registry = JustifierRegistry()
        registry.register("a2dpss", a2dpss_justifier_manager)
        wait_predicate = TypedWaitPredicate(registry)
        a2dpss_public_finalizer = A2DPSSPublicFinalizer(
            a2dpss_justifier_manager
        )
        a2dpss_reconstruction_manager = (
            A2DPSSSuccessorReconstructionManager()
        )
        a2dpss_lifecycle = A2DPSSHandoverLifecycle(
            my_node_id=my_id,
            justifier=a2dpss_justifier_manager,
            reconstruction=a2dpss_reconstruction_manager,
        )
        a2dpss_lifecycle.register_initial(
            a2dpss_context,
            my_rsk=(
                dpss_local.my_successor_rsk
                if dpss_local.am_successor
                else None
            ),
            on_new_share=(
                (
                    lambda _s, _sh: a2dpss_events.append(
                        {
                            "event": "new_share",
                            "algorithm": "Algorithm 4",
                            "protocol": "a2dpss",
                            "my_node_id": my_id,
                        }
                    )
                )
                if dpss_local.am_successor
                else None
            ),
        )
        if dpss_local.am_holder and a2dpss_fixture is None:
            a2dpss_holder_state = A2DPSSHolderState(
                *dpss_local.my_holder_share
            )
        a2dpss_handover_metrics = A2DPSSHandoverMetrics(
            protocol="dag-tob",
            node=my_id,
            qid=a2dpss_context.encode_qid(),
            is_holder=my_id in a2dpss_source_holder_ids,
            is_successor=dpss_local.am_successor,
        )
        dpss_local = None
    elif ocdpss_smoke:
        if ocdpss_context_override is not None:
            ocdpss_context = ocdpss_context_override
            ocdpss_adapter = ocdpss_operation_adapter_override
            if ocdpss_adapter is None:
                raise ValueError("OCDPSS context override requires its operation adapter")
            ocdpss_source_holder_ids = tuple(
                sorted(ocdpss_context.holder_eval_points)
            )
            ocdpss_successor_role_indices = dict(
                ocdpss_context.successor_role_indices
            )
            ocdpss_successor_rpk_list = tuple(
                ocdpss_context.successor_rpk_list
            )
            ocdpss_am_holder = my_id in ocdpss_context.holder_eval_points
            ocdpss_am_successor = my_id in ocdpss_successor_role_indices
            ocdpss_my_successor_rsk = ocdpss_successor_rsk_override
            if ocdpss_am_holder:
                if ocdpss_holder_share_override is None:
                    raise ValueError("OCDPSS holder override is missing its share")
                ocdpss_holder_state = OCDPSSHolderState(
                    *ocdpss_holder_share_override
                )
        else:
            ocdpss_am_successor = dpss_local.am_successor
            ocdpss_am_holder = dpss_local.am_holder
            ocdpss_source_holder_ids = tuple(dpss_local.source_committee)
            ocdpss_successor_rpk_list = tuple(
                dpss_local.ctx.successor_rpk_list
            )
            ocdpss_successor_role_indices = dict(
                dpss_local.ctx.successor_role_indices
            )
            ocdpss_my_successor_rsk = dpss_local.my_successor_rsk
            operation_input = os.environ.get(
                "DAG_TOB_OCDPSS_OPERATION_INPUT",
                "atcaas-ocdpss-network-smoke",
            ).encode("utf-8")
            operation_generator = G1Point.from_hex(dpss_g_hex)

            def _operation_base(value: bytes) -> G1Point:
                return operation_generator * hash_to_scalar(
                    "atcaas-ocdpss-network-operation-base-v1",
                    bytes(value).hex(),
                    nonzero=True,
                )

            ocdpss_adapter = ThresholdExponentiationOperation(
                operation_id="atcaas-threshold-sign-v1",
                g_hex=dpss_g_hex,
                h_hex=dpss_h_hex,
                base_deriver=_operation_base,
            )
            legacy_context = dpss_local.ctx
            ocdpss_context = OCDPSSContext(
                epoch=legacy_context.epoch,
                sid=str(legacy_context.sid).replace("dpss", "ocdpss"),
                source_committee=legacy_context.source_committee,
                successor_committee=legacy_context.successor_committee,
                transfer_seq=legacy_context.transfer_seq,
                g_hex=legacy_context.g_hex,
                h_hex=legacy_context.h_hex,
                key_base=legacy_context.key_base,
                successor_rpk_list=legacy_context.successor_rpk_list,
                service_t=legacy_context.service_t,
                old_commitment_vector=legacy_context.old_commitment_vector,
                holder_eval_points=legacy_context.holder_eval_points,
                successor_role_indices=legacy_context.successor_role_indices,
                operation_id=ocdpss_adapter.operation_id,
                operation_input=operation_input,
                operation_public_parameters=ocdpss_adapter.public_context(
                    operation_input
                ),
            )
        ocdpss_justifier_manager = OCDPSSJustifierManager()
        registry = JustifierRegistry()
        registry.register("ocdpss", ocdpss_justifier_manager)
        wait_predicate = TypedWaitPredicate(registry)
        ocdpss_public_finalizer = OCDPSSPublicFinalizer(
            ocdpss_justifier_manager
        )
        ocdpss_reconstruction_manager = OCDPSSSuccessorReconstructionManager()
        ocdpss_lifecycle = OCDPSSHandoverLifecycle(
            my_node_id=my_id,
            justifier=ocdpss_justifier_manager,
            reconstruction=ocdpss_reconstruction_manager,
        )
        ocdpss_lifecycle.register_initial(
            ocdpss_context,
            ocdpss_adapter,
            my_rsk=(
                ocdpss_my_successor_rsk
                if ocdpss_am_successor
                else None
            ),
            on_new_share=(
                (
                    lambda _s, _sh: ocdpss_events.append(
                        {"event": "new_share", "my_node_id": my_id}
                    )
                )
                if ocdpss_am_successor
                else None
            ),
        )
        if ocdpss_am_holder and ocdpss_holder_state is None:
            ocdpss_holder_state = OCDPSSHolderState(
                *dpss_local.my_holder_share
            )
        ocdpss_handover_metrics = A2DPSSHandoverMetrics(
            protocol="dag-tob",
            node=my_id,
            qid=ocdpss_context.encode_qid(),
            is_holder=ocdpss_am_holder,
            is_successor=ocdpss_am_successor,
        )
        # ``build_handover_local_config`` is only an initial-state fixture bridge
        # and contains every genesis holder share.  The OCDPSS runner retains
        # only public committee data, this node's role key, and its one-shot
        # holder state; no later hop may read intermediate shares from config.
        dpss_local = None

    if a2dpss_pipeline:
        contexts = workload.registrations()
        if any(
            context.service_t != t_r
            or len(context.holder_eval_points) != n_r
            or len(context.successor_role_indices) != n_r
            for context in contexts
        ):
            raise ValueError("A2DPSS pipeline fixture committee parameters differ")
        pipeline_justifier = A2DPSSJustifierManager(contexts)
        pipeline_registry = JustifierRegistry()
        pipeline_registry.register("a2dpss", pipeline_justifier)
        wait_predicate = TypedWaitPredicate(pipeline_registry)
        pipeline_finalizer = A2DPSSPublicFinalizer(pipeline_justifier)
        pipeline_reconstruction = A2DPSSSuccessorReconstructionManager()
        pipeline_local = build_handover_local_config(
            HbmpcConfig.extras,
            g_hex=dpss_g_hex,
            h_hex=dpss_h_hex,
            my_id=my_id,
            M=M,
            service_t=t_r,
            epoch=0,
        )
        for context in contexts:
            pipeline_reconstruction.register_context(
                context,
                my_node_id=my_id,
                my_rsk=(
                    pipeline_local.my_successor_rsk
                    if my_id in context.successor_role_indices
                    else None
                ),
            )
        pipeline_local = None

    if ocdpss_paired:
        paired_registrations = workload.registrations()
        paired_ocdpss_justifier = OCDPSSJustifierManager(paired_registrations)
        registry = JustifierRegistry()
        registry.register("ocdpss", paired_ocdpss_justifier)
        wait_predicate = TypedWaitPredicate(registry)
        paired_ocdpss_finalizer = OCDPSSPublicFinalizer(
            paired_ocdpss_justifier
        )
        paired_ocdpss_reconstruction = OCDPSSSuccessorReconstructionManager()
        paired_local = build_handover_local_config(
            HbmpcConfig.extras,
            g_hex=dpss_g_hex,
            h_hex=dpss_h_hex,
            my_id=my_id,
            M=M,
            service_t=t_r,
            epoch=0,
        )
        for context, _adapter in paired_registrations:
            paired_ocdpss_reconstruction.register_context(
                context,
                my_node_id=my_id,
                my_rsk=(
                    paired_local.my_successor_rsk
                    if my_id in context.successor_role_indices
                    else None
                ),
            )
        paired_local = None

    bhk22_phase_tasks = set()
    bhk22_phase_errors = []

    async def _handle_bhk22_delivery(event):
        try:
            if bhk22_runtime is None or bhk22_rbc_manager is None:
                raise RuntimeError("BHK22 network path is not initialized")
            outbound = bhk22_runtime.handle_delivery(
                event,
                on_phase_registered=(
                    bhk22_rbc_manager.ensure_phase_instances
                ),
            )
            if outbound is not None:
                await bhk22_rbc_manager.broadcast_role_batch(
                    outbound.phase_context,
                    role_id=outbound.role_id,
                    batch_bytes=outbound.batch_bytes,
                )
                bhk22_runtime.mark_sent(outbound.phase_context.phase)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            bhk22_phase_errors.append(exc)
            logging.exception("[BHK22] phase delivery processing failed")

    def _deliver_cb(qid, entries):
        nonlocal first_deliver_time
        nonlocal paired_public_finalize_seconds
        nonlocal paired_public_finalize_invocations
        nonlocal paired_reconstruction_seconds
        nonlocal paired_reconstruction_invocations
        deliver_time = workload_clock()
        if first_deliver_time is None:
            first_deliver_time = deliver_time
        deliver_events.append(
            {
                "sequence": len(deliver_events) + 1,
                "qid": qid,
                "entry_count": len(entries),
                "deliver_time": deliver_time,
                "latency_since_broadcast_sec": (
                    deliver_time - broadcast_start_time
                    if broadcast_start_time is not None
                    else None
                ),
            }
        )
        if lifecycle is not None and qid in set(workload.qids()):
            event = dag.delivery_events[-1]
            lifecycle.mark_delivered(
                qid,
                deliver_time,
                {
                    "committed_wave": event.committed_wave,
                    "ordered_vertex": {
                        "round": event.ordered_vertex[0],
                        "source": event.ordered_vertex[1],
                    },
                },
            )
        logging.info("[T_DELIVER] qid=%s entries=%d", qid, len(entries))
        if (
            bhk22_runtime is not None
            and qid in {
                context.qid
                for context in bhk22_runtime.controller.phase_contexts.values()
            }
        ):
            task = asyncio.create_task(
                _handle_bhk22_delivery(dag.delivery_events[-1])
            )
            bhk22_phase_tasks.add(task)
            task.add_done_callback(bhk22_phase_tasks.discard)
        if dpss_reconstruction_manager is not None:
            for result in dpss_reconstruction_manager.on_deliver(qid, entries):
                dpss_events.append(
                    {
                        "event": "reconstructed",
                        "qid": result.qid,
                        "my_node_id": result.my_node_id,
                        "my_role_idx": result.my_role_idx,
                        "dealer_eval_points": list(result.dealer_eval_points),
                    }
                )
        if (
            ru22_runtime is not None
            and qid == ru22_runtime.qid
        ):
            event = dag.delivery_events[-1]
            public_result, private_result = (
                ru22_runtime.handle_delivery(event)
            )
            logging.info(
                "[RU22:FINALIZED] qid=%s result=%s new_vps=%s "
                "successor_done=%s",
                qid,
                public_result.delivery_result_digest,
                public_result.new_vps_id,
                private_result is not None,
            )
        if (
            a2dpss_public_finalizer is not None
            and Qid.decode(qid).kind == "a2dpss"
        ):
            if qid == a2dpss_handover_metrics.qid:
                a2dpss_handover_metrics.mark_ordered()
            # DagTOB invokes this callback only after the exact same entries
            # pass the committed-order justifier.  Algorithm 4 therefore
            # proceeds directly to public commitment interpolation instead
            # of repeating PVSS and anchor verification after t_deliver.
            public_result = a2dpss_public_finalizer.on_verified_deliver(
                qid,
                entries,
            )
            if public_result is None:
                raise RuntimeError(
                    "justified A2DPSS delivery failed public finalization"
                )
            delivered_context = a2dpss_justifier_manager.contexts[
                public_result.context_digest
            ]
            a2dpss_completed_public_results[
                public_result.context_digest
            ] = public_result
            if qid == a2dpss_handover_metrics.qid:
                a2dpss_handover_metrics.mark_public_finalized()
            a2dpss_events.append(
                {
                    "event": "public_finalized",
                    "algorithm": "Algorithm 4",
                    "protocol": "a2dpss",
                    "qid": public_result.qid,
                    "context_digest": public_result.context_digest,
                    "transfer_seq": delivered_context.transfer_seq,
                    "source_committee": (
                        delivered_context.source_committee
                    ),
                    "successor_committee": (
                        delivered_context.successor_committee
                    ),
                    "context_old_commitment_vector": list(
                        delivered_context.old_commitment_vector
                    ),
                    "accepted_contribution_digests": list(
                        public_result.accepted_contribution_digests
                    ),
                    "successor_commitment_vector": list(
                        public_result.successor_commitment_vector
                    ),
                }
            )
            private_result = (
                a2dpss_reconstruction_manager.reconstruct(
                    public_result,
                    entries,
                )
            )
            if private_result is not None:
                if qid == a2dpss_handover_metrics.qid:
                    a2dpss_handover_metrics.mark_successor_done()
                a2dpss_events.append(
                    {
                        "event": "reconstructed",
                        "algorithm": "Algorithm 4",
                        "protocol": "a2dpss",
                        "qid": private_result.qid,
                        "my_node_id": private_result.my_node_id,
                        "my_role_idx": private_result.my_role_idx,
                        "accepted_eval_points": list(
                            private_result.accepted_eval_points
                        ),
                        "accepted_contribution_digests": list(
                            private_result.accepted_contribution_digests
                        ),
                    }
                )
        if (
            ocdpss_public_finalizer is not None
            and Qid.decode(qid).kind == "ocdpss"
        ):
            if (
                ocdpss_handover_metrics is not None
                and qid == ocdpss_handover_metrics.qid
            ):
                ocdpss_handover_metrics.mark_ordered()
            public_result = ocdpss_public_finalizer.on_verified_deliver(
                qid, entries
            )
            if public_result is None:
                raise RuntimeError("justified OCDPSS delivery failed public finalization")
            delivered_context = ocdpss_justifier_manager.contexts[
                public_result.context_digest
            ]
            ocdpss_completed_public_results[
                public_result.context_digest
            ] = public_result
            ocdpss_events.append(
                {
                    "event": "public_finalized",
                    "algorithm": "Algorithm 5",
                    "protocol": "ocdpss",
                    "transfer_seq": delivered_context.transfer_seq,
                    "source_committee": delivered_context.source_committee,
                    "successor_committee": delivered_context.successor_committee,
                    "context_old_commitment_vector": list(
                        delivered_context.old_commitment_vector
                    ),
                    "public_result": public_result.to_obj(),
                }
            )
            if (
                ocdpss_handover_metrics is not None
                and qid == ocdpss_handover_metrics.qid
            ):
                ocdpss_handover_metrics.mark_public_finalized()
            private_result = None
            if my_id in delivered_context.successor_role_indices:
                private_result = ocdpss_reconstruction_manager.reconstruct(
                    public_result, entries
                )
            if private_result is not None:
                if (
                    ocdpss_handover_metrics is not None
                    and qid == ocdpss_handover_metrics.qid
                ):
                    ocdpss_handover_metrics.mark_successor_done()
                ocdpss_events.append(
                    {
                        "event": "reconstructed",
                        "algorithm": "Algorithm 5",
                        "protocol": "ocdpss",
                        "qid": private_result.qid,
                        "transfer_seq": delivered_context.transfer_seq,
                        "my_node_id": private_result.my_node_id,
                        "my_role_idx": private_result.my_role_idx,
                        "accepted_eval_points": list(
                            private_result.accepted_eval_points
                        ),
                        "accepted_bundle_digests": list(
                            private_result.accepted_bundle_digests
                        ),
                    }
                )
        if a2dpss_pipeline and qid in set(workload.qids()):
            if any(event["qid"] == qid for event in pipeline_events):
                raise RuntimeError("A2DPSS pipeline qid finalized more than once")
            # DagTOB has just verified these exact immutable threshold entries.
            # Reuse that evidence, including for the multi-query benchmark.
            cpu_started = time.process_time()
            try:
                public_result = pipeline_finalizer.on_verified_deliver(qid, entries)
            finally:
                stage = pipeline_cpu_stages["public_finalization"]
                stage["cpu_seconds"] += time.process_time() - cpu_started
                stage["invocations"] += 1
            if public_result is None:
                raise RuntimeError("justified pipeline A2DPSS public finalization failed")
            public_at = workload_clock()
            expected_successor = (
                my_id in workload.context_for_qid(qid).successor_role_indices
            )
            private_result = None
            if expected_successor:
                cpu_started = time.process_time()
                try:
                    private_result = pipeline_reconstruction.reconstruct(public_result, entries)
                finally:
                    stage = pipeline_cpu_stages["successor_reconstruction"]
                    stage["cpu_seconds"] += time.process_time() - cpu_started
                    stage["invocations"] += 1
            if (private_result is not None) != expected_successor:
                raise RuntimeError("pipeline A2DPSS successor reconstruction incomplete")
            pipeline_events.append({
                "event": "public_finalized",
                "qid": qid,
                "context_digest": public_result.context_digest,
                "accepted_contribution_digests": list(
                    public_result.accepted_contribution_digests
                ),
                "successor_commitment_vector": list(
                    public_result.successor_commitment_vector
                ),
                "private_reconstructed": private_result is not None,
                "t_public": public_at,
                "t_successor": workload_clock() if private_result is not None else None,
            })
        if ocdpss_paired and qid in set(workload.qids()):
            started = time.perf_counter()
            public_result = paired_ocdpss_finalizer.on_deliver(qid, entries)
            paired_public_finalize_seconds += time.perf_counter() - started
            paired_public_finalize_invocations += 1
            if public_result is None:
                raise RuntimeError("paired OCDPSS delivery failed public finalization")
            paired_event = {
                "event": "public_finalized",
                "qid": qid,
                "context_digest": public_result.context_digest,
                "accepted_bundle_digests": list(
                    public_result.accepted_bundle_digests
                ),
                "operation_output": public_result.operation_output,
                "successor_commitment_vector": list(
                    public_result.successor_commitment_vector
                ),
            }
            started = time.perf_counter()
            private_result = paired_ocdpss_reconstruction.reconstruct(
                public_result, entries
            )
            paired_reconstruction_seconds += time.perf_counter() - started
            paired_reconstruction_invocations += 1
            paired_event["private_reconstructed"] = private_result is not None
            if private_result is not None:
                paired_event["my_role_idx"] = private_result.my_role_idx
            paired_ocdpss_events.append(paired_event)

    workload_qids = set(workload.qids()) if workload is not None else set()

    def _vertex_admitted_cb(vertex):
        if lifecycle is None or not vertex.block:
            return
        observed_at = workload_clock()
        for qid in vertex.block:
            if qid in workload_qids:
                lifecycle.mark_included(
                    qid,
                    observed_at,
                    {"round": vertex.round, "source": vertex.source},
                )

    dag = DagTOB(
        my_id=my_id,
        M=M,
        quorum_h=quorum_h,
        agreement_t=agreement_t,
        epoch=epoch,
        wait_predicate=wait_predicate,
        leader_oracle=leader_oracle,
        deliver_cb=_deliver_cb,
        vertex_admitted_cb=_vertex_admitted_cb,
        vertex_verifier=_verify_vertex,
        logger=logging.getLogger(f"dag_tob.phase3.{my_id}"),
    )
    injected_qids = []
    if multi_qid_smoke:
        qid_a = workload.batches_for_trigger(START)[0].instances[0]
        qid_a = workload.qid(qid_a)
        qid_b = qid_c = None
    else:
        qid_a = qid_b = qid_c = None

    async def wait_workload_ready(instances, trigger, timeout):
        gate = None
        if a2dpss_pipeline:
            gate = {"label": trigger, "enter": workload_clock(), "exit": None, "duration": None}
            pipeline_gates.append(gate)
        try:
            await _wait_for_workload_pool_ready(
                dag, workload, lifecycle, instances, timeout=timeout,
                **({"clock": workload_clock, "cpu_metrics": pipeline_ready_cpu}
                   if a2dpss_pipeline else {}),
            )
        finally:
            if gate is not None:
                gate["exit"] = workload_clock()
                gate["duration"] = gate["exit"] - gate["enter"]
    logging.info(
        "[START] node=%d M=%d expected_committee_size=%d committee_m=%d "
        "n_r=%d t_r=%d n_a=%d t_a=%d m_a=%d quorum_h=%d "
        "agreement_t_impl=%d coverage_threshold=%d "
        "threshold_fraction=%.6f epoch=%s",
        my_id,
        M,
        expected_committee_size,
        committee_m,
        n_r,
        t_r,
        n_a,
        t_a,
        m_a,
        quorum_h,
        agreement_t_impl,
        agreement_t + 1,
        threshold_fraction,
        epoch,
    )

    async with ProcessProgramRunner(
        peers,
        n,
        agreement_t_impl,
        my_id,
        linger_timeout=1 if experiment_barrier is not None else 0,
        drain_on_exit=experiment_barrier is not None,
    ) as runner:
        send, recv = runner.get_send_recv("")
        shared_task, shared_subscribe = subscribe_recv(recv)
        shared = (shared_task, shared_subscribe)
        threshold_coin_oracle = None
        coin_observer = None
        threshold_leader_oracle = None
        wave_coin_tasks = {}
        wave_coin_errors = []
        if effective_coin_mode == "threshold-bls":
            threshold_coin_oracle = ThresholdBLSCoinOracle(
                epoch=epoch,
                config=threshold_coin_keys.config,
                public_key_bytes=threshold_coin_keys.public_key_bytes,
                public_key=threshold_coin_keys.public_key,
                private_key=threshold_coin_keys.private_key,
                send=send,
                shared_subscribe=shared,
                signing_enabled=my_id not in coin_silent_nodes,
                logger=logging.getLogger(f"threshold_bls_coin.{my_id}"),
            )
            await threshold_coin_oracle.start()
            coin_observer = ObservedCoinOracle(
                threshold_coin_oracle,
                timeout=coin_timeout,
                logger=logging.getLogger(f"dag_common_coin.{my_id}"),
            )
            threshold_leader_oracle = ThresholdCoinLeaderOracle(
                coin_observer,
                leader_committees,
            )
            dag.leader_oracle = threshold_leader_oracle

            async def _prepare_wave_coin(wave):
                try:
                    leader = await threshold_leader_oracle.prepare_wave(wave)
                    resumed = dag.notify_wave_coin_ready(wave)
                    logging.info(
                        "[WAVE-COIN] ready wave=%d leader=%s resumed=%s",
                        wave,
                        leader,
                        resumed,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    wave_coin_errors.append(exc)
                    logging.exception(
                        "[WAVE-COIN] failed wave=%d", wave
                    )
                finally:
                    wave_coin_tasks.pop(wave, None)

            def _request_wave_coin(wave):
                if wave not in wave_coin_tasks:
                    wave_coin_tasks[wave] = asyncio.create_task(
                        _prepare_wave_coin(wave)
                    )

            dag.wave_coin_needed_cb = _request_wave_coin
        manager = DagRBCManager(
            dag=dag,
            epoch=epoch,
            my_id=my_id,
            M=M,
            t=agreement_t_impl,
            threshold_fraction=threshold_fraction,
            expected_committee_size=expected_committee_size,
            committee_m=committee_m,
            quorum_h=quorum_h,
            send=send,
            shared_subscribe=shared,
            crypto=crypto,
            vrf_sk=vrf_sk,
            vrf_pks=vrf_pks,
            logger=logging.getLogger(f"dag_rbc_manager.phase3.{my_id}"),
        )
        # P-ref.2: demo / multi-qid payloads (each node broadcasts its own mid)
        # travel on their own message-RBC keyed by mid.  The DPSS path keeps its
        # specialised transcript-RBC (DPSSTranscriptRBCManager), so no generic
        # hook there -- otherwise a received transcript would be re-broadcast.
        msg_mgr = None
        if (
            not dpss_smoke
            and not a2dpss_smoke
            and not ocdpss_smoke
            and not ru22_smoke
            and not bhk22_smoke
        ):
            msg_mgr = MessageRBCManager(
                dag=dag,
                epoch=epoch,
                my_id=my_id,
                M=M,
                t=agreement_t_impl,
                threshold_fraction=threshold_fraction,
                expected_committee_size=expected_committee_size,
                committee_m=committee_m,
                quorum_h=quorum_h,
                send=send,
                shared_subscribe=shared,
                crypto=crypto,
                vrf_sk=vrf_sk,
                vrf_pks=vrf_pks,
                logger=logging.getLogger(f"msg_rbc.phase3.{my_id}"),
            )
            if multi_qid_smoke:
                for q in workload.qids():
                    msg_mgr.ensure_window(q, range(M))
            elif unique_qid_per_node:
                for i in range(M):
                    msg_mgr.ensure_window(f"qid-demo-{i}", [i])
            else:
                msg_mgr.ensure_window("qid-demo", range(M))
            dag.broadcast_hook = lambda qid, mid, m: asyncio.create_task(
                msg_mgr.broadcast_message(qid, mid, m)
            )
        dpss_rbc_manager = None
        if dpss_transcript_rbc and dpss_local is not None:
            dpss_rbc_manager = DPSSTranscriptRBCManager(
                dag=dag,
                epoch=epoch,
                my_id=my_id,
                M=M,
                t=agreement_t_impl,
                threshold_fraction=threshold_fraction,
                expected_committee_size=expected_committee_size,
                committee_m=committee_m,
                quorum_h=quorum_h,
                send=send,
                shared_subscribe=shared,
                crypto=crypto,
                vrf_sk=vrf_sk,
                vrf_pks=vrf_pks,
                logger=logging.getLogger(f"dpss_rbc_manager.phase3.{my_id}"),
            )
            dpss_rbc_manager.ensure_context(
                dpss_local.ctx,
                dpss_local.source_committee,
            )
        a2dpss_rbc_manager = None
        if a2dpss_smoke and a2dpss_context is not None:
            a2dpss_rbc_manager = A2DPSSContributionRBCManager(
                dag=dag,
                epoch=epoch,
                my_id=my_id,
                M=M,
                t=agreement_t_impl,
                threshold_fraction=threshold_fraction,
                expected_committee_size=expected_committee_size,
                committee_m=committee_m,
                quorum_h=quorum_h,
                send=send,
                shared_subscribe=shared,
                crypto=crypto,
                vrf_sk=vrf_sk,
                vrf_pks=vrf_pks,
                logger=logging.getLogger(
                    f"a2dpss_rbc_manager.phase3.{my_id}"
                ),
            )
            a2dpss_rbc_manager.ensure_context(
                a2dpss_context,
                a2dpss_source_holder_ids,
            )
        bhk22_rbc_manager = None
        if bhk22_runtime is not None:
            default_bhk22_payload_limit = _env_int(
                "BHK22_RBC_PAYLOAD_LIMIT", 16 * 1024 * 1024
            )
            bhk22_rbc_manager = BHK22PhaseRBCManager(
                dag=dag,
                controller=bhk22_runtime.controller,
                epoch=epoch,
                my_id=my_id,
                M=M,
                t=agreement_t_impl,
                threshold_fraction=threshold_fraction,
                expected_committee_size=expected_committee_size,
                committee_m=committee_m,
                quorum_h=quorum_h,
                send=send,
                shared_subscribe=shared,
                crypto=crypto,
                vrf_sk=vrf_sk,
                vrf_pks=vrf_pks,
                payload_limits={
                    BHK22_PHASE_DEAL: _env_int(
                        "BHK22_DEAL_PAYLOAD_LIMIT", default_bhk22_payload_limit
                    ),
                    BHK22_PHASE_VERIFY: _env_int(
                        "BHK22_VERIFY_PAYLOAD_LIMIT", default_bhk22_payload_limit
                    ),
                    BHK22_PHASE_RESPONSE: _env_int(
                        "BHK22_RESPONSE_PAYLOAD_LIMIT", default_bhk22_payload_limit
                    ),
                    BHK22_PHASE_FINALIZE: _env_int(
                        "BHK22_FINALIZE_PAYLOAD_LIMIT", default_bhk22_payload_limit
                    ),
                },
                logger=logging.getLogger(
                    f"bhk22_rbc_manager.phase3.{my_id}"
                ),
            )
            bhk22_rbc_manager.ensure_phase_instances(
                bhk22_runtime.controller.current
            )
        ru22_rbc_manager = None
        if ru22_runtime is not None:
            ru22_manager_type = (
                RU22OperationContributionRBCManager
                if ru22_operation_context is not None
                else RU22ContributionRBCManager
            )
            ru22_manager_kwargs = {}
            if ru22_operation_context is not None:
                ru22_manager_kwargs.update(
                    operation_context=ru22_operation_context,
                    trial_id=ru22_runtime.manifest.trial_id,
                )
            ru22_rbc_manager = ru22_manager_type(
                dag=dag,
                epoch=epoch,
                my_id=my_id,
                source_role_nodes=(
                    ru22_runtime.manifest.source_role_nodes
                ),
                M=M,
                t=agreement_t_impl,
                threshold_fraction=threshold_fraction,
                expected_committee_size=expected_committee_size,
                committee_m=committee_m,
                quorum_h=quorum_h,
                send=send,
                shared_subscribe=shared,
                crypto=crypto,
                vrf_sk=vrf_sk,
                vrf_pks=vrf_pks,
                logger=logging.getLogger(
                    f"ru22_rbc_manager.phase3.{my_id}"
                ),
                **ru22_manager_kwargs,
            )
            # All source-role RBC receivers exist before the release time.
            ru22_rbc_manager.ensure_context(
                ru22_runtime.context,
                ru22_runtime.source_role_ids,
            )
        ocdpss_rbc_manager = None
        if ocdpss_smoke and ocdpss_context is not None:
            ocdpss_rbc_manager = OCDPSSBundleRBCManager(
                dag=dag,
                epoch=epoch,
                my_id=my_id,
                M=M,
                t=agreement_t_impl,
                threshold_fraction=threshold_fraction,
                expected_committee_size=expected_committee_size,
                committee_m=committee_m,
                quorum_h=quorum_h,
                send=send,
                shared_subscribe=shared,
                crypto=crypto,
                vrf_sk=vrf_sk,
                vrf_pks=vrf_pks,
                logger=logging.getLogger(
                    f"ocdpss_rbc_manager.phase3.{my_id}"
                ),
            )
            ocdpss_rbc_manager.ensure_context(
                ocdpss_context,
                ocdpss_source_holder_ids,
            )
        a2dpss_scheduled_hops = 0
        ocdpss_scheduled_hops = 0
        role_checked = set()
        summary = None

        def _ru22_compact_ledger():
            return [
                {
                    qid: [
                        {"mid": mid, "payload_bytes": len(payload)}
                        for mid, payload in entries
                    ]
                    for qid, entries in block.items()
                }
                for block in dag.ledger
            ]

        def _ru22_compact_delivery_events():
            return [
                {
                    "sequence": event.sequence,
                    "qid": event.qid,
                    "entries": [
                        {"mid": mid, "payload_bytes": len(payload)}
                        for mid, payload in event.entries
                    ],
                    "ordered_vertex": {
                        "round": event.ordered_vertex[0],
                        "source": event.ordered_vertex[1],
                    },
                    "committed_wave": event.committed_wave,
                }
                for event in dag.delivery_events
            ]

        def _build_summary(bytes_sent, cleanup_completed=None):
            transport_by_tag = {
                str(tag): int(value)
                for tag, value in sorted(
                    getattr(runner.node_communicator, "bytes_by_tag_path", {}).items()
                )
            }

            transport_family_totals = _transport_family_totals(
                transport_by_tag, total_bytes=int(bytes_sent)
            )
            bhk22_payload_schedule = None
            if bhk22_runtime is not None:
                manifest = bhk22_runtime.manifest
                phase_count = len(
                    bhk22_path_phases(
                        manifest.path_mode, manifest.completion_mode
                    )
                )
                bhk22_payload_schedule = (
                    f"bhk22-{manifest.path_mode}-{phase_count}-phase-"
                    f"{manifest.completion_mode}"
                )
            registry = getattr(dag.wait_predicate, "registry", None)
            justifier_metrics = (
                registry.metrics_snapshot() if registry is not None else None
            )
            a2dpss_computation_metrics = None
            if a2dpss_smoke:
                validation = (
                    a2dpss_justifier_manager.validation_metrics_snapshot()
                )
                finalization = a2dpss_public_finalizer.metrics_snapshot()
                reconstruction = (
                    a2dpss_reconstruction_manager.metrics_snapshot()
                )
                holder_events = [
                    event
                    for event in a2dpss_events
                    if event.get("event") == "holder_broadcast"
                ]
                a2dpss_computation_metrics = (
                    build_a2dpss_computation_metrics(
                        protocol="dag-tob",
                        node=my_id,
                        proposer_predicate={
                            "operation": "select_justified_subset",
                            **justifier_metrics["select"],
                        },
                        receiver_predicate={
                            "operation": "committed_order_verify",
                            **justifier_metrics["verify"],
                        },
                        candidate_formation=None,
                        pvss_generation={
                            "invocations": len(holder_events),
                            "cpu_seconds": sum(
                                float(event["pvss_cpu_seconds"])
                                for event in holder_events
                            ),
                        },
                        pvss_verification={
                            **validation["pvss"],
                            "detail": validation,
                        },
                        public_finalization={
                            **finalization["total"],
                            "detail": finalization,
                        },
                        successor_reconstruction={
                            **reconstruction["total"],
                            "detail": reconstruction,
                        },
                    )
                )
            return {
                "protocol": "atcaas-dag-tob",
                "r6_metrics_schema": 1,
                "agreement_parameters_schema": 1,
                "dag_agreement_parameters_schema": 1,
                "node": my_id,
                "M": M,
                "expected_committee_size": expected_committee_size,
                "committee_m": committee_m,
                "quorum_h": quorum_h,
                "n_r": n_r,
                "t_r": t_r,
                "n_a": n_a,
                "t_a": t_a,
                "m_a": m_a,
                "agreement_t_impl": agreement_t_impl,
                "coin_c": coin_c,
                "coin_t": coin_t,
                "coin_k": coin_k,
                "agreement_t": agreement_t,
                "coverage_threshold": agreement_t + 1,
                "threshold_fraction": threshold_fraction,
                "committee_selection_mode": committee_selection_mode,
                "fixed_committee_seed": (
                    fixed_committee_seed
                    if committee_selection_mode == FIXED_SIZE_COMMITTEE_MODE
                    else None
                ),
                "coin_mode": effective_coin_mode,
                "coin_mode_requested": coin_mode_requested,
                "coin_mode_forced_by_replay": coin_mode_forced_by_replay,
                "ideal_coin": effective_coin_mode == "ideal",
                "coin_cost": (
                    "included"
                    if effective_coin_mode == "threshold-bls"
                    else "excluded"
                ),
                "coin_committee": (
                    list(threshold_coin_keys.config.committee)
                    if threshold_coin_keys is not None
                    else []
                ),
                "coin_committee_size": (
                    len(threshold_coin_keys.config.committee)
                    if threshold_coin_keys is not None
                    else None
                ),
                "coin_threshold": (
                    threshold_coin_keys.config.threshold
                    if threshold_coin_keys is not None
                    else None
                ),
                "coin_membership_digest": (
                    threshold_coin_keys.config.membership_digest.hex()
                    if threshold_coin_keys is not None
                    else None
                ),
                "coin_public_key_digest": (
                    threshold_coin_keys.config.public_key_digest.hex()
                    if threshold_coin_keys is not None
                    else None
                ),
                "coin_silent_nodes": sorted(coin_silent_nodes),
                "coin_observations": (
                    coin_observer.snapshot()
                    if coin_observer is not None
                    else None
                ),
                "coin_metrics": (
                    threshold_coin_oracle.snapshot()
                    if threshold_coin_oracle is not None
                    else None
                ),
                "wave_coin": (
                    threshold_leader_oracle.snapshot()
                    if threshold_leader_oracle is not None
                    else None
                ),
                "wave_coin_state": dag.snapshot_wave_coin_state(),
                "protocol_trace_source": trace_source,
                "protocol_trace_digest": role_trace_digest,
                "role_trace_source": trace_source,
                "role_trace_format": role_trace_format,
                "role_trace_digest": role_trace_digest,
                "ideal_coin_trace_source": (
                    "recorded" if replay_trace is not None else "deterministic"
                ),
                "ideal_coin_trace_digest": (
                    replay_trace.digest() if replay_trace is not None else None
                ),
                "leader_committees": {
                    str(index): list(leader_committees.roles(index))
                    for index in range(1, max(1, (rounds + 3) // 4) + 1)
                },
                "epoch": epoch,
                "rounds": rounds,
                "handover_mode": (
                    "a2dpss-pipeline"
                    if a2dpss_pipeline
                    else "bhk22"
                    if bhk22_smoke
                    else "ru22"
                    if ru22_smoke
                    else "a2dpss"
                    if a2dpss_smoke
                    else "ocdpss"
                    if ocdpss_smoke
                    else "legacy-dpss"
                    if dpss_smoke
                    else "none"
                ),
                "handover_algorithm": (
                    "BHK+22 Section 4"
                    if bhk22_smoke
                    else "RU22"
                    if ru22_smoke
                    else "Algorithm 4"
                    if a2dpss_smoke or a2dpss_pipeline
                    else "Algorithm 5"
                    if ocdpss_smoke
                    else "legacy transcript-only compatibility"
                    if dpss_smoke
                    else None
                ),
                "payload_schedule": "a2dpss-pipeline"
                if a2dpss_pipeline
                else "ocdpss-paired"
                if ocdpss_paired
                else (bhk22_payload_schedule or "bhk22-uninitialized")
                if bhk22_smoke
                else "ru22-single-vps"
                if ru22_smoke
                else "ocdpss-smoke"
                if ocdpss_smoke
                else "a2dpss-smoke"
                if a2dpss_smoke
                else "legacy-dpss-transcript-smoke"
                if dpss_smoke
                else "multi-qid-smoke"
                if multi_qid_smoke
                else "single-demo-unique-qid"
                if unique_qid_per_node
                else "single-demo",
                "unique_qid_per_node": unique_qid_per_node,
                "injected_qids": injected_qids,
                "workload_trace_source": workload_source,
                "workload_trace": workload.to_dict() if workload is not None else None,
                "ocdpss_paired": ocdpss_paired,
                "ocdpss_fixture_digest": (
                    workload.digest() if ocdpss_paired else None
                ),
                "ocdpss_context_digests": (
                    [
                        workload.context_for_qid(qid).context_digest
                        for qid in workload.qids()
                    ]
                    if ocdpss_paired
                    else []
                ),
                "ocdpss_paired_events": paired_ocdpss_events,
                "ocdpss_phase_metrics": (
                    {
                        "holder_generation": workload.holder_metrics_summary(),
                        "justifier_validation": (
                            paired_ocdpss_justifier.validation_metrics_snapshot()
                        ),
                        "public_finalize": {
                            "invocations": paired_public_finalize_invocations,
                            "elapsed_sec": paired_public_finalize_seconds,
                            "detail": paired_ocdpss_finalizer.metrics_snapshot(),
                        },
                        "successor_reconstruction": {
                            "invocations": paired_reconstruction_invocations,
                            "elapsed_sec": paired_reconstruction_seconds,
                            "detail": (
                                paired_ocdpss_reconstruction.metrics_snapshot()
                            ),
                        },
                    }
                    if ocdpss_paired
                    else None
                ),
                **({
                    "a2dpss_pipeline": True,
                    "a2dpss_pipeline_clock": "perf_counter-relative",
                    "a2dpss_pipeline_fixture_digest": workload.digest(),
                    "a2dpss_pipeline_events": list(pipeline_events),
                    "a2dpss_pipeline_gates": [dict(gate) for gate in pipeline_gates],
                    "a2dpss_pipeline_cpu": pipeline_cpu_snapshot(),
                    "a2dpss_pipeline_validation_metrics": (
                        pipeline_justifier.validation_metrics_snapshot()
                    ),
                    "a2dpss_pipeline_finalization_metrics": (
                        pipeline_finalizer.metrics_snapshot()
                    ),
                    "a2dpss_pipeline_reconstruction_metrics": (
                        pipeline_reconstruction.metrics_snapshot()
                    ),
                } if a2dpss_pipeline else {}),
                "query_lifecycle": lifecycle.summary() if lifecycle is not None else None,
                "justifier_metrics": justifier_metrics,
                "dag_qid_causal_coverage": (
                    dag.snapshot_qid_causal_coverage()
                ),
                "dag_selectblock_metrics": (
                    dag.dag_selectblock_metrics_snapshot()
                ),
                "lookahead_rounds": lookahead_rounds,
                "retention_rounds": retention_rounds,
                "stop_grace_seconds": stop_grace_seconds,
                "cleanup_completed": cleanup_completed,
                "active_rbc_instances": manager.active_count(),
                "delivered_rbc_vertices": len(manager.delivered_vertices),
                "accepted_rbc_vertices": len(manager.accepted_vertices),
                "broadcast_start_time": broadcast_start_time,
                "broadcast_end_time": broadcast_end_time,
                "first_deliver_time": first_deliver_time,
                "first_delivery_latency_sec": (
                    first_deliver_time - broadcast_start_time
                    if first_deliver_time is not None
                    and broadcast_start_time is not None
                    else None
                ),
                "broadcast_duration_sec": (
                    broadcast_end_time - broadcast_start_time
                    if broadcast_end_time is not None
                    and broadcast_start_time is not None
                    else None
                ),
                "buffered_vertices": len(dag.buffer),
                "dag_round_sizes": {
                    str(r): len(dag.DAG.get(r, {}))
                    for r in range(0, rounds + 1)
                },
                "ledger": (
                    _ru22_compact_ledger()
                    if ru22_smoke or bhk22_smoke
                    else dag.snapshot_ledger()
                ),
                "protocol_delivery_events": (
                    _ru22_compact_delivery_events()
                    if ru22_smoke or bhk22_smoke
                    else dag.snapshot_delivery_events()
                ),
                "delivery_event_count": dag.delivery_count,
                "committed_wave": dag.decided_wave,
                "ordered_vertices": [
                    {"round": r, "source": s}
                    for r, s in dag.snapshot_ordered_vertices()
                ],
                "deliver_count": dag.delivery_count,
                "delivered_events": deliver_events,
                "legacy_dpss_smoke": dpss_smoke,
                "dpss_smoke": dpss_smoke,
                "dpss_transcript_rbc": dpss_transcript_rbc,
                "dpss_smoke_inject_all": dpss_smoke_inject_all,
                "dpss_events": dpss_events,
                "bhk22_smoke": bhk22_smoke,
                "bhk22_summary": (
                    bhk22_runtime.summary(
                        rbc_metrics=(
                            bhk22_rbc_manager.metrics_snapshot()
                            if bhk22_rbc_manager is not None
                            else None
                        ),
                        cleanup={
                            "completed": cleanup_completed,
                            "registered_contexts": len(
                                bhk22_runtime.justifier.contexts
                            ),
                            "active_rbc_instances": (
                                bhk22_rbc_manager.active_count()
                                if bhk22_rbc_manager is not None
                                else 0
                            ),
                        },
                    )
                    if bhk22_runtime is not None
                    else None
                ),
                "active_bhk22_rbc_instances": (
                    bhk22_rbc_manager.active_count()
                    if bhk22_rbc_manager is not None
                    else 0
                ),
                "ru22_smoke": ru22_smoke,
                "ru22_summary": (
                    ru22_runtime.summary(
                        rbc_metrics=(
                            ru22_rbc_manager.metrics.snapshot()
                            if ru22_rbc_manager is not None
                            else None
                        )
                    )
                    if ru22_runtime is not None
                    else None
                ),
                "active_ru22_rbc_instances": (
                    ru22_rbc_manager.active_count()
                    if ru22_rbc_manager is not None
                    else 0
                ),
                "delivered_ru22_batches": (
                    len(ru22_rbc_manager.delivered_batches)
                    if ru22_rbc_manager is not None
                    else 0
                ),
                "a2dpss_smoke": a2dpss_smoke,
                "a2dpss_paired": a2dpss_fixture is not None,
                "a2dpss_latency_view": (
                    "paired-tob"
                    if a2dpss_fixture is not None
                    else "live-end-to-end"
                ) if a2dpss_smoke else None,
                "a2dpss_fixture_digest": (
                    a2dpss_fixture.digest()
                    if a2dpss_fixture is not None
                    else None
                ),
                "a2dpss_trial_id": (
                    a2dpss_fixture.trial_id
                    if a2dpss_fixture is not None
                    else None
                ),
                "a2dpss_trial_epoch": (
                    a2dpss_fixture.trial_epoch
                    if a2dpss_fixture is not None
                    else None
                ),
                "a2dpss_qid": (
                    a2dpss_context.encode_qid()
                    if a2dpss_context is not None
                    else None
                ),
                "a2dpss_context_digest": (
                    a2dpss_context.context_digest
                    if a2dpss_context is not None
                    else None
                ),
                "a2dpss_source_holder_ids": list(
                    a2dpss_source_holder_ids
                ),
                "a2dpss_successor_node_ids": sorted(
                    a2dpss_successor_role_indices
                ),
                "a2dpss_fixture_contribution_digests": (
                    list(a2dpss_fixture.contribution_digests())
                    if a2dpss_fixture is not None
                    else []
                ),
                "a2dpss_observed_contribution_digests": (
                    sorted(
                        a2dpss_contribution_digest(payload)
                        for _mid, payload in dag._messages_for_qid(
                            a2dpss_context.encode_qid()
                        )
                    )
                    if a2dpss_fixture is not None
                    else []
                ),
                "a2dpss_fixture_holder_metrics": (
                    a2dpss_fixture.holder_metrics_summary()
                    if a2dpss_fixture is not None
                    else None
                ),
                "a2dpss_handover_metrics": (
                    a2dpss_handover_metrics.snapshot()
                    if a2dpss_handover_metrics is not None
                    else None
                ),
                "a2dpss_computation_metrics": (
                    a2dpss_computation_metrics
                ),
                "a2dpss_requested_hops": a2dpss_hops,
                "a2dpss_scheduled_hops": a2dpss_scheduled_hops,
                "a2dpss_events": a2dpss_events,
                "a2dpss_lifecycle": (
                    a2dpss_lifecycle.snapshot()
                    if a2dpss_lifecycle is not None
                    else None
                ),
                "a2dpss_rbc_metrics": (
                    a2dpss_rbc_manager.metrics.snapshot()
                    if a2dpss_rbc_manager is not None
                    else None
                ),
                "active_a2dpss_rbc_instances": (
                    a2dpss_rbc_manager.active_count()
                    if a2dpss_rbc_manager is not None
                    else 0
                ),
                "delivered_a2dpss_contributions": (
                    len(
                        a2dpss_rbc_manager.delivered_contributions
                    )
                    if a2dpss_rbc_manager is not None
                    else 0
                ),
                "ocdpss_smoke": ocdpss_smoke,
                "ocdpss_requested_hops": ocdpss_hops,
                "ocdpss_scheduled_hops": ocdpss_scheduled_hops,
                "ocdpss_events": ocdpss_events,
                "ocdpss_lifecycle": (
                    ocdpss_lifecycle.snapshot()
                    if ocdpss_lifecycle is not None
                    else None
                ),
                "ocdpss_handover_metrics": (
                    ocdpss_handover_metrics.snapshot()
                    if ocdpss_handover_metrics is not None
                    else None
                ),
                "ocdpss_validation_metrics": (
                    ocdpss_justifier_manager.validation_metrics_snapshot()
                    if ocdpss_justifier_manager is not None
                    else None
                ),
                "ocdpss_finalization_metrics": (
                    ocdpss_public_finalizer.metrics_snapshot()
                    if ocdpss_public_finalizer is not None
                    else None
                ),
                "ocdpss_reconstruction_metrics": (
                    ocdpss_reconstruction_manager.metrics_snapshot()
                    if ocdpss_reconstruction_manager is not None
                    else None
                ),
                "ocdpss_rbc_metrics": (
                    ocdpss_rbc_manager.metrics.snapshot()
                    if ocdpss_rbc_manager is not None
                    else None
                ),
                "active_ocdpss_rbc_instances": (
                    ocdpss_rbc_manager.active_count()
                    if ocdpss_rbc_manager is not None
                    else 0
                ),
                "delivered_ocdpss_bundles": (
                    len(ocdpss_rbc_manager.delivered_bundles)
                    if ocdpss_rbc_manager is not None
                    else 0
                ),
                "active_dpss_rbc_instances": (
                    dpss_rbc_manager.active_count()
                    if dpss_rbc_manager is not None
                    else 0
                ),
                "delivered_dpss_transcripts": (
                    len(dpss_rbc_manager.delivered_transcripts)
                    if dpss_rbc_manager is not None
                    else 0
                ),
                "bytes_sent": bytes_sent,
                "communication_bits": int(bytes_sent) * 8,
                "transport_bytes_by_tag": transport_by_tag,
                "transport_bytes_by_family": transport_family_totals,
                "excluded_costs": (
                    ["ideal_common_coin", "role_assignment"]
                    if effective_coin_mode == "ideal"
                    else ["role_assignment"]
                ),
            }

        async def _submit_a2dpss_holder(
            context,
            holder_state,
            *,
            holder_state_source,
        ):
            if a2dpss_fixture is not None:
                if my_id not in a2dpss_fixture.source_holder_ids():
                    logging.info(
                        "[ALGORITHM-4:A2DPSS] transfer_seq=%d non-holder "
                        "waits for paired fixture contribution RBC",
                        context.transfer_seq,
                    )
                    return None
                if a2dpss_rbc_manager is None:
                    raise RuntimeError(
                        "A2DPSS contribution RBC path is not initialized"
                    )
                entry = a2dpss_fixture.entry_for_sender(my_id)
                started = time.perf_counter()
                await a2dpss_rbc_manager.broadcast_contribution(
                    context,
                    holder_id=my_id,
                    contribution_bytes=entry.payload,
                )
                broadcast_seconds = time.perf_counter() - started
                a2dpss_events.append(
                    {
                        "event": "holder_broadcast",
                        "algorithm": "Algorithm 4",
                        "protocol": "a2dpss",
                        "qid": context.encode_qid(),
                        "transfer_seq": context.transfer_seq,
                        "mid": entry.mid,
                        "holder_state_source": "paired-fixture",
                        "fixture_replay": True,
                        "contribution_digest": (
                            a2dpss_contribution_digest(entry.payload)
                        ),
                        "contribution_bytes": len(entry.payload),
                        "pvss_seconds": 0.0,
                        "pvss_cpu_seconds": 0.0,
                        "serialization_seconds": 0.0,
                        "broadcast_seconds": broadcast_seconds,
                        "fixture_holder_metrics": dict(entry.holder_metrics),
                    }
                )
                return entry
            if holder_state is None:
                logging.info(
                    "[ALGORITHM-4:A2DPSS] transfer_seq=%d non-holder "
                    "waits for contribution RBC",
                    context.transfer_seq,
                )
                return None
            if a2dpss_rbc_manager is None:
                raise RuntimeError(
                    "A2DPSS contribution RBC path is not initialized"
                )
            share_s = holder_state.share_s
            share_s_hat = holder_state.share_s_hat

            async def _broadcast_a2dpss(qid, mid, payload):
                if qid != context.encode_qid():
                    raise ValueError(
                        "holder emitted an unexpected A2DPSS qid"
                    )
                if mid != f"node_{my_id}":
                    raise ValueError(
                        "holder emitted an unexpected A2DPSS mid"
                    )
                await a2dpss_rbc_manager.broadcast_contribution(
                    context,
                    holder_id=my_id,
                    contribution_bytes=payload,
                )

            holder_result = await run_a2dpss_handover_role(
                context=context,
                holder_id=my_id,
                share_s=share_s,
                share_s_hat=share_s_hat,
                broadcast_contribution=_broadcast_a2dpss,
                holder_state=holder_state,
            )
            del share_s, share_s_hat
            a2dpss_events.append(
                {
                    "event": "holder_broadcast",
                    "algorithm": "Algorithm 4",
                    "protocol": "a2dpss",
                    "qid": holder_result.qid,
                    "transfer_seq": context.transfer_seq,
                    "mid": holder_result.mid,
                    "holder_state_source": holder_state_source,
                    "contribution_bytes": (
                        holder_result.metrics.contribution_bytes
                    ),
                    "pvss_seconds": (
                        holder_result.metrics.pvss_seconds
                    ),
                    "pvss_cpu_seconds": (
                        holder_result.metrics.pvss_cpu_seconds
                    ),
                    "serialization_seconds": (
                        holder_result.metrics.serialization_seconds
                    ),
                    "broadcast_seconds": (
                        holder_result.metrics.broadcast_seconds
                    ),
                }
            )
            return holder_result

        async def _submit_ocdpss_holder(
            context,
            operation_adapter,
            holder_state,
            *,
            holder_state_source,
        ):
            if holder_state is None:
                logging.info(
                    "[POOL-SCHEDULE] qid=OCDPSS transfer_seq=%d "
                    "non-holder waits for bundle RBC",
                    context.transfer_seq,
                )
                return None
            if ocdpss_rbc_manager is None:
                raise RuntimeError("OCDPSS bundle RBC manager missing")
            share_s = holder_state.share_s
            share_s_hat = holder_state.share_s_hat

            async def _broadcast_ocdpss(qid, mid, payload):
                if qid != context.encode_qid():
                    raise ValueError("holder emitted an unexpected OCDPSS qid")
                if mid != f"node_{my_id}":
                    raise ValueError("holder emitted an unexpected OCDPSS mid")
                await ocdpss_rbc_manager.broadcast_bundle(
                    context,
                    holder_id=my_id,
                    bundle_bytes=payload,
                )

            holder_result = await run_ocdpss_handover_role(
                context=context,
                holder_id=my_id,
                share_s=share_s,
                share_s_hat=share_s_hat,
                operation_adapter=operation_adapter,
                broadcast_bundle=_broadcast_ocdpss,
                holder_state=holder_state,
            )
            del share_s, share_s_hat
            ocdpss_events.append(
                {
                    "event": "holder_broadcast",
                    "algorithm": "Algorithm 5",
                    "protocol": "ocdpss",
                    "qid": holder_result.qid,
                    "transfer_seq": context.transfer_seq,
                    "mid": holder_result.mid,
                    "holder_state_source": holder_state_source,
                    "bundle_bytes": holder_result.metrics.bundle_bytes,
                    "operation_seconds": holder_result.metrics.operation_seconds,
                    "operation_cpu_seconds": (
                        holder_result.metrics.operation_cpu_seconds
                    ),
                    "pvss_seconds": holder_result.metrics.pvss_seconds,
                    "pvss_cpu_seconds": (
                        holder_result.metrics.pvss_cpu_seconds
                    ),
                    "serialization_seconds": (
                        holder_result.metrics.serialization_seconds
                    ),
                    "serialization_cpu_seconds": (
                        holder_result.metrics.serialization_cpu_seconds
                    ),
                    "broadcast_seconds": holder_result.metrics.broadcast_seconds,
                    "broadcast_cpu_seconds": (
                        holder_result.metrics.broadcast_cpu_seconds
                    ),
                }
            )
            return holder_result

        try:
            while time.time() < start_time:
                await asyncio.sleep(0.05)

            if experiment_barrier is not None:
                await experiment_barrier(runner, "ready")
            if a2dpss_pipeline:
                pipeline_started_at = time.perf_counter()
            start_run = workload_clock()
            broadcast_start_time = start_run
            if bhk22_smoke:
                if bhk22_runtime is None or bhk22_rbc_manager is None:
                    raise RuntimeError("BHK22 DAG network path is not initialized")
                outbound = bhk22_runtime.start()
                if outbound is not None:
                    await bhk22_rbc_manager.broadcast_role_batch(
                        outbound.phase_context,
                        role_id=outbound.role_id,
                        batch_bytes=outbound.batch_bytes,
                    )
                    bhk22_runtime.mark_sent(outbound.phase_context.phase)
                injected_qids.append(
                    {"round": 1, "qid": bhk22_runtime.controller.current.qid}
                )
            elif ru22_smoke:
                if ru22_runtime is None or ru22_rbc_manager is None:
                    raise RuntimeError(
                        "RU22 DAG network path is not initialized"
                    )
                ru22_runtime.start()
                await ru22_runtime.broadcast_local(ru22_rbc_manager)
                state = await _wait_for_ru22_structural_pool(
                    dag,
                    ru22_runtime,
                    dpss_pool_wait_seconds,
                )
                required_batches = (
                    len(ru22_runtime.source_role_ids)
                    if ru22_runtime.requires_complete_source_pool
                    else ru22_runtime.context.resharing_threshold + 1
                )
                if (
                    not state.ready
                    or state.valid_batch_count < required_batches
                ):
                    raise RuntimeError(
                        "timed out waiting for RU22 structural pool: "
                        f"ready={state.ready} "
                        f"valid_batches={state.valid_batch_count} "
                        f"required_batches={required_batches} "
                        f"rejected={state.rejected_entry_count}"
                    )
                ru22_runtime.mark_pool_ready(state)
                injected_qids.append(
                    {"round": 1, "qid": ru22_runtime.qid}
                )
                logging.info(
                    "[RU22:POOL] round=1 via contribution RBC "
                    "valid_batches=%d parent=%s roles=%s",
                    state.valid_batch_count,
                    state.common_parent_vps_id,
                    list(state.common_parent_role_ids),
                )
            elif a2dpss_smoke:
                a2dpss_handover_metrics.start()
                if (
                    a2dpss_context is None
                    or a2dpss_rbc_manager is None
                ):
                    raise RuntimeError(
                        "A2DPSS network path is not initialized"
                    )
                holder_result = await _submit_a2dpss_holder(
                    a2dpss_context,
                    a2dpss_holder_state,
                    holder_state_source=(
                        "paired-fixture"
                        if a2dpss_fixture is not None
                        else "initial-config"
                    ),
                )
                if holder_result is not None:
                    a2dpss_handover_metrics.mark_contribution_sent()
                a2dpss_scheduled_hops = 1
                qid = a2dpss_context.encode_qid()
                min_entries = (
                    len(a2dpss_fixture.entries)
                    if a2dpss_fixture is not None
                    else a2dpss_context.service_t + 1
                )
                pool_size = await _wait_for_pool_entries(
                    dag,
                    qid,
                    min_entries,
                    dpss_pool_wait_seconds,
                )
                a2dpss_handover_metrics.mark_pool_ready()
                injected_qids.append({"round": 1, "qid": qid})
                logging.info(
                    "[ALGORITHM-4:A2DPSS:POOL] round=1 "
                    "via contribution RBC pool_size=%d min=%d",
                    pool_size,
                    min_entries,
                )
            elif ocdpss_smoke:
                ocdpss_handover_metrics.start()
                if (
                    ocdpss_context is None
                    or ocdpss_adapter is None
                    or ocdpss_rbc_manager is None
                ):
                    raise RuntimeError("OCDPSS network path is not initialized")
                holder_result = await _submit_ocdpss_holder(
                    ocdpss_context,
                    ocdpss_adapter,
                    ocdpss_holder_state,
                    holder_state_source="initial-config",
                )
                if holder_result is not None:
                    ocdpss_handover_metrics.mark_contribution_sent()
                ocdpss_scheduled_hops = 1
                qid = ocdpss_context.encode_qid()
                min_entries = ocdpss_context.service_t + 1
                pool_size = await _wait_for_pool_entries(
                    dag,
                    qid,
                    min_entries,
                    ocdpss_pool_wait_seconds,
                )
                ocdpss_handover_metrics.mark_pool_ready()
                injected_qids.append({"round": 1, "qid": qid})
                logging.info(
                    "[ALGORITHM-5:OCDPSS:POOL] round=1 via bundle RBC "
                    "pool_size=%d min=%d",
                    pool_size,
                    min_entries,
                )
            elif dpss_smoke:
                if dpss_local is not None:
                    if dpss_transcript_rbc:
                        if dpss_rbc_manager is None:
                            raise RuntimeError("DPSS transcript RBC manager missing")
                        if dpss_local.am_holder:
                            share_s, share_s_hat = dpss_local.my_holder_share
                            transcript = make_resharing_transcript(
                                g_hex=dpss_g_hex,
                                h_hex=dpss_h_hex,
                                my_share_s=share_s,
                                my_share_s_hat=share_s_hat,
                                successor_rpk_list=(
                                    dpss_local.ctx.successor_rpk_list
                                ),
                                dealer_eval_point=dpss_local.my_holder_eval_point,
                                service_t=dpss_local.ctx.service_t,
                                my_id=my_id,
                            )
                            await dpss_rbc_manager.broadcast_transcript(
                                dpss_local.ctx,
                                holder_id=my_id,
                                transcript=transcript,
                            )
                            logging.info(
                                "[POOL-SCHEDULE] qid=DPSS holder=%d via transcript RBC",
                                my_id,
                            )
                        else:
                            logging.info(
                                "[POOL-SCHEDULE] qid=DPSS non-holder waits for transcript RBC"
                            )
                        qid = dpss_local.ctx.encode_qid()
                        min_entries = dpss_local.ctx.service_t + 1
                        pool_size = await _wait_for_pool_entries(
                            dag,
                            qid,
                            min_entries,
                            dpss_pool_wait_seconds,
                        )
                        injected_qids.append({"round": 1, "qid": qid})
                        logging.info(
                            "[POOL-SCHEDULE] round=1 qid=DPSS via RBC pool_size=%d min=%d",
                            pool_size,
                            min_entries,
                        )
                    else:
                        injected_holders = list(dpss_local.source_committee)
                        if not dpss_smoke_inject_all:
                            injected_holders = [my_id] if dpss_local.am_holder else []
                        for holder_id in injected_holders:
                            share_s, share_s_hat = dpss_local.holder_shares[holder_id]
                            await run_resharing(
                                dag_tob=dag,
                                g_hex=dpss_g_hex,
                                h_hex=dpss_h_hex,
                                my_share_s=share_s,
                                my_share_s_hat=share_s_hat,
                                successor_rpk_list=(
                                    dpss_local.ctx.successor_rpk_list
                                ),
                                dealer_eval_point=(
                                    dpss_local.ctx.holder_eval_points[holder_id]
                                ),
                                service_t=dpss_local.ctx.service_t,
                                epoch=dpss_local.ctx.epoch,
                                sid=dpss_local.ctx.sid,
                                source_committee=dpss_local.ctx.source_committee,
                                successor_committee=(
                                    dpss_local.ctx.successor_committee
                                ),
                                transfer_seq=dpss_local.ctx.transfer_seq,
                                my_id=holder_id,
                            )
                        injected_qids.append(
                            {"round": 1, "qid": dpss_local.ctx.encode_qid()}
                        )
                        logging.info("[POOL-SCHEDULE] round=1 injected qid=DPSS")
            elif multi_qid_smoke:
                _inject_workload_batches(
                    dag, workload, lifecycle, START, my_id, injected_qids,
                    **({"clock": workload_clock} if a2dpss_pipeline else {}),
                )
                initial_instances = [
                    instance
                    for batch in workload.batches_for_trigger(START)
                    for instance in batch.instances
                ]
                await wait_workload_ready(
                    initial_instances, START, dpss_pool_wait_seconds,
                )
                logging.info(
                    "[WORKLOAD] trigger=start initial requests are J-ready"
                )
            else:
                qid_single = f"qid-demo-{my_id}" if unique_qid_per_node else "qid-demo"
                dag.t_broadcast(
                    qid_single,
                    f"mid-{my_id}",
                    f"phase3-payload-from-{my_id}".encode("utf-8"),
                )
                injected_qids.append({"round": 1, "qid": qid_single})
            broadcast_end_time = workload_clock()
            # Give the message-RBC a brief window to disseminate contributions so
            # proposed blocks reference a non-trivial, locally-available set.
            if msg_mgr is not None and not unique_qid_per_node and not a2dpss_pipeline:
                primary_qid = qid_a if multi_qid_smoke else "qid-demo"
                await _wait_for_pool_entries(
                    dag, primary_qid, min(M, quorum_h), timeout=10.0
                )
            manager.ensure_window(0, lookahead_rounds)
            injected_dynamic_batch = False
            while True:
                locally_complete = (
                    dag.delivery_count >= target_deliveries
                    and (bhk22_runtime is None or bhk22_runtime.completed)
                    and (ru22_runtime is None or ru22_runtime.completed)
                    and (not a2dpss_pipeline or len(pipeline_events) == target_deliveries)
                )
                now = workload_clock()
                if locally_complete:
                    if completion_service_started_at is None:
                        completion_service_started_at = now
                        # Preserve historical handover counter freezing.  The
                        # pipeline instead measures until every participant has
                        # completed its public results and successor work.
                        if not a2dpss_pipeline:
                            frozen_summary = _build_summary(
                                runner.node_communicator.bytes_sent,
                                cleanup_completed=None,
                            )
                        logging.info(
                            "[COMPLETION-SERVICE] local output fixed; "
                            "serving peers for %.3fs",
                            completion_service_grace_seconds,
                        )
                    if a2dpss_pipeline or getattr(experiment_barrier, "keep_serving_until_done", False):
                        if experiment_done_task is None:
                            experiment_done_task = asyncio.create_task(
                                experiment_barrier(runner, "done")
                            )
                        if experiment_done_task.done():
                            await experiment_done_task
                            break
                    elif (
                        now - completion_service_started_at
                        >= completion_service_grace_seconds
                    ):
                        break
                if wave_coin_errors:
                    raise RuntimeError("DAG wave common coin failed") from wave_coin_errors[0]
                if bhk22_phase_errors:
                    raise RuntimeError(
                        "BHK22 phase processing failed"
                    ) from bhk22_phase_errors[0]
                if not locally_complete and now - start_run > max_runtime:
                    highest_ready = _highest_ready_round(dag)
                    stop_reason = _incomplete_runtime_reason(
                        highest_ready_round=highest_ready,
                        max_rounds=rounds,
                    )
                    raise RuntimeError(
                        f"{stop_reason}: deliveries={dag.delivery_count} "
                        f"target={target_deliveries} highest_ready={highest_ready} "
                        f"max_rounds={rounds} "
                        f"active_rbc={manager.active_count()} buffer={len(dag.buffer)}"
                    )

                if bhk22_runtime is not None:
                    phase_context = bhk22_runtime.controller.current
                    if phase_context is not None and len(
                        dag.Pool.get(phase_context.qid, set())
                    ) >= len(phase_context.expected_role_ids):
                        bhk22_runtime.mark_pool_ready(phase_context.phase)

                if (
                    a2dpss_smoke
                    and a2dpss_scheduled_hops < a2dpss_hops
                    and a2dpss_lifecycle is not None
                ):
                    current_context = a2dpss_lifecycle.active_context
                    completed_public = (
                        a2dpss_completed_public_results.get(
                            current_context.context_digest
                        )
                    )
                    if completed_public is not None:
                        next_transfer_seq = (
                            current_context.transfer_seq + 1
                        )
                        next_successor_committee = (
                            f"{a2dpss_context.successor_committee}-hop-"
                            f"{next_transfer_seq + 1}"
                        )
                        advanced = a2dpss_lifecycle.advance(
                            completed_public,
                            next_successor_committee=(
                                next_successor_committee
                            ),
                            next_successor_rpk_list=(
                                a2dpss_successor_rpk_list
                            ),
                            next_successor_role_indices=(
                                a2dpss_successor_role_indices
                            ),
                            my_next_rsk=(
                                a2dpss_my_successor_rsk
                                if my_id
                                in a2dpss_successor_role_indices
                                else None
                            ),
                            on_next_share=(
                                lambda _s, _sh, _seq=next_transfer_seq: a2dpss_events.append(
                                    {
                                        "event": "new_share",
                                        "algorithm": "Algorithm 4",
                                        "protocol": "a2dpss",
                                        "my_node_id": my_id,
                                        "transfer_seq": _seq,
                                    }
                                )
                                if my_id
                                in a2dpss_successor_role_indices
                                else None
                            ),
                        )
                        next_context = advanced.next_context
                        a2dpss_rbc_manager.ensure_context(
                            next_context,
                            sorted(next_context.holder_eval_points),
                        )
                        a2dpss_events.append(
                            {
                                "event": "lifecycle_advanced",
                                "algorithm": "Algorithm 4",
                                "protocol": "a2dpss",
                                "previous_qid": advanced.previous_qid,
                                "next_qid": next_context.encode_qid(),
                                "next_transfer_seq": (
                                    next_context.transfer_seq
                                ),
                                "anchor_source_context_digest": (
                                    completed_public.context_digest
                                ),
                                "has_holder_state": (
                                    advanced.am_next_holder
                                ),
                            }
                        )
                        await _submit_a2dpss_holder(
                            next_context,
                            advanced.holder_state,
                            holder_state_source=(
                                "successor-result:"
                                + completed_public.context_digest
                            ),
                        )
                        a2dpss_scheduled_hops += 1
                        next_qid = next_context.encode_qid()
                        pool_size = await _wait_for_pool_entries(
                            dag,
                            next_qid,
                            next_context.service_t + 1,
                            dpss_pool_wait_seconds,
                        )
                        injected_qids.append(
                            {
                                "round": (
                                    _highest_ready_round(dag) + 1
                                ),
                                "qid": next_qid,
                            }
                        )
                        logging.info(
                            "[ALGORITHM-4:A2DPSS:POOL] "
                            "transfer_seq=%d via protocol outputs "
                            "pool_size=%d min=%d",
                            next_context.transfer_seq,
                            pool_size,
                            next_context.service_t + 1,
                        )

                if (
                    ocdpss_smoke
                    and ocdpss_scheduled_hops < ocdpss_hops
                    and ocdpss_lifecycle is not None
                ):
                    current_context = ocdpss_lifecycle.active_context
                    completed_public = ocdpss_completed_public_results.get(
                        current_context.context_digest
                    )
                    if completed_public is not None:
                        next_transfer_seq = current_context.transfer_seq + 1
                        next_operation_input = (
                            os.environ.get(
                                "DAG_TOB_OCDPSS_OPERATION_INPUT",
                                "atcaas-ocdpss-network-smoke",
                            )
                            + f":hop:{next_transfer_seq}"
                        ).encode("utf-8")
                        next_successor_committee = (
                            f"{ocdpss_context.successor_committee}-hop-"
                            f"{next_transfer_seq + 1}"
                        )
                        advanced = ocdpss_lifecycle.advance(
                            completed_public,
                            next_successor_committee=next_successor_committee,
                            next_successor_rpk_list=ocdpss_successor_rpk_list,
                            next_successor_role_indices=(
                                ocdpss_successor_role_indices
                            ),
                            next_operation_adapter=ocdpss_adapter,
                            next_operation_input=next_operation_input,
                            my_next_rsk=(
                                ocdpss_my_successor_rsk
                                if my_id
                                in ocdpss_successor_role_indices
                                else None
                            ),
                            on_next_share=(
                                lambda _s, _sh, _seq=next_transfer_seq: ocdpss_events.append(
                                    {
                                        "event": "new_share",
                                        "my_node_id": my_id,
                                        "transfer_seq": _seq,
                                    }
                                )
                                if my_id
                                in ocdpss_successor_role_indices
                                else None
                            ),
                        )
                        next_context = advanced.next_context
                        ocdpss_rbc_manager.ensure_context(
                            next_context,
                            sorted(next_context.holder_eval_points),
                        )
                        ocdpss_events.append(
                            {
                                "event": "lifecycle_advanced",
                                "previous_qid": advanced.previous_qid,
                                "next_qid": next_context.encode_qid(),
                                "next_transfer_seq": next_context.transfer_seq,
                                "anchor_source_context_digest": (
                                    completed_public.context_digest
                                ),
                                "has_holder_state": advanced.am_next_holder,
                            }
                        )
                        await _submit_ocdpss_holder(
                            next_context,
                            ocdpss_adapter,
                            advanced.holder_state,
                            holder_state_source=(
                                "successor-result:"
                                + completed_public.context_digest
                            ),
                        )
                        ocdpss_scheduled_hops += 1
                        next_qid = next_context.encode_qid()
                        pool_size = await _wait_for_pool_entries(
                            dag,
                            next_qid,
                            next_context.service_t + 1,
                            ocdpss_pool_wait_seconds,
                        )
                        injected_qids.append(
                            {
                                "round": _highest_ready_round(dag) + 1,
                                "qid": next_qid,
                            }
                        )
                        logging.info(
                            "[POOL-SCHEDULE] qid=OCDPSS transfer_seq=%d "
                            "via protocol outputs pool_size=%d min=%d",
                            next_context.transfer_seq,
                            pool_size,
                            next_context.service_t + 1,
                        )

                highest_ready = _highest_ready_round(dag)
                manager.ensure_window(highest_ready, lookahead_rounds)
                if (
                    multi_qid_smoke
                    and not injected_dynamic_batch
                    and dag.is_round_ready(1)
                ):
                    boundary_at = workload_clock()
                    later_instances = [
                        instance
                        for batch in workload.batches_for_trigger(
                            INITIAL_PROPOSAL_FIXED
                        )
                        for instance in batch.instances
                    ]
                    for instance in later_instances:
                        lifecycle.mark_proposal_boundary(
                            workload.qid(instance),
                            boundary_at,
                            {"protocol": "dag-tob", "round_ready": 1},
                        )
                    _inject_workload_batches(
                        dag,
                        workload,
                        lifecycle,
                        INITIAL_PROPOSAL_FIXED,
                        my_id,
                        injected_qids,
                        **({"clock": workload_clock} if a2dpss_pipeline else {}),
                    )
                    await wait_workload_ready(
                        later_instances, INITIAL_PROPOSAL_FIXED,
                        ocdpss_pool_wait_seconds,
                    )
                    injected_dynamic_batch = True
                    logging.info(
                        "[WORKLOAD] trigger=initial_proposal_fixed "
                        "dag_round=1 later requests are J-ready"
                    )

                for parent_round in range(0, min(highest_ready, rounds - 1) + 1):
                    if parent_round in role_checked:
                        continue
                    if not dag.is_round_ready(parent_round):
                        continue
                    role_checked.add(parent_round)
                    round_id = parent_round + 1
                    label = _vertex_committee_label(epoch, parent_round)
                    selected, proof = _local_vertex_sortition(label)
                    logging.info(
                        "[SORTITION] round=%d parent_round=%d selected=%s label=%s",
                        round_id,
                        parent_round,
                        selected,
                        label.decode("utf-8"),
                    )
                    if selected and round_id % 4 == 1:
                        wave = (round_id + 3) // 4
                        if my_id not in leader_committees.roles(wave):
                            logging.warning(
                                "[SORTITION] suppress v(%d,%d): role absent from replayed VRF committee wave=%d",
                                round_id,
                                my_id,
                                wave,
                            )
                            selected = False
                    if selected:
                        vertex = dag.create_new_vertex(
                            round_id,
                            source=my_id,
                            proof=proof,
                        )
                        await manager.broadcast_vertex(vertex)

                await manager.gc(highest_ready, retention_rounds)
                await asyncio.sleep(0.05)
            if experiment_barrier is not None:
                if a2dpss_pipeline and len(pipeline_events) != target_deliveries:
                    raise RuntimeError("A2DPSS pipeline did not complete every request")
                if a2dpss_smoke and not a2dpss_handover_metrics.snapshot()["frozen"]:
                    raise RuntimeError("A2DPSS did not complete before the runtime limit")
                if (
                    ocdpss_smoke
                    and not ocdpss_handover_metrics.snapshot()["frozen"]
                ):
                    raise RuntimeError("OCDPSS did not complete before the runtime limit")
                if experiment_done_task is None:
                    await experiment_barrier(runner, "done")
        finally:
            if experiment_done_task is not None and not experiment_done_task.done():
                experiment_done_task.cancel()
                await asyncio.gather(experiment_done_task, return_exceptions=True)
            bytes_sent = runner.node_communicator.bytes_sent
            summary = frozen_summary or _build_summary(
                bytes_sent, cleanup_completed=None
            )
            summary["completion_service_grace_seconds"] = (
                completion_service_grace_seconds
            )
            summary["completion_service_elapsed_sec"] = (
                max(0.0, workload_clock() - completion_service_started_at)
                if completion_service_started_at is not None
                else 0.0
            )
            logging.info("[SUMMARY] %s", json.dumps(summary, sort_keys=True))
            pending_bhk22_tasks = list(bhk22_phase_tasks)
            for task in pending_bhk22_tasks:
                if not task.done():
                    task.cancel()
            if pending_bhk22_tasks:
                await asyncio.gather(
                    *pending_bhk22_tasks, return_exceptions=True
                )
            if bhk22_runtime is not None:
                summary["bhk22_controller_before_cleanup"] = (
                    bhk22_runtime.controller.snapshot()
                )
                bhk22_runtime.retire()
                summary["bhk22_controller_after_cleanup"] = (
                    bhk22_runtime.controller.snapshot()
                )
                summary["registered_bhk22_contexts_after_cleanup"] = len(
                    bhk22_runtime.justifier.contexts
                )
                summary[
                    "delivered_bhk22_pool_entry_counts_after_context_cleanup"
                ] = {
                    qid: len(entries)
                    for qid, entries in dag.DeliveredPool.items()
                    if qid in {
                        context.qid
                        for context in bhk22_runtime.controller.phase_contexts.values()
                    }
                }
            if ru22_runtime is not None and hasattr(ru22_runtime, "retire"):
                ru22_runtime.retire()
                summary["registered_ru22_contexts_after_cleanup"] = len(
                    getattr(ru22_runtime.justifier, "_registrations", {})
                )
            if a2dpss_lifecycle is not None:
                summary["a2dpss_lifecycle_before_cleanup"] = (
                    a2dpss_lifecycle.snapshot()
                )
                a2dpss_lifecycle.retire_active()
                summary["a2dpss_lifecycle"] = (
                    a2dpss_lifecycle.snapshot()
                )
                summary[
                    "registered_a2dpss_contexts_after_cleanup"
                ] = len(a2dpss_justifier_manager.contexts)
                summary[
                    "registered_a2dpss_reconstruction_contexts_after_cleanup"
                ] = len(
                    a2dpss_reconstruction_manager.registrations
                )
                summary[
                    "delivered_a2dpss_pool_entry_counts_after_context_cleanup"
                ] = {
                    qid: len(entries)
                    for qid, entries in dag.DeliveredPool.items()
                    if Qid.decode(qid).kind == "a2dpss"
                }
            if ocdpss_lifecycle is not None:
                summary["ocdpss_lifecycle_before_cleanup"] = (
                    ocdpss_lifecycle.snapshot()
                )
                ocdpss_lifecycle.retire_active()
                summary["ocdpss_lifecycle"] = ocdpss_lifecycle.snapshot()
                summary["registered_ocdpss_contexts_after_cleanup"] = len(
                    ocdpss_justifier_manager.contexts
                )
                summary[
                    "registered_ocdpss_reconstruction_contexts_after_cleanup"
                ] = len(ocdpss_reconstruction_manager.registrations)
                summary["delivered_pool_entry_counts_after_context_cleanup"] = {
                    qid: len(entries)
                    for qid, entries in dag.DeliveredPool.items()
                    if Qid.decode(qid).kind == "ocdpss"
                }
            dpss_cleanup_completed = True
            if dpss_rbc_manager is not None:
                dpss_cleanup_completed = await dpss_rbc_manager.stop(
                    timeout=stop_grace_seconds
                )
            a2dpss_cleanup_completed = True
            if a2dpss_rbc_manager is not None:
                a2dpss_cleanup_completed = (
                    await a2dpss_rbc_manager.stop(
                        timeout=stop_grace_seconds
                    )
                )
            ru22_cleanup_completed = True
            if ru22_rbc_manager is not None:
                ru22_cleanup_completed = await ru22_rbc_manager.stop(
                    timeout=stop_grace_seconds
                )
            bhk22_cleanup_completed = True
            if bhk22_rbc_manager is not None:
                bhk22_cleanup_completed = await bhk22_rbc_manager.stop(
                    timeout=stop_grace_seconds
                )
            ocdpss_cleanup_completed = True
            if ocdpss_rbc_manager is not None:
                ocdpss_cleanup_completed = await ocdpss_rbc_manager.stop(
                    timeout=stop_grace_seconds
                )
            cleanup_completed = await manager.stop(timeout=stop_grace_seconds)
            tasks = list(wave_coin_tasks.values())
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if threshold_leader_oracle is not None:
                await threshold_leader_oracle.close()
            if threshold_coin_oracle is not None:
                await threshold_coin_oracle.close()
            summary["coin_observations"] = (
                coin_observer.snapshot()
                if coin_observer is not None
                else None
            )
            summary["coin_metrics"] = (
                threshold_coin_oracle.snapshot()
                if threshold_coin_oracle is not None
                else None
            )
            summary["wave_coin"] = (
                threshold_leader_oracle.snapshot()
                if threshold_leader_oracle is not None
                else None
            )
            summary["wave_coin_state"] = dag.snapshot_wave_coin_state()
            msg_cleanup_completed = True
            if msg_mgr is not None:
                summary["delivered_messages"] = len(msg_mgr.delivered_messages)
                msg_cleanup_completed = await msg_mgr.stop(timeout=stop_grace_seconds)
                summary["active_msg_rbc_instances_after_cleanup"] = (
                    msg_mgr.active_count()
                )
            if a2dpss_pipeline:
                for context in workload.registrations():
                    pipeline_reconstruction.unregister_context(context)
                    pipeline_justifier.unregister_context(context)
                summary["a2dpss_pipeline_contexts_after_cleanup"] = len(
                    pipeline_justifier.contexts
                )
                summary["a2dpss_pipeline_reconstruction_contexts_after_cleanup"] = len(
                    pipeline_reconstruction.registrations
                )
            summary["cleanup_completed"] = (
                cleanup_completed
                and dpss_cleanup_completed
                and a2dpss_cleanup_completed
                and ru22_cleanup_completed
                and bhk22_cleanup_completed
                and ocdpss_cleanup_completed
                and msg_cleanup_completed
            )
            summary["active_rbc_instances_after_cleanup"] = manager.active_count()
            summary["active_dpss_rbc_instances_after_cleanup"] = (
                dpss_rbc_manager.active_count()
                if dpss_rbc_manager is not None
                else 0
            )
            summary[
                "active_a2dpss_rbc_instances_after_cleanup"
            ] = (
                a2dpss_rbc_manager.active_count()
                if a2dpss_rbc_manager is not None
                else 0
            )
            summary["active_ru22_rbc_instances_after_cleanup"] = (
                ru22_rbc_manager.active_count()
                if ru22_rbc_manager is not None
                else 0
            )
            summary["active_bhk22_rbc_instances_after_cleanup"] = (
                bhk22_rbc_manager.active_count()
                if bhk22_rbc_manager is not None
                else 0
            )
            if bhk22_runtime is not None:
                summary["bhk22_summary"] = bhk22_runtime.summary(
                    rbc_metrics=(
                        bhk22_rbc_manager.metrics_snapshot()
                        if bhk22_rbc_manager is not None
                        else None
                    ),
                    cleanup={
                        "completed": summary["cleanup_completed"],
                        "registered_contexts": len(
                            bhk22_runtime.justifier.contexts
                        ),
                        "active_rbc_instances": summary[
                            "active_bhk22_rbc_instances_after_cleanup"
                        ],
                    },
                )
            summary["active_ocdpss_rbc_instances_after_cleanup"] = (
                ocdpss_rbc_manager.active_count()
                if ocdpss_rbc_manager is not None
                else 0
            )
            logging.info(
                "[CLEANUP] completed=%s active_after=%d "
                "active_a2dpss_after=%d active_dpss_after=%d "
                "active_ocdpss_after=%d active_ru22_after=%d "
                "active_bhk22_after=%d",
                summary["cleanup_completed"],
                manager.active_count(),
                summary[
                    "active_a2dpss_rbc_instances_after_cleanup"
                ],
                summary["active_dpss_rbc_instances_after_cleanup"],
                summary["active_ocdpss_rbc_instances_after_cleanup"],
                summary["active_ru22_rbc_instances_after_cleanup"],
                summary["active_bhk22_rbc_instances_after_cleanup"],
            )
            # Emit the cleanup-complete summary before communicator teardown.
            # A peer may close its socket first, so waiting to print only after
            # leaving ProcessProgramRunner can hide an otherwise completed run.
            logging.info(
                "[SUMMARY] %s",
                json.dumps(summary, sort_keys=True),
            )
            shared_task.cancel()
            # ``wait_for(cancelled_task)`` can itself wait indefinitely when
            # the underlying receive coroutine suppresses cancellation.  A
            # bounded ``wait`` lets final-summary emission and process exit
            # proceed after all protocol/RBC managers have already stopped.
            done, pending = await asyncio.wait({shared_task}, timeout=1.0)
            if pending:
                logging.warning(
                    "[CLEANUP] shared receive task ignored cancellation; "
                    "leaving it cancelled for event-loop teardown"
                )
            else:
                try:
                    shared_task.result()
                except asyncio.CancelledError:
                    pass

    return summary


def main() -> None:
    _install_cli_handover_mode()
    HbmpcConfig.load_config()
    _configure_logging(HbmpcConfig.my_id)

    M = getattr(HbmpcConfig, "M", HbmpcConfig.N)
    expected_committee_size = getattr(
        HbmpcConfig,
        "expected_committee_size",
        getattr(HbmpcConfig, "committee_n", HbmpcConfig.N),
    )
    committee_m = getattr(HbmpcConfig, "committee_m", expected_committee_size)
    quorum_h = getattr(HbmpcConfig, "quorum_h", 2 * HbmpcConfig.t + 1)
    epoch = os.environ.get(
        "ATCAAS_TRIAL_EPOCH",
        HbmpcConfig.extras.get("run_id", "phase3"),
    )
    lookahead_rounds = _env_int("DAG_TOB_LOOKAHEAD_ROUNDS", 4)
    retention_rounds = _env_int("DAG_TOB_RETENTION_ROUNDS", 8)
    multi_qid_smoke = _env_bool("DAG_TOB_MULTI_QID_SMOKE", False)
    unique_qid_per_node = _env_bool("DAG_TOB_UNIQUE_QID_PER_NODE", False)
    handover_mode = _handover_mode_from_env()
    dpss_smoke = handover_mode == "legacy-dpss"
    a2dpss_smoke = handover_mode == "a2dpss"
    ocdpss_smoke = handover_mode == "ocdpss"
    if _deprecated_dpss_alias_enabled():
        logging.warning(
            "[DEPRECATED] DAG_TOB_DPSS_SMOKE now selects formal "
            "Algorithm 4 A2DPSS; use DAG_TOB_A2DPSS_SMOKE or "
            "DAG_TOB_HANDOVER_MODE=a2dpss"
        )
    ocdpss_paired = _env_bool("DAG_TOB_OCDPSS_PAIRED", False)
    fixture_path = os.environ.get("ATCAAS_A2DPSS_FIXTURE")
    a2dpss_fixture = (
        A2DPSSPairedFixture.from_file(fixture_path)
        if fixture_path
        else None
    )
    target_deliveries = _env_int(
        "DAG_TOB_TARGET_DELIVERIES",
        (
            _env_int("DAG_TOB_A2DPSS_HOPS", 1)
            if a2dpss_smoke
            else _env_int("DAG_TOB_OCDPSS_HOPS", 1)
            if ocdpss_smoke
            else 1
        ),
    )
    max_runtime = _env_float("DAG_TOB_MAX_RUNTIME", 90.0)
    rounds = _env_int("DAG_TOB_MAX_ROUNDS", 12)
    stop_grace_seconds = _env_float("DAG_TOB_STOP_GRACE_SECONDS", 10.0)
    completion_service_grace_seconds = _env_float(
        "DAG_TOB_COMPLETION_SERVICE_GRACE_SECONDS", 0.0
    )
    coin_mode_requested = os.environ.get("DAG_TOB_COIN_MODE", "ideal")
    coin_silent_nodes = _node_id_set(
        os.environ.get("DAG_TOB_COIN_SILENT_NODES", ""),
        M=M,
        name="DAG_TOB_COIN_SILENT_NODES",
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        summary = loop.run_until_complete(
            _run(
                peers=HbmpcConfig.peers,
                n=HbmpcConfig.N,
                t=HbmpcConfig.t,
                my_id=HbmpcConfig.my_id,
                start_time=HbmpcConfig.time,
                M=M,
                expected_committee_size=expected_committee_size,
                committee_m=committee_m,
                quorum_h=quorum_h,
                epoch=epoch,
                rounds=rounds,
                target_deliveries=target_deliveries,
                max_runtime=max_runtime,
                lookahead_rounds=lookahead_rounds,
                retention_rounds=retention_rounds,
                stop_grace_seconds=stop_grace_seconds,
                completion_service_grace_seconds=(
                    completion_service_grace_seconds
                ),
                multi_qid_smoke=multi_qid_smoke,
                unique_qid_per_node=unique_qid_per_node,
                dpss_smoke=dpss_smoke,
                a2dpss_smoke=a2dpss_smoke,
                ocdpss_smoke=ocdpss_smoke,
                ocdpss_paired=ocdpss_paired,
                coin_mode_requested=coin_mode_requested,
                coin_timeout=_env_float("DAG_TOB_COIN_TIMEOUT", 30.0),
                coin_silent_nodes=coin_silent_nodes,
                a2dpss_fixture=a2dpss_fixture,
                n_r=HbmpcConfig.n_r,
                t_r=HbmpcConfig.t_r,
                n_a=HbmpcConfig.n_a,
                t_a=HbmpcConfig.t_a,
                m_a=HbmpcConfig.m_a,
                coin_c=HbmpcConfig.coin_c,
                coin_t=HbmpcConfig.coin_t,
                coin_k=HbmpcConfig.coin_k,
            )
        )
        print(json.dumps(summary, sort_keys=True))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
