from __future__ import annotations

import asyncio
import base64
import functools
import json
import logging
import os
import pickle
import sys
import time

from beaver.broadcast.crypto.boldyreva import TBLSPublicKey  # noqa:F401
from beaver.broadcast.crypto.boldyreva import TBLSPrivateKey  # noqa:F401
from beaver.a2dpss import (
    A2DPSSContext,
    A2DPSSContribution,
    A2DPSSHolderState,
    A2DPSSJustifierManager,
    A2DPSSPublicFinalizer,
    A2DPSSSuccessorReconstructionManager,
    a2dpss_contribution_digest,
    run_a2dpss_handover_role,
)
from beaver.a2dpss_acs_adapter import A2DPSSACSAdapter
from beaver.a2dpss_experiment import A2DPSSPairedFixture
from beaver.a2dpss_pipeline_workload import A2DPSSPairedWorkload
from beaver.async_yoso_acs import (
    CoreBlock,
    NetworkAsyncYosoACS,
    ProposalAdmission,
    ProposedBlock,
)
from beaver.async_yoso_coin import (
    DeterministicCoinOracle,
    ObservedCoinOracle,
    SharedIdealCoinOracle,
    ThresholdBLSCoinOracle,
    load_threshold_coin_key_material,
)
from beaver.async_yoso_tob import (
    AsyncYosoMessageRBCManager,
    AsyncYosoTOB,
    MessageRef,
)
from beaver.config import HbmpcConfig
from beaver.dag_tob import JustifierRegistry, TypedWaitPredicate
from beaver.experiment_trace import (
    A2DPSSHandoverMetrics,
    INITIAL_PROPOSAL_FIXED,
    START,
    QueryLifecycleMetrics,
    build_a2dpss_computation_metrics,
    load_dynamic_workload,
)
from beaver.ipc import ProcessProgramRunner
from beaver.ocdpss import (
    OCDPSSJustifierManager,
    OCDPSSPublicFinalizer,
    OCDPSSSuccessorReconstructionManager,
)
from beaver.ocdpss_experiment import OCDPSSPairedWorkload
from beaver.protocol_trace import (
    CommitteeRoleTraceArtifact,
    ProtocolReplayTrace,
)
from beaver.utils.misc import subscribe_recv
from beaver.vrf import (
    BERNOULLI_COMMITTEE_MODE,
    FIXED_SIZE_COMMITTEE_MODE,
    normalize_committee_selection_mode,
)
from beaver.handover_fixtures import build_handover_local_config


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _env_optional_int(name: str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return int(raw)


def _env_optional_float(name: str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return float(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _coin_mode(raw: str, *, replay_enabled: bool) -> tuple[str, bool]:
    requested = str(raw).strip().lower()
    if requested not in {"ideal", "threshold-bls"}:
        raise ValueError(
            "ASYNC_YOSO_TOB_COIN_MODE must be 'ideal' or 'threshold-bls'"
        )
    # A replay artifact must never mutate the explicitly requested coin mode.
    # ``replay_enabled`` remains in the signature for compatibility with the
    # Phase-4 tests and summary schema; the second result is now always false.
    return requested, False


def _node_id_set(raw: str, *, M: int, name: str) -> frozenset[int]:
    if not raw.strip():
        return frozenset()
    try:
        node_ids = frozenset(int(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated node-id list") from exc
    invalid = sorted(node_id for node_id in node_ids if not 0 <= node_id < M)
    if invalid:
        raise ValueError(f"{name} contains out-of-range node ids: {invalid}")
    return node_ids


def _configure_logging(my_id: int) -> None:
    os.makedirs("log", exist_ok=True)
    logfile = f"log/async-yoso-tob-{my_id}.log"
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


def _gh_from_srs(srs: dict):
    pk = json.loads(srs["Pk"].decode("utf-8"))
    return _g1_raw(pk["G1_g"][0]), _g1_raw(pk["G1_h"][0])


def _build_a2dpss_dissemination_state(
    *,
    extras,
    srs,
    my_id: int,
    M: int,
    service_t: int,
    fixture: A2DPSSPairedFixture | None = None,
):
    """Project the same neutral handover fixture used by the DAG runner."""

    g_hex, h_hex = _gh_from_srs(srs)
    local = build_handover_local_config(
        extras,
        g_hex=g_hex,
        h_hex=h_hex,
        my_id=my_id,
        M=M,
        service_t=service_t,
        epoch=0,
    )
    legacy_context = local.ctx
    if fixture is not None:
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
            getattr(fixture.context, field) != getattr(legacy_context, field)
            for field in comparable_fields
        ):
            raise ValueError(
                "paired fixture is incompatible with local handover config"
            )
        context = fixture.context
    else:
        context = A2DPSSContext(
            epoch=legacy_context.epoch,
            sid=str(legacy_context.sid).replace("dpss", "a2dpss"),
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
        )
    justifier = A2DPSSJustifierManager([context])
    return {
        "context": context,
        "source_holder_ids": (
            fixture.source_holder_ids()
            if fixture is not None
            else tuple(local.source_committee)
        ),
        "successor_node_ids": (
            fixture.successor_node_ids()
            if fixture is not None
            else tuple(local.successor_committee)
        ),
        "my_successor_rsk": local.my_successor_rsk,
        "my_successor_role_idx": local.my_successor_role_idx,
        "is_holder": my_id in context.holder_eval_points,
        "fixture_entry": (
            fixture.entry_for_sender(my_id)
            if fixture is not None and my_id in fixture.source_holder_ids()
            else None
        ),
        "holder_state": (
            A2DPSSHolderState(*local.my_holder_share)
            if local.am_holder and fixture is None
            else None
        ),
        "justifier": justifier,
        "adapter": A2DPSSACSAdapter(justifier),
    }


def _a2dpss_pending_snapshot(tob, adapter, context) -> list:
    """Summarize public contribution identities without running a predicate."""

    records = []
    for ref in sorted(tob.pending):
        entry = tob.pending[ref]
        record = {
            "ref": ref.encode(),
            "mid": entry.mid,
            "payload_digest": a2dpss_contribution_digest(entry.payload),
            "payload_bytes": len(entry.payload),
            "structurally_valid": False,
            "predicate_checked": False,
        }
        try:
            contribution = A2DPSSContribution.from_bytes(entry.payload)
            qid = adapter.qid_for_payload(entry.payload)
            structurally_valid = (
                contribution.context_digest == context.context_digest
                and entry.mid == f"node_{contribution.holder_id}"
                and context.holder_eval_points.get(
                    contribution.holder_id
                )
                == contribution.holder_eval_point
            )
            record.update(
                {
                    "qid": qid.encode(),
                    "context_digest": contribution.context_digest,
                    "holder_id": contribution.holder_id,
                    "holder_eval_point": (
                        contribution.holder_eval_point
                    ),
                    "structurally_valid": structurally_valid,
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            record["error"] = str(exc)
        records.append(record)
    return records


def _a2dpss_entry_set_snapshot(entries) -> list:
    """Summarize public transcript identities without logging payload bytes."""

    records = []
    for mid, payload in sorted(entries):
        contribution = A2DPSSContribution.from_bytes(payload)
        records.append(
            {
                "mid": mid,
                "holder_id": contribution.holder_id,
                "holder_eval_point": contribution.holder_eval_point,
                "context_digest": contribution.context_digest,
                "contribution_digest": a2dpss_contribution_digest(
                    contribution.to_bytes()
                ),
            }
        )
    return records


def _demo_qid(epoch: str, instance: str) -> str:
    return json.dumps(
        {
            "kind": "demo",
            "fields": {
                "epoch": epoch,
                "service": "multi-qid-smoke",
                "instance": instance,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _demo_payload(
    epoch: str,
    my_id: int,
    index: int,
    *,
    qid: str = "qid-demo",
    label: str = "demo",
) -> bytes:
    return json.dumps(
        {
            "kind": "demo",
            "qid": qid,
            "epoch": epoch,
            "sender": my_id,
            "index": index,
            "payload": {
                "label": label,
                "node": my_id,
                "idx": index,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_qid(payload: bytes) -> str:
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ""
    return str(data.get("qid", ""))


def _pending_refs_for_qids(
    tob: AsyncYosoTOB,
    qids,
    qid_resolver=_payload_qid,
) -> list:
    wanted = set(qids)
    refs = []
    for ref in tob.pending_refs():
        entry = tob.pending[ref]
        if qid_resolver(entry.payload) in wanted:
            refs.append(ref)
    return refs


async def _wait_for_refs(
    *,
    tob: AsyncYosoTOB,
    qids,
    expected: int,
    start_run: float,
    max_runtime: float,
    label: str,
    qid_resolver=_payload_qid,
):
    while True:
        refs = _pending_refs_for_qids(tob, qids, qid_resolver)
        if len(refs) >= expected:
            return refs
        if time.time() - start_run > max_runtime:
            raise RuntimeError(
                f"max runtime exceeded waiting for {label}: "
                f"refs={len(refs)} expected={expected} pending={len(tob.pending)}"
            )
        await asyncio.sleep(0.05)


async def _wait_for_a2dpss_candidate(
    *,
    tob,
    adapter,
    start_run: float,
    max_runtime: float,
):
    """Wait for a structurally ready honest proposal without running PVSS."""

    while True:
        refs = adapter.build_structurally_ready_candidate_refs(
            tob.pending,
            delivered_qids=(),
        )
        if refs:
            return refs
        if time.time() - start_run > max_runtime:
            raise RuntimeError(
                "max runtime exceeded waiting for A2DPSS candidate "
                f"pending={len(tob.pending)}"
            )
        await asyncio.sleep(0.05)


def _validate_a2dpss_proposal_once(
    *,
    node_id,
    session_id,
    proposer,
    proposal_refs,
    adapter,
    pending,
    records,
    result_cache,
):
    """Run and account for one receiver-side predicate per ACSPROP block."""

    proposal = ProposedBlock.build(proposer, proposal_refs)
    block_id = CoreBlock.build((proposal,)).digest()
    cache_key = (str(session_id), int(proposer), block_id)
    cached = result_cache.get(cache_key)
    if cached is not None:
        return cached

    # MissingTranscriptReferences is intentionally allowed to propagate.  A
    # retry before all referenced RBC transcripts arrive has not invoked the
    # formal predicate and therefore must not be charged as one.
    groups = adapter.group_entries_by_qid(proposal.refs, pending)
    accepted = bool(groups)
    for qid, entries in groups.items():
        started = time.process_time()
        predicate_accepted = adapter.justifier.verify_candidate_set(
            qid,
            entries,
        )
        cpu_seconds = time.process_time() - started
        records.append(
            {
                "protocol": "acs-tob",
                "node": int(node_id),
                "session_id": str(session_id),
                "qid": qid.encode(),
                "block_id": block_id,
                "proposer": int(proposer),
                "entry_count": len(entries),
                "cpu_seconds": cpu_seconds,
                "accepted": bool(predicate_accepted),
                "role": "receiver-admission",
                "is_local_proposer": int(proposer) == int(node_id),
            }
        )
        accepted = accepted and predicate_accepted

    admission = ProposalAdmission(
        accepted=accepted,
        qid_count=len(groups),
        transcript_count=sum(len(entries) for entries in groups.values()),
    )
    if admission.accepted:
        adapter.record_verified_candidate(
            str(session_id),
            int(proposer),
            proposal.refs,
            pending,
        )
    result_cache[cache_key] = admission
    return admission


async def _inject_workload_batches(
    tob,
    workload,
    lifecycle,
    trigger,
    my_id,
    events,
    clock=time.time,
):
    now = clock()
    for batch in workload.batches_for_trigger(trigger):
        for instance in batch.instances:
            qid = workload.qid(instance)
            lifecycle.mark_released(qid, now)
            should_broadcast = getattr(workload, "should_broadcast", None)
            if should_broadcast is None or should_broadcast(instance, my_id):
                for index in range(workload.messages_per_node):
                    await tob.t_broadcast(
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
    *,
    tob,
    workload,
    lifecycle,
    instances,
    wait_predicate,
    qid_resolver,
    start_run: float,
    max_runtime: float,
    clock=time.time,
    ready_cpu=None,
):
    """Wait until the real ACS-side J(qid,S) accepts every query."""

    pending = {str(instance) for instance in instances}
    while pending:
        for instance in tuple(sorted(pending)):
            qid = workload.qid(instance)
            entries = [
                (entry.mid, entry.payload)
                for entry in tob.pending.values()
                if qid_resolver(entry.payload) == qid
            ]
            expected = (
                workload.expected_entry_count(instance)
                if hasattr(workload, "expected_entry_count")
                else workload.messages_per_node * tob.M
            )
            if len(entries) < expected:
                continue
            cpu_started = time.process_time()
            selected = wait_predicate.select_block(
                {qid: set(entries)}, set()
            ) or {}
            if ready_cpu is not None:
                ready_cpu["cpu_seconds"] += time.process_time() - cpu_started
                ready_cpu["invocations"] += 1
            if qid in selected:
                lifecycle.mark_pool_ready(qid, clock())
                pending.remove(instance)
        if not pending:
            return
        if time.time() - start_run > max_runtime:
            raise RuntimeError(
                "max runtime exceeded waiting for J(qid,S)=true: "
                + ",".join(sorted(pending))
            )
        await asyncio.sleep(0.05)


async def _run(
    *,
    peers,
    n: int,
    t: int,
    my_id: int,
    start_time: int,
    M: int,
    epoch: str,
    messages_per_node: int = 1,
    max_runtime: float = 60.0,
    stop_grace_seconds: float = 5.0,
    enable_ledger: bool = True,
    acs_max_rounds: int = 10,
    acs_collect_all: bool = True,
    acs_activation_mode: str = "all",
    expected_committee_size: int = None,
    committee_m: int = None,
    quorum_h: int = None,
    acs_collect_threshold: int = None,
    acs_stage_timeout: float = None,
    acs_echo_adoption_wait: float = 0.25,
    acs_output_service_grace: float = 0.25,
    multi_qid_smoke: bool = False,
    ocdpss_paired: bool = False,
    a2dpss_pipeline: bool = False,
    a2dpss_smoke: bool = False,
    a2dpss_acsprop_smoke: bool = False,
    a2dpss_e2e_smoke: bool = False,
    coin_mode_requested: str = "ideal",
    coin_timeout: float = 30.0,
    coin_silent_nodes=(),
    a2dpss_fixture: A2DPSSPairedFixture | None = None,
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
    if a2dpss_pipeline:
        if not multi_qid_smoke or ocdpss_paired or a2dpss_smoke:
            raise ValueError("A2DPSS pipeline requires exclusive multi-qid mode")
        if messages_per_node != 1:
            raise ValueError("A2DPSS pipeline emits one contribution per holder/qid")
        enable_ledger = True
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
        if acs_activation_mode != "vrf":
            raise ValueError("paired ACS-TOB runs require VRF activation")
    if ocdpss_paired and not multi_qid_smoke:
        raise ValueError("OCDPSS paired mode requires multi-qid mode")
    if a2dpss_smoke and (multi_qid_smoke or ocdpss_paired):
        raise ValueError(
            "A2DPSS dissemination mode is mutually exclusive with "
            "demo/OCDPSS workloads"
        )
    if a2dpss_smoke and messages_per_node != 1:
        raise ValueError(
            "A2DPSS dissemination mode emits one contribution per holder"
        )
    if a2dpss_acsprop_smoke and not a2dpss_smoke:
        raise ValueError(
            "A2DPSS ACSPROP mode requires A2DPSS dissemination mode"
        )
    if a2dpss_e2e_smoke and not a2dpss_smoke:
        raise ValueError(
            "A2DPSS end-to-end mode requires A2DPSS dissemination mode"
        )
    if a2dpss_acsprop_smoke and a2dpss_e2e_smoke:
        raise ValueError(
            "A2DPSS ACSPROP-only and end-to-end modes are exclusive"
        )
    if a2dpss_e2e_smoke:
        enable_ledger = True
    elif a2dpss_smoke:
        enable_ledger = False
    if expected_committee_size is None:
        expected_committee_size = M
    if committee_m is None:
        committee_m = expected_committee_size
    if quorum_h is None:
        quorum_h = M - t
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
        raise ValueError("ACS agreement aliases do not match n_a/m_a")
    if 3 * t_a >= n_a or quorum_h != n_a - t_a:
        raise ValueError(
            "ACS paper parameters require t_a < n_a/3 and "
            "quorum_h = n_a-t_a"
        )
    if not n_a <= m_a <= M or 2 * quorum_h - m_a <= t_a:
        raise ValueError("ACS quorum intersection requires 2*h-m_a > t_a")
    if not 0 <= t_r or not 2 * t_r < n_r <= M:
        raise ValueError("ACS resharing parameters require t_r < n_r/2 <= M")
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
                "ACS coin parameters require "
                "0 <= coin_t < coin_k <= coin_c-coin_t and coin_c <= M"
            )
    elif coin_t is not None or coin_k is not None:
        raise ValueError("ACS coin_t/coin_k require coin_c")
    if (
        a2dpss_fixture is not None
        and acs_collect_threshold not in (None, quorum_h)
    ):
        raise ValueError(
            "paired ACS collect threshold must equal protocol quorum_h"
        )
    agreement_t_impl = m_a - quorum_h
    if agreement_t_impl < 0:
        raise ValueError("committee_m must be at least quorum_h")
    crypto = _load_crypto_from_config()
    a2dpss_state = None
    if a2dpss_smoke:
        a2dpss_state = _build_a2dpss_dissemination_state(
            extras=HbmpcConfig.extras,
            srs=crypto[4],
            my_id=my_id,
            M=M,
            service_t=t_r,
            fixture=a2dpss_fixture,
        )
        if (
            len(a2dpss_state["source_holder_ids"]) != n_r
            or len(a2dpss_state["successor_node_ids"]) != n_r
            or a2dpss_state["context"].service_t != t_r
        ):
            raise ValueError(
                "resolved DPSS context does not match n_r/t_r parameters"
            )
        logging.getLogger(f"async_yoso_tob.{my_id}").info(
            "[PHASE-4:A2DPSS] dissemination-only mode qid=%s holder=%s",
            a2dpss_state["context"].encode_qid(),
            a2dpss_state["is_holder"],
        )
    vrf_sk = HbmpcConfig.extras.get("vrf_sk")
    vrf_pks = HbmpcConfig.extras.get("vrf_pks")
    if vrf_sk is None or vrf_pks is None:
        raise ValueError("Generate node VRF keys with scripts.run_key_gen_atcaas before running")
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
        if committee_selection_mode == FIXED_SIZE_COMMITTEE_MODE:
            wrong_sizes = {
                index: len(roles)
                for index, roles in (
                    role_trace_artifact.committees.roles_by_index.items()
                )
                if len(roles) != expected_committee_size
            }
            if wrong_sizes:
                raise ValueError(
                    "fixed-size role trace committees must all contain "
                    f"exactly n_a={expected_committee_size} roles: "
                    f"{wrong_sizes}"
                )
    replay_trace = (
        ProtocolReplayTrace.from_file(replay_trace_path)
        if replay_trace_path
        else None
    )
    effective_coin_mode, coin_mode_forced_by_replay = _coin_mode(
        coin_mode_requested,
        replay_enabled=replay_trace is not None,
    )
    coin_silent_nodes = frozenset(coin_silent_nodes)
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
    ideal_coin_oracle = (
        SharedIdealCoinOracle(replay_trace.coin)
        if replay_trace is not None
        else DeterministicCoinOracle(epoch)
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
    if role_trace_artifact is not None:
        committee_trace = role_trace_artifact.committees
        role_trace_source = "recorded"
        role_trace_format = "role-only"
        role_trace_digest = role_trace_artifact.digest()
    elif replay_trace is not None:
        committee_trace = replay_trace.committees
        legacy_roles = CommitteeRoleTraceArtifact(
            replay_trace.committees.roles_by_index
        )
        legacy_roles.validate_population(M)
        role_trace_source = "recorded"
        role_trace_format = "legacy-protocol-replay"
        role_trace_digest = legacy_roles.digest()
    else:
        committee_trace = None
        role_trace_source = "none"
        role_trace_format = None
        role_trace_digest = None
    workload = None
    workload_source = None
    lifecycle = None
    if multi_qid_smoke:
        if a2dpss_pipeline:
            fixture_path = os.environ.get("ATCAAS_A2DPSS_PIPELINE_FIXTURE")
            if not fixture_path:
                raise ValueError("ATCAAS_A2DPSS_PIPELINE_FIXTURE is required")
            workload = A2DPSSPairedWorkload.from_file(fixture_path)
            workload_source = "recorded-a2dpss-pipeline-fixture"
        elif ocdpss_paired:
            fixture_path = os.environ.get("ATCAAS_OCDPSS_FIXTURE")
            if not fixture_path:
                raise ValueError("ATCAAS_OCDPSS_FIXTURE is required in paired mode")
            workload = OCDPSSPairedWorkload.from_file(fixture_path)
            workload_source = "recorded-ocdpss-fixture"
        else:
            workload, workload_source = load_dynamic_workload(
                os.environ.get("ATCAAS_WORKLOAD_TRACE"),
                messages_per_node=messages_per_node,
            )
        if workload.messages_per_node != messages_per_node:
            raise ValueError(
                "workload trace messages_per_node does not match "
                "ASYNC_YOSO_TOB_MESSAGES_PER_NODE"
            )
        lifecycle = QueryLifecycleMetrics(workload)

    if a2dpss_smoke:
        payload_qid = lambda payload: (
            a2dpss_state["adapter"].qid_for_payload(payload).encode()
        )
    else:
        payload_qid = (
            workload.qid_for_payload
            if ocdpss_paired or a2dpss_pipeline
            else _payload_qid
        )

    tob = AsyncYosoTOB(
        my_id=my_id,
        M=M,
        epoch=epoch,
        logger=logging.getLogger(f"async_yoso_tob.{my_id}"),
    )
    paired_ocdpss_justifier = None
    paired_ocdpss_finalizer = None
    paired_ocdpss_reconstruction = None
    paired_ocdpss_events = []
    paired_public_finalize_seconds = 0.0
    paired_public_finalize_invocations = 0
    paired_reconstruction_seconds = 0.0
    paired_reconstruction_invocations = 0
    a2dpss_events = []
    a2dpss_candidate_refs = []
    a2dpss_predicate_records = []
    a2dpss_predicate_result_cache = {}
    a2dpss_acsprop_core = None
    a2dpss_pending_records_at_ready = []
    a2dpss_s_out_refs = []
    a2dpss_s_out_entries = []
    a2dpss_selected_T = []
    a2dpss_public_result = None
    a2dpss_private_result = None
    a2dpss_public_finalize_seconds = 0.0
    a2dpss_reconstruction_seconds = 0.0
    a2dpss_public_finalizer = None
    a2dpss_reconstruction = None
    a2dpss_handover_metrics = None
    pipeline_origin = time.perf_counter()
    pipeline_events = []
    pipeline_gates = []
    pipeline_cpu_stages = {
        stage: {"invocations": 0, "cpu_seconds": 0.0}
        for stage in (
            "ready_check", "candidate_formation", "formal_predicate",
            "public_finalization", "successor_reconstruction",
        )
    }
    pipeline_justifier = None
    pipeline_adapter = None
    pipeline_finalizer = None
    pipeline_reconstruction = None

    def _event_clock():
        return (
            time.perf_counter() - pipeline_origin
            if a2dpss_pipeline else time.time()
        )

    def _pipeline_cpu_call(stage, callback, *args, **kwargs):
        started = time.process_time()
        try:
            return callback(*args, **kwargs)
        finally:
            metric = pipeline_cpu_stages[stage]
            metric["invocations"] += 1
            metric["cpu_seconds"] += time.process_time() - started

    registry = JustifierRegistry()
    if a2dpss_pipeline:
        contexts = workload.registrations()
        pipeline_justifier = A2DPSSJustifierManager(contexts)
        pipeline_adapter = A2DPSSACSAdapter(pipeline_justifier)
        registry.register("a2dpss", pipeline_justifier)
        pipeline_finalizer = A2DPSSPublicFinalizer(pipeline_justifier)
        pipeline_reconstruction = A2DPSSSuccessorReconstructionManager()
        first_context = contexts[0]
        local = build_handover_local_config(
            HbmpcConfig.extras,
            g_hex=first_context.g_hex,
            h_hex=first_context.h_hex,
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
                    local.my_successor_rsk
                    if my_id in context.successor_role_indices else None
                ),
            )
        local = None
    elif a2dpss_smoke:
        registry.register("a2dpss", a2dpss_state["justifier"])
        a2dpss_handover_metrics = A2DPSSHandoverMetrics(
            protocol="acs-tob",
            node=my_id,
            qid=a2dpss_state["context"].encode_qid(),
            is_holder=a2dpss_state["is_holder"],
            is_successor=(
                a2dpss_state["my_successor_role_idx"] is not None
            ),
        )
        if a2dpss_e2e_smoke:
            a2dpss_public_finalizer = A2DPSSPublicFinalizer(
                a2dpss_state["justifier"]
            )
            a2dpss_reconstruction = (
                A2DPSSSuccessorReconstructionManager()
            )
            a2dpss_reconstruction.register_context(
                a2dpss_state["context"],
                my_node_id=my_id,
                my_rsk=a2dpss_state["my_successor_rsk"],
            )
    elif ocdpss_paired:
        paired_registrations = workload.registrations()
        paired_ocdpss_justifier = OCDPSSJustifierManager(paired_registrations)
        registry.register("ocdpss", paired_ocdpss_justifier)
        paired_ocdpss_finalizer = OCDPSSPublicFinalizer(
            paired_ocdpss_justifier
        )
        paired_ocdpss_reconstruction = OCDPSSSuccessorReconstructionManager()
        first_context = paired_registrations[0][0]
        paired_local = build_handover_local_config(
            HbmpcConfig.extras,
            g_hex=first_context.g_hex,
            h_hex=first_context.h_hex,
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
    acs_wait_predicate = TypedWaitPredicate(registry)

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
            base_coin_oracle = threshold_coin_oracle
        else:
            base_coin_oracle = ideal_coin_oracle
        coin_oracle = ObservedCoinOracle(
            base_coin_oracle,
            timeout=coin_timeout,
            logger=logging.getLogger(f"common_coin.{my_id}"),
        )
        manager = AsyncYosoMessageRBCManager(
            tob=tob,
            epoch=epoch,
            my_id=my_id,
            M=M,
            t=agreement_t_impl,
            send=send,
            shared_subscribe=shared,
            crypto=crypto,
            vrf_sk=vrf_sk,
            vrf_pks=vrf_pks,
            threshold_fraction=expected_committee_size / M,
            expected_committee_size=expected_committee_size,
            committee_m=committee_m,
            quorum_h=quorum_h,
            require_vrf_proofs=(acs_activation_mode == "vrf"),
            logger=logging.getLogger(f"async_yoso_msg_rbc.{my_id}"),
        )
        tob.set_message_manager(manager)
        summary = None
        acs = None
        acs_output = None
        delivered_event = None
        acs_outputs = []
        delivered_events = []
        query_delivery_events = []
        injected_qids = []
        broadcast_start_time = None
        broadcast_end_time = None
        pending_ready_time = None
        first_deliver_time = None
        acs_start_times = []
        acs_end_times = []
        qid_count = len(workload.qids()) if workload is not None else 1
        if a2dpss_smoke:
            expected_pending_total = (
                a2dpss_state["context"].service_t + 1
            )
        else:
            expected_pending_total = (
                sum(
                    workload.expected_entry_count(instance)
                    for batch in workload.batches
                    for instance in batch.instances
                )
                if ocdpss_paired or a2dpss_pipeline
                else M * messages_per_node * qid_count
            )

        def _build_summary(cleanup_completed=None):
            acs_threshold_fraction = None
            if acs is not None:
                acs_threshold_fraction = acs.threshold_fraction
            a2dpss_computation_metrics = None
            if a2dpss_smoke:
                validation = a2dpss_state[
                    "justifier"
                ].validation_metrics_snapshot()
                adapter_metrics = a2dpss_state[
                    "adapter"
                ].metrics_snapshot()
                finalization = (
                    a2dpss_public_finalizer.metrics_snapshot()
                    if a2dpss_public_finalizer is not None
                    else {}
                )
                reconstruction = (
                    a2dpss_reconstruction.metrics_snapshot()
                    if a2dpss_reconstruction is not None
                    else {}
                )
                holder_events = [
                    event
                    for event in a2dpss_events
                    if event.get("event") == "holder_broadcast"
                ]
                receiver_cpu = sum(
                    float(record["cpu_seconds"])
                    for record in a2dpss_predicate_records
                )
                finalizer_total = finalization.get(
                    "total",
                    {"invocations": 0, "cpu_seconds": 0.0},
                )
                preparation = adapter_metrics["delivery_preparation"]
                reconstruction_total = reconstruction.get(
                    "total",
                    {"invocations": 0, "cpu_seconds": 0.0},
                )
                a2dpss_computation_metrics = (
                    build_a2dpss_computation_metrics(
                        protocol="acs-tob",
                        node=my_id,
                        proposer_predicate={
                            "operation": "structural_candidate_only",
                            "invocations": 0,
                            "cpu_seconds": 0.0,
                            "records": [],
                        },
                        receiver_predicate={
                            "operation": "acsprop_receiver_admission",
                            "invocations": len(
                                a2dpss_predicate_records
                            ),
                            "cpu_seconds": receiver_cpu,
                            "records": list(
                                a2dpss_predicate_records
                            ),
                        },
                        candidate_formation=adapter_metrics[
                            "candidate_formation"
                        ],
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
                            "invocations": finalizer_total[
                                "invocations"
                            ],
                            "cpu_seconds": (
                                float(finalizer_total["cpu_seconds"])
                                + float(preparation["cpu_seconds"])
                            ),
                            "detail": {
                                "delivery_preparation": preparation,
                                "finalizer": finalization,
                            },
                        },
                        successor_reconstruction={
                            **reconstruction_total,
                            "detail": reconstruction,
                        },
                    )
                )
            return {
                "protocol": "async-yoso-tob",
                "phase": (
                    "P7B-a2dpss-vrf-network-e2e"
                    if a2dpss_e2e_smoke and acs_activation_mode == "vrf"
                    else "P6-a2dpss-network-e2e"
                    if a2dpss_e2e_smoke
                    else "P5-a2dpss-acsprop"
                    if a2dpss_acsprop_smoke
                    else "P4-a2dpss-message-rbc"
                    if a2dpss_smoke
                    else "B5-vrf-activation"
                    if enable_ledger and acs_activation_mode == "vrf"
                    else "B4-multi-qid-ledger-driver"
                    if enable_ledger and multi_qid_smoke
                    else "B3-ledger-driver"
                    if enable_ledger
                    else "B1-message-rbc"
                ),
                "phase_detail": (
                    "P7B-a2dpss-vrf-quorum-unique-coin"
                    if a2dpss_e2e_smoke and acs_activation_mode == "vrf"
                    else "P6-a2dpss-acs-union-finalize-reconstruct"
                    if a2dpss_e2e_smoke
                    else "P5-a2dpss-predicate-gated-acsprop"
                    if a2dpss_acsprop_smoke
                    else "P4-a2dpss-dissemination-only"
                    if a2dpss_smoke
                    else "B4-multi-qid-ledger-driver"
                    if enable_ledger and multi_qid_smoke
                    else "B3-ledger-driver"
                    if enable_ledger
                    else "B1-message-rbc"
                ),
                "node": my_id,
                "r6_metrics_schema": 1,
                "agreement_parameters_schema": 1,
                "M": M,
                "t": t,
                "expected_committee_size": expected_committee_size,
                "committee_m": committee_m,
                "quorum_h": quorum_h,
                "committee_selection_mode": committee_selection_mode,
                "fixed_committee_seed": (
                    fixed_committee_seed
                    if committee_selection_mode == FIXED_SIZE_COMMITTEE_MODE
                    else None
                ),
                "n_r": n_r,
                "t_r": t_r,
                "n_a": n_a,
                "t_a": t_a,
                "m_a": m_a,
                "agreement_t": t_a,
                "agreement_t_impl": agreement_t_impl,
                "coin_c": coin_c,
                "coin_t": coin_t,
                "coin_k": coin_k,
                "coverage_threshold": t_a + 1,
                "dag_selectblock_metrics": {
                    "schema": "dag-selectblock-metrics/v1",
                    "applicable": False,
                    "reason": "ACS-TOB does not use DAG SelectBlock",
                    "agreement_t": t_a,
                    "coverage_threshold": t_a + 1,
                },
                "message_rbc_parameters": manager.parameters_snapshot(),
                "epoch": epoch,
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
                "coin_observations": coin_oracle.snapshot(),
                "coin_metrics": (
                    threshold_coin_oracle.snapshot()
                    if threshold_coin_oracle is not None
                    else None
                ),
                "protocol_trace_source": (
                    role_trace_source
                ),
                "protocol_trace_digest": (
                    role_trace_digest
                ),
                "role_trace_source": role_trace_source,
                "role_trace_format": role_trace_format,
                "role_trace_digest": role_trace_digest,
                "ideal_coin_trace_source": (
                    "recorded" if replay_trace is not None else "deterministic"
                ),
                "ideal_coin_trace_digest": (
                    replay_trace.digest() if replay_trace is not None else None
                ),
                "messages_per_node": messages_per_node,
                "expected_pending": expected_pending_total,
                "deliver_count": len(tob.ledger),
                "ledger": tob.snapshot_ledger(),
                "ledger_refs": tob.snapshot_ledger_refs(),
                "pending_count": len(tob.pending),
                "pending": tob.snapshot_pending(),
                "message_rbc_deliveries": len(manager.delivered_messages),
                "broadcast_start_time": broadcast_start_time,
                "broadcast_end_time": broadcast_end_time,
                "pending_ready_time": pending_ready_time,
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
                "pending_wait_latency_sec": (
                    pending_ready_time - broadcast_start_time
                    if pending_ready_time is not None
                    and broadcast_start_time is not None
                    else None
                ),
                "acs_start_times": acs_start_times,
                "acs_end_times": acs_end_times,
                "active_rbc_instances": manager.active_count(),
                "enable_ledger": enable_ledger,
                "payload_schedule": (
                    "a2dpss-pipeline"
                    if a2dpss_pipeline
                    else "a2dpss"
                    if a2dpss_smoke
                    else "ocdpss-paired"
                    if ocdpss_paired
                    else "multi-qid-smoke"
                    if multi_qid_smoke
                    else "single-demo"
                ),
                "justifier_kind": (
                    "a2dpss"
                    if a2dpss_smoke or a2dpss_pipeline
                    else "ocdpss"
                    if ocdpss_paired
                    else "demo"
                ),
                "injected_qids": injected_qids,
                "workload_trace_source": workload_source,
                "workload_trace": workload.to_dict() if workload is not None else None,
                "ocdpss_paired": ocdpss_paired,
                "a2dpss_pipeline": a2dpss_pipeline,
                "a2dpss_pipeline_clock": (
                    "perf_counter-relative" if a2dpss_pipeline else None
                ),
                "a2dpss_pipeline_fixture_digest": (
                    workload.digest() if a2dpss_pipeline else None
                ),
                "a2dpss_pipeline_events": pipeline_events,
                "a2dpss_pipeline_gates": pipeline_gates,
                "a2dpss_pipeline_cpu": (
                    {
                        "clock": "process_time",
                        "stages": pipeline_cpu_stages,
                        "total_cpu_seconds": sum(
                            item["cpu_seconds"]
                            for item in pipeline_cpu_stages.values()
                        ),
                    }
                    if a2dpss_pipeline else None
                ),
                "a2dpss_pipeline_validation": (
                    pipeline_justifier.validation_metrics_snapshot()
                    if a2dpss_pipeline else None
                ),
                "a2dpss_pipeline_predicate_records": (
                    list(a2dpss_predicate_records) if a2dpss_pipeline else []
                ),
                "query_lifecycle_clock": (
                    "local-relative-perf_counter" if a2dpss_pipeline else "time.time"
                ),
                "a2dpss_smoke": a2dpss_smoke,
                "a2dpss_acsprop_smoke": a2dpss_acsprop_smoke,
                "a2dpss_e2e_smoke": a2dpss_e2e_smoke,
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
                "a2dpss_fixture_contribution_digests": (
                    list(a2dpss_fixture.contribution_digests())
                    if a2dpss_fixture is not None
                    else []
                ),
                "a2dpss_observed_contribution_digests": (
                    sorted(
                        record["payload_digest"]
                        for record in (
                            a2dpss_pending_records_at_ready
                            or _a2dpss_pending_snapshot(
                                tob,
                                a2dpss_state["adapter"],
                                a2dpss_state["context"],
                            )
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
                "a2dpss_qid": (
                    a2dpss_state["context"].encode_qid()
                    if a2dpss_smoke
                    else None
                ),
                "a2dpss_context_digest": (
                    a2dpss_state["context"].context_digest
                    if a2dpss_smoke
                    else None
                ),
                "a2dpss_source_holder_ids": (
                    list(a2dpss_state["source_holder_ids"])
                    if a2dpss_smoke
                    else []
                ),
                "a2dpss_successor_node_ids": (
                    list(a2dpss_state["successor_node_ids"])
                    if a2dpss_smoke
                    else []
                ),
                "a2dpss_candidate_ready": bool(
                    a2dpss_candidate_refs
                ),
                "a2dpss_candidate_refs": [
                    ref.encode() for ref in a2dpss_candidate_refs
                ],
                "a2dpss_pending_records": (
                    a2dpss_pending_records_at_ready
                    or _a2dpss_pending_snapshot(
                            tob,
                            a2dpss_state["adapter"],
                            a2dpss_state["context"],
                        )
                    if a2dpss_smoke
                    else []
                ),
                "a2dpss_events": a2dpss_events,
                "a2dpss_validation_metrics": (
                    a2dpss_state[
                        "justifier"
                    ].validation_metrics_snapshot()
                    if a2dpss_smoke
                    else None
                ),
                "a2dpss_acs_predicate_metrics": (
                    {
                        "invocations": len(a2dpss_predicate_records),
                        "cpu_seconds": sum(
                            record["cpu_seconds"]
                            for record in a2dpss_predicate_records
                        ),
                        "records": list(a2dpss_predicate_records),
                    }
                    if a2dpss_smoke
                    else None
                ),
                "a2dpss_acsprop_core": (
                    {
                        "digest": a2dpss_acsprop_core.digest(),
                        "proposers": [
                            proposal.proposer
                            for proposal in a2dpss_acsprop_core.proposals
                        ],
                        "refs": [
                            ref.encode()
                            for ref in a2dpss_acsprop_core.flatten_refs()
                        ],
                    }
                    if a2dpss_acsprop_core is not None
                    else None
                ),
                "a2dpss_s_out_refs": [
                    ref.encode() for ref in a2dpss_s_out_refs
                ],
                "a2dpss_s_out": _a2dpss_entry_set_snapshot(
                    a2dpss_s_out_entries
                ),
                "a2dpss_selected_T": _a2dpss_entry_set_snapshot(
                    a2dpss_selected_T
                ),
                "a2dpss_public_result": (
                    a2dpss_public_result.to_obj()
                    if a2dpss_public_result is not None
                    else None
                ),
                "a2dpss_private_reconstruction": (
                    {
                        "completed": True,
                        "my_node_id": a2dpss_private_result.my_node_id,
                        "my_role_idx": a2dpss_private_result.my_role_idx,
                        "accepted_eval_points": list(
                            a2dpss_private_result.accepted_eval_points
                        ),
                        "accepted_contribution_digests": list(
                            a2dpss_private_result
                            .accepted_contribution_digests
                        ),
                    }
                    if a2dpss_private_result is not None
                    else {
                        "completed": False,
                        "my_node_id": my_id,
                        "my_role_idx": a2dpss_state[
                            "my_successor_role_idx"
                        ] if a2dpss_smoke else None,
                    }
                ),
                "a2dpss_phase_metrics": (
                    {
                        "public_finalize": {
                            "elapsed_sec": a2dpss_public_finalize_seconds,
                            "detail": (
                                a2dpss_public_finalizer.metrics_snapshot()
                            ),
                        },
                        "successor_reconstruction": {
                            "elapsed_sec": a2dpss_reconstruction_seconds,
                            "detail": a2dpss_reconstruction.metrics_snapshot(),
                        },
                    }
                    if a2dpss_e2e_smoke
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
                "registered_a2dpss_reconstruction_contexts": (
                    len(a2dpss_reconstruction.registrations)
                    if a2dpss_reconstruction is not None
                    else 0
                ),
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
                "query_lifecycle": lifecycle.summary() if lifecycle is not None else None,
                "query_delivery_events": query_delivery_events,
                "query_delivery_count": len(query_delivery_events),
                "justifier_metrics": (
                    {
                        "applicable": True,
                        "source": "acs-receiver-admission",
                        "select": {
                            "invocations": 0,
                            "elapsed_sec": 0.0,
                            "cpu_seconds": 0.0,
                            "by_qid": {},
                            "records": [],
                        },
                        "verify": {
                            "invocations": len(
                                a2dpss_predicate_records
                            ),
                            "elapsed_sec": None,
                            "cpu_seconds": sum(
                                float(record["cpu_seconds"])
                                for record in a2dpss_predicate_records
                            ),
                            "by_qid": {},
                            "records": list(
                                a2dpss_predicate_records
                            ),
                        },
                    }
                    if a2dpss_smoke
                    else {
                        "applicable": True,
                        **acs_wait_predicate.registry.metrics_snapshot(),
                    }
                ),
                "acs": acs_output.to_summary() if acs_output is not None else None,
                "acs_outputs": [
                    output.to_summary() for output in acs_outputs
                ],
                "acs_rbc_instance_count": (
                    acs.rbc_instance_count if acs is not None else 0
                ),
                "acs_activation_mode": acs_activation_mode,
                "acs_collect_threshold": (
                    acs.collect_threshold if acs is not None else acs_collect_threshold
                ),
                "acs_protocol_quorum": (
                    acs.protocol_quorum if acs is not None else None
                ),
                "acs_coin_indices": (
                    acs.coin_indices.snapshot()
                    if acs is not None
                    else None
                ),
                "acs_stage_timeout": (
                    acs.stage_timeout if acs is not None else acs_stage_timeout
                ),
                "acs_threshold_fraction": acs_threshold_fraction,
                "acs_activation_send_count": (
                    acs.activation_send_count if acs is not None else 0
                ),
                "acs_stage_collect_counts": (
                    acs.stage_collect_counts if acs is not None else []
                ),
                "acs_proposal_metrics": (
                    acs.proposal_metrics_snapshot()
                    if acs is not None
                    else None
                ),
                "acs_select_block_input_metrics": (
                    acs.select_block_input_metrics_snapshot()
                    if acs is not None
                    else None
                ),
                "acs_echo_metrics": (
                    acs.echo_metrics_snapshot()
                    if acs is not None
                    else None
                ),
                "acs_echo_adoption_wait": acs_echo_adoption_wait,
                "acs_output_service_grace": acs_output_service_grace,
                "delivered_event": delivered_event,
                "delivered_events": delivered_events,
                "cleanup_completed": cleanup_completed,
                "bytes_sent": runner.node_communicator.bytes_sent,
                "communication_bits": int(runner.node_communicator.bytes_sent) * 8,
                "excluded_costs": (
                    ["role_assignment"]
                    if effective_coin_mode == "threshold-bls"
                    else ["ideal_common_coin", "role_assignment"]
                ),
            }

        try:
            manager.ensure_counter_window(messages_per_node * qid_count)
            while time.time() < start_time:
                await asyncio.sleep(0.05)

            if experiment_barrier is not None:
                await experiment_barrier(runner, "ready")
            start_run = time.time()
            if a2dpss_pipeline:
                pipeline_origin = time.perf_counter()
            broadcast_start_time = start_run
            if a2dpss_smoke:
                a2dpss_handover_metrics.start()
                context = a2dpss_state["context"]
                holder_state = a2dpss_state["holder_state"]
                initial_qids = [context.encode_qid()]
                later_qids = []
                fixture_entry = a2dpss_state["fixture_entry"]
                if fixture_entry is not None:
                    started = time.perf_counter()
                    await tob.t_broadcast(
                        fixture_entry.mid,
                        fixture_entry.payload,
                    )
                    broadcast_seconds = time.perf_counter() - started
                    a2dpss_handover_metrics.mark_contribution_sent()
                    a2dpss_events.append(
                        {
                            "event": "holder_broadcast",
                            "qid": context.encode_qid(),
                            "mid": fixture_entry.mid,
                            "holder_id": my_id,
                            "holder_state_source": "paired-fixture",
                            "fixture_replay": True,
                            "contribution_digest": (
                                a2dpss_contribution_digest(
                                    fixture_entry.payload
                                )
                            ),
                            "contribution_bytes": len(fixture_entry.payload),
                            "pvss_seconds": 0.0,
                            "pvss_cpu_seconds": 0.0,
                            "serialization_seconds": 0.0,
                            "broadcast_seconds": broadcast_seconds,
                            "fixture_holder_metrics": dict(
                                fixture_entry.holder_metrics
                            ),
                        }
                    )
                elif holder_state is not None:
                    share_s = holder_state.share_s
                    share_s_hat = holder_state.share_s_hat

                    async def _broadcast_a2dpss(
                        qid,
                        mid,
                        contribution_bytes,
                    ):
                        if qid != context.encode_qid():
                            raise ValueError(
                                "holder emitted an unexpected A2DPSS qid"
                            )
                        if mid != f"node_{my_id}":
                            raise ValueError(
                                "holder emitted an unexpected A2DPSS mid"
                            )
                        await tob.t_broadcast(
                            mid,
                            contribution_bytes,
                        )

                    holder_result = await run_a2dpss_handover_role(
                        context=context,
                        holder_id=my_id,
                        share_s=share_s,
                        share_s_hat=share_s_hat,
                        broadcast_contribution=_broadcast_a2dpss,
                        holder_state=holder_state,
                    )
                    a2dpss_handover_metrics.mark_contribution_sent()
                    del share_s, share_s_hat
                    a2dpss_events.append(
                        {
                            "event": "holder_broadcast",
                            "qid": holder_result.qid,
                            "mid": holder_result.mid,
                            "holder_id": my_id,
                            "contribution_digest": (
                                a2dpss_contribution_digest(
                                    holder_result.contribution_bytes
                                )
                            ),
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
                else:
                    a2dpss_events.append(
                        {
                            "event": "non_holder",
                            "qid": context.encode_qid(),
                            "node": my_id,
                        }
                    )
            elif multi_qid_smoke:
                initial_batches = workload.batches_for_trigger(START)
                initial_qids = [
                    workload.qid(instance)
                    for batch in initial_batches
                    for instance in batch.instances
                ]
                later_batches = workload.batches_for_trigger(
                    INITIAL_PROPOSAL_FIXED
                )
                later_qids = [
                    workload.qid(instance)
                    for batch in later_batches
                    for instance in batch.instances
                ]
                await _inject_workload_batches(
                    tob, workload, lifecycle, START, my_id, injected_qids,
                    clock=_event_clock,
                )
                logging.info("[WORKLOAD] trigger=start injected initial batch")
            else:
                initial_qids = later_qids = []
                for index in range(messages_per_node):
                    await tob.t_broadcast(
                        f"mid-{my_id}-{index}",
                        _demo_payload(epoch, my_id, index),
                    )
            broadcast_end_time = time.time()

            if a2dpss_smoke:
                a2dpss_candidate_refs = (
                    await _wait_for_a2dpss_candidate(
                        tob=tob,
                        adapter=a2dpss_state["adapter"],
                        start_run=start_run,
                        max_runtime=max_runtime,
                    )
                )
                if a2dpss_fixture is not None:
                    a2dpss_candidate_refs = await _wait_for_refs(
                        tob=tob,
                        qids=[a2dpss_state["context"].encode_qid()],
                        expected=len(a2dpss_fixture.entries),
                        start_run=start_run,
                        max_runtime=max_runtime,
                        label="complete paired A2DPSS fixture",
                        qid_resolver=payload_qid,
                    )
                a2dpss_pending_records_at_ready = (
                    _a2dpss_pending_snapshot(
                        tob,
                        a2dpss_state["adapter"],
                        a2dpss_state["context"],
                    )
                )
                a2dpss_handover_metrics.mark_pool_ready()
                injected_qids.append(
                    {
                        "round": 1,
                        "qid": a2dpss_state[
                            "context"
                        ].encode_qid(),
                    }
                )
                a2dpss_events.append(
                    {
                        "event": "candidate_ready",
                        "qid": a2dpss_state[
                            "context"
                        ].encode_qid(),
                        "pending_count": len(tob.pending),
                        "candidate_refs": [
                            ref.encode()
                            for ref in a2dpss_candidate_refs
                        ],
                    }
                )
            elif multi_qid_smoke:
                initial_gate_enter = _event_clock()
                await _wait_for_workload_pool_ready(
                    tob=tob,
                    workload=workload,
                    lifecycle=lifecycle,
                    instances=[
                        instance
                        for batch in initial_batches
                        for instance in batch.instances
                    ],
                    wait_predicate=acs_wait_predicate,
                    qid_resolver=payload_qid,
                    start_run=start_run,
                    max_runtime=max_runtime,
                    clock=_event_clock,
                    ready_cpu=(
                        pipeline_cpu_stages["ready_check"]
                        if a2dpss_pipeline else None
                    ),
                )
                if a2dpss_pipeline:
                    initial_gate_exit = _event_clock()
                    pipeline_gates.append({
                        "label": "initial_pool_ready",
                        "enter": initial_gate_enter,
                        "exit": initial_gate_exit,
                        "duration": initial_gate_exit - initial_gate_enter,
                    })
            else:
                while len(tob.pending) < expected_pending_total:
                    if time.time() - start_run > max_runtime:
                        raise RuntimeError(
                            f"max runtime exceeded: pending={len(tob.pending)} "
                            f"expected={expected_pending_total} "
                            f"active_rbc={manager.active_count()}"
                    )
                    await asyncio.sleep(0.05)
            pending_ready_time = time.time()

            if a2dpss_acsprop_smoke or a2dpss_e2e_smoke:
                acs = NetworkAsyncYosoACS(
                    my_id=my_id,
                    M=M,
                    t=agreement_t_impl,
                    epoch=epoch,
                    send=send,
                    shared_subscribe=shared,
                    crypto=crypto,
                    vrf_sk=vrf_sk,
                    vrf_pks=vrf_pks,
                    coin_oracle=coin_oracle,
                    committee_trace=committee_trace,
                    collect_all=acs_collect_all,
                    activation_mode=acs_activation_mode,
                    expected_committee_size=expected_committee_size,
                    committee_m=committee_m,
                    quorum_h=quorum_h,
                    collect_threshold=acs_collect_threshold,
                    stage_timeout=acs_stage_timeout,
                    echo_adoption_wait=acs_echo_adoption_wait,
                    output_service_grace=acs_output_service_grace,
                    committee_selection_mode=committee_selection_mode,
                    fixed_committee_seed=fixed_committee_seed,
                    logger=logging.getLogger(
                        f"network_async_yoso_acs.{my_id}"
                    ),
                )

                a2dpss_acs_session_id = (
                    "a2dpss-e2e-1"
                    if a2dpss_e2e_smoke
                    else "a2dpss-acsprop-1"
                )

                def _validate_a2dpss_proposal(proposer, proposal_refs):
                    return _validate_a2dpss_proposal_once(
                        node_id=my_id,
                        session_id=a2dpss_acs_session_id,
                        proposer=proposer,
                        proposal_refs=proposal_refs,
                        adapter=a2dpss_state["adapter"],
                        pending=tob.pending,
                        records=a2dpss_predicate_records,
                        result_cache=a2dpss_predicate_result_cache,
                    )

                acs_started = time.time()
                acs_start_times.append(acs_started)
                remaining = max(
                    0.1,
                    max_runtime - (acs_started - start_run),
                )
                if a2dpss_e2e_smoke:
                    async def _complete_a2dpss_output(logical_output):
                        nonlocal acs_output
                        nonlocal a2dpss_acsprop_core
                        nonlocal a2dpss_s_out_refs
                        nonlocal a2dpss_s_out_entries
                        nonlocal a2dpss_selected_T
                        nonlocal a2dpss_public_result
                        nonlocal a2dpss_private_result
                        nonlocal a2dpss_public_finalize_seconds
                        nonlocal a2dpss_reconstruction_seconds
                        nonlocal first_deliver_time
                        nonlocal delivered_event

                        acs_output = logical_output
                        a2dpss_handover_metrics.mark_ordered()
                        a2dpss_acsprop_core = (
                            logical_output.selected_core
                        )
                        loop = asyncio.get_running_loop()
                        # A selected core can originate at another observer.
                        # Usually SB_INPUT already caused this node to validate
                        # all of its proposer blocks.  If causal delivery brings
                        # a genuinely unseen block, validate that distinct block
                        # exactly once here before using the verified fast path.
                        missing_evidence = a2dpss_state[
                            "adapter"
                        ].unverified_core_proposals(
                            logical_output.selected_core,
                            tob.pending,
                            a2dpss_acs_session_id,
                        )
                        for proposal in missing_evidence:
                            admission = await loop.run_in_executor(
                                None,
                                functools.partial(
                                    _validate_a2dpss_proposal_once,
                                    node_id=my_id,
                                    session_id=a2dpss_acs_session_id,
                                    proposer=proposal.proposer,
                                    proposal_refs=proposal.refs,
                                    adapter=a2dpss_state["adapter"],
                                    pending=tob.pending,
                                    records=a2dpss_predicate_records,
                                    result_cache=(
                                        a2dpss_predicate_result_cache
                                    ),
                                ),
                            )
                            if not admission.accepted:
                                raise RuntimeError(
                                    "selected ACS core contains an invalid "
                                    "proposer block"
                                )
                        delivery_plan = await loop.run_in_executor(
                            None,
                            a2dpss_state[
                                "adapter"
                            ].prepare_verified_single_qid_delivery,
                            logical_output.selected_core,
                            tob.pending,
                            a2dpss_acs_session_id,
                        )
                        a2dpss_s_out_refs = list(delivery_plan.refs)
                        if a2dpss_s_out_refs != sorted(
                            set(logical_output.block_refs)
                        ):
                            raise RuntimeError(
                                "ACS output refs do not match selected core union"
                            )

                        expected_qid = delivery_plan.qid
                        if expected_qid != a2dpss_state[
                            "context"
                        ].build_qid():
                            raise RuntimeError(
                                "Phase 6 ACS output has an unexpected A2DPSS qid"
                            )
                        a2dpss_s_out_entries = list(delivery_plan.s_out)
                        a2dpss_selected_T = list(delivery_plan.selected_T)

                        started = time.perf_counter()
                        # Every proposer block in the selected core already
                        # passed the complete A²DPSS predicate at receiver
                        # admission.  The delivery plan reuses that exact,
                        # session-bound evidence and fixes canonical T without
                        # repeating PVSS/anchor checks.
                        a2dpss_public_result = await loop.run_in_executor(
                            None,
                            a2dpss_public_finalizer.on_verified_deliver,
                            expected_qid,
                            a2dpss_selected_T,
                        )
                        a2dpss_public_finalize_seconds += (
                            time.perf_counter() - started
                        )
                        if a2dpss_public_result is None:
                            raise RuntimeError(
                                "canonical A2DPSS T failed public finalization"
                            )
                        a2dpss_handover_metrics.mark_public_finalized()

                        started = time.perf_counter()
                        a2dpss_private_result = await loop.run_in_executor(
                            None,
                            a2dpss_reconstruction.reconstruct,
                            a2dpss_public_result,
                            a2dpss_selected_T,
                        )
                        a2dpss_reconstruction_seconds += (
                            time.perf_counter() - started
                        )
                        if a2dpss_private_result is not None:
                            a2dpss_handover_metrics.mark_successor_done()

                        ledger_index, block = tob.append_acs_block(
                            a2dpss_s_out_refs
                        )
                        first_deliver_time = time.time()
                        delivered_event = {
                            "ledger_index": ledger_index,
                            "label": "a2dpss-single-qid",
                            "qid": expected_qid.encode(),
                            "refs": [
                                ref.encode()
                                for ref in a2dpss_s_out_refs
                            ],
                            "block_size": len(block),
                            "s_out_size": len(a2dpss_s_out_entries),
                            "selected_T_size": len(a2dpss_selected_T),
                            "deliver_time": first_deliver_time,
                            "latency_since_broadcast_sec": (
                                first_deliver_time - broadcast_start_time
                            ),
                            "acs_latency_sec": (
                                first_deliver_time - acs_started
                            ),
                        }
                        delivered_events.append(delivered_event)
                        query_delivery_events.append(
                            {
                                "sequence": 1,
                                "qid": expected_qid.encode(),
                                "entry_count": len(a2dpss_s_out_entries),
                                "ledger_index": ledger_index,
                                "deliver_time": first_deliver_time,
                            }
                        )
                        a2dpss_events.extend(
                            [
                                {
                                    "event": "acs_output",
                                    "core_digest": (
                                        a2dpss_acsprop_core.digest()
                                    ),
                                    "output_round": (
                                        logical_output.output_round
                                    ),
                                    "refs": [
                                        ref.encode()
                                        for ref in a2dpss_s_out_refs
                                    ],
                                },
                                {
                                    "event": "public_finalized",
                                    "qid": a2dpss_public_result.qid,
                                    "context_digest": (
                                        a2dpss_public_result.context_digest
                                    ),
                                    "accepted_contribution_digests": list(
                                        a2dpss_public_result
                                        .accepted_contribution_digests
                                    ),
                                    "successor_commitment_vector": list(
                                        a2dpss_public_result
                                        .successor_commitment_vector
                                    ),
                                },
                                {
                                    "event": (
                                        "reconstructed"
                                        if a2dpss_private_result is not None
                                        else "non_successor_observer"
                                    ),
                                    "qid": a2dpss_public_result.qid,
                                    "my_node_id": my_id,
                                    "my_role_idx": a2dpss_state[
                                        "my_successor_role_idx"
                                    ],
                                },
                            ]
                        )

                    acs_output = await asyncio.wait_for(
                        acs.run(
                            a2dpss_candidate_refs,
                            session_id=a2dpss_acs_session_id,
                            max_rounds=acs_max_rounds,
                            proposal_validator=(
                                _validate_a2dpss_proposal
                            ),
                            output_fixed_cb=_complete_a2dpss_output,
                        ),
                        timeout=remaining,
                    )
                    acs_end_times.append(time.time())
                    acs_outputs.append(acs_output)
                else:
                    a2dpss_acsprop_core = await asyncio.wait_for(
                        acs.collect_proposals(
                            a2dpss_candidate_refs,
                            session_id=a2dpss_acs_session_id,
                            proposal_validator=(
                                _validate_a2dpss_proposal
                            ),
                        ),
                        timeout=remaining,
                    )
                    acs_end_times.append(time.time())
                    a2dpss_events.append(
                        {
                            "event": "acsprop_core_ready",
                            "core_digest": a2dpss_acsprop_core.digest(),
                            "proposers": [
                                proposal.proposer
                                for proposal in (
                                    a2dpss_acsprop_core.proposals
                                )
                            ],
                            "refs": [
                                ref.encode()
                                for ref in (
                                    a2dpss_acsprop_core.flatten_refs()
                                )
                            ],
                        }
                    )

            if enable_ledger and not a2dpss_e2e_smoke:
                acs = NetworkAsyncYosoACS(
                    my_id=my_id,
                    M=M,
                    t=agreement_t_impl,
                    epoch=epoch,
                    send=send,
                    shared_subscribe=shared,
                    crypto=crypto,
                    vrf_sk=vrf_sk,
                    vrf_pks=vrf_pks,
                    coin_oracle=coin_oracle,
                    committee_trace=committee_trace,
                    collect_all=acs_collect_all,
                    activation_mode=acs_activation_mode,
                    expected_committee_size=expected_committee_size,
                    committee_m=committee_m,
                    quorum_h=quorum_h,
                    collect_threshold=acs_collect_threshold,
                    stage_timeout=acs_stage_timeout,
                    echo_adoption_wait=acs_echo_adoption_wait,
                    output_service_grace=acs_output_service_grace,
                    committee_selection_mode=committee_selection_mode,
                    fixed_committee_seed=fixed_committee_seed,
                    logger=logging.getLogger(f"network_async_yoso_acs.{my_id}"),
                )

                async def _run_pipeline_acs(
                    candidate_refs, label, proposal_fixed_cb=None,
                ):
                    """Order independent A2DPSS qids with real ACSPROP admission.

                    The shared adapter already supports multi-qid proposal
                    validation.  Its single-qid delivery convenience method is
                    deliberately avoided: each verified union gets its own T,
                    public result, and successor reconstruction below.
                    """
                    nonlocal acs_output, delivered_event, first_deliver_time
                    session_id = f"ledger-{len(tob.ledger) + 1}"
                    candidate_refs = _pipeline_cpu_call(
                        "candidate_formation",
                        pipeline_adapter.build_structurally_ready_candidate_refs,
                        {ref: tob.pending[ref] for ref in candidate_refs},
                    )
                    if not candidate_refs:
                        raise RuntimeError("no ready A2DPSS pipeline candidate")

                    def _validate_pipeline_proposal(proposer, proposal_refs):
                        before = len(a2dpss_predicate_records)
                        try:
                            return _pipeline_cpu_call(
                                "formal_predicate",
                                _validate_a2dpss_proposal_once,
                                node_id=my_id,
                                session_id=session_id,
                                proposer=proposer,
                                proposal_refs=proposal_refs,
                                adapter=pipeline_adapter,
                                pending=tob.pending,
                                records=a2dpss_predicate_records,
                                result_cache=a2dpss_predicate_result_cache,
                            )
                        finally:
                            # Count actual per-qid formal checks, excluding
                            # cached admission and missing-transcript retries.
                            pipeline_cpu_stages["formal_predicate"]["invocations"] += (
                                len(a2dpss_predicate_records) - before - 1
                            )

                    acs_start_time = time.time()
                    acs_start_times.append(acs_start_time)

                    async def _complete_pipeline_output(output):
                        nonlocal acs_output, delivered_event, first_deliver_time
                        acs_output = output
                        # A selected remote core may contain proposer blocks
                        # not previously observed here.  Verify only those
                        # distinct blocks before consuming the trusted path.
                        while tob.missing_refs(output.block_refs):
                            if time.time() - start_run > max_runtime:
                                raise RuntimeError("A2DPSS pipeline output refs timed out")
                            await asyncio.sleep(0.05)
                        for proposal in pipeline_adapter.unverified_core_proposals(
                            output.selected_core, tob.pending, session_id
                        ):
                            admission = _validate_pipeline_proposal(
                                proposal.proposer, proposal.refs
                            )
                            if not admission.accepted:
                                raise RuntimeError("invalid A2DPSS pipeline ACS core")

                        def _prepare_verified_output():
                            refs = pipeline_adapter.union_core_refs(output.selected_core)
                            groups = pipeline_adapter.group_entries_by_qid(
                                refs, tob.pending
                            )
                            selected = {}
                            for qid, entries in groups.items():
                                threshold_entries = pipeline_adapter.select_verified_delivery_T(
                                    qid, entries
                                )
                                if threshold_entries is None:
                                    raise RuntimeError("verified A2DPSS union has no threshold T")
                                selected[qid.encode()] = threshold_entries
                            return refs, selected

                        refs, selected_by_qid = _pipeline_cpu_call(
                            "public_finalization", _prepare_verified_output
                        )
                        ledger_index, block = tob.append_acs_block(refs)
                        ordered_at = _event_clock()
                        delivered_wall = time.time()
                        if first_deliver_time is None:
                            first_deliver_time = delivered_wall
                        delivered_event = {
                            "ledger_index": ledger_index,
                            "label": label,
                            "refs": [ref.encode() for ref in refs],
                            "block_size": len(block),
                            "deliver_time": delivered_wall,
                            "t_ordered": ordered_at,
                            "acs_latency_sec": delivered_wall - acs_start_time,
                        }
                        delivered_events.append(delivered_event)
                        # Mark every qid at the common ordering boundary before
                        # doing any per-qid public/private work.
                        for qid in sorted(selected_by_qid):
                            lifecycle.mark_delivered(
                                qid, ordered_at,
                                {"ledger_index": ledger_index, "acs_session": session_id},
                            )
                            query_delivery_events.append({
                                "sequence": len(query_delivery_events) + 1,
                                "qid": qid,
                                "entry_count": len(selected_by_qid[qid]),
                                "ledger_index": ledger_index,
                                "deliver_time": ordered_at,
                            })
                        for qid in sorted(selected_by_qid):
                            entries = selected_by_qid[qid]
                            public = _pipeline_cpu_call(
                                "public_finalization",
                                pipeline_finalizer.on_verified_deliver,
                                qid, entries,
                            )
                            if public is None:
                                raise RuntimeError("A2DPSS pipeline public finalization failed")
                            event = {
                                "event": "public_finalized",
                                "qid": qid,
                                "context_digest": public.context_digest,
                                "accepted_contribution_digests": list(
                                    public.accepted_contribution_digests
                                ),
                                "successor_commitment_vector": list(
                                    public.successor_commitment_vector
                                ),
                                "private_reconstructed": False,
                                "t_public": _event_clock(),
                                "t_successor": None,
                            }
                            context = workload.context_for_qid(qid)
                            if my_id in context.successor_role_indices:
                                private = _pipeline_cpu_call(
                                    "successor_reconstruction",
                                    pipeline_reconstruction.reconstruct,
                                    public, entries,
                                )
                                if private is None:
                                    raise RuntimeError("A2DPSS pipeline successor reconstruction failed")
                                event["private_reconstructed"] = True
                                event["my_role_idx"] = private.my_role_idx
                                event["t_successor"] = _event_clock()
                            pipeline_events.append(event)

                    acs_output = await acs.run(
                        candidate_refs,
                        session_id=session_id,
                        max_rounds=acs_max_rounds,
                        proposal_validator=_validate_pipeline_proposal,
                        proposal_fixed_cb=proposal_fixed_cb,
                        output_fixed_cb=_complete_pipeline_output,
                    )
                    acs_end_times.append(time.time())
                    acs_outputs.append(acs_output)

                async def _run_ledger_acs(
                    candidate_refs,
                    label,
                    proposal_fixed_cb=None,
                ):
                    nonlocal acs_output, delivered_event, first_deliver_time
                    nonlocal paired_public_finalize_seconds
                    nonlocal paired_public_finalize_invocations
                    nonlocal paired_reconstruction_seconds
                    nonlocal paired_reconstruction_invocations
                    if a2dpss_pipeline:
                        return await _run_pipeline_acs(
                            candidate_refs, label, proposal_fixed_cb
                        )
                    pool = {}
                    refs_by_qid_mid = {}
                    for ref in candidate_refs:
                        entry = tob.pending[ref]
                        qid = payload_qid(entry.payload)
                        if not qid:
                            continue
                        pool.setdefault(qid, set()).add((entry.mid, entry.payload))
                        refs_by_qid_mid[(qid, entry.mid)] = ref
                    selected = acs_wait_predicate.select_block(pool, set()) or {}
                    candidate_refs = [
                        refs_by_qid_mid[(qid, mid)]
                        for qid in sorted(selected)
                        for mid in selected[qid]
                        if (qid, mid) in refs_by_qid_mid
                    ]
                    if not candidate_refs:
                        raise RuntimeError("no ACS candidate qid satisfies J(qid, S)")
                    session_id = f"ledger-{len(tob.ledger) + 1}"
                    logging.info(
                        "[B4] starting ACS session=%s label=%s candidate_refs=%s",
                        session_id,
                        label,
                        [ref.encode() for ref in candidate_refs],
                    )
                    acs_start_time = time.time()
                    acs_start_times.append(acs_start_time)
                    acs_output = await acs.run(
                        candidate_refs,
                        session_id=session_id,
                        max_rounds=acs_max_rounds,
                        proposal_fixed_cb=proposal_fixed_cb,
                    )
                    acs_end_times.append(time.time())
                    acs_outputs.append(acs_output)
                    while True:
                        missing = tob.missing_refs(acs_output.block_refs)
                        if not missing:
                            break
                        if time.time() - start_run > max_runtime:
                            raise RuntimeError(
                                "max runtime exceeded waiting for ACS refs: "
                                + ",".join(ref.encode() for ref in missing)
                            )
                        await asyncio.sleep(0.05)
                    resolved_by_qid = {}
                    refs_by_qid = {}
                    for ref in acs_output.block_refs:
                        entry = tob.pending[ref]
                        qid = payload_qid(entry.payload)
                        if not qid:
                            continue
                        resolved_by_qid.setdefault(qid, []).append(
                            (entry.mid, entry.payload)
                        )
                        refs_by_qid.setdefault(qid, []).append(ref)
                    valid_refs = []
                    for qid in sorted(resolved_by_qid):
                        entries = sorted(resolved_by_qid[qid])
                        if acs_wait_predicate.verify_qid_set(qid, entries):
                            valid_refs.extend(sorted(refs_by_qid[qid]))
                    if not valid_refs:
                        raise RuntimeError("selected ACS block has no qid satisfying J(qid, S)")
                    ledger_index, block = tob.append_acs_block(valid_refs)
                    deliver_time = time.time()
                    if first_deliver_time is None:
                        first_deliver_time = deliver_time
                    delivered_event = {
                        "ledger_index": ledger_index,
                        "label": label,
                        "refs": [ref.encode() for ref in valid_refs],
                        "block_size": len(block),
                        "deliver_time": deliver_time,
                        "latency_since_broadcast_sec": (
                            deliver_time - broadcast_start_time
                            if broadcast_start_time is not None
                            else None
                        ),
                        "acs_latency_sec": deliver_time - acs_start_time,
                    }
                    delivered_events.append(delivered_event)
                    by_qid = {}
                    for ref, payload in zip(
                        tob.ledger_refs[ledger_index - 1],
                        block,
                    ):
                        qid = payload_qid(payload)
                        if qid:
                            by_qid[qid] = by_qid.get(qid, 0) + 1
                    for qid in sorted(by_qid):
                        query_event = {
                            "sequence": len(query_delivery_events) + 1,
                            "qid": qid,
                            "entry_count": by_qid[qid],
                            "ledger_index": ledger_index,
                            "deliver_time": deliver_time,
                        }
                        query_delivery_events.append(query_event)
                        if ocdpss_paired:
                            entries = sorted(resolved_by_qid[qid])
                            started = time.perf_counter()
                            public_result = paired_ocdpss_finalizer.on_deliver(
                                qid, entries
                            )
                            paired_public_finalize_seconds += (
                                time.perf_counter() - started
                            )
                            paired_public_finalize_invocations += 1
                            if public_result is None:
                                raise RuntimeError(
                                    "paired OCDPSS ACS delivery failed public finalization"
                                )
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
                            paired_reconstruction_seconds += (
                                time.perf_counter() - started
                            )
                            paired_reconstruction_invocations += 1
                            paired_event["private_reconstructed"] = (
                                private_result is not None
                            )
                            if private_result is not None:
                                paired_event["my_role_idx"] = (
                                    private_result.my_role_idx
                                )
                            paired_ocdpss_events.append(paired_event)
                        if lifecycle is not None and qid in set(workload.qids()):
                            lifecycle.mark_delivered(
                                qid,
                                deliver_time,
                                {"ledger_index": ledger_index},
                            )

                if multi_qid_smoke:
                    async def _mark_fixed_and_inject_later(metadata):
                        fixed_at = _event_clock()
                        for ref_raw in metadata.get("refs", []):
                            ref = MessageRef.decode(ref_raw)
                            entry = tob.pending.get(ref)
                            if entry is None:
                                continue
                            qid = payload_qid(entry.payload)
                            if qid in set(workload.qids()):
                                lifecycle.mark_included(
                                    qid,
                                    fixed_at,
                                    {"acs_session": metadata["session_id"]},
                                )
                        later_instances = [
                            instance
                            for batch in later_batches
                            for instance in batch.instances
                        ]
                        for instance in later_instances:
                            lifecycle.mark_proposal_boundary(
                                workload.qid(instance),
                                fixed_at,
                                {
                                    "protocol": "acs-tob",
                                    "acs_session": metadata["session_id"],
                                    "core_digest": metadata.get("core_digest"),
                                    "refs": list(metadata.get("refs", [])),
                                },
                            )
                        await _inject_workload_batches(
                            tob,
                            workload,
                            lifecycle,
                            INITIAL_PROPOSAL_FIXED,
                            my_id,
                            injected_qids,
                            clock=_event_clock,
                        )
                        gate_enter = _event_clock()
                        await _wait_for_workload_pool_ready(
                            tob=tob,
                            workload=workload,
                            lifecycle=lifecycle,
                            instances=later_instances,
                            wait_predicate=acs_wait_predicate,
                            qid_resolver=payload_qid,
                            start_run=start_run,
                            max_runtime=max_runtime,
                            clock=_event_clock,
                            ready_cpu=(
                                pipeline_cpu_stages["ready_check"]
                                if a2dpss_pipeline else None
                            ),
                        )
                        if a2dpss_pipeline:
                            gate_exit = _event_clock()
                            pipeline_gates.append({
                                "label": "initial_proposal_fixed",
                                "enter": gate_enter,
                                "exit": gate_exit,
                                "duration": gate_exit - gate_enter,
                                "acs_session": metadata["session_id"],
                            })
                        logging.info(
                            "[WORKLOAD] trigger=initial_proposal_fixed "
                            "acs_session=%s later requests are J-ready",
                            metadata["session_id"],
                        )

                    await _run_ledger_acs(
                        _pending_refs_for_qids(
                            tob, initial_qids, payload_qid
                        ),
                        "initial",
                        proposal_fixed_cb=_mark_fixed_and_inject_later,
                    )
                    if later_qids:
                        refs_later = await _wait_for_refs(
                            tob=tob,
                            qids=later_qids,
                            expected=(
                                sum(
                                    workload.expected_entry_count(instance)
                                    for batch in later_batches
                                    for instance in batch.instances
                                )
                                if ocdpss_paired or a2dpss_pipeline
                                else len(later_qids) * M * messages_per_node
                            ),
                            start_run=start_run,
                            max_runtime=max_runtime,
                            label="qid B/C",
                            qid_resolver=payload_qid,
                        )
                        async def _mark_later_fixed(metadata):
                            fixed_at = _event_clock()
                            included_qids = (
                                {
                                    payload_qid(tob.pending[MessageRef.decode(raw)].payload)
                                    for raw in metadata.get("refs", [])
                                    if MessageRef.decode(raw) in tob.pending
                                }
                                if a2dpss_pipeline else later_qids
                            )
                            for qid in included_qids:
                                if qid not in later_qids:
                                    continue
                                lifecycle.mark_included(
                                    qid,
                                    fixed_at,
                                    {"acs_session": metadata["session_id"]},
                                )

                        await _run_ledger_acs(
                            refs_later,
                            "later",
                            proposal_fixed_cb=_mark_later_fixed,
                        )
                else:
                    await _run_ledger_acs(tob.pending_refs(), "single-demo")
            if experiment_barrier is not None:
                if a2dpss_pipeline and len(pipeline_events) != qid_count:
                    raise RuntimeError("A2DPSS pipeline did not finish every qid")
                if (
                    a2dpss_handover_metrics is not None
                    and not a2dpss_handover_metrics.snapshot()["frozen"]
                ):
                    raise RuntimeError("A2DPSS did not complete before the runtime limit")
                await experiment_barrier(runner, "done")
        finally:
            cleanup_completed = await manager.stop(timeout=stop_grace_seconds)
            if a2dpss_reconstruction is not None:
                a2dpss_reconstruction.unregister_context(
                    a2dpss_state["context"]
                )
            if pipeline_reconstruction is not None:
                for context in workload.registrations():
                    pipeline_reconstruction.unregister_context(context)
            logging.info(
                "[CLEANUP] completed=%s active_after=%d",
                cleanup_completed,
                manager.active_count(),
            )
            if threshold_coin_oracle is not None:
                await threshold_coin_oracle.close()
            summary = _build_summary(cleanup_completed=cleanup_completed)
            summary["active_rbc_instances_after_cleanup"] = manager.active_count()
            logging.info("[SUMMARY] %s", json.dumps(summary, sort_keys=True))
            shared_task.cancel()
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
    HbmpcConfig.load_config()
    _configure_logging(HbmpcConfig.my_id)

    M = getattr(HbmpcConfig, "M", HbmpcConfig.N)
    expected_committee_size = getattr(
        HbmpcConfig,
        "expected_committee_size",
        None,
    )
    if expected_committee_size is None:
        expected_committee_size = HbmpcConfig.N
    committee_m = getattr(HbmpcConfig, "committee_m", None)
    if committee_m is None:
        committee_m = expected_committee_size
    quorum_h = getattr(HbmpcConfig, "quorum_h", None)
    if quorum_h is None:
        quorum_h = 2 * HbmpcConfig.t + 1
    epoch = os.environ.get(
        "ATCAAS_TRIAL_EPOCH",
        HbmpcConfig.extras.get("run_id", "async-yoso-tob"),
    )
    ocdpss_paired = _env_bool("ASYNC_YOSO_TOB_OCDPSS_PAIRED", False)
    a2dpss_e2e_smoke = _env_bool(
        "ASYNC_YOSO_TOB_A2DPSS_E2E_SMOKE",
        False,
    )
    a2dpss_smoke = _env_bool(
        "ASYNC_YOSO_TOB_A2DPSS_SMOKE",
        False,
    ) or a2dpss_e2e_smoke
    fixture_path = os.environ.get("ATCAAS_A2DPSS_FIXTURE")
    a2dpss_fixture = (
        A2DPSSPairedFixture.from_file(fixture_path)
        if fixture_path
        else None
    )
    a2dpss_acsprop_smoke = _env_bool(
        "ASYNC_YOSO_TOB_A2DPSS_ACSPROP_SMOKE",
        False,
    )
    coin_mode_requested = os.environ.get(
        "ASYNC_YOSO_TOB_COIN_MODE",
        "ideal",
    )
    coin_silent_nodes = _node_id_set(
        os.environ.get("ASYNC_YOSO_TOB_COIN_SILENT_NODES", ""),
        M=M,
        name="ASYNC_YOSO_TOB_COIN_SILENT_NODES",
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
                epoch=epoch,
                messages_per_node=_env_int("ASYNC_YOSO_TOB_MESSAGES_PER_NODE", 1),
                max_runtime=_env_float("ASYNC_YOSO_TOB_MAX_RUNTIME", 60.0),
                stop_grace_seconds=_env_float(
                    "ASYNC_YOSO_TOB_STOP_GRACE_SECONDS",
                    5.0,
                ),
                enable_ledger=_env_bool("ASYNC_YOSO_TOB_ENABLE_LEDGER", True),
                acs_max_rounds=_env_int("ASYNC_YOSO_TOB_ACS_MAX_ROUNDS", 10),
                acs_collect_all=_env_bool("ASYNC_YOSO_TOB_ACS_COLLECT_ALL", True),
                acs_activation_mode=os.environ.get(
                    "ASYNC_YOSO_TOB_ACS_ACTIVATION_MODE",
                    "all",
                ),
                expected_committee_size=expected_committee_size,
                committee_m=committee_m,
                quorum_h=quorum_h,
                acs_collect_threshold=_env_optional_int(
                    "ASYNC_YOSO_TOB_ACS_COLLECT_THRESHOLD"
                ),
                acs_stage_timeout=_env_optional_float(
                    "ASYNC_YOSO_TOB_ACS_STAGE_TIMEOUT"
                ),
                acs_echo_adoption_wait=_env_float(
                    "ASYNC_YOSO_TOB_ACS_ECHO_ADOPTION_WAIT",
                    0.25,
                ),
                acs_output_service_grace=_env_float(
                    "ASYNC_YOSO_TOB_ACS_OUTPUT_SERVICE_GRACE",
                    0.25,
                ),
                multi_qid_smoke=_env_bool(
                    "ASYNC_YOSO_TOB_MULTI_QID_SMOKE",
                    False,
                ),
                ocdpss_paired=ocdpss_paired,
                a2dpss_pipeline=_env_bool("ATCAAS_A2DPSS_PIPELINE", False),
                a2dpss_smoke=a2dpss_smoke,
                a2dpss_acsprop_smoke=a2dpss_acsprop_smoke,
                a2dpss_e2e_smoke=a2dpss_e2e_smoke,
                coin_mode_requested=coin_mode_requested,
                coin_timeout=_env_float(
                    "ASYNC_YOSO_TOB_COIN_TIMEOUT",
                    30.0,
                ),
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
