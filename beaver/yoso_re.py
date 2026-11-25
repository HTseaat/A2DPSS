"""Legacy RE harness retained for resharing-only regression tests.

P1.1 was a reusable **single DPSS handoff** over the DAG-TOB harness — the unit a
service request triggers in the role-execution stage. One invocation reshares a
key's sharing from the current holder committee to its designated successor
committee (the older draft's resharing-only protocol), reconstructs the successor shares, and derives the
successor's commitment vector (P1.2, `interpolate_commitment_vector`) so the
successor can hold the sharing for the *next* request in the same epoch. The
underlying secret is preserved across the handoff.

This module does not implement formal Algorithm-4 A²DPSS or Algorithm-5
OCDPSS. It remains only for historical RE/smoke compatibility; new protocol
work uses ``beaver.a2dpss`` or the ``beaver.ocdpss`` extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from beaver.chunk_elgamal import derive_public_key
from beaver.dag_tob import (
    DagTOB,
    DeterministicLeaderOracle,
    JustifierRegistry,
    TypedWaitPredicate,
)
from beaver.pvss import (
    PVSSTranscript,
    interpolate_commitment_vector,
    pvss_interpolate,
)
from beaver.legacy_yoso_dpss import (
    DPSSContext,
    DPSSJustifierManager,
    DPSSLocalConfig,
    DPSSReconstructionManager,
    build_dpss_local_config,
    run_resharing,
)


@dataclass
class HandoffResult:
    """Outcome of one DPSS handoff (current holder -> designated successor)."""

    ctx: DPSSContext
    successor_shares: Dict[int, Tuple[int, int]]   # node_id -> (s, s_hat)
    successor_eval_points: Dict[int, int]          # node_id -> eval point (role_idx + 1)
    new_commitment_vector: List[str]               # derived v_{s,u'} for the next handoff
    recovered_secret: Tuple[int, int]              # interpolated (s, s_hat); invariant oracle


async def run_dpss_handoff(
    *,
    locals_by_node: Sequence[DPSSLocalConfig],
    g_hex: str,
    h_hex: str,
    M: int,
    quorum_h: int,
    epoch_label: str,
    rounds: int = 4,
) -> HandoffResult:
    """Run one DPSS handoff and return the successor sharing state.

    `locals_by_node[i]` is party `i`'s `DPSSLocalConfig` (same `ctx` for all);
    it carries whether `i` is a source holder and/or a successor, its share, and
    its successor role key. The local harness runs no network broadcast layer
    (the real system's Bracha RBC / payload dispersal), so every node holds the
    configured holder shares and reshares them.
    """

    ctx = locals_by_node[0].ctx
    captured: Dict[int, Tuple[int, int, int]] = {}  # node_id -> (role_idx, s, s_hat)
    callback_counts: Dict[int, int] = {}
    reconstruction_managers: List[DPSSReconstructionManager] = []
    nodes: List[DagTOB] = []

    for node_id, local in enumerate(locals_by_node):
        registry = JustifierRegistry()
        registry.register("dpss", DPSSJustifierManager([ctx]))
        manager = DPSSReconstructionManager()
        if local.am_successor:

            def _on_share(s, sh, nid=node_id, ridx=local.my_successor_role_idx):
                callback_counts[nid] = callback_counts.get(nid, 0) + 1
                captured[nid] = (int(ridx), int(s), int(sh))

            manager.register_context(
                ctx,
                my_node_id=node_id,
                my_rsk=local.my_successor_rsk,
                on_new_share=_on_share,
            )
        reconstruction_managers.append(manager)

        def _deliver_cb(qid, entries, mgr=manager):
            mgr.on_deliver(qid, entries)

        nodes.append(
            DagTOB(
                my_id=node_id,
                M=M,
                quorum_h=quorum_h,
                agreement_t=0,  # in-process harness models an all-honest committee
                epoch=epoch_label,
                wait_predicate=TypedWaitPredicate(registry),
                leader_oracle=DeterministicLeaderOracle(epoch_label),
                deliver_cb=_deliver_cb,
            )
        )

    # The local harness has no message-RBC.  Each holder produces one random
    # transcript and every observer receives the exact same payload.
    for holder_raw in locals_by_node[0].source_committee:
        holder_id = int(holder_raw)
        local = locals_by_node[holder_id]
        share_s, share_s_hat = local.holder_shares[holder_id]
        result = await run_resharing(
            dag_tob=nodes[holder_id],
            g_hex=g_hex,
            h_hex=h_hex,
            my_share_s=share_s,
            my_share_s_hat=share_s_hat,
            successor_rpk_list=ctx.successor_rpk_list,
            dealer_eval_point=ctx.holder_eval_points[holder_id],
            service_t=ctx.service_t,
            epoch=ctx.epoch,
            sid=ctx.sid,
            source_committee=ctx.source_committee,
            successor_committee=ctx.successor_committee,
            transfer_seq=ctx.transfer_seq,
            key_base_hex=ctx.key_base,
            my_id=holder_id,
        )
        payload = result.transcript.to_bytes()
        for node_id, node in enumerate(nodes):
            if node_id != holder_id:
                node.deliver_message(result.qid, result.mid, payload)

    # drive the DAG-TOB to deliver the transfer block
    selected_sources = list(range(quorum_h))
    for round_id in range(1, rounds + 1):
        vertices = [
            nodes[src].create_new_vertex(round_id, source=src)
            for src in selected_sources
        ]
        for node in nodes:
            for vertex in vertices:
                node.on_r_deliver(vertex)

    if not all(node.delivery_count == 1 for node in nodes):
        raise RuntimeError("DPSS transfer qid not delivered by all DAG-TOB nodes")
    delivery_views = [node.snapshot_delivery_events() for node in nodes]
    if any(view != delivery_views[0] for view in delivery_views[1:]):
        raise RuntimeError("DPSS delivery events differ across observers")
    if delivery_views[0][0]["qid"] != ctx.encode_qid():
        raise RuntimeError("DAG-TOB delivered an unexpected DPSS qid")
    for node_id, local in enumerate(locals_by_node):
        expected = 1 if local.am_successor else 0
        if callback_counts.get(node_id, 0) != expected:
            raise RuntimeError(
                f"node {node_id}: reconstruction callback count "
                f"{callback_counts.get(node_id, 0)} != {expected}"
            )
        duplicate = reconstruction_managers[node_id].on_deliver(
            ctx.encode_qid(),
            nodes[node_id].delivery_events[0].entries,
        )
        if duplicate or callback_counts.get(node_id, 0) != expected:
            raise RuntimeError(f"node {node_id}: duplicate DPSS reconstruction")

    # gather reconstructed successor shares
    successor_shares: Dict[int, Tuple[int, int]] = {}
    successor_eval_points: Dict[int, int] = {}
    for node_id, (role_idx, s, sh) in captured.items():
        successor_shares[node_id] = (s, sh)
        successor_eval_points[node_id] = role_idx + 1
    if len(successor_shares) < ctx.service_t + 1:
        raise RuntimeError("fewer than t+1 successors reconstructed")

    # secret-preservation oracle: interpolate the new shares back to the secret
    items = sorted(successor_shares.items())[: ctx.service_t + 1]
    recovered = pvss_interpolate(
        [successor_eval_points[nid] for nid, _ in items],
        [share for _, share in items],
        ctx.service_t,
    )

    # Derive the successor commitment vector from the SAME t+1 transcripts used
    # by reconstruction (the resharing rule retained from the older draft).
    recon = next(
        (mgr.results[ctx.key] for mgr in reconstruction_managers if ctx.key in mgr.results),
        None,
    )
    if recon is None:
        raise RuntimeError("no reconstruction result available")
    qid_raw = ctx.encode_qid()
    transcripts_by_ep: Dict[int, PVSSTranscript] = {}
    for event in nodes[0].delivery_events:
        if event.qid != qid_raw:
            continue
        for _mid, payload in event.entries:
            tr = PVSSTranscript.from_bytes(payload)
            transcripts_by_ep[int(tr.dealer_eval_point)] = tr
    eval_pts = [int(ep) for ep in recon.dealer_eval_points]
    cvs = [transcripts_by_ep[ep].commitment_vector for ep in eval_pts]
    new_cv = interpolate_commitment_vector(eval_pts, cvs, ctx.service_t)

    return HandoffResult(
        ctx=ctx,
        successor_shares=successor_shares,
        successor_eval_points=successor_eval_points,
        new_commitment_vector=new_cv,
        recovered_secret=recovered,
    )


@dataclass
class ReHopResult:
    """One served request in the RE walk: a handoff source -> successor."""

    hop: int
    source_committee: Tuple[int, ...]
    successor_committee: Tuple[int, ...]
    recovered_secret: Tuple[int, int]


async def run_re_service_loop(
    *,
    committees: Sequence[Sequence[int]],
    role_rsks: Sequence[int],
    g_hex: str,
    h_hex: str,
    M: int,
    quorum_h: int,
    service_t: int,
    tau: int,
    secret: int,
    secret_hat: int,
    sid: str = "key-1",
    epoch: int = 0,
) -> Tuple[List[ReHopResult], Tuple[int, ...]]:
    """Walk the RE-stage committee list ``h_e(sid) = committees`` (paper §3.1).

    Each iteration serves one request by handing the sharing off from the current
    holder ``committees[0]`` to its designated successor ``committees[1]`` via one
    DPSS reshare (P1.1 :func:`run_dpss_handoff`), then advances the holder. The
    successor sharing state (shares + commitment vector, via P1.2) becomes the
    next hop's input, so the underlying secret is preserved across the walk.
    Serving continues while ``len(h_e) - 1 >= tau`` (the reserve threshold);
    once violated the epoch would enter role assignment (out of scope here).

    Returns ``(hops, final_holder_committee)``.
    """

    role_keys = [
        {"rsk": int(role_rsks[i]), "rpk": derive_public_key(int(role_rsks[i]), g_hex)}
        for i in range(M)
    ]
    h_e: List[List[int]] = [list(c) for c in committees]

    hops: List[ReHopResult] = []
    holder_shares: Dict[int, Tuple[int, int]] | None = None
    holder_eval_points: Dict[int, int] | None = None
    old_cv: List[str] | None = None
    hop = 0

    while len(h_e) - 1 >= tau:
        source = h_e[0]
        successor = h_e[1]
        if holder_eval_points is None:  # genesis hop: assign 1..n to the holders
            holder_eval_points = {node: idx + 1 for idx, node in enumerate(source)}

        cfg: dict = {
            "service_t": service_t,
            "sid": sid,
            "epoch": epoch,
            "transfer_seq": hop,
            "source_committee": list(source),
            "successor_committee": list(successor),
            "source_committee_id": f"{sid}-c{hop}",
            "successor_committee_id": f"{sid}-c{hop + 1}",
            "holder_eval_points": {str(n): int(ep) for n, ep in holder_eval_points.items()},
            "successor_role_indices": {str(n): i for i, n in enumerate(successor)},
            "role_keys": role_keys,
        }
        if holder_shares is None:
            cfg["secret"] = int(secret)
            cfg["secret_hat"] = int(secret_hat)
        else:
            cfg["holder_shares"] = {
                str(n): {"s": int(s), "s_hat": int(sh)}
                for n, (s, sh) in holder_shares.items()
            }
            cfg["old_commitment_vector"] = old_cv

        locals_by_node = [
            build_dpss_local_config(
                {"dpss_smoke": cfg},
                g_hex=g_hex,
                h_hex=h_hex,
                my_id=i,
                M=M,
                service_t=service_t,
                epoch=epoch,
            )
            for i in range(M)
        ]
        res = await run_dpss_handoff(
            locals_by_node=locals_by_node,
            g_hex=g_hex,
            h_hex=h_hex,
            M=M,
            quorum_h=quorum_h,
            epoch_label=f"{sid}-hop{hop}",
        )
        hops.append(
            ReHopResult(
                hop=hop,
                source_committee=tuple(source),
                successor_committee=tuple(successor),
                recovered_secret=res.recovered_secret,
            )
        )

        # advance the holder and chain the sharing state forward
        holder_shares = {n: res.successor_shares[n] for n in successor}
        holder_eval_points = {n: res.successor_eval_points[n] for n in successor}
        old_cv = res.new_commitment_vector
        h_e.pop(0)
        hop += 1

    final_holder = tuple(h_e[0]) if h_e else ()
    return hops, final_holder


async def run_concurrent_handoffs(
    *,
    sid_locals: Dict[str, Sequence[DPSSLocalConfig]],
    g_hex: str,
    h_hex: str,
    M: int,
    quorum_h: int,
    epoch_label: str,
    rounds: int = 4,
) -> Tuple[Dict[str, Tuple[int, int]], set]:
    """Run one handoff for several `sid`s **concurrently over a single TOB**.

    Validates that K managed keys + mu coin slots (paper §3.1: coin generation is
    RE.Req through the same machinery) hand off in the same epoch without
    interfering: each `sid` has its own `DPSSContext` (keyed by sid, so qids do
    not collide), all are registered in every node's justifier/reconstruction
    managers, and all resharing transcripts are delivered through one ledger.

    `sid_locals[sid][i]` is party `i`'s `DPSSLocalConfig` for `sid`. Committees
    are expected pairwise disjoint across sids. Returns
    `(recovered_secret_by_sid, delivered_qids)`.
    """

    all_ctxs = [locs[0].ctx for locs in sid_locals.values()]
    captured: Dict[Tuple, Tuple[int, int, int]] = {}  # (ctx.key, node_id) -> (role_idx, s, sh)
    callback_counts: Dict[Tuple, int] = {}
    reconstruction_managers: List[DPSSReconstructionManager] = []
    nodes: List[DagTOB] = []

    for node_id in range(M):
        registry = JustifierRegistry()
        registry.register("dpss", DPSSJustifierManager(all_ctxs))
        manager = DPSSReconstructionManager()
        for locs in sid_locals.values():
            local = locs[node_id]
            if local.am_successor:

                def _cb(s, sh, key=local.ctx.key, nid=node_id, ridx=local.my_successor_role_idx):
                    callback_counts[(key, nid)] = (
                        callback_counts.get((key, nid), 0) + 1
                    )
                    captured[(key, nid)] = (int(ridx), int(s), int(sh))

                manager.register_context(
                    local.ctx,
                    my_node_id=node_id,
                    my_rsk=local.my_successor_rsk,
                    on_new_share=_cb,
                )

        reconstruction_managers.append(manager)

        def _deliver_cb(qid, entries, mgr=manager):
            mgr.on_deliver(qid, entries)

        nodes.append(
            DagTOB(
                my_id=node_id,
                M=M,
                quorum_h=quorum_h,
                agreement_t=0,  # in-process harness models an all-honest committee
                epoch=epoch_label,
                wait_predicate=TypedWaitPredicate(registry),
                leader_oracle=DeterministicLeaderOracle(epoch_label),
                deliver_cb=_deliver_cb,
            )
        )

    # Generate each (sid, holder) transcript once, then reproduce message-RBC's
    # identical delivery at every observer in this in-process harness.
    for locs in sid_locals.values():
        ctx = locs[0].ctx
        for holder_raw in locs[0].source_committee:
            holder_id = int(holder_raw)
            local = locs[holder_id]
            share_s, share_s_hat = local.holder_shares[holder_id]
            result = await run_resharing(
                dag_tob=nodes[holder_id],
                g_hex=g_hex,
                h_hex=h_hex,
                my_share_s=share_s,
                my_share_s_hat=share_s_hat,
                successor_rpk_list=ctx.successor_rpk_list,
                dealer_eval_point=ctx.holder_eval_points[holder_id],
                service_t=ctx.service_t,
                epoch=ctx.epoch,
                sid=ctx.sid,
                source_committee=ctx.source_committee,
                successor_committee=ctx.successor_committee,
                transfer_seq=ctx.transfer_seq,
                key_base_hex=ctx.key_base,
                my_id=holder_id,
            )
            payload = result.transcript.to_bytes()
            for node_id, node in enumerate(nodes):
                if node_id != holder_id:
                    node.deliver_message(result.qid, result.mid, payload)

    selected_sources = list(range(quorum_h))
    for round_id in range(1, rounds + 1):
        vertices = [
            nodes[src].create_new_vertex(round_id, source=src)
            for src in selected_sources
        ]
        for node in nodes:
            for vertex in vertices:
                node.on_r_deliver(vertex)

    expected_qids = {ctx.encode_qid() for ctx in all_ctxs}
    if not all(node.delivery_count == len(expected_qids) for node in nodes):
        raise RuntimeError("concurrent DPSS qids not delivered by all nodes")
    delivery_views = [node.snapshot_delivery_events() for node in nodes]
    if any(view != delivery_views[0] for view in delivery_views[1:]):
        raise RuntimeError("concurrent DPSS delivery events differ across observers")
    delivered_sequence = [event["qid"] for event in delivery_views[0]]
    if len(delivered_sequence) != len(set(delivered_sequence)):
        raise RuntimeError("a concurrent DPSS qid was delivered more than once")
    if set(delivered_sequence) != expected_qids:
        raise RuntimeError("concurrent DPSS delivery qid set mismatch")

    for sid, locs in sid_locals.items():
        ctx = locs[0].ctx
        for node_id, local in enumerate(locs):
            expected = 1 if local.am_successor else 0
            key = (ctx.key, node_id)
            if callback_counts.get(key, 0) != expected:
                raise RuntimeError(
                    f"sid {sid} node {node_id}: reconstruction callback count "
                    f"{callback_counts.get(key, 0)} != {expected}"
                )
            event = next(
                item
                for item in nodes[node_id].delivery_events
                if item.qid == ctx.encode_qid()
            )
            duplicate = reconstruction_managers[node_id].on_deliver(
                event.qid,
                event.entries,
            )
            if duplicate or callback_counts.get(key, 0) != expected:
                raise RuntimeError(
                    f"sid {sid} node {node_id}: duplicate reconstruction"
                )

    recovered_by_sid: Dict[str, Tuple[int, int]] = {}
    for sid, locs in sid_locals.items():
        ctx = locs[0].ctx
        shares: Dict[int, Tuple[int, int]] = {}
        eval_points: Dict[int, int] = {}
        for (key, nid), (role_idx, s, sh) in captured.items():
            if key == ctx.key:
                shares[nid] = (s, sh)
                eval_points[nid] = role_idx + 1
        if len(shares) < ctx.service_t + 1:
            raise RuntimeError(f"sid {sid}: fewer than t+1 successors reconstructed")
        items = sorted(shares.items())[: ctx.service_t + 1]
        recovered_by_sid[sid] = pvss_interpolate(
            [eval_points[nid] for nid, _ in items],
            [share for _, share in items],
            ctx.service_t,
        )

    delivered_qids = set(delivered_sequence)

    return recovered_by_sid, delivered_qids


def generate_re_stage_config(
    *,
    M: int,
    service_t: int,
    K: int = 1,
    mu: int = 1,
    committee_size: int | None = None,
    list_len: int | None = None,
    tau: int = 1,
) -> dict:
    """Predefined Role-Execution layout for SID = {1..K} u {coin_1..coin_mu}.

    Each sid gets an ordered, pairwise-disjoint committee list ``h_e(sid)`` of
    ``list_len`` committees of size ``committee_size`` (default ``service_t+1``),
    plus a deterministic genesis secret. Committees are carved disjointly from
    the ``M`` long-lived parties; ``list_len`` auto-fits so the layout always
    fits (degrades to 1 committee/sid when ``M`` is too small). In the real
    system these committees come from role assignment (RA); here they stand in
    for it so the RE stage runs without RA (plan decision #3). Pure data; written
    to config under ``extra.re_stage`` and consumed by the RE smokes.
    """

    n = int(committee_size or (service_t + 1))
    sids = [f"key-{i + 1}" for i in range(K)] + [f"coin-{a + 1}" for a in range(mu)]
    num = len(sids)
    max_list_len = (M // (num * n)) if num * n > 0 else 0
    if list_len is None:
        list_len = min(3, max_list_len)
    list_len = max(1, min(int(list_len), max_list_len)) if max_list_len >= 1 else 1

    cursor = 0
    sid_cfgs: Dict[str, dict] = {}
    for idx, sid in enumerate(sids):
        committees: List[List[int]] = []
        for _ in range(list_len):
            if cursor + n <= M:
                committees.append(list(range(cursor, cursor + n)))
                cursor += n
        sid_cfgs[sid] = {
            "committees": committees,
            "secret": str(1000 * (idx + 1) + 1),
            "secret_hat": str(1000 * (idx + 1) + 7),
        }

    return {
        "version": 1,
        "epoch": 0,
        "service_t": int(service_t),
        "tau": int(tau),
        "committee_size": n,
        "K": int(K),
        "mu": int(mu),
        "list_len": list_len,
        "sids": sid_cfgs,
    }
