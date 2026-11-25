"""Authenticated full-population broadcast with dedicated one-send roles.

Audit messages observe a common reconstructible VPS for measurement only;
they never select protocol inputs, certify values, or release a role's send.
The experiment stops after that endpoint, as an external epoch-stop event.
"""
import asyncio
import math
import time
from dataclasses import asdict

from beaver.baselines.ru22_certificate import canonical, strict_hex
from beaver.baselines.ru22_dpss import RU22ContributionBatch, RU22VPS, RU22HolderMetrics, make_ru22_contribution_batch
from beaver.baselines.ru22_native import (
    NativeParentRegistry, NativeLeader, NativeVerifier, NativeSuccessor,
    NativeCombine, NativeMetrics, CRYPTO_STAGES,
)
from beaver.baselines.ru22_native_roles import NativeAuthentication, CONTROL
from beaver.baselines.ru22_native_experiment import load_native_fixture, BROADCAST_DELIVERY
from beaver.baselines.ru22_transport import NativeTransport


async def run_native_node(*, fixture, node, peers, timeout=300, startup_timeout=60):
    public, private, base, shard = load_native_fixture(fixture, node)
    c = base.context
    auth = NativeAuthentication(c, public["roles"], node, private["identity"])
    transport = NativeTransport(peers, node)
    all_nodes = tuple(range(public["roles"]["population"]))
    if len(peers) != len(all_nodes):
        raise ValueError("native population/peer count mismatch")
    authority = auth.authority
    leader_nodes = tuple(dict(authority.leader_keys))
    verifier_nodes = {leader: tuple(authority.verifiers_for(leader)) for leader in leader_nodes}
    successor_nodes = tuple(sorted(c.successor_role_indices))
    registry = lambda: NativeParentRegistry(c, genesis=base.parent_vps_catalog)
    leader_type, verifier_type, successor_type = NativeLeader, NativeVerifier, NativeSuccessor
    batch_type, package_type, generate = RU22ContributionBatch, NativeCombine, make_ru22_contribution_batch
    operation_args, generation_metrics = {}, RU22HolderMetrics()
    if "operation_context" in public:
        from beaver.threshold_operation_contract import ThresholdOperationContext
        from beaver.baselines.ru22_operation import (
            RU22OperationContributionBatch, RU22OperationHolderMetrics, make_ru22_operation_contribution_batch)
        from beaver.baselines.ru22_native_operation import (
            NativeOperationLeader, NativeOperationVerifier, NativeOperationSuccessor, NativeOperationPackage)
        operation_args = dict(operation_context=ThresholdOperationContext.from_obj(public["operation_context"]),
                              trial_id=public["operation_trial_id"])
        leader_type, verifier_type, successor_type = NativeOperationLeader, NativeOperationVerifier, NativeOperationSuccessor
        batch_type, package_type, generate = RU22OperationContributionBatch, NativeOperationPackage, make_ru22_operation_contribution_batch
        generation_metrics = RU22OperationHolderMetrics()
    leader = leader_type(authority, registry(), node, **operation_args) if node in leader_nodes else None
    verifiers = {i: verifier_type(authority, registry(), i, node,
        bytes.fromhex(private["identity"]["role_signing"][f"verifier/{i}"]), **operation_args)
        for i in leader_nodes if node in verifier_nodes[i]}
    successor = successor_type(authority, node, shard.successor_recipient_secret_key, **operation_args) if node in successor_nodes else None
    holder = shard.source.holder_state(c.resharing_n) if shard.source else None
    del private, shard
    metrics = NativeMetrics()
    events = {phase: asyncio.Event() for phase in ("ready", "ready-ack", "done", "done-ack")}
    received_control = {phase: set() for phase in events}
    states, arrivals = {}, {}
    common = asyncio.Event()
    protocol_inbox = asyncio.Queue()
    protocol_idle = asyncio.Event()
    protocol_idle.set()
    origin = None
    first_role_sent = None
    rejected = 0
    sent_payload = {phase: 0 for phase in ("contribution", "combine", "vote")}
    tasks = []
    result = None

    async def send(phase, body, recipients, instance=None):
        nonlocal first_role_sent
        envelope = auth.seal(phase, body, instance=instance)
        if phase in sent_payload:
            sent_payload[phase] += len(canonical(body))
        await transport.multicast(recipients, envelope)
        if phase in sent_payload and first_role_sent is None:
            first_role_sent = time.perf_counter() - origin

    def check_common():
        if set(states) == set(successor_nodes):
            if set.intersection(*(set(states[i]) for i in successor_nodes)):
                common.set()

    async def receive():
        nonlocal rejected
        while True:
            envelope = await transport.inbox.get()
            try:
                sender, phase, body = auth.open(envelope)
                if phase in events:
                    if body != {}:
                        raise ValueError("invalid barrier body")
                    received_control[phase].add(sender)
                    if received_control[phase] == set(all_nodes):
                        events[phase].set()
                elif phase == "audit-state":
                    if sender not in successor_nodes or not isinstance(body, dict) or len(body) > c.resharing_n:
                        raise ValueError("invalid successor audit message")
                    for vid, elapsed in body.items():
                        strict_hex(vid, 32)
                        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                            raise ValueError("invalid successor audit timestamp")
                    known = states.setdefault(sender, {})
                    if any(vid in known and known[vid] != elapsed for vid, elapsed in body.items()):
                        raise ValueError("successor changed an arrival timestamp")
                    known.update(body)
                    check_common()
                else:
                    # Buffer early protocol traffic until this node has passed
                    # the startup barrier and established its measurement t0.
                    protocol_inbox.put_nowait((sender, phase, body))
            except (TypeError, KeyError, ValueError):
                rejected += 1

    async def process_protocol():
        nonlocal rejected
        while True:
            sender, phase, body = await protocol_inbox.get()
            # The benchmark's external epoch stop must not cut an active
            # M-recipient send in half. Stop new work, then drain that send.
            if common.is_set():
                continue
            protocol_idle.clear()
            try:
                if phase == "contribution" and leader:
                    batch = batch_type.from_obj(body["batch"])
                    result = leader.receive(sender, batch, [RU22VPS.from_obj(v) for v in body["parents"]])
                    if result:
                        await send("combine", result.to_obj(), all_nodes)
                elif phase == "combine" and sender in verifiers:
                    result = verifiers[sender].receive(sender, package_type(canonical(body)))
                    if result:
                        await send("vote", result, all_nodes, instance=sender)
                elif phase == "vote" and successor:
                    result = successor.receive(sender, body)
                    if result:
                        arrivals.setdefault(result.vps.vps_id, time.perf_counter() - origin)
                        await send("audit-state", dict(arrivals), all_nodes)
                elif phase not in ("contribution", "combine", "vote"):
                    raise ValueError("unknown protocol phase")
                # Every node authenticates broadcasts; only assigned roles
                # execute the protocol body. Unrelated recipients are normal.
            except (TypeError, KeyError, ValueError):
                rejected += 1
            finally:
                protocol_idle.set()

    async def wait_event(event):
        waiter = asyncio.create_task(event.wait())
        try:
            completed, _ = await asyncio.wait([waiter] + tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in tasks:
                if task in completed:
                    await task
                    raise RuntimeError("native receiver stopped unexpectedly")
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    try:
        await transport.start()
        tasks.append(asyncio.create_task(receive()))
        for phase in ("ready", "ready-ack"):
            await asyncio.wait_for(send(phase, {}, all_nodes), startup_timeout)
            await asyncio.wait_for(wait_event(events[phase]), startup_timeout)
        origin = time.perf_counter()
        tasks.append(asyncio.create_task(process_protocol()))

        async def protocol():
            nonlocal holder
            if holder:
                holder.validate_public(c)
                role, point, index, rsk, local = holder.consume()
                with metrics.measure("source_generation"):
                    extra = dict(operation_args, metrics=generation_metrics)
                    batch = generate(context=c, holder_role_id=role,
                        holder_eval_point=point, recipient_index=index,
                        recipient_secret_key=rsk, local_vps_shares=local, **extra)
                    generation_metrics.batch_bytes = len(batch.to_bytes())
                await send("contribution", dict(batch=batch.to_obj(),
                    parents=[entry.vps.to_obj() for entry in local]), all_nodes)
                del local, rsk
                holder = None
            await wait_event(common)
            await wait_event(protocol_idle)
            for phase in ("done", "done-ack"):
                await send(phase, {}, all_nodes)
                await wait_event(events[phase])

        await asyncio.wait_for(protocol(), timeout)
        end = time.perf_counter() - origin
        if not common.is_set():
            raise RuntimeError("native benchmark has no common recoverable VPS")
        # Sum equal stage names across independent roles on the same machine.
        protocol_metrics = {}
        crypto_metrics = {stage: dict(invocations=0, wall_seconds=0., cpu_seconds=0.) for stage in CRYPTO_STAGES}
        for source in [metrics, auth.metrics] + [r.metrics for r in [leader, successor] + list(verifiers.values()) if r]:
            for stage, values in source.snapshot().items():
                target = protocol_metrics.setdefault(stage, {key: 0 for key in values})
                for key, value in values.items():
                    target[key] += value
            for stage, values in source.crypto.items():
                for key, value in values.items():
                    crypto_metrics[stage][key] += value
        base_generation = generation_metrics.base if operation_args else generation_metrics
        for kind in ("pvss", "dleq"):
            crypto_metrics["source_" + kind + "_generation"] = dict(
                invocations=base_generation.pvr_count,
                wall_seconds=getattr(base_generation, kind + "_seconds"),
                cpu_seconds=getattr(base_generation, kind + "_cpu_seconds"))
        common_ids = set.intersection(*(set(states[i]) for i in successor_nodes))
        earliest_common = min(max(states[i][vid] for i in successor_nodes) for vid in common_ids)
        result = dict(schema="ru22-native-node-metrics/v3", protocol="ru22", protocol_mode="native",
            protocol_delivery=BROADCAST_DELIVERY,
            status="success", node_id=node, native_manifest_digest=public["digest"],
            common_workload_digest=public["common_workload_digest"],
            arm_manifest_digest=public["arm_manifest_digest"],
            authority_digest=authority.committee_digest,
            successor_nodes=list(successor_nodes), leader_nodes=list(leader_nodes),
            verifier_committees={str(i): list(members) for i, members in verifier_nodes.items()},
            verifier_committee_count=c.resharing_n, verifier_role_count=c.resharing_n ** 2,
            verifier_send_scope=public["verifier_send_scope"], strict_refresh_wide_yoso=True,
            yoso_scope=public["yoso_scope"],
            completion="measurement-observer-common-reconstructible-VPS; not protocol agreement",
            protocol_stop="common endpoint; active role broadcasts drained before snapshot",
            timeline=dict(t0=0., t_first_role_message_sent=first_role_sent,
                          t_first_local_vps=min(arrivals.values()) if arrivals else None,
                          t_common_reconstructible_vps=earliest_common, frozen_at=end),
            vps_arrivals=dict(arrivals),
            certified_vps=[entry.vps.to_obj() for entry in successor.results.values()] if successor else [],
            local_opening_checks=len(successor.results) if successor else 0,
            verifier_actions=sum(v.sent for v in verifiers.values()),
            verifier_roles={str(i): dict(sent=v.sent, closed=v.closed) for i, v in verifiers.items()},
            leader_combines=int(leader.output is not None) if leader else 0,
            rejected_messages=rejected, transport_errors=list(transport.errors),
            cpu=protocol_metrics,
            crypto_metrics=crypto_metrics,
            crypto_accounting="nested subsets of cpu stages; do not add to total CPU; DLEQ is old-to-new binding only",
            communication=dict(total_network_bytes=sum(v for k, v in transport.bytes_sent.items() if k not in CONTROL),
                control_bytes=sum(v for k, v in transport.bytes_sent.items() if k in CONTROL),
                by_phase=dict(transport.bytes_sent), messages_by_phase=dict(transport.messages_sent),
                application_payload_bytes=sent_payload, dag_protocol_bytes=0, common_coin_bytes=0,
                yoso_rbc_envelope_bytes=0, accounting="serialized TCP frames; excludes self delivery and experiment control"))
        if operation_args:
            result["operation_context_digest"] = operation_args["operation_context"].context_digest
            result["operation_trial_id"] = operation_args["trial_id"]
            result["operation_results"] = successor.operation_results if successor else {}
            result["operation_generation"] = asdict(generation_metrics)
        return result
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for verifier in verifiers.values():
            verifier.close()
        auth.close()
        await transport.close()
        if result is not None:
            result["cleanup"] = dict(completed=all(t.done() for t in tasks)
                and not transport.tasks and not transport.server.is_serving())
