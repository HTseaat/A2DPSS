"""Threshold scheduling with explicit stubs, plus real application validity.

These gates check ordering and continued service, not wall-clock speedups.
"""
import asyncio
import copy
import functools
import time
from collections import defaultdict
from types import SimpleNamespace

import pytest

from beaver.a2dpss_benchmark_metrics import CPU_STAGES
from beaver.async_algorand import a2dpss, avss, election
from beaver.async_algorand.transport import Runtime


def async_test(function):
    @functools.wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


class SchedulingRuntime:
    # Exercise production notifications, failure propagation and cancellation.
    pulse, wait, spawn, close = Runtime.pulse, Runtime.wait, Runtime.spawn, Runtime.close

    def __init__(self):
        self.node = 127
        self.p = SimpleNamespace(population=128, h=26, m=38, degree=13,
                                 role=lambda view, name: "v%d/%s" % (view, name))
        self.event = asyncio.Event()
        self.failure = asyncio.get_running_loop().create_future()
        self.tasks = set()
        self.metrics = defaultdict(int)
        self.closed = False
        self.g = self.h = "stubbed"

    async def until(self, predicate):
        return await asyncio.wait_for(self.wait(predicate), 10)


class BroadcastProbe:
    def __init__(self, runtime, tag, role):
        self.delivered, self.sent = {}, []

    def broadcast(self, value):
        self.sent.append(copy.deepcopy(value))


class ProposalCaptured(Exception):
    pass


class AgreementProbe:
    def __init__(self, runtime, valid):
        pass

    async def run_prevalidated(self, proposal):
        raise ProposalCaptured(proposal)


def stub_adapter(monkeypatch, runtime, invalid=(), fixture=False):
    """Cheap validation isolates 17/9 scheduling from PVSS cost."""
    monkeypatch.setattr(a2dpss, "NWA", AgreementProbe)
    monkeypatch.setattr(a2dpss, "A2DPSSContribution", SimpleNamespace(
        from_bytes=lambda payload: SimpleNamespace(holder_id=int(payload))))
    monkeypatch.setattr(a2dpss, "validate_a2_contribution",
                        lambda ctx, contribution: contribution.holder_id not in invalid)
    adapter = a2dpss.A2DPSSAdapter.__new__(a2dpss.A2DPSSAdapter)
    adapter.r = runtime
    adapter.context = SimpleNamespace(service_t=8, holder_eval_points=dict.fromkeys(range(17)))
    adapter.fixture = SimpleNamespace(entries=list(range(17))) if fixture else None
    adapter.qid, adapter.pool, adapter.holders = "qid", {}, {}
    adapter.required = 17 if fixture else 9
    adapter.candidate_refs, adapter.created_at = None, time.perf_counter()
    adapter.cpu_stages = {name: {"cpu_seconds": 0., "invocations": 0} for name in CPU_STAGES}
    adapter.timeline = SimpleNamespace(start=lambda *args: None, mark_pool_ready=lambda: None)
    adapter.justifier = SimpleNamespace(select_preverified_subset=lambda qid, entries: entries[:9])
    adapter.broadcast = BroadcastProbe(runtime, "handover", None)
    adapter.broadcast.delivered = {i: {"payload": str(i).encode().hex()} for i in range(17)}
    runtime.spawn(adapter._receive())
    return adapter


@pytest.mark.parametrize("delayed_run", [False, True])
@async_test
async def test_live_batch_freezes_nine_even_when_run_resumes_after_all_seventeen(monkeypatch, delayed_run):
    runtime = SchedulingRuntime()
    adapter = stub_adapter(monkeypatch, runtime)
    try:
        if delayed_run:
            await runtime.until(lambda: len(adapter.holders) == 17)
        with pytest.raises(ProposalCaptured) as captured:
            await adapter.run()
        proposal = captured.value.args[0]
        assert len(proposal["refs"]) == 9
        assert {int(mid[5:]) for mid, _ in adapter.resolve(proposal)} == set(range(9))
        if not delayed_run:
            assert runtime.metrics["a2_receive_validation_calls"] == 9
        await runtime.until(lambda: len(adapter.holders) == 17)
        assert tuple(proposal["refs"]) == adapter.candidate_refs
        assert len(adapter.pool) == 17  # Later references remain available.
        assert runtime.metrics["candidate_ref_count"] == 9
        assert runtime.metrics["candidate_freeze_seconds"] >= 0
    finally:
        await runtime.close()


@pytest.mark.parametrize("invalid", [tuple(range(8)), tuple(range(9))])
@async_test
async def test_invalid_contributions_do_not_count_toward_frozen_threshold(monkeypatch, invalid):
    runtime = SchedulingRuntime()
    adapter = stub_adapter(monkeypatch, runtime, invalid=invalid)
    try:
        await runtime.until(lambda: runtime.metrics["a2_receive_validation_calls"] == 17)
        assert len(adapter.pool) == 17
        assert runtime.metrics["invalid_a2_contributions"] == len(invalid)
        if len(invalid) == 8:
            with pytest.raises(ProposalCaptured) as captured:
                await adapter.run()
            entries = adapter.resolve(captured.value.args[0])
            assert {int(mid[5:]) for mid, _ in entries} == set(range(8, 17))
        else:
            assert adapter.candidate_refs is None
    finally:
        await runtime.close()


@async_test
async def test_sender_mismatch_does_not_supply_a_second_valid_holder(monkeypatch):
    runtime = SchedulingRuntime()
    adapter = stub_adapter(monkeypatch, runtime, invalid=tuple(range(8)))
    adapter.broadcast.delivered[16] = {"payload": b"8".hex()}
    try:
        await runtime.until(lambda: runtime.metrics["a2_receive_validation_calls"] == 17)
        assert len(adapter.holders) == 8
        assert adapter.candidate_refs is None
    finally:
        await runtime.close()


@async_test
async def test_fixture_retains_all_input_policy(monkeypatch):
    runtime = SchedulingRuntime()
    adapter = stub_adapter(monkeypatch, runtime, fixture=True)
    try:
        with pytest.raises(ProposalCaptured) as captured:
            await adapter.run()
        assert len(captured.value.args[0]["refs"]) == 17
        assert runtime.metrics["candidate_required_contributions"] == 17
    finally:
        await runtime.close()


@async_test
async def test_real_pvss_late_superset_is_fully_validated_after_candidate_freeze(
        monkeypatch, a2dpss_acs_contract_setup):
    fixture = a2dpss_acs_contract_setup
    runtime = SchedulingRuntime()
    monkeypatch.setattr(a2dpss, "CommitteeBroadcast", BroadcastProbe)
    monkeypatch.setattr(a2dpss, "NWA", AgreementProbe)
    adapter = a2dpss.A2DPSSAdapter(runtime, fixture["context"])
    try:
        adapter.broadcast.delivered.update({
            i: {"payload": fixture["entries"][i][1].hex()} for i in (10, 11)})
        runtime.pulse()
        with pytest.raises(ProposalCaptured) as captured:
            await adapter.run()
        assert len(captured.value.args[0]["refs"]) == 2
        frozen = adapter.candidate_refs
        adapter.broadcast.delivered[12] = {"payload": fixture["entries"][12][1].hex()}
        runtime.pulse()
        await runtime.until(lambda: len(adapter.holders) == 3)
        superset = {"qid": adapter.qid, "refs": sorted(adapter.holders.values())}
        assert adapter.valid(superset) is True
        assert adapter.candidate_refs == frozen
        # A third, anchor-invalid contribution must still be checked: validating
        # only the first two entries of a received superset would accept this.
        mid, payload = fixture["wrong_anchor_entry"]
        bad_ref = adapter.add(mid, payload)
        bad = {"qid": adapter.qid, "refs": sorted(list(frozen) + [bad_ref])}
        assert adapter.valid(bad) is False
    finally:
        await runtime.close()


def stub_election(monkeypatch, runtime, predicate):
    monkeypatch.setattr(election, "CommitteeBroadcast", BroadcastProbe)
    monkeypatch.setattr(election, "CommitteeAVSS", lambda runtime, view: SimpleNamespace(payloads={}))
    return election.ProposalElection(runtime, 1, predicate)


@async_test
async def test_gather_starts_at_twenty_six_before_validating_twenty_seventh(monkeypatch):
    runtime = SchedulingRuntime()
    calls = []

    def valid(proposal):
        if len(calls) == 26:
            assert pe.gather.propose.sent == [list(range(26))]
        calls.append(proposal)
        return True

    pe = stub_election(monkeypatch, runtime, valid)
    pe.avss.payloads.update({i: {"proposal": i} for i in range(38)})
    try:
        await runtime.until(lambda: len(pe.accepted) == 38)
        assert pe.gather.propose.sent == [list(range(26))]
        assert pe.gather.initial_ids == tuple(range(26))
        assert calls == list(range(38))
        assert runtime.metrics["gather/1/initial_ids_count"] == 26
        assert runtime.metrics["pe/1/threshold_seconds"] >= 0
        assert runtime.metrics["pe/1/gather_start_delay_seconds"] >= 0
        # Other Gather participants may include the 12 later dealers. They
        # must remain valid, and the final union must not be truncated to h.
        all_ids = list(range(38))
        pe.gather.propose.delivered.update({i: all_ids for i in range(26)})
        pe.gather.aggregate.delivered.update({i: list(range(26)) for i in range(26)})
        runtime.pulse()
        await runtime.until(lambda: pe.gather.output is not None)
        assert pe.gather.output == all_ids
        assert pe.gather.verify(all_ids) is True
        assert pe.gather.verify(list(range(26))) is None
    finally:
        await runtime.close()


@async_test
async def test_pending_and_invalid_proposals_wait_for_real_threshold_then_keep_serving(monkeypatch):
    runtime = SchedulingRuntime()
    available = set(range(25))
    calls = defaultdict(int)

    def valid(proposal):
        calls[proposal] += 1
        return False if proposal == 38 else (True if proposal in available else None)

    pe = stub_election(monkeypatch, runtime, valid)
    pe.avss.payloads.update({i: {"proposal": i} for i in range(39)})
    try:
        await runtime.until(lambda: 38 in pe.invalid)
        assert len(pe.accepted) == 25
        assert pe.gather.propose.sent == []
        available.add(25)
        runtime.pulse()
        await runtime.until(lambda: len(pe.accepted) == 26)
        assert pe.gather.propose.sent == [list(range(26))]
        # Continue resolving pending dependencies even after Gather output.
        pe.gather.output = list(range(26))
        available.update(range(38))
        runtime.pulse()
        await runtime.until(lambda: len(pe.accepted) == 38)
        assert pe.gather.propose.sent == [list(range(26))]
        assert calls[0] == calls[38] == 1
        assert calls[37] >= 2
    finally:
        await runtime.close()


@async_test
async def test_gather_threshold_trigger_does_not_depend_on_yielding(monkeypatch):
    """Starve the Gather service deliberately: PE must invoke it explicitly."""
    runtime = SchedulingRuntime()
    pe = stub_election(monkeypatch, runtime, lambda proposal: True)
    try:
        # No tasks have run yet. Replace the initial spawned coroutines with
        # one direct validator and hold it at the first end-of-pass wait.
        await runtime.close()
        runtime.closed = False
        pe.avss.payloads.update({i: {"proposal": i} for i in range(38)})

        async def no_yield(_):
            return

        monkeypatch.setattr(election.asyncio, "sleep", no_yield)
        runtime.spawn(pe._validate())
        await runtime.until(lambda: len(pe.accepted) == 38)
        assert pe.gather.propose.sent == [list(range(26))]
    finally:
        await runtime.close()


@async_test
async def test_avss_yields_valid_payloads_to_pe_before_finishing_share_batch(monkeypatch):
    runtime = SchedulingRuntime()
    monkeypatch.setattr(avss, "CommitteeBroadcast", BroadcastProbe)
    monkeypatch.setattr(election, "CommitteeBroadcast", BroadcastProbe)
    monkeypatch.setattr(avss, "PVSSTranscript", SimpleNamespace(
        from_bytes=lambda encoded: SimpleNamespace(dealer_id=int(encoded), dealer_eval_point=int(encoded) + 1)))
    monkeypatch.setattr(avss, "pvss_verify", lambda *args: True)
    pe = election.ProposalElection(runtime, 1, lambda proposal: True)
    ids = list(range(26))
    pe.avss.setups.update({i: {"public": "stubbed"} for i in ids})
    pe.avss.share_rbc.delivered.update({
        i: {"ids": ids, "transcript": i, "proposal": i} for i in range(38)})
    observed = []
    broadcast = pe.gather.propose.broadcast

    def record(value):
        observed.append(len(pe.avss.payloads))
        broadcast(value)

    pe.gather.propose.broadcast = record
    try:
        await runtime.until(lambda: len(pe.accepted) == 38)
        assert observed == [26]
        assert len(pe.avss.payloads) == 38
        assert pe.gather.propose.sent == [list(range(26))]
    finally:
        await runtime.close()
