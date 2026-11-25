"""Independent protocol gates; all election cryptography is real unless named scripted."""
import asyncio
import base64
import copy
import json
import functools
import hashlib
import time
from pathlib import Path
import pytest
from beaver.async_algorand.model import Parameters, canonical, decode
from beaver.async_algorand.crypto import (Signer, Verifier, generate_identity,
    encrypt_recipient, decrypt_recipient)
from beaver.async_algorand.transport import Runtime, CommitteeBroadcast
from beaver.async_algorand.nwa import NWA
from beaver.async_algorand.election import Gather


def async_test(function):
    @functools.wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


def generators():
    config = json.loads(Path("conf/admpc_100_4_4/local.0.json").read_text())
    srs = json.loads(base64.b64decode(config["extra"]["SRS"]))["Pk"]
    def raw(point):
        return (b"\x04" + int(point["X"]).to_bytes(48, "big") + int(point["Y"]).to_bytes(48, "big")).hex()
    return raw(srs["G1_g"][0]), raw(srs["G1_h"][0])


def network(count=5, parameters=None, transform=None):
    g, h = generators()
    identities = [generate_identity(g) for _ in range(count)]
    public = [p for _, p in identities]
    queues = [asyncio.Queue() for _ in range(count)]
    runtimes = []
    for node in range(count):
        def send(peer, wire, sender=node):
            if transform is None or transform(sender, peer, wire):
                queues[peer].put_nowait((sender, wire))
        runtimes.append(Runtime("test-session", node, parameters or Parameters(count, count, count - 1),
                                identities[node][0], public, g, h, send, queues[node].get).start())
    return runtimes


def test_parameters_wire_and_signature_replay():
    with pytest.raises(ValueError, match="rounded"):
        Parameters(4, 4, 3)
    with pytest.raises(ValueError):
        decode(b'{"x":1,"x":2}')
    with pytest.raises(ValueError):
        canonical(float("nan"))
    g, _ = generators()
    private, public = generate_identity(g)
    signer = Signer("session-a", 0, private["signing"])
    verifier = Verifier("session-a", [public])
    first = signer.seal("first", {"value": 1})
    second = signer.seal("second", {"value": 2})
    assert verifier.envelope(second) and verifier.envelope(first)  # out of order
    assert not Verifier("session-b", [public]).envelope(first)
    forged = copy.deepcopy(first)
    forged["body"]["value"] = 2
    assert not verifier.envelope(forged)
    forged = copy.deepcopy(second)
    forged["signed"]["chain"][0]["link"]["next"] = public["signing"]
    assert not verifier.envelope(forged)
    with pytest.raises(ValueError, match="once"):
        signer.seal("first", {})
    # Simulate possession of the CURRENT key after an honest speech.
    # It must not authorize a new envelope for the already consumed role.
    link = {"topic": "first", "hash": hashlib.sha256(canonical({})).hexdigest(),
            "next": public["signing"]}
    retroactive = {"topic": "first", "body": {}, "link": link,
                   "signed": signer.attest("envelope", link)}
    assert not verifier.envelope(retroactive)


def test_recipient_ciphertext_context_and_integrity():
    g, _ = generators()
    private, public = generate_identity(g)
    other, _ = generate_identity(g)
    encrypted = encrypt_recipient(g, public["encryption"], b"hidden rank", b"view1")
    assert decrypt_recipient(private["encryption"], encrypted, b"view1") == b"hidden rank"
    for key, aad in [(other["encryption"], b"view1"), (private["encryption"], b"view2")]:
        with pytest.raises(ValueError):
            decrypt_recipient(key, encrypted, aad)


@async_test
async def test_committee_broadcast_with_one_silent_node():
    runtimes = network(transform=lambda sender, peer, wire: sender != 4)
    try:
        rbcs = [CommitteeBroadcast(r, "test", "source") for r in runtimes]
        rbcs[0].broadcast({"hello": "world"})
        await asyncio.wait_for(asyncio.gather(*(r.wait(lambda b=b: 0 in b.delivered)
                                                for r, b in zip(runtimes, rbcs))), 20)
        assert all(b.delivered[0] == {"hello": "world"} for b in rbcs)
    finally:
        await asyncio.gather(*(r.close() for r in runtimes))


@async_test
async def test_nwa_real_election_agreement_with_one_silent_node():
    runtimes = network(transform=lambda sender, peer, wire: sender != 4)
    protocols = [NWA(r, lambda value: type(value) is int and 0 <= value < 5) for r in runtimes]
    try:
        decisions = await asyncio.wait_for(asyncio.gather(*(p.run(i) for i, p in enumerate(protocols))), 240)
        assert len(set(decisions)) == 1
        assert all(r.metrics["election_pvss_deals"] >= 1 for r in runtimes)
        assert all(r.metrics["avss_rec_started"] >= 1 for r in runtimes)
    finally:
        await asyncio.gather(*(r.close() for r in runtimes))


@async_test
async def test_scripted_pe_conflict_advances_view_with_different_stage_committees():
    """Only PE is scripted: real network envelopes, VRF, certificates and NWA."""
    runtimes = network(6, Parameters(6, 5, 4, mode="fixed-size", seed="different-stages"))
    class ScriptedElection:
        def __init__(self, runtime, view, valid):
            self.r, self.view, self.valid = runtime, view, valid
        async def run(self, proposal):
            value = self.r.node % 2 if self.view == 1 else 0
            return {"proposal": {"view": 0, "value": value, "cert": []}, "ids": [value]}
        def verify(self, output):
            proposal = output.get("proposal", {})
            return (self.valid(proposal) is True and proposal["view"] == 0
                    and output["ids"] == [proposal["value"]]
                    and (self.view == 1 or proposal["value"] == 0))
    protocols = [NWA(r, lambda v: type(v) is int and v in (0, 1), ScriptedElection) for r in runtimes]
    try:
        outputs = await asyncio.wait_for(asyncio.gather(*(p.run(i % 2) for i, p in enumerate(protocols))), 30)
        assert outputs == [0] * 6
        assert all(r.metrics["nwa_failed_views"] == 1 for r in runtimes)
        assert all(p.key["view"] == 2 and p.lock["view"] == 2 for p in protocols)
        assert any(not r.committees.member(p.role(2, 2))[0] for r, p in zip(runtimes, protocols))
    finally:
        await asyncio.gather(*(r.close() for r in runtimes))


@async_test
async def test_certificate_rejects_duplicates_replay_wrong_value_stage_and_erased_role():
    runtimes = network()
    protocols = [NWA(r, lambda v: isinstance(v, int)) for r in runtimes]
    try:
        votes = [p.vote(1, 1, 7) for p in protocols[:4]]
        verifier = protocols[4]
        assert verifier.certificate_valid(1, 1, 7, votes)
        assert not verifier.certificate_valid(1, 1, 7, [votes[0]] * 4)
        assert not verifier.certificate_valid(1, 2, 7, votes)
        assert not verifier.certificate_valid(2, 1, 7, votes)
        assert not verifier.certificate_valid(1, 1, 8, votes)
        runtimes[0].multicast(protocols[0].topic(1, 1), {"test": "consume-role"})
        later = protocols[0].vote(1, 1, 7)
        assert not verifier.certificate_valid(1, 1, 7, [later] + votes[1:])
        key = {"view": 1, "value": 7, "cert": votes}
        assert verifier.key_valid(key)
        assert not verifier.key_valid(dict(key, value=8))
    finally:
        await asyncio.gather(*(r.close() for r in runtimes))


@async_test
async def test_gather_pending_dependencies_and_binding_core():
    runtimes = network()
    try:
        available = {0, 1, 2}
        gather = Gather(runtimes[0], 1, lambda i: True if i in available else None)
        assert gather.valid_set([0, 1, 2, 3]) is None
        assert gather.valid_set([0, 0, 1, 2]) is False
        available.add(3)
        assert gather.verify([0, 1, 2, 3]) is None
        gather.U = {i: {0, 1, 2, 3} for i in range(4)}
        assert gather.verify([0, 1, 2, 3]) is True
        available.add(4)
        assert gather.verify([0, 1, 2, 4]) is None
        assert gather.verify([0, 1, 2, 3, 4]) is True
    finally:
        await asyncio.gather(*(r.close() for r in runtimes))


@async_test
async def test_rbc_rejects_spoofed_designated_sender_and_bad_membership():
    runtimes = network()
    try:
        rbcs = [CommitteeBroadcast(r, "spoof", "source") for r in runtimes]
        # Node 0 is authenticated but cannot impersonate designated source 1.
        runtimes[0].multicast(rbcs[0].topic(1, "val"), {"wrong": True}, "source")
        # A source without the matching membership proof must not trigger ECHO.
        runtimes[1].multicast(rbcs[1].topic(1, "val"), {"wrong": True}, "wrong-domain")
        await asyncio.sleep(0.05)
        assert all(not b.delivered for b in rbcs)
        assert all(not r.inbox(rbcs[0].topic(1, "echo")) for r in runtimes)
    finally:
        await asyncio.gather(*(r.close() for r in runtimes))


@async_test
async def test_avss_holds_key_shares_until_rec_and_accepts_late_setup():
    from beaver.async_algorand.avss import CommitteeAVSS
    runtimes = network()
    try:
        instances = [CommitteeAVSS(r, 1) for r in runtimes]
        await asyncio.wait_for(asyncio.gather(*(a.setup() for a in instances)), 30)
        assert all(not a.keys and not a.rec_sent for a in instances)
        assert all(not any(topic.startswith("rec/") for topic in r.messages) for r in runtimes)
        for instance in instances:
            instance.reconstruct()
        await asyncio.wait_for(asyncio.gather(*(r.wait(lambda a=a: len(a.keys) == 5)
                                                for r, a in zip(runtimes, instances))), 30)
        assert all(len(a.setups) == 5 for a in instances)  # service continues after h=4
        for source in range(5):
            assert len({a.keys[source] for a in instances}) == 1
        share = runtimes[0].inbox("rec/" + instances[0].tag + "/0")[0][1]
        assert not instances[1]._valid_key_share(1, share["share"]["recipient"], share)
    finally:
        await asyncio.gather(*(r.close() for r in runtimes))


def test_a2_adapter_pending_and_all_valid_candidate_contract(a2dpss_acs_contract_setup):
    from collections import defaultdict
    from types import SimpleNamespace
    from beaver.a2dpss import A2DPSSJustifierManager
    from beaver.a2dpss_benchmark_metrics import CPU_STAGES
    from beaver.async_algorand.a2dpss import A2DPSSAdapter
    from beaver.async_algorand.model import digest
    fixture = a2dpss_acs_contract_setup
    adapter = A2DPSSAdapter.__new__(A2DPSSAdapter)
    adapter.cpu_stages = {name: {"cpu_seconds": 0.0, "invocations": 0} for name in CPU_STAGES}
    adapter.context = fixture["context"]
    adapter.qid = adapter.context.encode_qid()
    adapter.r = SimpleNamespace(pulse=lambda: None, metrics=defaultdict(int))
    adapter.justifier = A2DPSSJustifierManager([adapter.context])
    adapter.pool = {}
    entries = [fixture["entries"][i] for i in (10, 11, 12)]
    refs = [digest([mid, payload.hex()]) for mid, payload in entries]
    proposal = {"qid": adapter.qid, "refs": sorted(refs)}
    assert adapter.valid(proposal) is None
    for mid, payload in entries:
        adapter.add(mid, payload)
    assert adapter.valid(proposal) is True
    assert adapter.valid(dict(proposal, refs=[refs[0]] * 3)) is False
    assert adapter.valid(dict(proposal, qid="wrong-session")) is False
    bad = adapter.add("node_99", b"not a transcript")
    assert adapter.valid(dict(proposal, refs=sorted([refs[0], bad]))) is False


@async_test
async def test_blame_requires_verifying_pe_and_lock_evidence():
    from types import SimpleNamespace
    runtimes = network()
    protocols = [NWA(r, lambda v: isinstance(v, int)) for r in runtimes]
    try:
        protocol = protocols[4]
        protocol.elections[2] = SimpleNamespace(verify=lambda output: output.get("proof") == "valid")
        cert = [p.vote(1, 2, 7) for p in protocols[:4]]
        output = {"proposal": {"view": 0, "value": 8, "cert": []}, "proof": "valid"}
        error = {"kind": "blame", "output": output, "lock": {"view": 1, "value": 7, "cert": cert}}
        assert protocol._error_valid(2, error) is True
        bad = copy.deepcopy(error)
        bad["lock"]["value"] = 8
        assert protocol._error_valid(2, bad) is False
        bad = copy.deepcopy(error)
        bad["output"]["proof"] = "invalid"
        assert protocol._error_valid(2, bad) is False
        bad = copy.deepcopy(error)
        bad["output"]["proposal"]["view"] = 1
        assert protocol._error_valid(2, bad) is False
        future = {"kind": "blame", "output": {"proposal": {"view": 2}, "proof": "valid"},
                  "lock": {"view": 0, "value": None, "cert": []}}
        assert protocol._error_valid(2, future) is True
    finally:
        await asyncio.gather(*(r.close() for r in runtimes))


@async_test
async def test_termination_needs_distinct_members_and_valid_commit_certificate():
    runtimes = network()
    predicate_calls = []

    def valid(value):
        predicate_calls.append(value)
        return value == 7

    protocols = [NWA(r, valid) for r in runtimes]
    task = None
    try:
        certificate = [p.vote(1, 3, 7) for p in protocols[:4]]
        message = {"view": 1, "value": 7, "cert": certificate}
        task = asyncio.create_task(protocols[4]._termination())
        runtimes[0].multicast("termination", message, "termination")
        # Receiver forwards its own valid vote: still only 2 distinct members.
        await asyncio.sleep(0.1)
        assert not task.done()
        runtimes[1].multicast("termination", message, "termination")
        assert await asyncio.wait_for(task, 5) == 7
        # Algorithm 14 relies on the commit certificate rooted in Proposal
        # Election external validity; termination does not repeat valid(7).
        assert predicate_calls == []
    finally:
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(*(r.close() for r in runtimes))


@async_test
async def test_prevalidated_nwa_input_skips_only_the_initial_predicate():
    from types import SimpleNamespace

    calls = []
    protocol = NWA(
        SimpleNamespace(),
        lambda value: calls.append(value) or True,
    )

    async def decide():
        return 7

    async def never():
        await asyncio.Event().wait()

    protocol._termination = decide
    protocol._views = never

    assert await protocol.run_prevalidated(7) == 7
    assert calls == []


@async_test
async def test_malformed_multicast_and_round4_do_not_crash_receiver():
    runtimes = network()
    task = None
    try:
        bad = runtimes[0].signer.seal("bad-shape", [])
        for peer in range(5):
            runtimes[0].send(peer, canonical(bad))
        protocol = NWA(runtimes[4], lambda v: True)
        task = asyncio.create_task(protocol._termination())
        runtimes[1].multicast(protocol.topic(1, 4), ["not a round message"], protocol.role(1, 4))
        await asyncio.sleep(0.1)
        assert not task.done()
        assert runtimes[4].metrics["rejected_messages"] == 1
        assert not runtimes[4].failure.done()
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(*(r.close() for r in runtimes))


@async_test
async def test_observed_committee_overflow_fails_closed():
    runtimes = network(6, Parameters(6, 5, 4))
    try:
        for runtime in runtimes:
            runtime.multicast("oversized", True, "oversized-role")
        await asyncio.sleep(0.2)
        with pytest.raises(RuntimeError, match="exceeds configured m"):
            runtimes[0].inbox("oversized", "oversized-role")
    finally:
        await asyncio.gather(*(r.close() for r in runtimes))


@async_test
async def test_invalid_reconstructed_ranks_cannot_win_or_verify():
    from types import SimpleNamespace
    from beaver.async_algorand.election import ProposalElection
    runtimes = network()
    try:
        role = runtimes[0].p.role(1, "rank")
        rank = runtimes[0].committees.rank(role)
        assert runtimes[1].committees.verify_rank(role, 0, rank)
        assert not runtimes[1].committees.verify_rank(role, 1, rank)
        assert not runtimes[1].committees.verify_rank("v2/rank", 0, rank)
        altered = dict(rank, rank=rank["rank"] + 1)
        assert not runtimes[1].committees.verify_rank(role, 0, altered)
        pe = ProposalElection.__new__(ProposalElection)
        pe.r, pe.rank_role = runtimes[1], role
        pe.avss = SimpleNamespace(recovered={0: rank, 1: None, 2: altered, 3: {"rank": 2 ** 256}})
        pe.accepted = {0: "valid proposal", 1: "invalid rank", 2: "tampered", 3: "unproved"}
        pe.gather = SimpleNamespace(verify=lambda ids: True)
        assert pe._winner([0, 1, 2, 3]) == 0
        assert pe.verify({"ids": [0, 1, 2, 3], "proposal": "valid proposal"}) is True
        assert pe.verify({"ids": [0, 1, 2, 3], "proposal": "unproved"}) is False
    finally:
        await asyncio.gather(*(r.close() for r in runtimes))


@async_test
async def test_a2_received_invalid_transcript_is_rejected_instead_of_pending(a2dpss_acs_contract_setup):
    from types import SimpleNamespace
    from beaver.a2dpss import A2DPSSJustifierManager
    from beaver.a2dpss_benchmark_metrics import CPU_STAGES
    from beaver.async_algorand.a2dpss import A2DPSSAdapter
    from beaver.async_algorand.model import digest
    runtimes = network()
    try:
        adapter = A2DPSSAdapter.__new__(A2DPSSAdapter)
        adapter.cpu_stages = {name: {"cpu_seconds": 0.0, "invocations": 0} for name in CPU_STAGES}
        adapter.r, adapter.context = runtimes[0], a2dpss_acs_contract_setup["context"]
        adapter.qid = adapter.context.encode_qid()
        adapter.justifier = A2DPSSJustifierManager([adapter.context])
        adapter.pool, adapter.holders = {}, {}
        adapter.required = adapter.context.service_t + 1
        adapter.candidate_refs = None
        adapter.created_at = time.perf_counter()
        valid_payload = a2dpss_acs_contract_setup["entries"][10][1]
        bad_payload = b"received but not a valid A2DPSS contribution"
        valid_ref = digest(["node_10", valid_payload.hex()])
        bad_ref = digest(["node_11", bad_payload.hex()])
        adapter.broadcast = SimpleNamespace(delivered={10: {"payload": valid_payload.hex()}})
        runtimes[0].spawn(adapter._receive())
        proposal = {"qid": adapter.qid, "refs": sorted([valid_ref, bad_ref])}
        assert adapter.valid(proposal) is None
        await asyncio.wait_for(runtimes[0].wait(lambda: 10 in adapter.holders), 20)
        assert adapter.valid(proposal) is None  # the second transcript has not arrived
        adapter.broadcast.delivered[11] = {"payload": bad_payload.hex()}
        runtimes[0].pulse()
        await asyncio.wait_for(runtimes[0].wait(lambda: runtimes[0].metrics["invalid_a2_contributions"] == 1), 5)
        assert adapter.valid(proposal) is False  # availability is now known, validity is false
        assert 11 not in adapter.holders
        assert adapter.valid({"qid": adapter.qid, "refs": sorted([valid_ref, "z" * 64])}) is False
    finally:
        await asyncio.gather(*(r.close() for r in runtimes))
