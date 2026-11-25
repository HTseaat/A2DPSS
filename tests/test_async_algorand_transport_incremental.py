"""Scheduling/counting gates with explicit fake crypto; real crypto gates are separate."""
import asyncio
import functools

import pytest

from beaver.async_algorand import transport
from beaver.async_algorand.model import Parameters, canonical, decode


def async_test(function):
    @functools.wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


class FakeSigner:
    def __init__(self, session, node, seed):
        self.node = node

    def seal(self, topic, body):
        return {"topic": topic, "body": body, "signed": {"node": self.node}}


class FakeVerifier:
    def __init__(self, session, public):
        pass

    def envelope(self, message):
        return message.get("authenticated", True)


class FakeCommittees:
    def __init__(self, *args):
        self.calls = []

    def member(self, role):
        return True, True

    def verify(self, role, sender, proof):
        self.calls.append((role, sender, proof))
        return proof is True


class Harness:
    def __init__(self, monkeypatch):
        monkeypatch.setattr(transport, "Signer", FakeSigner)
        monkeypatch.setattr(transport, "Verifier", FakeVerifier)
        monkeypatch.setattr(transport, "Committees", FakeCommittees)
        self.queue = asyncio.Queue()
        self.sent = []
        self.runtime = transport.Runtime(
            "incremental-test", 6, Parameters(7, 5, 4), {"signing": "fake"},
            [{} for _ in range(7)], "unused-g", "unused-h",
            lambda peer, wire: self.sent.append((peer, decode(wire))),
            self.queue.get,
        ).start()

    def wire(self, sender, topic, value, membership=True, authenticated=True, fork=0):
        return canonical({"topic": topic, "body": {"value": value, "membership": membership},
                          "signed": {"node": sender}, "authenticated": authenticated, "fork": fork})

    async def receive(self, sender, topic, value, **kwargs):
        wire = self.wire(sender, topic, value, **kwargs)
        self.queue.put_nowait((sender, wire))
        await self.settle()
        return wire

    async def settle(self):
        # Drain runnable receive/instance/done callbacks without wall-clock sleeps.
        for _ in range(5):
            await asyncio.sleep(0)

    def emitted(self, topic):
        return [message["body"]["value"] for peer, message in self.sent
                if peer == 0 and message["topic"] == topic]


@async_test
async def test_unrelated_messages_and_duplicate_wires_do_not_rescan_rbc(monkeypatch):
    harness = Harness(monkeypatch)
    runtime = harness.runtime
    calls = []
    original = runtime.messages_since

    def observed(topic, cursor):
        messages, end = original(topic, cursor)
        calls.append((topic, len(messages)))
        return messages, end

    monkeypatch.setattr(runtime, "messages_since", observed)
    try:
        rbc = transport.CommitteeBroadcast(runtime, "test", "source")
        await harness.settle()
        calls.clear()
        await harness.receive(0, "unrelated/topic", {"large": "payload"})
        assert calls == []

        topic = rbc.topic(0, "echo")
        first = await harness.receive(1, topic, {"candidate": "A"})
        assert len(calls) == 3  # Only source 0, not all seven instances.
        assert sum(count for _, count in calls) == 1
        assert runtime.committees.calls == [(topic, 1, True)]
        calls.clear()
        event = runtime.event
        # No dispatch shortcut: duplicate uses the actual receive_loop.
        harness.queue.put_nowait((1, first))
        await harness.settle()
        assert calls == []
        assert runtime.event is event
        assert len(runtime._topic_messages[topic]) == 1

        await harness.receive(2, topic, {"candidate": "A"})
        assert sum(count for _, count in calls) == 1
        assert runtime.committees.calls == [(topic, 1, True), (topic, 2, True)]
        calls.clear()
        runtime.pulse()  # Application progress must not wake unrelated RBCs.
        await harness.settle()
        assert calls == []
    finally:
        await runtime.close()
    assert not runtime._topic_subscribers


@async_test
async def test_preexisting_votes_deliver_without_val_and_unsubscribe(monkeypatch):
    harness = Harness(monkeypatch)
    runtime = harness.runtime
    topic = "rbc/preexisting/0/vote"
    try:
        for sender in range(4):
            await harness.receive(sender, topic, {"chosen": 7})
        rbc = transport.CommitteeBroadcast(runtime, "preexisting", "source")
        await asyncio.wait_for(runtime.wait(lambda: 0 in rbc.delivered), 1)
        await harness.settle()
        assert rbc.delivered[0] == {"chosen": 7}
        assert harness.emitted(topic) == [{"chosen": 7}]
        assert runtime.metrics["rbc_deliveries"] == 1
        assert all(rbc.topic(0, phase) not in runtime._topic_subscribers
                   for phase in ("val", "echo", "vote"))
    finally:
        await runtime.close()
    assert not runtime._topic_subscribers


@async_test
async def test_out_of_order_forks_and_duplicates_preserve_sender_quorums(monkeypatch):
    harness = Harness(monkeypatch)
    runtime = harness.runtime
    value_a, value_b = {"candidate": "A"}, {"candidate": "B"}
    try:
        rbc = transport.CommitteeBroadcast(runtime, "forks", "source")
        topic = rbc.topic(0, "vote")
        first = await harness.receive(0, topic, value_a)
        await harness.receive(0, topic, value_b, fork=1)
        # A third authenticated fork is ignored, like the original inbox.
        await harness.receive(0, topic, value_a, fork=2)
        harness.queue.put_nowait((0, first))
        await harness.settle()
        assert len(runtime._topic_messages[topic]) == 2
        assert runtime.committees.calls.count((topic, 0, True)) == 2
        assert not harness.emitted(topic)

        await harness.receive(1, topic, value_a)
        await harness.receive(2, topic, value_a, membership=False)
        assert not harness.emitted(topic)
        await harness.receive(2, topic, value_a, membership=True, fork=1)
        assert harness.emitted(topic) == [value_a]  # reconstruct=3, before h=4.
        assert 0 not in rbc.delivered
        await harness.receive(3, topic, value_a)
        assert rbc.delivered[0] == value_a
        assert runtime.metrics["rbc_deliveries"] == 1
    finally:
        await runtime.close()


@async_test
async def test_only_source_val_echoes_and_echo_quorum_votes_once(monkeypatch):
    harness = Harness(monkeypatch)
    runtime = harness.runtime
    try:
        rbc = transport.CommitteeBroadcast(runtime, "val", "source")
        val, echo, vote = (rbc.topic(0, phase) for phase in ("val", "echo", "vote"))
        await harness.receive(1, val, "wrong source")
        await harness.receive(0, val, "invalid membership", membership=False)
        assert harness.emitted(echo) == []
        await harness.receive(0, val, "accepted", membership=True, fork=1)
        assert harness.emitted(echo) == ["accepted"]
        for sender in range(3):
            await harness.receive(sender, echo, "accepted")
        assert harness.emitted(vote) == []
        # Distinct authenticated forks for the same value still count once.
        await harness.receive(0, echo, "accepted", fork=1)
        assert harness.emitted(vote) == []
        await harness.receive(3, echo, "accepted")
        await harness.receive(4, echo, "accepted")
        assert harness.emitted(vote) == ["accepted"]
        assert not rbc.delivered
    finally:
        await runtime.close()


@pytest.mark.parametrize("phase", ["val", "echo", "vote"])
@async_test
async def test_committee_limit_across_batches_and_values_propagates_failure(monkeypatch, phase):
    harness = Harness(monkeypatch)
    runtime = harness.runtime
    try:
        rbc = transport.CommitteeBroadcast(runtime, "limit", "source")
        topic = rbc.topic(0, phase)
        for sender in range(5):
            await harness.receive(sender, topic, {"different": sender})
            assert not runtime.failure.done()
        await harness.receive(5, topic, {"different": 5})
        with pytest.raises(RuntimeError, match="committee exceeds"):
            await asyncio.wait_for(runtime.wait(lambda: False), 1)
        assert 0 not in rbc.delivered
        assert topic not in runtime._topic_subscribers
    finally:
        await runtime.close()
    assert not runtime._topic_subscribers


@async_test
async def test_invalid_envelope_never_enters_topic_log(monkeypatch):
    harness = Harness(monkeypatch)
    runtime = harness.runtime
    try:
        rbc = transport.CommitteeBroadcast(runtime, "invalid", "source")
        await harness.settle()
        event = runtime.event
        await harness.receive(0, rbc.topic(0, "val"), "untrusted", authenticated=False)
        assert runtime.metrics["rejected_messages"] == 1
        assert runtime.event is event
        assert not runtime._topic_messages
        assert not runtime.committees.calls
        assert not harness.sent
    finally:
        await runtime.close()
