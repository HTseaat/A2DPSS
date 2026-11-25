"""Authenticated multicast and Algorithm 4 committee broadcast.

Uses the project's send/recv interface and VRF, with independent state. The
full-value ECHO/VOTE wire representation matches the existing YOSO RBC style.
No timer changes protocol state or manufactures a missing quorum.
"""
import asyncio
from collections import defaultdict
from .model import canonical, decode, digest
from .crypto import Signer, Verifier, Committees


class Runtime:
    def __init__(self, session, node, parameters, private, public, g, h, send, recv):
        self.session, self.node, self.p = session, node, parameters
        self.private, self.public, self.g, self.h = private, public, g, h
        self.send, self.recv = send, recv
        # Consume the provisioned root seed; keeping it next to the ratchet
        # would make even logical forward erasure meaningless.
        signing_seed = private.pop("signing")
        self.signer = Signer(session, node, signing_seed)
        del signing_seed
        self.verifier = Verifier(session, public)
        self.committees = Committees(session, node, parameters, private, public, g)
        self.messages = defaultdict(dict)
        # Append only authenticated, previously unseen envelopes. The legacy
        # inbox remains available to protocols that need a complete snapshot.
        self._topic_messages = defaultdict(list)
        self._topic_subscribers = defaultdict(set)
        self.event = asyncio.Event()
        self.tasks = set()
        self.failure = asyncio.get_running_loop().create_future()
        self.metrics = defaultdict(int)
        self.closed = False
        self.sender_topics = defaultdict(set)

    def pulse(self):
        old, self.event = self.event, asyncio.Event()
        old.set()

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        def done(completed):
            self.tasks.discard(completed)
            if not completed.cancelled() and completed.exception() is not None:
                if not self.failure.done():
                    self.failure.set_result(completed.exception())
                self.pulse()
        task.add_done_callback(done)
        return task

    async def wait(self, predicate):
        while True:
            event = self.event
            if self.failure.done():
                raise self.failure.result()
            value = predicate()
            if value:
                return value
            await event.wait()

    async def receive_loop(self):
        while True:
            sender, wire = await self.recv()
            self.metrics["received_bytes"] += len(wire) if isinstance(wire, bytes) else 0
            try:
                if not isinstance(wire, bytes) or len(wire) > 64 * 1024 * 1024:
                    raise ValueError("invalid wire size")
                message = decode(wire)
                if message["signed"]["node"] != sender or not self.verifier.envelope(message):
                    raise ValueError("invalid authenticated envelope")
                if not isinstance(message["body"], dict) or set(message["body"]) != {"value", "membership"}:
                    raise ValueError("invalid multicast envelope body")
                topic = message["topic"]
                if not isinstance(topic, str) or len(topic) > 512:
                    raise ValueError("invalid topic")
                topics = self.sender_topics[sender]
                if topic not in topics and len(topics) >= 64 * self.p.max_views * self.p.population + 32:
                    raise ValueError("sender exceeded configured topic budget")
                topics.add(topic)
                # Keep two forks for evidence, but never count one sender twice.
                bucket = self.messages[topic].setdefault(sender, {})
                if len(bucket) < 2:
                    message_id = digest(message)
                    if message_id not in bucket:
                        bucket[message_id] = message
                        self._topic_messages[topic].append((sender, message))
                        for event in self._topic_subscribers.get(topic, ()):
                            event.set()
                        self.pulse()
            except (ValueError, KeyError, TypeError, RecursionError):
                self.metrics["rejected_messages"] += 1

    def start(self):
        self.spawn(self.receive_loop())
        return self

    def multicast(self, topic, value, role=None):
        membership = None
        if role is not None:
            selected, membership = self.committees.member(role)
            if not selected:
                return None
        message = self.signer.seal(topic, {"value": value, "membership": membership})
        wire = canonical(message)
        self.metrics["sent_messages"] += self.p.population
        self.metrics["sent_bytes"] += len(wire) * self.p.population
        self.metrics["sent_bytes/" + topic.split("/")[0]] += len(wire) * self.p.population
        for peer in range(self.p.population):
            self.send(peer, wire)
        return message

    def inbox(self, topic, role=None):
        result = []
        for sender, forks in self.messages.get(topic, {}).items():
            for envelope in forks.values():
                if role is None or self.committees.verify(
                        role, sender, envelope["body"]["membership"]):
                    result.append((sender, envelope["body"]["value"], envelope))
        if role is not None and len({sender for sender, _, _ in result}) > self.p.m:
            raise RuntimeError("observed committee exceeds configured m; conditioned quorum bounds failed")
        return result

    def subscribe_topics(self, topics):
        """Wake one consumer only when one of its topics receives new data."""
        event = asyncio.Event()
        for topic in topics:
            self._topic_subscribers[topic].add(event)
        return event

    def unsubscribe_topics(self, topics, event):
        for topic in topics:
            subscribers = self._topic_subscribers.get(topic)
            if subscribers is not None:
                subscribers.discard(event)
                if not subscribers:
                    del self._topic_subscribers[topic]

    def messages_since(self, topic, cursor):
        """Return new authenticated envelopes, including pre-subscription data."""
        messages = self._topic_messages.get(topic, ())
        return messages[cursor:], len(messages)

    async def close(self):
        self.closed = True
        pending = list(self.tasks)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


class CommitteeBroadcast:
    """One sender per instance; all instances remain separate from NWA stages."""
    def __init__(self, runtime, tag, source_role):
        self.r, self.tag, self.source_role = runtime, tag, source_role
        self.delivered = {}
        for source in range(runtime.p.population):
            runtime.spawn(self._instance(source))

    def topic(self, source, phase):
        return "rbc/%s/%d/%s" % (self.tag, source, phase)

    def role(self, source, phase):
        return self.source_role if phase == "val" else self.topic(source, phase)

    def broadcast(self, value):
        return self.r.multicast(self.topic(self.r.node, "val"), value, self.source_role)

    async def _instance(self, source):
        echoed = voted = False
        phases = ("val", "echo", "vote")
        topics = {phase: self.topic(source, phase) for phase in phases}
        cursors = dict.fromkeys(phases, 0)
        # Membership bounds apply across batches and conflicting values, not
        # separately to each candidate digest.
        members = {phase: set() for phase in phases}
        groups = {}
        event = self.r.subscribe_topics(topics.values())
        try:
            while True:
                # There is no await between clearing and draining the logs, so
                # arrivals cannot be lost at the transition to event.wait().
                event.clear()
                for phase in phases:
                    role = self.role(source, phase)
                    messages, cursors[phase] = self.r.messages_since(topics[phase], cursors[phase])
                    accepted = []
                    for sender, envelope in messages:
                        if role is not None and not self.r.committees.verify(
                                role, sender, envelope["body"]["membership"]):
                            continue
                        members[phase].add(sender)
                        accepted.append((sender, envelope["body"]["value"]))
                    if role is not None and len(members[phase]) > self.r.p.m:
                        raise RuntimeError("observed committee exceeds configured m; conditioned quorum bounds failed")
                    for sender, value in accepted:
                        if phase == "val":
                            if sender == source and not echoed:
                                self.r.multicast(topics["echo"], value, self.role(source, "echo"))
                                echoed = True
                        else:
                            group = groups.setdefault(digest(value), {"value": value, "echo": set(), "vote": set()})
                            group[phase].add(sender)
                for group in groups.values():
                    if not voted and (len(group["echo"]) >= self.r.p.h or
                                      len(group["vote"]) >= self.r.p.reconstruct):
                        self.r.multicast(topics["vote"], group["value"], self.role(source, "vote"))
                        voted = True
                    if len(group["vote"]) >= self.r.p.h:
                        self.delivered[source] = group["value"]
                        self.r.metrics["rbc_deliveries"] += 1
                        self.r.pulse()
                        return
                await event.wait()
        finally:
            self.r.unsubscribe_topics(topics.values(), event)
