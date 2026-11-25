"""Single-decision NWA, Algorithms 13–23.

Stage certificates contain h detached, domain-bound endorsements, as in the
paper. Their signers validate the preceding evidence before endorsing. This
avoids recursively duplicating the entire history inside each certificate.
"""
import asyncio
import time
from .model import digest
from .election import ProposalElection


class NWA:
    def __init__(self, runtime, valid, election_factory=ProposalElection):
        self.r, self.valid, self.election_factory = runtime, valid, election_factory
        self.elections = {}
        self.key = self.lock = None
        self.view = 0
        self.cert_cache = {}

    def role(self, view, stage):
        return self.r.p.role(view, "round%d" % stage)

    def topic(self, view, stage):
        return "nwa/" + self.role(view, stage)

    def vote(self, view, stage, value):
        selected, proof = self.r.committees.member(self.role(view, stage))
        if not selected:
            return None
        body = {"view": view, "stage": stage, "value": value, "membership": proof}
        return {"body": body, "signed": self.r.signer.attest("nwa-vote", body)}

    def certificate_valid(self, stage, view, value, votes):
        try:
            if type(view) is not int or not 1 <= view <= self.r.p.max_views:
                return False
            if not isinstance(votes, list) or not self.r.p.h <= len(votes) <= self.r.p.m:
                return False
            cache = (stage, view, digest(value), digest(votes))
            if cache in self.cert_cache:
                return self.cert_cache[cache]
            seen = set()
            for vote in votes:
                body, signed = vote["body"], vote["signed"]
                sender = signed["node"]
                if (sender in seen or body["view"] != view or body["stage"] != stage
                        or digest(body["value"]) != digest(value)
                        or not self.r.verifier.verify("nwa-vote", body, signed)
                        or not self.r.committees.verify(self.role(view, stage), sender, body["membership"])):
                    return False
                if any(link["link"]["topic"] == self.topic(view, stage) for link in signed["chain"]):
                    return False  # a later key cannot retroactively endorse an erased role
                seen.add(sender)
            self.cert_cache[cache] = True
            return True
        except (KeyError, ValueError, TypeError):
            return False

    def key_valid(self, key):
        try:
            view = key["view"]
            if type(view) is not int or view < 0:
                return False
            if view == 0:
                return self.valid(key["value"]) if key["cert"] == [] else False
            if not self.certificate_valid(1, view, key["value"], key["cert"]):
                return False
            return self.valid(key["value"])
        except (KeyError, ValueError, TypeError):
            return False

    def _error_valid(self, view, error):
        try:
            pe = self.elections[view]
            if error["kind"] == "equivocate":
                left, right = error["left"], error["right"]
                if digest(left["proposal"]) == digest(right["proposal"]):
                    return False
                statuses = [pe.verify(left), pe.verify(right)]
                return False if False in statuses else (True if all(x is True for x in statuses) else None)
            if error["kind"] != "blame":
                return False
            output, lock = error["output"], error["lock"]
            key_view = output["proposal"]["view"]
            if not (key_view >= view or key_view < lock["view"]):
                return False
            if lock["view"] == 0:
                if lock != {"view": 0, "value": None, "cert": []}:
                    return False
            elif not (0 < lock["view"] <= view and
                      self.certificate_valid(2, lock["view"], lock["value"], lock["cert"])):
                return False
            return pe.verify(output)
        except (KeyError, ValueError, TypeError):
            return False

    def message_valid(self, sender, view, stage, message):
        try:
            if message["kind"] == "error":
                return self._error_valid(view, message["error"])
            kind = {1: "echo", 2: "key", 3: "lock", 4: "commit"}[stage]
            if message["kind"] != kind:
                return False
            value, vote = message["value"], message["vote"]
            body = vote["body"]
            if (vote["signed"]["node"] != sender or body["view"] != view
                    or body["stage"] != stage or digest(body["value"]) != digest(value)
                    or not self.r.verifier.verify("nwa-vote", body, vote["signed"])
                    or not self.r.committees.verify(self.role(view, stage), sender, body["membership"])):
                return False
            if any(link["link"]["topic"] == self.topic(view, stage) for link in vote["signed"]["chain"]):
                return False
            if stage == 1:
                if digest(message["output"]["proposal"]["value"]) != digest(value):
                    return False
                return self.elections[view].verify(message["output"])
            return self.certificate_valid(stage - 1, view, value, message["cert"])
        except (KeyError, ValueError, TypeError):
            return False

    def _send(self, view, stage, message):
        if not self.r.committees.member(self.role(view, stage))[0]:
            return
        if message["kind"] != "error":
            message = dict(message, vote=self.vote(view, stage, message["value"]))
        self.r.multicast(self.topic(view, stage), message, self.role(view, stage))

    async def _quorum(self, view, stage):
        accepted = {}
        invalid = set()
        def ready():
            for sender, message, envelope in self.r.inbox(self.topic(view, stage), self.role(view, stage)):
                message_id = digest(envelope)
                if sender in accepted or message_id in invalid:
                    continue
                status = self.message_valid(sender, view, stage, message)
                if status is False:
                    invalid.add(message_id)
                if status is not True:
                    continue
                accepted[sender] = message
                # Observers must perform these updates even if never selected.
                if message["kind"] == "key" and self.key["view"] < view:
                    self.key = {"view": view, "value": message["value"], "cert": message["cert"]}
                if message["kind"] == "lock" and self.lock["view"] < view:
                    self.lock = {"view": view, "value": message["value"], "cert": message["cert"]}
                if len(accepted) >= self.r.p.h:
                    return list(accepted.values())
            return None
        return await self.r.wait(ready)

    def _next(self, view, stage, messages):
        for message in messages:
            if message["kind"] == "error":
                return {"kind": "error", "error": message["error"]}
        first = messages[0]
        for other in messages[1:]:
            if stage == 1 and digest(first["output"]["proposal"]) != digest(other["output"]["proposal"]):
                return {"kind": "error", "error": {"kind": "equivocate", "left": first["output"],
                                                       "right": other["output"]}}
            if digest(first["value"]) != digest(other["value"]):
                raise RuntimeError("conflicting quorum certificates: committee security assumption violated")
        return {"kind": {1: "key", 2: "lock", 3: "commit"}[stage],
                "value": first["value"], "cert": [m["vote"] for m in messages]}

    async def _views(self):
        for view in range(1, self.r.p.max_views + 1):
            self.view = view
            self.r.metrics["nwa_views_started"] = view
            started = time.perf_counter()
            pe = self.election_factory(self.r, view, self.key_valid)
            self.elections[view] = pe
            output = await pe.run(self.key)
            self.r.metrics["proposal_election_seconds"] += time.perf_counter() - started
            proposed = output["proposal"]
            if self.lock["view"] <= proposed["view"] < view:
                message = {"kind": "echo", "value": proposed["value"], "output": output}
            else:
                message = {"kind": "error", "error": {"kind": "blame", "output": output, "lock": self.lock}}
            self._send(view, 1, message)
            for stage in (1, 2, 3):
                messages = await self._quorum(view, stage)
                self._send(view, stage + 1, self._next(view, stage, messages))
            # A single valid round-4 error suffices, AFTER reading earlier quorums.
            def failed_view():
                for sender, message, _ in self.r.inbox(self.topic(view, 4), self.role(view, 4)):
                    if isinstance(message, dict) and message.get("kind") == "error" and self.message_valid(sender, view, 4, message) is True:
                        return True
                return False
            await self.r.wait(failed_view)
            self.r.metrics["nwa_failed_views"] += 1
        raise RuntimeError("configured view budget exhausted without decision")

    async def _termination(self):
        forwarded = False
        while True:
            event = self.r.event
            candidates = []
            for view in range(1, self.r.p.max_views + 1):
                for sender, message, _ in self.r.inbox(self.topic(view, 4), self.role(view, 4)):
                    if isinstance(message, dict) and message.get("kind") == "commit" and self.message_valid(sender, view, 4, message) is True:
                        candidates.append({"view": view, "value": message["value"], "cert": message["cert"]})
            groups = {}
            for sender, message, _ in self.r.inbox("termination", "termination"):
                try:
                    if not self.certificate_valid(3, message["view"], message["value"], message["cert"]):
                        continue
                    candidates.append(message)
                    group = groups.setdefault(digest(message["value"]), (message, set()))
                    group[1].add(sender)
                except (KeyError, TypeError, ValueError):
                    continue
            if candidates and not forwarded:
                self.r.multicast("termination", candidates[0], "termination")
                forwarded = True
            for message, senders in groups.values():
                if len(senders) >= self.r.p.reconstruct:
                    # The commit certificate is rooted in Proposal Election's
                    # external-validity check.  Algorithm 14 verifies that
                    # certificate; it does not rerun the application predicate
                    # over the same decided value at termination.
                    self.r.metrics["decision_view"] = message["view"]
                    return message["value"]
            if self.r.failure.done():
                raise self.r.failure.result()
            await event.wait()

    async def run(self, value):
        status = self.valid(value)
        if status is not True:
            raise ValueError("initial value must be available and externally valid")
        return await self._run(value)

    async def run_prevalidated(self, value):
        """Run with an exact local input validated by the application layer.

        Proposal Election still independently applies ``self.valid`` to every
        proposer value, including this node's proposal.  This entry point only
        avoids evaluating the same local input once before PE starts.
        """

        return await self._run(value)

    async def _run(self, value):
        self.key = {"view": 0, "value": value, "cert": []}
        self.lock = {"view": 0, "value": None, "cert": []}
        views = asyncio.create_task(self._views())
        termination = asyncio.create_task(self._termination())
        try:
            done, _ = await asyncio.wait([views, termination], return_when=asyncio.FIRST_COMPLETED)
            if termination in done:
                return termination.result()
            return views.result()
        finally:
            for task in (views, termination):
                if not task.done():
                    task.cancel()
            await asyncio.gather(views, termination, return_exceptions=True)
