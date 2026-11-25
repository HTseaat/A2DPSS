"""Verifiable Gather (8–9) and Proposal Election (10–12).

Verification is an interactive, pending-aware predicate: None means wait for
dependencies. A PE result is NOT itself a consensus decision.
"""
import asyncio
import time
from .model import node_set, digest
from .transport import CommitteeBroadcast
from .avss import CommitteeAVSS


class Gather:
    def __init__(self, runtime, view, valid):
        self.r, self.view, self.valid = runtime, view, valid
        self.tag = runtime.p.role(view, "gather")
        self.propose = CommitteeBroadcast(runtime, self.tag + "/propose", self.tag + "/propose")
        self.aggregate = CommitteeBroadcast(runtime, self.tag + "/aggregate", self.tag + "/aggregate")
        self.R, self.T, self.U = set(), {}, {}
        self.sent_proposal = self.sent_aggregate = False
        self.initial_ids = None
        self.output = None
        runtime.spawn(self._service())

    def valid_set(self, value):
        if not node_set(value, self.r.p.population, self.r.p.h):
            return False
        statuses = [self.valid(i) for i in value]
        if any(status is False for status in statuses):
            return False
        return True if all(status is True for status in statuses) else None

    def verify(self, ids):
        valid = self.valid_set(ids)
        if valid is not True:
            return valid
        return True if sum(group.issubset(ids) for group in self.U.values()) >= self.r.p.h else None

    def start(self, ids):
        """Latch and broadcast the first valid input before more J checks run.

        Only the initial input is frozen. R/T/U and proof validation continue
        to accept later valid dealers, including ones outside this snapshot.
        """
        if self.sent_proposal or self.valid_set(ids) is not True:
            return
        self.initial_ids = tuple(ids)
        self.propose.broadcast(list(self.initial_ids))
        self.sent_proposal = True
        self.r.metrics["gather/%d/initial_ids_count" % self.view] = len(self.initial_ids)

    async def _service(self):
        while True:
            event = self.r.event
            valid = [i for i in range(self.r.p.population) if self.valid(i) is True]
            if not self.sent_proposal and len(valid) >= self.r.p.h:
                self.start(valid)
            for source, ids in list(self.propose.delivered.items()):
                if source not in self.T and self.valid_set(ids) is True:
                    self.R.update(ids)
                    self.T[source] = set(ids)
                    self.r.pulse()
            if not self.sent_aggregate and len(self.T) >= self.r.p.h:
                self.aggregate.broadcast(sorted(self.T)[:self.r.p.h])
                self.sent_aggregate = True
            for source, ids in list(self.aggregate.delivered.items()):
                if (source not in self.U and node_set(ids, self.r.p.population, self.r.p.h)
                        and set(ids).issubset(self.T)):
                    self.U[source] = set().union(*(self.T[i] for i in ids))
                    self.r.pulse()
            if self.output is None and len(self.U) >= self.r.p.h:
                self.output = sorted(self.R)
                self.r.pulse()
            await event.wait()


class ProposalElection:
    def __init__(self, runtime, view, valid):
        self.r, self.view, self.valid = runtime, view, valid
        self.started = time.perf_counter()
        self.avss = CommitteeAVSS(runtime, view)
        self.accepted, self.invalid = {}, set()
        self.gather = Gather(runtime, view, self._dealer_valid)
        self.rank_role = runtime.p.role(view, "rank")
        runtime.spawn(self._validate())

    def _dealer_valid(self, source):
        return True if source in self.accepted else (False if source in self.invalid else None)

    async def _validate(self):
        while True:
            event = self.r.event
            for source, payload in list(self.avss.payloads.items()):
                if source in self.accepted or source in self.invalid:
                    continue
                try:
                    proposal = payload["proposal"]
                    status = self.valid(proposal)
                except (KeyError, ValueError, TypeError):
                    status = False
                if status is True:
                    self.accepted[source] = proposal
                    if len(self.accepted) == self.r.p.h:
                        reached = time.perf_counter()
                        self.r.metrics["pe/%d/threshold_seconds" % self.view] = reached - self.started
                        # Synchronous trigger: the (h+1)th J cannot postpone
                        # or enlarge the first Gather proposal.
                        self.gather.start(sorted(self.accepted))
                        self.r.metrics["pe/%d/gather_start_delay_seconds" % self.view] = time.perf_counter() - reached
                    self.r.pulse()
                elif status is False:
                    self.invalid.add(source)
                    self.r.pulse()
                # Keep serving all proposers, including pending dependencies,
                # without monopolizing the event loop after the threshold.
                await asyncio.sleep(0)
            await event.wait()

    async def run(self, proposal):
        await self.avss.setup()
        if self.r.committees.member(self.avss.share_role)[0]:
            await self.avss.share(self.r.committees.rank(self.rank_role), proposal)
        ids = await self.r.wait(lambda: self.gather.output)
        self.avss.reconstruct()
        await self.r.wait(lambda: all(i in self.avss.recovered for i in ids))
        winner = self._winner(ids)
        if winner is None:
            raise RuntimeError("Gather contained no valid reconstructed rank")
        return {"proposal": self.accepted[winner], "ids": ids}

    def _winner(self, ids):
        ranks = [(self.avss.recovered[i]["rank"], i) for i in ids
                 if self.r.committees.verify_rank(self.rank_role, i, self.avss.recovered[i])]
        return max(ranks)[1] if ranks else None

    def verify(self, output):
        try:
            ids = output["ids"]
            status = self.gather.verify(ids)
            if status is not True:
                return status
            if not all(i in self.avss.recovered for i in ids):
                return None
            winner = self._winner(ids)
            return winner is not None and digest(self.accepted[winner]) == digest(output["proposal"])
        except (KeyError, ValueError, TypeError):
            return False
