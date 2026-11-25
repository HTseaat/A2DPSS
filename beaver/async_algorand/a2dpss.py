"""Single-handover adapter: immutable references -> one proposal -> exact T."""
import asyncio
import time
from contextlib import contextmanager
from beaver.a2dpss import (A2DPSSContribution, A2DPSSJustifierManager,
    A2DPSSPublicFinalizer, A2DPSSSuccessorReconstructionManager,
    A2DPSSHolderMetrics, make_a2dpss_contribution, validate_a2_contribution)
from beaver.a2dpss_benchmark_metrics import CPU_STAGES
from .model import digest
from .transport import CommitteeBroadcast
from .nwa import NWA
from beaver.experiment_trace import A2DPSSHandoverMetrics


class A2DPSSAdapter:
    def __init__(self, runtime, context, successor_key=None, fixture=None):
        self.r, self.context = runtime, context
        if fixture is not None and fixture.context.context_digest != context.context_digest:
            raise ValueError("Algorand fixture/context mismatch")
        self.fixture = fixture
        self.cpu_stages = {name: {"cpu_seconds": 0.0, "invocations": 0}
                           for name in CPU_STAGES}
        self.delivery_preparation_cpu = 0.0
        self.qid = context.encode_qid()
        self.timeline = A2DPSSHandoverMetrics(
            protocol="async-algorand", node=runtime.node, qid=self.qid,
            is_holder=runtime.node in context.holder_eval_points,
            is_successor=runtime.node in context.successor_role_indices)
        self.justifier = A2DPSSJustifierManager([context])
        self.finalizer = A2DPSSPublicFinalizer(self.justifier)
        self.reconstruction = A2DPSSSuccessorReconstructionManager()
        self.reconstruction.register_context(context, my_node_id=runtime.node, my_rsk=successor_key)
        self.pool, self.holders = {}, {}
        self.required = len(fixture.entries) if fixture is not None else context.service_t + 1
        # Freeze at the state transition, independently of when run() resumes.
        # Fixture replay deliberately retains its all-inputs policy.
        self.candidate_refs = None
        self.created_at = time.perf_counter()
        self.broadcast = CommitteeBroadcast(runtime, "handover", None)
        runtime.spawn(self._receive())

    @contextmanager
    def _charge_cpu(self, stage):
        # Only synchronous application operations are enclosed: an await would
        # incorrectly charge other tasks' CPU to the same stage.
        started = time.process_time()
        try:
            yield
        finally:
            metric = self.cpu_stages[stage]
            metric["cpu_seconds"] += time.process_time() - started
            metric["invocations"] += 1

    def cpu_snapshot(self):
        stages = {name: dict(value) for name, value in self.cpu_stages.items()}
        stages["public_finalization"] = dict(self.finalizer.metrics_snapshot()["total"])
        stages["public_finalization"]["cpu_seconds"] += self.delivery_preparation_cpu
        stages["successor_reconstruction"] = dict(self.reconstruction.metrics_snapshot()["total"])
        return stages

    def add(self, mid, payload):
        reference = digest([mid, payload.hex()])
        entry = (mid, bytes(payload))
        previous = self.pool.get(reference)
        if previous is not None and previous != entry:
            raise ValueError("immutable transcript reference collision")
        self.pool[reference] = entry
        self.r.pulse()
        return reference

    def resolve(self, proposal):
        try:
            refs = proposal["refs"]
            if (set(proposal) != {"qid", "refs"} or proposal["qid"] != self.qid
                    or not isinstance(refs, list)
                    or not self.context.service_t + 1 <= len(refs) <= len(self.context.holder_eval_points)
                    or any(not isinstance(ref, str) or len(ref) != 64
                           or any(char not in "0123456789abcdef" for char in ref) for ref in refs)
                    or refs != sorted(set(refs))):
                return False
            if not all(ref in self.pool for ref in refs):
                return None
            return [self.pool[ref] for ref in refs]
        except (KeyError, ValueError, TypeError):
            return False

    def valid(self, proposal):
        entries = self.resolve(proposal)
        if entries is False or entries is None:
            return entries
        start = time.perf_counter()
        with self._charge_cpu("formal_predicate"):
            valid = self.justifier.verify_candidate_set(self.qid, entries)
        self.r.metrics["a2_predicate_seconds"] += time.perf_counter() - start
        self.r.metrics["a2_predicate_calls"] += 1
        self.r.metrics["a2_predicate_cpu_seconds"] = self.cpu_stages["formal_predicate"]["cpu_seconds"]
        return valid

    async def _receive(self):
        seen = set()
        while True:
            event = self.r.event
            for source, value in list(self.broadcast.delivered.items()):
                if source in seen:
                    continue
                seen.add(source)
                try:
                    payload = bytes.fromhex(value["payload"])
                    # Availability and validity are separate. Keep the immutable
                    # bytes even when the contribution is invalid, so a proposal
                    # referring to them is rejected instead of pending forever.
                    reference = self.add("node_%d" % source, payload)
                    contribution = A2DPSSContribution.from_bytes(payload)
                    started, cpu = time.perf_counter(), time.process_time()
                    valid = source == contribution.holder_id and validate_a2_contribution(self.context, contribution)
                    cpu_elapsed = time.process_time() - cpu
                    self.r.metrics["a2_receive_validation_seconds"] += time.perf_counter() - started
                    self.r.metrics["a2_receive_validation_cpu_seconds"] += cpu_elapsed
                    self.r.metrics["a2_receive_validation_calls"] += 1
                    self.cpu_stages["auxiliary_validation"]["cpu_seconds"] += cpu_elapsed
                    self.cpu_stages["auxiliary_validation"]["invocations"] += 1
                    if not valid:
                        self.r.metrics["invalid_a2_contributions"] += 1
                        continue
                    self.holders[source] = reference
                    if self.candidate_refs is None and len(self.holders) >= self.required:
                        self.candidate_refs = tuple(sorted(self.holders.values()))
                        self.r.metrics["candidate_ref_count"] = len(self.candidate_refs)
                        self.r.metrics["candidate_required_contributions"] = self.required
                        self.r.metrics["candidate_freeze_seconds"] = time.perf_counter() - self.created_at
                    self.r.pulse()
                except (ValueError, KeyError, TypeError):
                    self.r.metrics["invalid_a2_contributions"] += 1
                finally:
                    # Continue receiving after freezing, but let agreement and
                    # reference-dependent validators advance between checks.
                    await asyncio.sleep(0)
            await event.wait()

    async def run(self, old_share=None):
        started = time.perf_counter()
        self.timeline.start(started)
        if self.fixture is not None and self.r.node in self.context.holder_eval_points:
            if old_share is not None:
                raise ValueError("fixture replay cannot also generate a live contribution")
            entry = self.fixture.entry_for_sender(self.r.node)
            self.broadcast.broadcast({"payload": entry.payload.hex()})
            self.timeline.mark_contribution_sent()
        elif old_share is not None:
            holder_metrics = A2DPSSHolderMetrics()
            generation_cpu_started = time.process_time()
            contribution = make_a2dpss_contribution(context=self.context, holder_id=self.r.node,
                share_s=old_share[0], share_s_hat=old_share[1], metrics=holder_metrics)
            self.r.metrics["a2_contribution_generation_cpu_seconds"] = time.process_time() - generation_cpu_started
            self.r.metrics["a2_contribution_generation_calls"] = 1
            self.r.metrics["a2_contribution_generation_seconds"] = time.perf_counter() - started
            # Align the primary stage with DAG/ACS HolderMetrics: pvss_deal.
            # The enclosing generation timing above is diagnostic, not added.
            self.cpu_stages["pvss_generation"] = {
                "cpu_seconds": holder_metrics.pvss_cpu_seconds, "invocations": 1}
            self.broadcast.broadcast({"payload": contribution.to_bytes().hex()})
            self.timeline.mark_contribution_sent()
        await self.r.wait(lambda: self.candidate_refs is not None)
        self.timeline.mark_pool_ready()
        with self._charge_cpu("candidate_formation"):
            proposal = {"qid": self.qid, "refs": list(self.candidate_refs)}
            local_entries = self.resolve(proposal)
            if (local_entries is False or local_entries is None
                    or self.justifier.select_preverified_subset(self.qid, local_entries) is None):
                raise RuntimeError("local A2DPSS proposal is not prevalidated")
        self.r.metrics["candidate_ready_seconds"] = time.perf_counter() - started
        agreed = await NWA(self.r, self.valid).run_prevalidated(proposal)
        entries = self.resolve(agreed)
        if entries is False:
            raise RuntimeError("decided A2DPSS proposal is malformed")
        if entries is None:
            entries = await self.r.wait(
                lambda: self.resolve(agreed)
            )
        preparation_cpu_started = time.process_time()
        selected = self.justifier.select_preverified_subset(self.qid, entries)
        self.delivery_preparation_cpu += time.process_time() - preparation_cpu_started
        if selected is None:
            raise RuntimeError("cannot select exact canonical threshold T")
        self.r.metrics["t_deliver_seconds"] = time.perf_counter() - started
        self.timeline.mark_ordered()
        public = self.finalizer.on_verified_deliver(self.qid, selected)
        if public is None:
            raise RuntimeError("A2DPSS public finalization failed")
        self.r.metrics["t_finalize_seconds"] = time.perf_counter() - started
        self.timeline.mark_public_finalized()
        successor = self.reconstruction.reconstruct(public, selected)
        if self.r.node in self.context.successor_role_indices and successor is None:
            raise RuntimeError("A2DPSS successor reconstruction failed")
        self.r.metrics["t_reconstruct_seconds"] = time.perf_counter() - started
        if successor is not None:
            self.timeline.mark_successor_done()
        return agreed, public, successor
