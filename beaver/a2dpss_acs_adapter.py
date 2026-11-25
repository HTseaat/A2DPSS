"""Pure local boundary between A²DPSS data and Async-YOSO ACS blocks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import time
from typing import Dict, List, Tuple

from beaver.a2dpss import (
    A2DPSSContext,
    A2DPSSContribution,
    A2DPSSJustifierManager,
)
from beaver.async_yoso_tob import MessageRef, PendingEntry
from beaver.dag_tob import PoolEntry, Qid


class MissingTranscriptReferences(LookupError):
    """A candidate cannot yet be resolved from the local Pending store."""

    def __init__(self, refs: Iterable[MessageRef]) -> None:
        self.refs = tuple(sorted(set(refs)))
        encoded = ", ".join(ref.encode() for ref in self.refs)
        super().__init__(f"missing or unavailable transcript refs: {encoded}")


@dataclass(frozen=True)
class A2DPSSACSDeliveryPlan:
    """Resolved single-qid ACS output before public/private finalization."""

    qid: Qid
    refs: Tuple[MessageRef, ...]
    s_out: Tuple[PoolEntry, ...]
    selected_T: Tuple[PoolEntry, ...]


class A2DPSSACSAdapter:
    """Deterministic, network-free A²DPSS adapter for ACS candidate blocks."""

    def __init__(self, justifier: A2DPSSJustifierManager) -> None:
        if not isinstance(justifier, A2DPSSJustifierManager):
            raise TypeError(
                "justifier must be an A2DPSSJustifierManager"
            )
        self.justifier = justifier
        # Successful formal ACSPROP checks are bound to the protocol session,
        # proposer identity, exact ref list, and immutable pending bytes.  The
        # optimized delivery path consumes this local evidence instead of
        # repeating the candidate predicate over the selected core union.
        self._verified_candidates: Dict[
            Tuple[str, int, Tuple[MessageRef, ...]],
            Tuple[Tuple[MessageRef, str, str], ...],
        ] = {}
        self._phase_metrics = {
            "candidate_formation": {
                "invocations": 0,
                "cpu_seconds": 0.0,
            },
            "delivery_preparation": {
                "invocations": 0,
                "cpu_seconds": 0.0,
            },
        }

    def _record_cpu(self, phase: str, started: float) -> None:
        metric = self._phase_metrics[phase]
        metric["invocations"] += 1
        metric["cpu_seconds"] += time.process_time() - started

    def metrics_snapshot(self) -> dict:
        return {
            phase: {
                "invocations": int(metric["invocations"]),
                "cpu_seconds": float(metric["cpu_seconds"]),
            }
            for phase, metric in self._phase_metrics.items()
        }

    def register_context(self, context: A2DPSSContext) -> None:
        """Register an A²DPSS context in the shared justifier registry."""

        self.justifier.register_context(context)

    def qid_for_payload(
        self,
        payload: bytes | bytearray | memoryview,
    ) -> Qid:
        """Resolve an unwrapped contribution payload to its registered qid."""

        contribution = A2DPSSContribution.from_bytes(payload)
        try:
            context = self.justifier.contexts[
                contribution.context_digest
            ]
        except KeyError as exc:
            raise KeyError(
                "unknown A2DPSS context for contribution payload: "
                f"{contribution.context_digest}"
            ) from exc
        qid = context.build_qid()
        self.justifier._registration(qid)
        return qid

    @staticmethod
    def _ordered_refs(refs: Iterable[MessageRef]) -> List[MessageRef]:
        try:
            ordered = list(refs)
        except TypeError as exc:
            raise TypeError("refs must be an iterable of MessageRef") from exc
        if any(not isinstance(ref, MessageRef) for ref in ordered):
            raise TypeError("refs must contain only MessageRef values")
        return sorted(ordered)

    def _resolve_ref_items(
        self,
        refs: Iterable[MessageRef],
        pending: Mapping[MessageRef, PendingEntry],
    ) -> List[Tuple[MessageRef, PoolEntry]]:
        if not isinstance(pending, Mapping):
            raise TypeError("pending must be a MessageRef mapping")
        ordered_refs = self._ordered_refs(refs)
        missing = [ref for ref in ordered_refs if ref not in pending]
        if missing:
            raise MissingTranscriptReferences(missing)

        resolved: List[Tuple[MessageRef, PoolEntry]] = []
        for ref in ordered_refs:
            entry = pending[ref]
            if not isinstance(entry, PendingEntry):
                raise TypeError(
                    "pending values must be PendingEntry instances"
                )
            if entry.ref != ref:
                raise ValueError(
                    "pending entry ref does not match its mapping key"
                )
            if not isinstance(entry.mid, str) or not entry.mid:
                raise ValueError("pending entry has an invalid mid")
            if not isinstance(
                entry.payload, (bytes, bytearray, memoryview)
            ):
                raise TypeError("pending entry payload must be bytes")
            resolved.append(
                (ref, (entry.mid, bytes(entry.payload)))
            )
        return resolved

    def resolve_refs(
        self,
        refs: Iterable[MessageRef],
        pending: Mapping[MessageRef, PendingEntry],
    ) -> List[PoolEntry]:
        """Resolve every ref, preserving duplicates so predicates reject them."""

        return [
            entry
            for _ref, entry in self._resolve_ref_items(refs, pending)
        ]

    def group_entries_by_qid(
        self,
        refs: Iterable[MessageRef],
        pending: Mapping[MessageRef, PendingEntry],
    ) -> Dict[Qid, List[PoolEntry]]:
        """Resolve refs and group their original contribution bytes by qid."""

        groups: Dict[Qid, List[PoolEntry]] = {}
        for entry in self.resolve_refs(refs, pending):
            qid = self.qid_for_payload(entry[1])
            groups.setdefault(qid, []).append(entry)
        return {
            qid: groups[qid]
            for qid in sorted(groups, key=lambda value: value.encode())
        }

    def validate_candidate_refs(
        self,
        refs: Iterable[MessageRef],
        pending: Mapping[MessageRef, PendingEntry],
    ) -> bool:
        """Apply the block predicate to all qid segments in a candidate."""

        try:
            groups = self.group_entries_by_qid(refs, pending)
        except MissingTranscriptReferences:
            raise
        except (KeyError, TypeError, ValueError):
            return False
        if not groups:
            return False
        return all(
            self.justifier.verify_candidate_set(qid, entries)
            for qid, entries in groups.items()
        )

    @staticmethod
    def _verified_candidate_key(
        session_id: str,
        proposer: int,
        refs: Iterable[MessageRef],
    ) -> Tuple[str, int, Tuple[MessageRef, ...]]:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if (
            isinstance(proposer, bool)
            or not isinstance(proposer, int)
            or proposer < 0
        ):
            raise ValueError("proposer must be a non-negative integer")
        return session_id, proposer, tuple(
            A2DPSSACSAdapter._ordered_refs(refs)
        )

    def record_verified_candidate(
        self,
        session_id: str,
        proposer: int,
        refs: Iterable[MessageRef],
        pending: Mapping[MessageRef, PendingEntry],
    ) -> None:
        """Record one successful formal proposer-block validation.

        This must be called only by the receiver admission callback immediately
        after the exact proposal returned true.  Payload fingerprints prevent a
        later mutation of the pending mapping from inheriting that validation.
        """

        key = self._verified_candidate_key(session_id, proposer, refs)
        resolved = self._resolve_ref_items(key[2], pending)
        self._verified_candidates[key] = tuple(
            (
                ref,
                entry[0],
                hashlib.sha256(entry[1]).hexdigest(),
            )
            for ref, entry in resolved
        )

    def _candidate_has_verified_evidence(
        self,
        session_id: str,
        proposer: int,
        refs: Iterable[MessageRef],
        pending: Mapping[MessageRef, PendingEntry],
    ) -> bool:
        try:
            key = self._verified_candidate_key(
                session_id, proposer, refs
            )
            expected = self._verified_candidates[key]
            resolved = self._resolve_ref_items(key[2], pending)
        except (KeyError, TypeError, ValueError, MissingTranscriptReferences):
            return False
        actual = tuple(
            (
                ref,
                entry[0],
                hashlib.sha256(entry[1]).hexdigest(),
            )
            for ref, entry in resolved
        )
        return actual == expected

    def unverified_core_proposals(
        self,
        core,
        pending: Mapping[MessageRef, PendingEntry],
        session_id: str,
    ) -> List[object]:
        """Return core proposals lacking exact local admission evidence."""

        try:
            proposals = tuple(core.proposals)
        except (AttributeError, TypeError) as exc:
            raise TypeError("core must provide proposals") from exc
        if not proposals:
            raise ValueError("verified ACS core must contain proposals")

        missing = []
        for proposal in proposals:
            try:
                verified = self._candidate_has_verified_evidence(
                    session_id,
                    proposal.proposer,
                    proposal.refs,
                    pending,
                )
            except AttributeError as exc:
                raise TypeError(
                    "core proposals must provide proposer and refs"
                ) from exc
            if not verified:
                missing.append(proposal)
        return missing

    def build_candidate_refs(
        self,
        pending: Mapping[MessageRef, PendingEntry],
        delivered_qids: Iterable[Qid | str] = (),
    ) -> List[MessageRef]:
        """Return refs for every locally ready, not-yet-delivered qid."""

        delivered = set()
        for value in delivered_qids:
            if isinstance(value, str):
                value = Qid.decode(value)
            if not isinstance(value, Qid):
                raise TypeError(
                    "delivered_qids must contain Qid or encoded qid strings"
                )
            delivered.add(value.encode())

        refs_by_qid: Dict[Qid, List[MessageRef]] = {}
        entries_by_qid: Dict[Qid, List[PoolEntry]] = {}
        for ref, entry in self._resolve_ref_items(
            pending.keys(), pending
        ):
            qid = self.qid_for_payload(entry[1])
            refs_by_qid.setdefault(qid, []).append(ref)
            entries_by_qid.setdefault(qid, []).append(entry)

        candidate_refs: List[MessageRef] = []
        for qid in sorted(refs_by_qid, key=lambda value: value.encode()):
            if qid.encode() in delivered:
                continue
            if not self.justifier.verify_candidate_set(
                qid, entries_by_qid[qid]
            ):
                continue
            candidate_refs.extend(refs_by_qid[qid])
        return sorted(candidate_refs)

    def build_structurally_ready_candidate_refs(
        self,
        pending: Mapping[MessageRef, PendingEntry],
        delivered_qids: Iterable[Qid | str] = (),
    ) -> List[MessageRef]:
        """Build an honest proposal without executing the formal predicate.

        Candidate readiness is only a scheduling condition.  It decodes public
        contribution identities and waits for a distinct-holder threshold, but
        deliberately does not run PVSS or anchor verification.  Each received
        ACSPROP block is checked exactly once by ``verify_candidate_set`` at the
        protocol admission boundary.
        """

        cpu_started = time.process_time()
        delivered = set()
        for value in delivered_qids:
            if isinstance(value, str):
                value = Qid.decode(value)
            if not isinstance(value, Qid):
                raise TypeError(
                    "delivered_qids must contain Qid or encoded qid strings"
                )
            delivered.add(value.encode())

        refs_by_qid: Dict[Qid, List[MessageRef]] = {}
        seen_by_qid: Dict[Qid, set] = {}
        for ref, entry in self._resolve_ref_items(
            pending.keys(), pending
        ):
            try:
                contribution = A2DPSSContribution.from_bytes(entry[1])
                qid = self.qid_for_payload(entry[1])
                context = self.justifier._registration(qid)
            except (KeyError, TypeError, ValueError):
                continue
            if qid.encode() in delivered:
                continue
            if entry[0] != f"node_{contribution.holder_id}":
                continue
            if (
                contribution.context_digest != context.context_digest
                or context.holder_eval_points.get(contribution.holder_id)
                != contribution.holder_eval_point
            ):
                continue
            identity = (
                contribution.holder_id,
                contribution.holder_eval_point,
            )
            seen = seen_by_qid.setdefault(qid, set())
            if identity in seen:
                continue
            seen.add(identity)
            refs_by_qid.setdefault(qid, []).append(ref)

        candidate_refs: List[MessageRef] = []
        for qid in sorted(refs_by_qid, key=lambda value: value.encode()):
            refs = refs_by_qid[qid]
            context = self.justifier._registration(qid)
            if len(refs) >= context.service_t + 1:
                candidate_refs.extend(refs)
        result = sorted(candidate_refs)
        self._record_cpu("candidate_formation", cpu_started)
        return result

    def union_core_refs(self, core) -> List[MessageRef]:
        """Return the deterministic deduplicated ref union of an ACS core."""

        flatten_refs = getattr(core, "flatten_refs", None)
        if not callable(flatten_refs):
            raise TypeError("core must provide flatten_refs()")
        refs = self._ordered_refs(flatten_refs())
        return sorted(set(refs))

    def select_delivery_T(
        self,
        qid: Qid | str,
        s_out: Sequence[PoolEntry],
    ) -> List[PoolEntry] | None:
        """Validate an ACS union and select its exact canonical threshold T."""

        if not self.justifier.verify_candidate_set(qid, s_out):
            return None
        return self.justifier.select_justified_subset(qid, s_out)

    def select_verified_delivery_T(
        self,
        qid: Qid | str,
        s_out: Sequence[PoolEntry],
    ) -> List[PoolEntry] | None:
        """Select ``T`` after the core's proposer blocks were all verified."""

        return self.justifier.select_preverified_subset(qid, s_out)

    def _build_delivery_plan(
        self,
        core,
        pending: Mapping[MessageRef, PendingEntry],
        *,
        preverified: bool,
    ) -> A2DPSSACSDeliveryPlan:
        refs = self.union_core_refs(core)
        groups = self.group_entries_by_qid(refs, pending)
        if len(groups) != 1:
            raise ValueError(
                "single-qid ACS delivery requires exactly one qid"
            )
        qid, s_out = next(iter(groups.items()))
        selected = (
            self.select_verified_delivery_T(qid, s_out)
            if preverified
            else self.select_delivery_T(qid, s_out)
        )
        if selected is None:
            raise ValueError(
                "ACS core transcript union does not satisfy A2DPSS predicate"
            )
        return A2DPSSACSDeliveryPlan(
            qid=qid,
            refs=tuple(refs),
            s_out=tuple(s_out),
            selected_T=tuple(selected),
        )

    def prepare_single_qid_delivery(
        self,
        core,
        pending: Mapping[MessageRef, PendingEntry],
    ) -> A2DPSSACSDeliveryPlan:
        """Resolve a common ACS core into ``S_out`` and canonical exact ``T``.

        Phase 6 intentionally supports exactly one qid.  Every reference must
        be locally available, the complete deduplicated union must satisfy the
        candidate predicate, and finalization receives only the selected
        ``service_t + 1`` entries.
        """

        cpu_started = time.process_time()
        result = self._build_delivery_plan(
            core,
            pending,
            preverified=False,
        )
        self._record_cpu("delivery_preparation", cpu_started)
        return result

    def prepare_verified_single_qid_delivery(
        self,
        core,
        pending: Mapping[MessageRef, PendingEntry],
        session_id: str,
    ) -> A2DPSSACSDeliveryPlan:
        """Prepare delivery using exact receiver-admission evidence.

        Every distinct proposer block in ``core`` must have a successful local
        admission record for this session, and its referenced pending bytes
        must still match the recorded fingerprints.  Once those checks pass,
        only structural union checks and deterministic threshold selection are
        repeated; PVSS and anchor checks are not.
        """

        cpu_started = time.process_time()
        if self.unverified_core_proposals(core, pending, session_id):
            raise ValueError(
                "ACS core contains a proposer block without matching "
                "local validation evidence"
            )
        result = self._build_delivery_plan(
            core,
            pending,
            preverified=True,
        )
        self._record_cpu("delivery_preparation", cpu_started)
        return result
