"""Lifecycle control for consecutive Algorithm-4 A²DPSS handovers.

The public half advances from the protocol output ``vc'`` independently of
local successor membership.  The full lifecycle additionally moves a local
successor result into a one-shot holder state for the next handover.

Python can provide logical erasure by dropping references and clearing mutable
containers; it cannot guarantee physical zeroization of immutable integers or
interpreter-managed memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

from beaver.a2dpss import (
    A2DPSSContext,
    A2DPSSHolderState,
    A2DPSSJustifierManager,
    A2DPSSPublicResult,
    A2DPSSSuccessorReconstructionManager,
)


LOGICAL_ERASURE_LIMITATION = (
    "Python logical erasure drops references and clears mutable buffers, but "
    "cannot guarantee physical zeroization of immutable integers or runtime memory."
)


def wipe_mutable_buffer(buffer) -> None:
    """Best-effort zeroing for a caller-owned mutable byte buffer."""

    if not isinstance(buffer, (bytearray, memoryview)):
        raise TypeError(
            "erasable buffer must be a bytearray or writable memoryview"
        )
    if isinstance(buffer, memoryview) and buffer.readonly:
        raise TypeError("erasable memoryview must be writable")
    buffer[:] = b"\x00" * len(buffer)


@dataclass(frozen=True)
class A2DPSSPreparedAdvance:
    """Validated public transition awaiting an atomic owner commit."""

    previous_qid: str
    previous_context_digest: str
    public_result: A2DPSSPublicResult
    next_context: A2DPSSContext


@dataclass(frozen=True)
class A2DPSSAdvanceResult:
    """Committed public transition plus this node's optional private state."""

    previous_qid: str
    public_result: A2DPSSPublicResult
    next_context: A2DPSSContext
    holder_state: Optional[A2DPSSHolderState]

    @property
    def am_next_holder(self) -> bool:
        return self.holder_state is not None


def build_next_a2dpss_context(
    current: A2DPSSContext,
    public_result: A2DPSSPublicResult,
    *,
    next_successor_committee: str,
    next_successor_rpk_list: Sequence[str],
    next_successor_role_indices: Mapping[int, int],
) -> A2DPSSContext:
    """Build the next public statement solely from the previous output."""

    if not isinstance(current, A2DPSSContext):
        raise TypeError("current must be an A2DPSSContext")
    if not isinstance(public_result, A2DPSSPublicResult):
        raise TypeError("public_result must be an A2DPSSPublicResult")
    if public_result.context_digest != current.context_digest:
        raise ValueError("public result does not match active context")
    if public_result.qid != current.encode_qid():
        raise ValueError("public result qid does not match active context")
    if not str(next_successor_committee):
        raise ValueError("next_successor_committee must be non-empty")

    # Algorithm 4 assigns successor role j the evaluation point j+1.
    # Those public coordinates become the next source-holder coordinates.
    next_holder_eval_points = {
        int(node_id): int(role_idx) + 1
        for node_id, role_idx in current.successor_role_indices.items()
    }
    next_context = A2DPSSContext(
        epoch=current.epoch,
        sid=current.sid,
        source_committee=current.successor_committee,
        successor_committee=str(next_successor_committee),
        transfer_seq=current.transfer_seq + 1,
        g_hex=current.g_hex,
        h_hex=current.h_hex,
        key_base=current.key_base,
        successor_rpk_list=next_successor_rpk_list,
        service_t=current.service_t,
        old_commitment_vector=public_result.successor_commitment_vector,
        holder_eval_points=next_holder_eval_points,
        successor_role_indices=next_successor_role_indices,
    )
    if next_context.transfer_seq <= current.transfer_seq:
        raise AssertionError("A2DPSS transfer sequence did not advance")
    if next_context.encode_qid() == current.encode_qid():
        raise AssertionError("A2DPSS advance did not produce a fresh qid")
    return next_context


class A2DPSSPublicLifecycle:
    """Operation-independent public state machine shared by A²DPSS/OCDPSS."""

    def __init__(self) -> None:
        self.active_context: Optional[A2DPSSContext] = None
        self.retired_context_digests = set()
        self.advanced_context_digests = set()

    def register_initial(self, context: A2DPSSContext) -> None:
        if not isinstance(context, A2DPSSContext):
            raise TypeError("context must be an A2DPSSContext")
        if self.active_context is not None:
            raise RuntimeError("an A2DPSS public lifecycle is already active")
        self.active_context = context

    def prepare_advance(
        self,
        public_result: A2DPSSPublicResult,
        *,
        next_successor_committee: str,
        next_successor_rpk_list: Sequence[str],
        next_successor_role_indices: Mapping[int, int],
    ) -> A2DPSSPreparedAdvance:
        current = self.active_context
        if current is None:
            raise RuntimeError("no active A2DPSS context to advance")
        if current.context_digest in self.advanced_context_digests:
            raise RuntimeError("active A2DPSS context was already advanced")
        next_context = build_next_a2dpss_context(
            current,
            public_result,
            next_successor_committee=next_successor_committee,
            next_successor_rpk_list=next_successor_rpk_list,
            next_successor_role_indices=next_successor_role_indices,
        )
        return A2DPSSPreparedAdvance(
            previous_qid=current.encode_qid(),
            previous_context_digest=current.context_digest,
            public_result=public_result,
            next_context=next_context,
        )

    def commit_advance(
        self,
        prepared: A2DPSSPreparedAdvance,
    ) -> None:
        if not isinstance(prepared, A2DPSSPreparedAdvance):
            raise TypeError("prepared must be an A2DPSSPreparedAdvance")
        current = self.active_context
        if (
            current is None
            or current.context_digest
            != prepared.previous_context_digest
        ):
            raise RuntimeError(
                "prepared A2DPSS transition is no longer active"
            )
        self.advanced_context_digests.add(current.context_digest)
        self.retired_context_digests.add(current.context_digest)
        self.active_context = prepared.next_context

    def retire_active(self) -> None:
        context = self.active_context
        if context is None:
            return
        self.retired_context_digests.add(context.context_digest)
        self.active_context = None

    def snapshot(self) -> dict:
        context = self.active_context
        return {
            "active_qid": (
                context.encode_qid() if context is not None else None
            ),
            "active_transfer_seq": (
                context.transfer_seq if context is not None else None
            ),
            "retired_context_digests": sorted(
                self.retired_context_digests
            ),
            "advanced_context_digests": sorted(
                self.advanced_context_digests
            ),
        }


class A2DPSSHandoverLifecycle:
    """Own public registrations and private state across A²DPSS hops."""

    def __init__(
        self,
        *,
        my_node_id: int,
        justifier: A2DPSSJustifierManager,
        reconstruction: A2DPSSSuccessorReconstructionManager,
    ) -> None:
        if isinstance(my_node_id, bool) or not isinstance(my_node_id, int):
            raise TypeError("my_node_id must be an integer")
        if not isinstance(justifier, A2DPSSJustifierManager):
            raise TypeError(
                "justifier must be an A2DPSSJustifierManager"
            )
        if not isinstance(
            reconstruction, A2DPSSSuccessorReconstructionManager
        ):
            raise TypeError(
                "reconstruction must be an "
                "A2DPSSSuccessorReconstructionManager"
            )
        self.my_node_id = my_node_id
        self.justifier = justifier
        self.reconstruction = reconstruction
        self.public_lifecycle = A2DPSSPublicLifecycle()

    @property
    def active_context(self) -> Optional[A2DPSSContext]:
        return self.public_lifecycle.active_context

    @property
    def retired_context_digests(self):
        return self.public_lifecycle.retired_context_digests

    @property
    def advanced_context_digests(self):
        return self.public_lifecycle.advanced_context_digests

    def register_initial(
        self,
        context: A2DPSSContext,
        *,
        my_rsk: Optional[int] = None,
        on_new_share: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        if self.active_context is not None:
            raise RuntimeError("an A2DPSS lifecycle context is already active")
        self.justifier.register_context(context)
        try:
            self.reconstruction.register_context(
                context,
                my_node_id=self.my_node_id,
                my_rsk=my_rsk,
                on_new_share=on_new_share,
            )
        except Exception:
            self.justifier.unregister_context(context)
            raise
        self.public_lifecycle.register_initial(context)

    def advance(
        self,
        public_result: A2DPSSPublicResult,
        *,
        next_successor_committee: str,
        next_successor_rpk_list: Sequence[str],
        next_successor_role_indices: Mapping[int, int],
        my_next_rsk: Optional[int] = None,
        on_next_share: Optional[Callable[[int, int], None]] = None,
    ) -> A2DPSSAdvanceResult:
        current = self.active_context
        if current is None:
            raise RuntimeError("no active A2DPSS context to advance")
        prepared = self.public_lifecycle.prepare_advance(
            public_result,
            next_successor_committee=next_successor_committee,
            next_successor_rpk_list=next_successor_rpk_list,
            next_successor_role_indices=next_successor_role_indices,
        )
        next_context = prepared.next_context

        am_current_successor = (
            self.my_node_id in current.successor_role_indices
        )
        has_private_result = (
            current.context_digest in self.reconstruction.results
        )
        if am_current_successor != has_private_result:
            raise RuntimeError(
                "successor private result availability does not match "
                "active role"
            )

        # Register the complete public next state before irreversibly moving
        # the old private result.
        self.justifier.register_context(next_context)
        try:
            self.reconstruction.register_context(
                next_context,
                my_node_id=self.my_node_id,
                my_rsk=my_next_rsk,
                on_new_share=on_next_share,
            )
        except Exception:
            self.justifier.unregister_context(next_context)
            raise

        holder_state = None
        try:
            if am_current_successor:
                holder_state = (
                    self.reconstruction.consume_result_to_holder_state(
                        current
                    )
                )
        except Exception:
            self.reconstruction.unregister_context(next_context)
            self.justifier.unregister_context(next_context)
            raise

        self.reconstruction.unregister_context(current)
        self.justifier.unregister_context(current)
        self.public_lifecycle.commit_advance(prepared)
        return A2DPSSAdvanceResult(
            previous_qid=prepared.previous_qid,
            public_result=public_result,
            next_context=next_context,
            holder_state=holder_state,
        )

    def retire_active(self) -> None:
        context = self.active_context
        if context is None:
            return
        self.reconstruction.unregister_context(context)
        self.justifier.unregister_context(context)
        self.public_lifecycle.retire_active()

    def snapshot(self) -> dict:
        snapshot = self.public_lifecycle.snapshot()
        snapshot.update(
            {
                "my_node_id": self.my_node_id,
                "logical_erasure_limitation": (
                    LOGICAL_ERASURE_LIMITATION
                ),
            }
        )
        return snapshot


__all__ = [
    "A2DPSSAdvanceResult",
    "A2DPSSHandoverLifecycle",
    "A2DPSSPreparedAdvance",
    "A2DPSSPublicLifecycle",
    "LOGICAL_ERASURE_LIMITATION",
    "build_next_a2dpss_context",
    "wipe_mutable_buffer",
]
