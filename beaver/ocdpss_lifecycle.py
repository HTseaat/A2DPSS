"""Lifecycle control for consecutive Algorithm-5 OCDPSS handovers.

OCDPSS composes the operation-independent A²DPSS public lifecycle and retains
only its operation adapter/input/output extension plus OCDPSS registrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

from beaver.a2dpss import A2DPSSContext
from beaver.a2dpss_lifecycle import (
    A2DPSSPreparedAdvance,
    A2DPSSPublicLifecycle,
    LOGICAL_ERASURE_LIMITATION,
    wipe_mutable_buffer,
)
from beaver.ocdpss import (
    OCDPSSContext,
    OCDPSSHolderState,
    OCDPSSJustifierManager,
    OCDPSSPublicResult,
    OCDPSSSuccessorReconstructionManager,
)
from beaver.operation_share import OperationAdapter


def _project_a2_context(context: OCDPSSContext) -> A2DPSSContext:
    """Project an OCDPSS context onto the Algorithm-4 public state."""

    return A2DPSSContext(
        epoch=context.epoch,
        sid=context.sid,
        source_committee=context.source_committee,
        successor_committee=context.successor_committee,
        transfer_seq=context.transfer_seq,
        g_hex=context.g_hex,
        h_hex=context.h_hex,
        key_base=context.key_base,
        successor_rpk_list=context.successor_rpk_list,
        service_t=context.service_t,
        old_commitment_vector=context.old_commitment_vector,
        holder_eval_points=context.holder_eval_points,
        successor_role_indices=context.successor_role_indices,
    )


@dataclass(frozen=True)
class OCDPSSAdvanceResult:
    previous_qid: str
    next_context: OCDPSSContext
    holder_state: Optional[OCDPSSHolderState]
    operation_output: object
    a2_advance: A2DPSSPreparedAdvance

    @property
    def am_next_holder(self) -> bool:
        return self.holder_state is not None


class OCDPSSHandoverLifecycle:
    """Own registrations and monotonic state transitions across handovers."""

    def __init__(
        self,
        *,
        my_node_id: int,
        justifier: OCDPSSJustifierManager,
        reconstruction: OCDPSSSuccessorReconstructionManager,
    ) -> None:
        if not isinstance(justifier, OCDPSSJustifierManager):
            raise TypeError("justifier must be an OCDPSSJustifierManager")
        if not isinstance(
            reconstruction, OCDPSSSuccessorReconstructionManager
        ):
            raise TypeError(
                "reconstruction must be an OCDPSSSuccessorReconstructionManager"
            )
        if isinstance(my_node_id, bool) or not isinstance(my_node_id, int):
            raise TypeError("my_node_id must be an integer")
        self.my_node_id = my_node_id
        self.justifier = justifier
        self.reconstruction = reconstruction
        self.active_context: Optional[OCDPSSContext] = None
        self.active_adapter: Optional[OperationAdapter] = None
        self.a2_lifecycle = A2DPSSPublicLifecycle()
        self.retired_context_digests = set()
        self.advanced_context_digests = set()
        self.operation_outputs = {}

    def register_initial(
        self,
        context: OCDPSSContext,
        operation_adapter: OperationAdapter,
        *,
        my_rsk: Optional[int] = None,
        on_new_share: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        if self.active_context is not None:
            raise RuntimeError("an OCDPSS lifecycle context is already active")
        self.justifier.register_context(context, operation_adapter)
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
        self.a2_lifecycle.register_initial(
            _project_a2_context(context)
        )
        self.active_context = context
        self.active_adapter = operation_adapter

    def advance(
        self,
        public_result: OCDPSSPublicResult,
        *,
        next_successor_committee: str,
        next_successor_rpk_list: Sequence[str],
        next_successor_role_indices: Mapping[int, int],
        next_operation_adapter: OperationAdapter,
        next_operation_input: bytes | str,
        my_next_rsk: Optional[int] = None,
        on_next_share: Optional[Callable[[int, int], None]] = None,
    ) -> OCDPSSAdvanceResult:
        current = self.active_context
        if current is None:
            raise RuntimeError("no active OCDPSS context to advance")
        if not isinstance(public_result, OCDPSSPublicResult):
            raise TypeError("public_result must be an OCDPSSPublicResult")
        if public_result.context_digest != current.context_digest:
            raise ValueError("public result does not match active context")
        if public_result.qid != current.encode_qid():
            raise ValueError("public result qid does not match active context")
        if current.context_digest in self.advanced_context_digests:
            raise RuntimeError("active OCDPSS context was already advanced")
        if not str(next_successor_committee):
            raise ValueError("next_successor_committee must be non-empty")

        a2_public_result = public_result.a2_result
        if a2_public_result is None:
            raise ValueError(
                "OCDPSS public result lacks its finalized A2DPSS view"
            )
        prepared_a2 = self.a2_lifecycle.prepare_advance(
            a2_public_result,
            next_successor_committee=next_successor_committee,
            next_successor_rpk_list=next_successor_rpk_list,
            next_successor_role_indices=next_successor_role_indices,
        )
        a2_next = prepared_a2.next_context
        operation_input = (
            next_operation_input.encode("utf-8")
            if isinstance(next_operation_input, str)
            else bytes(next_operation_input)
        )
        next_context = OCDPSSContext(
            epoch=a2_next.epoch,
            sid=a2_next.sid,
            source_committee=a2_next.source_committee,
            successor_committee=a2_next.successor_committee,
            transfer_seq=a2_next.transfer_seq,
            g_hex=a2_next.g_hex,
            h_hex=a2_next.h_hex,
            key_base=a2_next.key_base,
            successor_rpk_list=a2_next.successor_rpk_list,
            service_t=a2_next.service_t,
            old_commitment_vector=a2_next.old_commitment_vector,
            holder_eval_points=a2_next.holder_eval_points,
            successor_role_indices=a2_next.successor_role_indices,
            operation_id=next_operation_adapter.operation_id,
            operation_input=operation_input,
            operation_public_parameters=next_operation_adapter.public_context(
                operation_input
            ),
        )
        if next_context.transfer_seq <= current.transfer_seq:
            raise AssertionError("OCDPSS transfer sequence did not advance")
        if next_context.encode_qid() == current.encode_qid():
            raise AssertionError("OCDPSS advance did not produce a fresh qid")

        am_current_successor = self.my_node_id in current.successor_role_indices
        has_private_result = current.context_digest in self.reconstruction.results
        if am_current_successor != has_private_result:
            raise RuntimeError(
                "successor private result availability does not match active role"
            )

        # Validate/register every public part of the next context before the
        # irreversible private-result ownership transfer.
        self.justifier.register_context(next_context, next_operation_adapter)
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
                holder_state = self.reconstruction.consume_result_to_holder_state(
                    current
                )
        except Exception:
            self.reconstruction.unregister_context(next_context)
            self.justifier.unregister_context(next_context)
            raise

        self.reconstruction.unregister_context(current)
        self.justifier.unregister_context(current)
        self.a2_lifecycle.commit_advance(prepared_a2)
        self.advanced_context_digests.add(current.context_digest)
        self.retired_context_digests.add(current.context_digest)
        self.operation_outputs[current.context_digest] = (
            public_result.operation_output
        )
        self.active_context = next_context
        self.active_adapter = next_operation_adapter
        return OCDPSSAdvanceResult(
            previous_qid=current.encode_qid(),
            next_context=next_context,
            holder_state=holder_state,
            operation_output=public_result.operation_output,
            a2_advance=prepared_a2,
        )

    def retire_active(self) -> None:
        context = self.active_context
        if context is None:
            return
        self.reconstruction.unregister_context(context)
        self.justifier.unregister_context(context)
        self.retired_context_digests.add(context.context_digest)
        self.a2_lifecycle.retire_active()
        self.active_context = None
        self.active_adapter = None

    def snapshot(self) -> dict:
        context = self.active_context
        return {
            "my_node_id": self.my_node_id,
            "active_qid": context.encode_qid() if context is not None else None,
            "active_transfer_seq": (
                context.transfer_seq if context is not None else None
            ),
            "retired_context_digests": sorted(self.retired_context_digests),
            "advanced_context_digests": sorted(self.advanced_context_digests),
            "operation_output_context_digests": sorted(
                self.operation_outputs
            ),
            "a2_public_lifecycle": self.a2_lifecycle.snapshot(),
            "logical_erasure_limitation": LOGICAL_ERASURE_LIMITATION,
        }


__all__ = [
    "LOGICAL_ERASURE_LIMITATION",
    "OCDPSSAdvanceResult",
    "OCDPSSHandoverLifecycle",
    "wipe_mutable_buffer",
]
