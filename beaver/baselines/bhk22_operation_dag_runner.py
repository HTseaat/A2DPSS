"""Real persistent-DAG runtime for the BHK+22 threshold-operation arm."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from beaver.baselines.bhk22_contract import (
    COMPLETION_ENCRYPTED_CARRY_FORWARD,
    PHASE_DEAL,
    PHASE_FINALIZE,
    PHASE_RESPONSE,
    PHASE_VERIFY,
    ROLE_DEALER,
    ROLE_VERIFIER,
)
from beaver.baselines.bhk22_dag_runner import (
    DEAL_PUBLIC_STATE_DOMAIN,
    BHK22OptimisticDAGRuntime,
    BHK22RuntimeOutbound,
    _inject_dealer_fault,
    _state_digest,
)
from beaver.baselines.bhk22_experiment import (
    BHK22DAGExperimentManifest,
    BHK22NodePrivateShard,
)
from beaver.baselines.bhk22_messages import (
    BHK22PublicFinalizeOutput,
    BHK22ResponderBatch,
    BHK22VerifierBatch,
)
from beaver.baselines.bhk22_operation import (
    BHK22EncryptedSourceCoordinate,
    BHK22EncryptedSourceVector,
    BHK22OperationDealerBatch,
    BHK22OperationDealerRole,
    BHK22OperationPublicContext,
    BHK22OperationPublicResult,
    BHK22OperationSourceState,
    build_bhk22_operation_qid,
    verify_bhk22_operation_dealer_batch,
)
from beaver.baselines.bhk22_phase import (
    BHK22PhaseController,
    BHK22PhaseOrderingJustifier,
    PHASE_MESSAGE_TYPES,
)
from beaver.baselines.bhk22_shareholder import BHK22EncryptedCommitmentVector
from beaver.baselines.bhk22_verifier import BHK22VerifierRole
from beaver.g1_backend import G1Point
from beaver.operation_vk import combine_vk_operation_shares
from beaver.pedersen_shamir import lagrange_coeff_at_zero
from beaver.threshold_operation_contract import operation_output_digest


class BHK22OperationDAGRuntime(BHK22OptimisticDAGRuntime):
    """BHK phase runtime with application proofs in the one-shot dealer speech."""

    def __init__(
        self,
        *,
        manifest: BHK22DAGExperimentManifest,
        local_shard: BHK22NodePrivateShard,
        operation_context: BHK22OperationPublicContext,
        operation_dealers: list[Mapping[str, Any]],
        my_node_id: int,
    ) -> None:
        if not isinstance(operation_context, BHK22OperationPublicContext):
            raise TypeError("operation_context has the wrong type")
        if operation_context.base_context_digest != manifest.context.root_context_digest:
            raise ValueError("BHK operation/base context mismatch")
        if operation_context.trial_id != manifest.context.trial_id:
            raise ValueError("BHK operation trial mismatch")
        if manifest.context.encrypted_source != operation_context.encrypted_source:
            raise ValueError("BHK operation requires the same encrypted protocol source")
        super().__init__(
            manifest=manifest, local_shard=local_shard, my_node_id=my_node_id
        )
        self.application_context = operation_context
        codecs = dict(PHASE_MESSAGE_TYPES)
        codecs[PHASE_DEAL] = BHK22OperationDealerBatch
        self.justifier = BHK22PhaseOrderingJustifier(codecs)
        self.controller = BHK22PhaseController(
            context=self.context,
            path_mode=manifest.path_mode,
            expected_active_roles=manifest.expected_active_roles,
            justifier=self.justifier,
            completion_mode=COMPLETION_ENCRYPTED_CARRY_FORWARD,
            qid_builder=lambda phase: build_bhk22_operation_qid(
                self.application_context, phase=phase
            ),
        )
        local_material = list(operation_dealers)
        if len(local_material) > 1:
            raise ValueError("one node cannot own multiple BHK operation dealers")
        self.operation_source_state = None
        if local_material:
            item = local_material[0]
            role = int(item["role_id"])
            if self.context.committees[ROLE_DEALER].role_to_node[role] != self.my_node_id:
                raise ValueError("BHK operation dealer node binding mismatch")
            self.operation_source_state = BHK22OperationSourceState(
                role_id=role,
                evaluation_point=int(item["evaluation_point"]),
                recipient_secret_key=int(item["recipient_secret_key"]),
                share=int(item["source_share"]),
                public_value_secret_key=int(item["public_value_secret_key"]),
                coordinate=operation_context.encrypted_source.coordinates[role],
                encrypted_source_digest=item["encrypted_source_digest"],
                terminal_state_identity=(
                    f"genesis:{operation_context.encrypted_source.digest}"
                ),
            )
        self.operation_public_result: BHK22OperationPublicResult | None = None
        self.operation_verifications = {}
        self.dealer_public_verifier = lambda batch: self.operation_verifications[
            batch.role_id
        ].accepted
        self._operation_dealer_batches = ()

    def start(self) -> BHK22RuntimeOutbound | None:
        if "t0" in self._clock:
            raise RuntimeError("BHK operation runtime already started")
        self._mark("t0")
        deal_phase = self.prepare()
        self._mark("deal_generation_started")
        outbound = None
        state = self.operation_source_state
        if state is not None:
            result = BHK22OperationDealerRole(
                context=self.application_context,
                protocol_context=self.context,
                state=state,
                phase_qid=deal_phase.qid,
            ).generate_batch()
            fault = self.manifest.fault_profile
            if fault is not None and state.role_id == fault.dealer_role_id:
                result = replace(
                    result,
                    batch=replace(
                        result.batch,
                        base_batch=_inject_dealer_fault(
                            result.batch.base_batch, fault
                        ),
                    ),
                )
            self.dealer_result = result
            outbound = BHK22RuntimeOutbound(
                phase_context=deal_phase,
                role_id=state.role_id,
                batch_bytes=result.batch.to_bytes(),
            )
        self._mark("deal_generation_finished")
        return outbound

    def _decode_dealer_messages(self, entries):
        operation_batches = self._decode_messages(
            entries, BHK22OperationDealerBatch
        )
        self._operation_dealer_batches = operation_batches
        return tuple(item.base_batch for item in operation_batches)

    def _process_deal(self, event, artifact_digest, on_phase_registered):
        operation_batches = self._decode_messages(
            event.entries, BHK22OperationDealerBatch
        )
        verifications = {
            batch.role_id: verify_bhk22_operation_dealer_batch(
                self.application_context,
                batch,
                expected_phase_qid=event.qid,
            )
            for batch in operation_batches
        }
        if not all(item.accepted for item in verifications.values()):
            raise RuntimeError("BHK operation deal contains invalid proof tuple")
        self.operation_verifications = verifications
        self._operation_dealer_batches = operation_batches
        dealers = tuple(item.base_batch for item in operation_batches)
        public_state_digest = _state_digest(
            DEAL_PUBLIC_STATE_DOMAIN,
            {
                "root_context_digest": self.context.root_context_digest,
                "operation_context_digest": self.application_context.context_digest,
                "deal_artifact_digest": artifact_digest,
                "accepted_dealer_ids": [item.role_id for item in operation_batches],
                "dealer_message_digests": [
                    item.message_digest for item in operation_batches
                ],
            },
        )
        verify_phase = self.controller.register_next(
            public_state_digest=public_state_digest
        )
        if on_phase_registered is not None:
            on_phase_registered(verify_phase)
        self._mark("verify_generation_started")
        state = self.states.get(ROLE_VERIFIER)
        outbound = None
        if state is not None:
            self.verifier_result = BHK22VerifierRole(
                context=self.context,
                state=state,
                dealer_batches=dealers,
                deal_phase_qid=event.qid,
                verify_phase_qid=verify_phase.qid,
                public_batch_verifier=lambda batch: self.operation_verifications[
                    batch.role_id
                ].accepted,
            ).generate_batch()
            outbound = BHK22RuntimeOutbound(
                phase_context=verify_phase,
                role_id=state.role_id,
                batch_bytes=self.verifier_result.batch.to_bytes(),
            )
        self._mark("verify_generation_finished")
        return outbound

    def _process_verify(self, event, artifact_digest, on_phase_registered):
        return super()._process_verify(
            event, artifact_digest, on_phase_registered
        )

    def _process_response(self, event, artifact_digest, on_phase_registered):
        operation_entries = tuple(getattr(self, "_deal_entries", ()))
        base_entries = tuple(
            (mid, BHK22OperationDealerBatch.from_bytes(payload).base_batch.to_bytes())
            for mid, payload in operation_entries
        )
        self._deal_entries = base_entries
        try:
            return super()._process_response(
                event, artifact_digest, on_phase_registered
            )
        finally:
            self._deal_entries = operation_entries

    def _delivered_payload(self, phase: str, message_digest: str) -> bytes:
        if phase != PHASE_DEAL:
            return super()._delivered_payload(phase, message_digest)
        for _mid, payload in getattr(self, "_deal_entries", ()):
            message = BHK22OperationDealerBatch.from_bytes(payload)
            if message.message_digest == message_digest:
                return message.base_batch.to_bytes()
        raise KeyError("ordered BHK operation dealer message is unavailable")

    def _complete_encrypted_carry_forward(self, **kwargs) -> None:
        super()._complete_encrypted_carry_forward(**kwargs)
        if not isinstance(self.public_commitment, BHK22EncryptedCommitmentVector):
            raise RuntimeError("BHK operation completion must remain encrypted")
        by_role = {item.role_id: item for item in self._operation_dealer_batches}
        selected = tuple(self.qualified_dealer_ids[: self.context.parameters.t_r + 1])
        points = tuple(role + 1 for role in selected)
        coefficients = tuple(
            lagrange_coeff_at_zero(points, point) for point in points
        )
        output = combine_vk_operation_shares(
            points,
            [by_role[role].operation_statement.operation_share_hex for role in selected],
        ).to_hex()
        output_digest = operation_output_digest(
            operation_id=self.application_context.operation_context.operation_id,
            operation_input=self.application_context.operation_context.operation_input,
            operation_output=output,
        )
        if output_digest != self.application_context.operation_context.expected_operation_output_digest:
            raise RuntimeError("BHK operation output digest mismatch")
        encrypted_successor = BHK22EncryptedSourceVector(tuple(
            BHK22EncryptedSourceCoordinate(
                coordinate_index=role,
                evaluation_point=role + 1,
                generator_hex=self.context.crs.generators[role + 1],
                public_key_hex=self.context.crs.public_value_keys[role + 1],
                ciphertext=self.public_commitment.aggregate_ciphertexts[role],
            )
            for role in range(self.context.parameters.n_r)
        ))
        self.operation_public_result = BHK22OperationPublicResult(
            arm_context_digest=self.application_context.context_digest,
            operation_context_digest=(
                self.application_context.operation_context.context_digest
            ),
            path_mode=self.manifest.path_mode,
            block_count=len(self.controller.artifacts),
            terminal_phase_qid=kwargs["terminal_phase_qid"],
            qualified_dealer_ids=self.qualified_dealer_ids,
            selected_dealer_ids=selected,
            dealer_lagrange_coefficients=coefficients,
            operation_output_hex=output,
            operation_output_digest=output_digest,
            encrypted_successor=encrypted_successor,
            speech_counts={role: 1 for role in by_role},
        )

    def retire(self) -> None:
        if self.operation_source_state is not None:
            self.operation_source_state.wipe()
        super().retire()

    def summary(self, *, rbc_metrics=None, cleanup=None) -> dict[str, Any]:
        result = super().summary(rbc_metrics=rbc_metrics, cleanup=cleanup)
        public = self.operation_public_result
        batch = self.dealer_result.batch if self.dealer_result is not None else None
        operation_binding_bytes = {
            "link_proof": (
                len(batch.link_proof.to_bytes()) if batch is not None else 0
            ),
            "operation_proof": (
                len(batch.operation_proof.to_bytes()) if batch is not None else 0
            ),
            "vk": (
                len(batch.operation_statement.operation_vk_hex.encode("ascii"))
                if batch is not None else 0
            ),
            "operation_share": (
                len(batch.operation_statement.operation_share_hex.encode("ascii"))
                if batch is not None else 0
            ),
        }
        operation_binding_bytes["classified_total"] = sum(
            operation_binding_bytes.values()
        )
        result.update({
            "application": "threshold-operation",
            "metrics_schema": "bhk22-operation-dag-node-metrics/v1",
            "operation_context_digest": self.application_context.operation_context.context_digest,
            "application_context_digest": self.application_context.context_digest,
            "public_result_digest": public.result_digest if public else None,
            "operation_output": public.operation_output_hex if public else None,
            "operation_output_digest": public.operation_output_digest if public else None,
            "public_successor_state_digest": (
                public.encrypted_successor.digest if public else None
            ),
            "selected_dealer_ids": (
                list(public.selected_dealer_ids) if public else []
            ),
            "operation_proof_verification": {
                str(role): dict(verification.checks)
                for role, verification in self.operation_verifications.items()
            },
            "operation_proof_verification_timings_ms": {
                str(role): dict(verification.timings_ms)
                for role, verification in self.operation_verifications.items()
            },
            "operation_proof_verification_cpu_seconds": {
                str(role): dict(verification.cpu_seconds)
                for role, verification in self.operation_verifications.items()
            },
            # Exact serialized proof sizes plus the two public group elements
            # added by the operation extension.  The enclosing statement and
            # context headers remain part of the protocol payload total.
            "operation_binding_bytes": operation_binding_bytes,
            "operation_dealer_batch_bytes": (
                len(batch.to_bytes()) if batch is not None else 0
            ),
            "base_dealer_batch_bytes": (
                len(batch.base_batch.to_bytes()) if batch is not None else 0
            ),
        })
        return result


__all__ = ["BHK22OperationDAGRuntime"]
