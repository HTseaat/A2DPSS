"""Step 9/10 node-local BHK+22 runtime for DAG-TOB."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Sequence

from beaver.baselines.bhk22_context import canonical_json
from beaver.baselines.bhk22_contract import (
    BHK22_QID_KIND,
    COMPLETION_ENCRYPTED_CARRY_FORWARD,
    PATH_ONE_ACCUSATION,
    PATH_OPTIMISTIC,
    PHASE_DEAL,
    PHASE_FINALIZE,
    PHASE_RESPONSE,
    PHASE_VERIFY,
    ROLE_DEALER,
    ROLE_RESPONDER,
    ROLE_SHAREHOLDER,
    ROLE_VERIFIER,
)
from beaver.baselines.bhk22_crypto import BHK22AEADCiphertext
from beaver.baselines.bhk22_dealer import (
    BHK22DealerGenerationResult,
    BHK22DealerRole,
    verify_dealer_batch_public,
)
from beaver.baselines.bhk22_experiment import (
    BHK22CarryForwardSuccessorState,
    BHK22DAGExperimentManifest,
    BHK22FaultProfile,
    BHK22NodePrivateShard,
    BHK22SuccessorResult,
    bhk22_coin_config_digest,
    bhk22_dag_config_digest,
)
from beaver.baselines.bhk22_messages import (
    BHK22DealerBatch,
    BHK22PublicFinalizeOutput,
    BHK22ResponderBatch,
    BHK22VerifierBatch,
)
from beaver.baselines.bhk22_phase import (
    BHK22DeliveryMetadata,
    BHK22PhaseContext,
    BHK22PhaseController,
    BHK22PhaseOrderingJustifier,
)
from beaver.baselines.bhk22_response import (
    BHK22AccusationPreprocessing,
    BHK22ResponderGenerationResult,
    BHK22ResponderRole,
    BHK22ShareholderDisqualificationResult,
    preprocess_accusations,
    resolve_disqualifications,
    verify_responder_batch_public,
)
from beaver.baselines.bhk22_shareholder import (
    BHK22CarryForwardGenerationResult,
    BHK22EncryptedCommitmentVector,
    BHK22PublicCommitmentVector,
    BHK22ShareholderGenerationResult,
    BHK22ShareholderRole,
    build_encrypted_commitment_vector,
    process_finalize_outputs,
)
from beaver.baselines.bhk22_verifier import (
    BHK22VerifierGenerationResult,
    BHK22VerifierRole,
)
from beaver.dag_tob import DeliveryEvent, PoolEntry, Qid


DEAL_PUBLIC_STATE_DOMAIN = b"ATCaaS/BHK22/deal-public-state/v1\x00"
VERIFY_PUBLIC_STATE_DOMAIN = b"ATCaaS/BHK22/verify-public-state/v1\x00"
RESPONSE_PUBLIC_STATE_DOMAIN = b"ATCaaS/BHK22/response-public-state/v1\x00"
BHK22_DAG_NODE_METRICS_SCHEMA = "bhk22-dag-node-metrics/v3"


def _state_digest(domain: bytes, material: dict[str, Any]) -> str:
    return hashlib.sha256(domain + canonical_json(material)).hexdigest()


def _inject_dealer_fault(
    batch: BHK22DealerBatch, profile: BHK22FaultProfile
) -> BHK22DealerBatch:
    if batch.role_id != profile.dealer_role_id:
        return batch
    columns = list(batch.verifier_columns)
    verifier = profile.verifier_role_id
    column = columns[verifier]
    ciphertext = column.encrypted_row
    bad_tag = "00" * 16 if ciphertext.tag != "00" * 16 else "11" * 16
    columns[verifier] = replace(
        column,
        encrypted_row=BHK22AEADCiphertext(
            nonce=ciphertext.nonce,
            body=ciphertext.body,
            tag=bad_tag,
        ),
    )
    return replace(batch, verifier_columns=tuple(columns))


@dataclass(frozen=True)
class BHK22RuntimeOutbound:
    phase_context: BHK22PhaseContext
    role_id: int
    batch_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.phase_context, BHK22PhaseContext):
            raise TypeError("runtime outbound requires BHK22PhaseContext")
        if isinstance(self.role_id, bool) or not isinstance(self.role_id, int):
            raise TypeError("runtime outbound role_id must be an integer")
        object.__setattr__(self, "batch_bytes", bytes(self.batch_bytes))


class BHK22OptimisticDAGRuntime:
    """Own one node's role states across one configured BHK22 handover."""

    def __init__(
        self,
        *,
        manifest: BHK22DAGExperimentManifest,
        local_shard: BHK22NodePrivateShard,
        my_node_id: int,
    ) -> None:
        if not isinstance(manifest, BHK22DAGExperimentManifest):
            raise TypeError("manifest must be BHK22DAGExperimentManifest")
        if not isinstance(local_shard, BHK22NodePrivateShard):
            raise TypeError("local_shard must be BHK22NodePrivateShard")
        if manifest.path_mode not in {PATH_OPTIMISTIC, PATH_ONE_ACCUSATION}:
            raise ValueError("unsupported BHK22 DAG runtime path")
        if local_shard.node_id != int(my_node_id):
            raise ValueError("BHK22 local shard belongs to another node")
        local_shard.validate(manifest)
        self.manifest = manifest
        self.local_shard = local_shard
        self.my_node_id = int(my_node_id)
        self.context = manifest.context
        self.justifier = BHK22PhaseOrderingJustifier()
        self.controller = BHK22PhaseController(
            context=self.context,
            path_mode=manifest.path_mode,
            expected_active_roles=manifest.expected_active_roles,
            justifier=self.justifier,
            completion_mode=manifest.completion_mode,
        )
        self.states = {
            material.role_type: material.new_state()
            for material in local_shard.role_materials
        }
        self.dealer_result: BHK22DealerGenerationResult | None = None
        self.verifier_result: BHK22VerifierGenerationResult | None = None
        self.responder_result: BHK22ResponderGenerationResult | None = None
        self.shareholder_result: BHK22ShareholderGenerationResult | None = None
        self.carry_forward_result: BHK22CarryForwardGenerationResult | None = None
        self.successor_result: (
            BHK22SuccessorResult | BHK22CarryForwardSuccessorState | None
        ) = None
        self.preprocessing: BHK22AccusationPreprocessing | None = None
        self.disqualification: BHK22ShareholderDisqualificationResult | None = None
        self.public_commitment: (
            BHK22PublicCommitmentVector | BHK22EncryptedCommitmentVector | None
        ) = None
        self.qualified_dealer_ids: tuple[int, ...] = ()
        self.events: list[dict[str, Any]] = []
        self._clock: dict[str, float] = {}
        self._cpu: dict[str, float] = {}
        self._failure_reason: str | None = None
        self.application_verification_metrics: dict[str, dict[str, Any]] = {}

    @property
    def is_successor(self) -> bool:
        return ROLE_SHAREHOLDER in self.states

    @property
    def completed(self) -> bool:
        return self.public_commitment is not None and (
            not self.is_successor or self.successor_result is not None
        )

    def validate_network_parameters(
        self,
        *,
        M: int,
        n_r: int,
        t_r: int,
        n_a: int,
        t_a: int,
        m_a: int,
        quorum_h: int,
        coin_c: int,
        coin_t: int,
        coin_k: int,
        trial_epoch: str | int,
    ) -> None:
        expected = self.context.parameters
        actual = (
            M,
            n_r,
            t_r,
            n_a,
            t_a,
            m_a,
            quorum_h,
            coin_c,
            coin_t,
            coin_k,
        )
        frozen = (
            expected.M,
            expected.n_r,
            expected.t_r,
            expected.n_a,
            expected.t_a,
            expected.m_a,
            expected.quorum_h,
            expected.coin_c,
            expected.coin_t,
            expected.coin_k,
        )
        if tuple(map(int, actual)) != frozen:
            raise ValueError("BHK22 manifest parameters do not match DAG process")
        if str(trial_epoch) != str(self.context.trial_epoch):
            raise ValueError("BHK22 trial epoch does not match DAG run")

    def validate_context_bindings(
        self,
        *,
        role_trace_digest: str,
        coin_config_extras: dict[str, Any],
    ) -> None:
        if (
            bhk22_dag_config_digest(self.context.parameters)
            != self.context.dag_config_digest
        ):
            raise ValueError("BHK22 manifest DAG config digest mismatch")
        if (
            bhk22_coin_config_digest(coin_config_extras)
            != self.context.coin_config_digest
        ):
            raise ValueError("BHK22 manifest coin config digest mismatch")
        if role_trace_digest != self.context.role_trace_digest:
            raise ValueError("BHK22 manifest role-trace digest mismatch")

    def _material_role_id(self, role_type: str) -> int | None:
        state = self.states.get(role_type)
        return state.role_id if state is not None else None

    def _mark(self, name: str) -> None:
        self._clock[name] = time.perf_counter()
        self._cpu[name] = time.process_time()

    def _record_application_verification(
        self,
        stage: str,
        *,
        elapsed_ms: float,
        cpu_seconds: float,
        invocations: int,
        detail: dict[str, Any] | None = None,
    ) -> None:
        metric = self.application_verification_metrics.setdefault(
            stage,
            {
                "elapsed_ms": 0.0,
                "cpu_seconds": 0.0,
                "invocations": 0,
                "detail": {},
            },
        )
        metric["elapsed_ms"] += float(elapsed_ms)
        metric["cpu_seconds"] += float(cpu_seconds)
        metric["invocations"] += int(invocations)
        for name, value in (detail or {}).items():
            metric["detail"][name] = metric["detail"].get(name, 0.0) + float(value)

    def prepare(self) -> BHK22PhaseContext:
        """Register deal before the release barrier without consuming a role."""

        if self.controller.state == "INIT":
            return self.controller.start()
        if (
            self.controller.state == "DEAL_REGISTERED"
            and self.controller.current is not None
        ):
            return self.controller.current
        raise RuntimeError("BHK22 runtime cannot prepare after deal processing")

    def start(self) -> BHK22RuntimeOutbound | None:
        if "t0" in self._clock:
            raise RuntimeError("BHK22 runtime already started")
        self._mark("t0")
        deal_phase = self.prepare()
        self._mark("deal_generation_started")
        state = self.states.get(ROLE_DEALER)
        outbound = None
        if state is not None:
            self.dealer_result = BHK22DealerRole(
                context=self.context,
                state=state,
                phase_qid=deal_phase.qid,
            ).generate_batch()
            fault = self.manifest.fault_profile
            if fault is not None and state.role_id == fault.dealer_role_id:
                self.dealer_result = replace(
                    self.dealer_result,
                    batch=_inject_dealer_fault(self.dealer_result.batch, fault),
                )
                self.events.append(
                    {
                        "event": "fixture_fault_injected",
                        "phase": PHASE_DEAL,
                        **fault.to_obj(),
                    }
                )
            outbound = BHK22RuntimeOutbound(
                phase_context=deal_phase,
                role_id=state.role_id,
                batch_bytes=self.dealer_result.batch.to_bytes(),
            )
        self._mark("deal_generation_finished")
        self.events.append(
            {
                "event": "phase_generated",
                "phase": PHASE_DEAL,
                "qid": deal_phase.qid,
                "local_role_id": self._material_role_id(ROLE_DEALER),
                "has_outbound": outbound is not None,
            }
        )
        return outbound

    def mark_sent(self, phase: str) -> None:
        self._mark(f"{phase}_sent")

    def mark_pool_ready(self, phase: str) -> None:
        name = f"{phase}_pool_ready"
        if name not in self._clock:
            self._mark(name)

    @staticmethod
    def _delivery_metadata(event: DeliveryEvent) -> BHK22DeliveryMetadata:
        return BHK22DeliveryMetadata(
            sequence=event.sequence,
            ordered_round=event.ordered_vertex[0],
            ordered_source=event.ordered_vertex[1],
            committed_wave=event.committed_wave,
        )

    @staticmethod
    def _decode_messages(
        entries: Sequence[PoolEntry], message_type: type
    ) -> tuple[Any, ...]:
        messages = tuple(message_type.from_bytes(payload) for _mid, payload in entries)
        return tuple(sorted(messages, key=lambda message: message.role_id))

    def _process_deal(
        self,
        event: DeliveryEvent,
        artifact_digest: str,
        on_phase_registered: Callable[[BHK22PhaseContext], None] | None,
    ) -> BHK22RuntimeOutbound | None:
        dealers = self._decode_messages(event.entries, BHK22DealerBatch)
        verification = tuple(
            verify_dealer_batch_public(
                self.context,
                dealer,
                expected_phase_qid=event.qid,
            )
            for dealer in dealers
        )
        self._record_application_verification(
            "dealer_batch_public",
            elapsed_ms=sum(item.verify_ms for item in verification),
            cpu_seconds=sum(item.verify_cpu_seconds for item in verification),
            invocations=len(verification),
            detail={
                "figure7_proof_verify_ms": sum(
                    item.figure7_proof_verify_ms for item in verification
                ),
                "figure7_proof_verify_cpu_seconds": sum(
                    item.figure7_proof_verify_cpu_seconds for item in verification
                ),
                "figure8_proof_verify_ms": sum(
                    item.figure8_proof_verify_ms for item in verification
                ),
                "figure8_proof_verify_cpu_seconds": sum(
                    item.figure8_proof_verify_cpu_seconds for item in verification
                ),
            },
        )
        if not all(item.accepted for item in verification):
            raise RuntimeError("BHK22 deal delivery contains invalid public dealer")
        public_state_digest = _state_digest(
            DEAL_PUBLIC_STATE_DOMAIN,
            {
                "root_context_digest": self.context.root_context_digest,
                "deal_artifact_digest": artifact_digest,
                "accepted_dealer_ids": [dealer.role_id for dealer in dealers],
                "dealer_message_digests": [dealer.message_digest for dealer in dealers],
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
            ).generate_batch()
            outbound = BHK22RuntimeOutbound(
                phase_context=verify_phase,
                role_id=state.role_id,
                batch_bytes=self.verifier_result.batch.to_bytes(),
            )
        self._mark("verify_generation_finished")
        self.events.append(
            {
                "event": "phase_generated",
                "phase": PHASE_VERIFY,
                "qid": verify_phase.qid,
                "local_role_id": self._material_role_id(ROLE_VERIFIER),
                "has_outbound": outbound is not None,
            }
        )
        return outbound

    def _generate_shareholder_outbound(
        self,
        *,
        dealers: Sequence[BHK22DealerBatch],
        verifiers: Sequence[BHK22VerifierBatch],
        disqualification: BHK22ShareholderDisqualificationResult,
        deal_phase_qid: str,
        verify_phase_qid: str,
        finalize_phase: BHK22PhaseContext,
    ) -> BHK22RuntimeOutbound | None:
        self.disqualification = disqualification
        self.qualified_dealer_ids = tuple(disqualification.qualified_dealer_ids)
        if len(self.qualified_dealer_ids) < self.context.parameters.t_r + 1:
            raise RuntimeError("BHK22 path has fewer than t_r+1 qualified dealers")
        self._mark("finalize_generation_started")
        state = self.states.get(ROLE_SHAREHOLDER)
        outbound = None
        if state is not None:
            self.shareholder_result = BHK22ShareholderRole(
                context=self.context,
                state=state,
                dealer_batches=dealers,
                verifier_batches=verifiers,
                disqualification=disqualification,
                deal_phase_qid=deal_phase_qid,
                verify_phase_qid=verify_phase_qid,
                finalize_phase_qid=finalize_phase.qid,
                dealer_public_verifier=getattr(
                    self, "dealer_public_verifier", None
                ),
            ).generate_output()
            self.successor_result = self.shareholder_result.successor_result
            outbound = BHK22RuntimeOutbound(
                phase_context=finalize_phase,
                role_id=state.role_id,
                batch_bytes=self.shareholder_result.public_output.to_bytes(),
            )
        self._mark("finalize_generation_finished")
        self.events.append(
            {
                "event": "phase_generated",
                "phase": PHASE_FINALIZE,
                "qid": finalize_phase.qid,
                "local_role_id": self._material_role_id(ROLE_SHAREHOLDER),
                "has_outbound": outbound is not None,
            }
        )
        return outbound

    def _complete_encrypted_carry_forward(
        self,
        *,
        dealers: Sequence[BHK22DealerBatch],
        verifiers: Sequence[BHK22VerifierBatch],
        disqualification: BHK22ShareholderDisqualificationResult,
        deal_phase_qid: str,
        verify_phase_qid: str,
        terminal_phase_qid: str,
        terminal_artifact_digest: str,
    ) -> None:
        """Finish without shareholder speech and retain next-dealer secrets."""

        self.disqualification = disqualification
        self.qualified_dealer_ids = tuple(disqualification.qualified_dealer_ids)
        if len(self.qualified_dealer_ids) < self.context.parameters.t_r + 1:
            raise RuntimeError("BHK22 path has fewer than t_r+1 qualified dealers")
        self.public_commitment = build_encrypted_commitment_vector(
            self.context,
            dealers,
            deal_phase_qid=deal_phase_qid,
            terminal_phase_qid=terminal_phase_qid,
            terminal_artifact_digest=terminal_artifact_digest,
            expected_qualified_dealer_ids=self.qualified_dealer_ids,
        )
        self._mark("public_finalized")
        state = self.states.get(ROLE_SHAREHOLDER)
        if state is not None:
            self._mark("carry_forward_reconstruction_started")
            self.carry_forward_result = BHK22ShareholderRole(
                context=self.context,
                state=state,
                dealer_batches=dealers,
                verifier_batches=verifiers,
                disqualification=disqualification,
                deal_phase_qid=deal_phase_qid,
                verify_phase_qid=verify_phase_qid,
                finalize_phase_qid=terminal_phase_qid,
                dealer_public_verifier=getattr(
                    self, "dealer_public_verifier", None
                ),
            ).generate_carry_forward_state(
                terminal_phase_qid=terminal_phase_qid
            )
            self.successor_result = self.carry_forward_result.successor_state
            role_id = self.successor_result.shareholder_role_id
            expected_ciphertext = self.public_commitment.aggregate_ciphertexts[role_id]
            if self.successor_result.aggregate_ciphertext != expected_ciphertext:
                raise RuntimeError("local carry-forward ciphertext differs from public result")
            self._mark("carry_forward_reconstruction_finished")
            self._mark("successor_done")
            self._clock["frozen_at"] = self._clock["successor_done"]
            self._cpu["frozen_at"] = self._cpu["successor_done"]
        else:
            self._clock["frozen_at"] = self._clock["public_finalized"]
            self._cpu["frozen_at"] = self._cpu["public_finalized"]
        self.events.append(
            {
                "event": "encrypted_carry_forward_completed",
                "phase": self.controller.current.phase,
                "qid": terminal_phase_qid,
                "public_commitment_digest": self.public_commitment.commitment_digest,
                "local_successor": state is not None,
            }
        )

    def _process_verify(
        self,
        event: DeliveryEvent,
        artifact_digest: str,
        on_phase_registered: Callable[[BHK22PhaseContext], None] | None,
    ) -> BHK22RuntimeOutbound | None:
        deal_phase = self.controller.phase_contexts[PHASE_DEAL]
        deal_artifact = self.controller.artifacts[PHASE_DEAL]
        dealer_entries = tuple(
            (
                evidence.mid,
                self._delivered_payload(PHASE_DEAL, evidence.message_digest),
            )
            for evidence in deal_artifact.entries
        )
        dealers = self._decode_messages(dealer_entries, BHK22DealerBatch)
        verifiers = self._decode_messages(event.entries, BHK22VerifierBatch)
        verification_started = time.perf_counter()
        verification_cpu_started = time.process_time()
        preprocessing = preprocess_accusations(
            self.context,
            dealers,
            verifiers,
            deal_phase_qid=deal_phase.qid,
            verify_phase_qid=event.qid,
            dealer_public_verifier=getattr(
                self, "dealer_public_verifier", None
            ),
        )
        self._record_application_verification(
            "verify_phase_preprocessing",
            elapsed_ms=(time.perf_counter() - verification_started) * 1000.0,
            cpu_seconds=time.process_time() - verification_cpu_started,
            invocations=len(verifiers),
        )
        self.preprocessing = preprocessing
        if (
            preprocessing.invalid_verifier_ids
            or preprocessing.ignored_verifier_ids
            or preprocessing.pre_disqualified_dealer_ids
        ):
            raise RuntimeError("BHK22 fixture produced unexpected preprocessing sets")
        fault = self.manifest.fault_profile
        expected_accusations = (
            fault.expected_accusations if fault is not None else ()
        )
        if tuple(preprocessing.remaining_accusations) != expected_accusations:
            raise RuntimeError("BHK22 accusation set differs from fixture")
        response_required = self.manifest.path_mode == PATH_ONE_ACCUSATION
        if preprocessing.response_required != response_required:
            raise RuntimeError("BHK22 response branch differs from fixture path")
        public_state_digest = _state_digest(
            VERIFY_PUBLIC_STATE_DOMAIN,
            {
                "root_context_digest": self.context.root_context_digest,
                "verify_artifact_digest": artifact_digest,
                "preprocessing_digest": preprocessing.preprocessing_digest,
            },
        )
        if response_required:
            response_phase = self.controller.register_next(
                public_state_digest=public_state_digest,
                response_required=True,
            )
            if on_phase_registered is not None:
                on_phase_registered(response_phase)
            self._mark("response_generation_started")
            state = self.states.get(ROLE_RESPONDER)
            outbound = None
            if state is not None:
                self.responder_result = BHK22ResponderRole(
                    context=self.context,
                    state=state,
                    dealer_batches=dealers,
                    preprocessing=preprocessing,
                    response_phase_qid=response_phase.qid,
                ).generate_batch()
                if self.responder_result.failed_dealer_ids:
                    raise RuntimeError("fixture responder failed to decrypt future state")
                outbound = BHK22RuntimeOutbound(
                    phase_context=response_phase,
                    role_id=state.role_id,
                    batch_bytes=self.responder_result.batch.to_bytes(),
                )
            self._mark("response_generation_finished")
            self.events.append(
                {
                    "event": "phase_generated",
                    "phase": PHASE_RESPONSE,
                    "qid": response_phase.qid,
                    "local_role_id": self._material_role_id(ROLE_RESPONDER),
                    "has_outbound": outbound is not None,
                }
            )
            return outbound

        state = self.states.get(ROLE_SHAREHOLDER)
        disqualification = resolve_disqualifications(
            self.context,
            dealers,
            (),
            preprocessing,
            response_phase_qid="optimistic-no-response",
            shareholder_role_id=state.role_id if state is not None else 0,
        )
        if self.manifest.completion_mode == COMPLETION_ENCRYPTED_CARRY_FORWARD:
            self._complete_encrypted_carry_forward(
                dealers=dealers,
                verifiers=verifiers,
                disqualification=disqualification,
                deal_phase_qid=deal_phase.qid,
                verify_phase_qid=event.qid,
                terminal_phase_qid=event.qid,
                terminal_artifact_digest=artifact_digest,
            )
            return None
        finalize_phase = self.controller.register_next(
            public_state_digest=public_state_digest,
            response_required=False,
        )
        if on_phase_registered is not None:
            on_phase_registered(finalize_phase)
        return self._generate_shareholder_outbound(
            dealers=dealers,
            verifiers=verifiers,
            disqualification=disqualification,
            deal_phase_qid=deal_phase.qid,
            verify_phase_qid=event.qid,
            finalize_phase=finalize_phase,
        )

    def _process_response(
        self,
        event: DeliveryEvent,
        artifact_digest: str,
        on_phase_registered: Callable[[BHK22PhaseContext], None] | None,
    ) -> BHK22RuntimeOutbound | None:
        if self.preprocessing is None or not self.preprocessing.response_required:
            raise RuntimeError("BHK22 response delivery lacks accusation preprocessing")
        deal_phase = self.controller.phase_contexts[PHASE_DEAL]
        verify_phase = self.controller.phase_contexts[PHASE_VERIFY]
        dealers = self._decode_messages(
            getattr(self, "_deal_entries", ()), BHK22DealerBatch
        )
        verifiers = self._decode_messages(
            getattr(self, "_verify_entries", ()), BHK22VerifierBatch
        )
        responses = self._decode_messages(event.entries, BHK22ResponderBatch)
        public_verification = tuple(
            verify_responder_batch_public(
                self.context,
                response,
                dealers,
                self.preprocessing,
                expected_response_phase_qid=event.qid,
            )
            for response in responses
        )
        self._record_application_verification(
            "responder_batch_public",
            elapsed_ms=sum(item.verify_ms for item in public_verification),
            cpu_seconds=sum(item.verify_cpu_seconds for item in public_verification),
            invocations=len(public_verification),
        )
        if not all(item.accepted for item in public_verification):
            raise RuntimeError("BHK22 response delivery contains invalid opening")
        state = self.states.get(ROLE_SHAREHOLDER)
        disqualification = resolve_disqualifications(
            self.context,
            dealers,
            responses,
            self.preprocessing,
            response_phase_qid=event.qid,
            shareholder_role_id=state.role_id if state is not None else 0,
        )
        fault = self.manifest.fault_profile
        if fault is None:
            raise RuntimeError("BHK22 response phase has no fixture fault")
        if (
            tuple(disqualification.disqualified_dealer_ids)
            != fault.expected_disqualified_dealer_ids
            or disqualification.accusation_outcomes.get(
                fault.expected_accusations[0]
            )
            != "dealer-row-invalid"
        ):
            raise RuntimeError("BHK22 response did not resolve fixture fault")
        public_state_digest = _state_digest(
            RESPONSE_PUBLIC_STATE_DOMAIN,
            {
                "root_context_digest": self.context.root_context_digest,
                "response_artifact_digest": artifact_digest,
                "preprocessing_digest": self.preprocessing.preprocessing_digest,
                "response_message_digests": [
                    response.message_digest for response in responses
                ],
                "qualified_dealer_ids": list(
                    disqualification.qualified_dealer_ids
                ),
                "disqualified_dealer_ids": list(
                    disqualification.disqualified_dealer_ids
                ),
                "accusation_outcomes": [
                    {
                        "dealer": dealer,
                        "verifier": verifier,
                        "status": status,
                    }
                    for (dealer, verifier), status in sorted(
                        disqualification.accusation_outcomes.items()
                    )
                ],
            },
        )
        if self.manifest.completion_mode == COMPLETION_ENCRYPTED_CARRY_FORWARD:
            self._complete_encrypted_carry_forward(
                dealers=dealers,
                verifiers=verifiers,
                disqualification=disqualification,
                deal_phase_qid=deal_phase.qid,
                verify_phase_qid=verify_phase.qid,
                terminal_phase_qid=event.qid,
                terminal_artifact_digest=artifact_digest,
            )
            return None
        finalize_phase = self.controller.register_next(
            public_state_digest=public_state_digest
        )
        if on_phase_registered is not None:
            on_phase_registered(finalize_phase)
        return self._generate_shareholder_outbound(
            dealers=dealers,
            verifiers=verifiers,
            disqualification=disqualification,
            deal_phase_qid=deal_phase.qid,
            verify_phase_qid=verify_phase.qid,
            finalize_phase=finalize_phase,
        )

    def _process_finalize(self, event: DeliveryEvent) -> None:
        deal_phase = self.controller.phase_contexts[PHASE_DEAL]
        deal_artifact = self.controller.artifacts[PHASE_DEAL]
        dealer_entries = tuple(
            (
                evidence.mid,
                self._delivered_payload(PHASE_DEAL, evidence.message_digest),
            )
            for evidence in deal_artifact.entries
        )
        dealers = self._decode_messages(dealer_entries, BHK22DealerBatch)
        outputs = self._decode_messages(event.entries, BHK22PublicFinalizeOutput)
        verification_started = time.perf_counter()
        verification_cpu_started = time.process_time()
        self.public_commitment = process_finalize_outputs(
            self.context,
            outputs,
            dealers,
            deal_phase_qid=deal_phase.qid,
            finalize_phase_qid=event.qid,
            expected_qualified_dealer_ids=self.qualified_dealer_ids
            or tuple(range(self.context.parameters.n_r)),
        )
        self._record_application_verification(
            "finalize_outputs_public",
            elapsed_ms=(time.perf_counter() - verification_started) * 1000.0,
            cpu_seconds=time.process_time() - verification_cpu_started,
            invocations=len(outputs),
        )
        if self.shareholder_result is not None:
            local_role = self.shareholder_result.public_output.role_id
            if outputs[local_role] != self.shareholder_result.public_output:
                raise RuntimeError("ordered finalize output differs from local speech")
        self._mark("public_finalized")
        if self.is_successor:
            if self.successor_result is None:
                raise RuntimeError("local successor result is missing")
            self._mark("successor_done")
            self._clock["frozen_at"] = self._clock["successor_done"]
            self._cpu["frozen_at"] = self._cpu["successor_done"]
        else:
            self._clock["frozen_at"] = self._clock["public_finalized"]
            self._cpu["frozen_at"] = self._cpu["public_finalized"]

    def handle_delivery(
        self,
        event: DeliveryEvent,
        *,
        on_phase_registered: Callable[[BHK22PhaseContext], None] | None = None,
    ) -> BHK22RuntimeOutbound | None:
        if not isinstance(event, DeliveryEvent):
            raise TypeError("BHK22 runtime requires DeliveryEvent")
        if event.qid not in {
            context.qid for context in self.controller.phase_contexts.values()
        }:
            raise ValueError("delivery is not a registered BHK22 phase")
        phase_context = self.controller.current
        if phase_context is None or event.qid != phase_context.qid:
            raise ValueError("BHK22 delivery does not match current phase")
        phase = phase_context.phase
        self._mark(f"{phase}_ordered")
        # Keep delivered public bytes for later processors. This is an audit
        # cache, never a private-state bridge.
        setattr(self, f"_{phase}_entries", tuple(event.entries))
        artifact = self.controller.on_delivery(
            event.qid, event.entries, self._delivery_metadata(event)
        )
        if phase == PHASE_DEAL:
            outbound = self._process_deal(
                event, artifact.artifact_digest, on_phase_registered
            )
        elif phase == PHASE_VERIFY:
            outbound = self._process_verify(
                event, artifact.artifact_digest, on_phase_registered
            )
        elif phase == PHASE_RESPONSE:
            outbound = self._process_response(
                event, artifact.artifact_digest, on_phase_registered
            )
        elif phase == PHASE_FINALIZE:
            self._process_finalize(event)
            outbound = None
        else:
            raise RuntimeError("BHK22 runtime received an unsupported phase")
        self._mark(f"{phase}_processed")
        self.events.append(
            {
                "event": "phase_delivered",
                "phase": phase,
                "qid": event.qid,
                "artifact_digest": artifact.artifact_digest,
                "entry_count": len(event.entries),
                "next_phase": (
                    outbound.phase_context.phase if outbound is not None else None
                ),
            }
        )
        return outbound

    def _delivered_payload(self, phase: str, message_digest: str) -> bytes:
        entries = getattr(self, f"_{phase}_entries", ())
        message_type = {
            PHASE_DEAL: BHK22DealerBatch,
            PHASE_VERIFY: BHK22VerifierBatch,
            PHASE_RESPONSE: BHK22ResponderBatch,
            PHASE_FINALIZE: BHK22PublicFinalizeOutput,
        }[phase]
        for _mid, payload in entries:
            message = message_type.from_bytes(payload)
            if message.message_digest == message_digest:
                return bytes(payload)
        raise KeyError("ordered BHK22 message digest is unavailable")

    def retire(self) -> None:
        self.controller.retire()
        for state in self.states.values():
            state.wipe()
        if self.successor_result is not None:
            self.successor_result.wipe()

    def summary(self, *, rbc_metrics=None, cleanup: dict[str, Any] | None = None) -> dict[str, Any]:
        t0 = self._clock.get("t0")

        def elapsed(name: str) -> float | None:
            value = self._clock.get(name)
            return value - t0 if value is not None and t0 is not None else None

        phases = (PHASE_DEAL, PHASE_VERIFY, PHASE_RESPONSE, PHASE_FINALIZE)
        timeline = {
            "t0": 0.0 if t0 is not None else None,
            **{
                f"t_{phase}_{stage}": elapsed(f"{phase}_{stage}")
                for phase in phases
                for stage in (
                    "generation_started",
                    "generation_finished",
                    "sent",
                    "pool_ready",
                    "ordered",
                    "processed",
                )
            },
            "t_public_finalized": elapsed("public_finalized"),
            "t_successor_done": elapsed("successor_done"),
            "frozen_at": elapsed("frozen_at"),
        }
        role_metrics = {}
        for role, result in (
            (ROLE_DEALER, self.dealer_result),
            (ROLE_VERIFIER, self.verifier_result),
            (ROLE_RESPONDER, self.responder_result),
            (ROLE_SHAREHOLDER, self.shareholder_result),
        ):
            if result is not None:
                metrics = getattr(result, "metrics", None)
                if metrics is not None:
                    role_metrics[role] = asdict(metrics)
                elif hasattr(result, "generation_ms"):
                    role_metrics[role] = {
                        "generation_ms": float(result.generation_ms),
                        "generation_cpu_seconds": float(
                            getattr(result, "generation_cpu_seconds", 0.0)
                        ),
                    }
        if self.carry_forward_result is not None:
            role_metrics[ROLE_SHAREHOLDER] = asdict(
                self.carry_forward_result.metrics
            )
        return {
            "protocol": "bhk22",
            "transport": "dag-tob",
            "metrics_schema": BHK22_DAG_NODE_METRICS_SCHEMA,
            "trial_id": self.context.trial_id,
            "trial_epoch": self.context.trial_epoch,
            "node": self.my_node_id,
            "manifest_digest": self.manifest.digest(),
            "root_context_digest": self.context.root_context_digest,
            "common_workload_digest": self.manifest.common_workload_digest,
            "path_mode": self.manifest.path_mode,
            "completion_mode": self.manifest.completion_mode,
            "fault_profile": (
                self.manifest.fault_profile.to_obj()
                if self.manifest.fault_profile is not None
                else None
            ),
            "parameters": self.context.parameters.to_config_obj(),
            "status": "success" if self.completed else "incomplete",
            "failure_reason": self._failure_reason,
            "committed_application_blocks": len(self.controller.artifacts),
            "phase_qids": {
                phase: context.qid
                for phase, context in self.controller.phase_contexts.items()
            },
            "artifact_digests": {
                phase: artifact.artifact_digest
                for phase, artifact in self.controller.artifacts.items()
            },
            "accusation_count": (
                len(self.preprocessing.remaining_accusations)
                if self.preprocessing is not None
                else None
            ),
            "accusation_pairs": (
                [list(pair) for pair in self.preprocessing.remaining_accusations]
                if self.preprocessing is not None
                else None
            ),
            "disqualified_dealer_ids": (
                list(self.disqualification.disqualified_dealer_ids)
                if self.disqualification is not None
                else []
            ),
            "accusation_outcomes": (
                [
                    {
                        "dealer": dealer,
                        "verifier": verifier,
                        "status": status,
                    }
                    for (dealer, verifier), status in sorted(
                        self.disqualification.accusation_outcomes.items()
                    )
                ]
                if self.disqualification is not None
                else []
            ),
            "response_entries_from_local_state": (
                self.responder_result is not None
                and not self.responder_result.failed_dealer_ids
                and self.responder_result.metrics.produced_openings
                == len(self.preprocessing.remaining_accusations)
                if self.preprocessing is not None
                and self.preprocessing.response_required
                else None
            ),
            "qualified_dealer_ids": list(self.qualified_dealer_ids),
            "public_commitment_digest": (
                self.public_commitment.commitment_digest
                if self.public_commitment is not None
                else None
            ),
            "public_commitment_values": (
                list(self.public_commitment.public_values)
                if isinstance(self.public_commitment, BHK22PublicCommitmentVector)
                else None
            ),
            "encrypted_commitment_ciphertexts": (
                [
                    item.to_obj()
                    for item in self.public_commitment.aggregate_ciphertexts
                ]
                if isinstance(self.public_commitment, BHK22EncryptedCommitmentVector)
                else None
            ),
            "public_commitment_kind": (
                "plaintext"
                if isinstance(self.public_commitment, BHK22PublicCommitmentVector)
                else "encrypted"
                if isinstance(self.public_commitment, BHK22EncryptedCommitmentVector)
                else None
            ),
            "is_successor": self.is_successor,
            "successor_completed": self.successor_result is not None,
            "timeline": timeline,
            "handover_latency_sec": elapsed(
                "successor_done" if self.is_successor else "public_finalized"
            ),
            "committee_selection": {
                role: {
                    "selection_mode": committee.selection_mode,
                    "selection_attempt": committee.selection_attempt,
                    "role_to_node": {
                        str(role_id): node_id
                        for role_id, node_id in committee.role_to_node.items()
                    },
                }
                for role, committee in self.context.committees.items()
            },
            "role_metrics": role_metrics,
            "application_verification_metrics": {
                name: {
                    **metric,
                    "detail": dict(metric["detail"]),
                }
                for name, metric in self.application_verification_metrics.items()
            },
            "structural_predicate_metrics": self.justifier.metrics_snapshot(),
            "rbc_metrics": rbc_metrics,
            "controller": self.controller.snapshot(),
            "cleanup": cleanup,
            "events": list(self.events),
        }


def run_mocked_handover(bundle) -> tuple[dict[str, Any], ...]:
    """Run one fixture path with shared public bytes and independent states."""

    runtimes = tuple(
        BHK22OptimisticDAGRuntime(
            manifest=bundle.manifest,
            local_shard=bundle.shards[node],
            my_node_id=node,
        )
        for node in sorted(bundle.shards)
    )
    initial = []
    for runtime in runtimes:
        item = runtime.start()
        if item is not None:
            runtime.mark_sent(PHASE_DEAL)
            initial.append(item)
    outbound = tuple(initial)
    sequence = 1
    round_number = 4
    while outbound:
        qids = {item.phase_context.qid for item in outbound}
        if len(qids) != 1:
            raise RuntimeError("mocked BHK22 observers derived different phase qids")
        phase = outbound[0].phase_context.phase
        expected = outbound[0].phase_context.expected_role_ids
        if tuple(sorted(item.role_id for item in outbound)) != expected:
            raise RuntimeError("mocked BHK22 phase lacks exact active roles")
        entries = tuple(
            sorted(
                (
                    (
                        f"node_{item.phase_context.context.committees[item.phase_context.role_type].role_to_node[item.role_id]}:"
                        f"bhk22:{item.phase_context.role_type}:{item.role_id}",
                        item.batch_bytes,
                    )
                    for item in outbound
                ),
                key=lambda entry: entry[0],
            )
        )
        for runtime in runtimes:
            runtime.mark_pool_ready(phase)
        event = DeliveryEvent(
            sequence=sequence,
            qid=next(iter(qids)),
            entries=entries,
            ordered_vertex=(round_number, 0),
            committed_wave=sequence,
        )
        next_outbound = []
        for runtime in runtimes:
            item = runtime.handle_delivery(event)
            if item is not None:
                runtime.mark_sent(item.phase_context.phase)
                next_outbound.append(item)
        outbound = tuple(next_outbound)
        sequence += 1
        round_number += 4
    if not all(runtime.completed for runtime in runtimes):
        raise RuntimeError("mocked BHK22 handover did not complete")
    summaries = tuple(runtime.summary() for runtime in runtimes)
    for runtime in runtimes:
        runtime.retire()
    return summaries


run_mocked_optimistic_handover = run_mocked_handover
BHK22DAGRuntime = BHK22OptimisticDAGRuntime


__all__ = [
    "BHK22OptimisticDAGRuntime",
    "BHK22DAGRuntime",
    "BHK22RuntimeOutbound",
    "run_mocked_handover",
    "run_mocked_optimistic_handover",
]
