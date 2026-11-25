"""BHK+22 shareholder reconstruction and Figure-8 finalization (Step 6)."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from time import perf_counter, process_time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from beaver.baselines.bhk22_context import (
    BHK22PublicContext,
    canonical_json,
    canonical_point,
    decode_object,
    digest_hex,
    exact_fields,
    nonempty_text,
    strict_int,
)
from beaver.baselines.bhk22_contract import (
    ROLE_SHAREHOLDER,
    ROLE_VERIFIER,
    validate_public_object,
)
from beaver.baselines.bhk22_crypto import (
    BHK22ElGamalCiphertext,
    BHK22ProofContext,
    combine_public_value_ciphertexts,
    decrypt_public_value,
    proof_size_bytes,
    prove_correct_decryption,
    verify_correct_decryption,
)
from beaver.baselines.bhk22_experiment import (
    BHK22CarryForwardSuccessorState,
    BHK22ShareholderState,
    BHK22SuccessorResult,
)
from beaver.baselines.bhk22_messages import (
    BHK22DealerBatch,
    BHK22PublicFinalizeOutput,
    BHK22VerifierBatch,
)
from beaver.baselines.bhk22_response import BHK22ShareholderDisqualificationResult
from beaver.baselines.bhk22_verifier import (
    decrypt_verifier_shareholder_payload,
    derive_verifier_challenges,
    verify_verifier_batch_public,
)
from beaver.g1_backend import FIELD_ORDER, G1Point
from beaver.pedersen_shamir import interpolate_at_zero, lagrange_coeff_at_zero


VC_KIND = "atcaas-bhk22-public-commitment-vector"
VC_VERSION = 1
VC_DIGEST_DOMAIN = b"ATCaaS/BHK22/public-commitment-vector/v1\x00"
CARRY_VC_KIND = "atcaas-bhk22-encrypted-commitment-vector"
CARRY_VC_VERSION = 1
CARRY_VC_DIGEST_DOMAIN = b"ATCaaS/BHK22/encrypted-commitment-vector/v1\x00"


def _canonical_ids(
    values: Sequence[int], *, n_r: int, name: str, allow_empty: bool = False
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an integer sequence")
    result = tuple(strict_int(value, name=f"{name} entry") for value in values)
    if (not allow_empty and not result) or result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    if any(value >= n_r for value in result):
        raise ValueError(f"{name} contains an out-of-range role id")
    return result


def _normalize_dealers(
    context: BHK22PublicContext,
    batches: Sequence[BHK22DealerBatch],
    *,
    deal_phase_qid: str,
) -> tuple[BHK22DealerBatch, ...]:
    result = tuple(batches)
    n_r = context.parameters.n_r
    if len(result) != n_r or not all(isinstance(item, BHK22DealerBatch) for item in result):
        raise ValueError("shareholder requires exactly n_r dealer batches")
    if tuple(item.role_id for item in result) != tuple(range(n_r)):
        raise ValueError("dealer batches must be ordered by role id")
    if any(
        item.root_context_digest != context.root_context_digest
        or item.phase_qid != deal_phase_qid
        or item.n_r != n_r
        for item in result
    ):
        raise ValueError("dealer batch context/qid mismatch")
    return result


def _normalize_verifiers(
    context: BHK22PublicContext,
    batches: Sequence[BHK22VerifierBatch],
    *,
    verify_phase_qid: str,
) -> tuple[BHK22VerifierBatch, ...]:
    result = tuple(batches)
    n_r = context.parameters.n_r
    if len(result) != n_r or not all(isinstance(item, BHK22VerifierBatch) for item in result):
        raise ValueError("shareholder requires exactly n_r verifier batches")
    if tuple(item.role_id for item in result) != tuple(range(n_r)):
        raise ValueError("verifier batches must be ordered by role id")
    if any(
        item.root_context_digest != context.root_context_digest
        or item.phase_qid != verify_phase_qid
        or item.n_r != n_r
        for item in result
    ):
        raise ValueError("verifier batch context/qid mismatch")
    return result


class BHK22InsufficientQualifiedDealers(RuntimeError):
    pass


class BHK22InsufficientValidVerifiers(RuntimeError):
    pass


@dataclass(frozen=True)
class BHK22ShareholderSelection:
    qualified_dealer_ids: Sequence[int]
    selected_dealer_ids: Sequence[int]
    valid_verifier_ids: Sequence[int]
    selected_verifier_ids: Sequence[int]
    dealer_lagrange_coefficients: Sequence[int]
    verifier_lagrange_coefficients: Sequence[int]

    def __post_init__(self) -> None:
        qualified = tuple(self.qualified_dealer_ids)
        selected_dealers = tuple(self.selected_dealer_ids)
        valid_verifiers = tuple(self.valid_verifier_ids)
        selected_verifiers = tuple(self.selected_verifier_ids)
        if (
            qualified != tuple(sorted(set(qualified)))
            or selected_dealers != tuple(sorted(set(selected_dealers)))
            or valid_verifiers != tuple(sorted(set(valid_verifiers)))
            or selected_verifiers != tuple(sorted(set(selected_verifiers)))
        ):
            raise ValueError("shareholder selection ids must be canonical")
        if not set(selected_dealers) <= set(qualified) or not set(selected_verifiers) <= set(valid_verifiers):
            raise ValueError("selected roles must be subsets of valid candidates")
        dealer_coefficients = tuple(int(value) % FIELD_ORDER for value in self.dealer_lagrange_coefficients)
        verifier_coefficients = tuple(int(value) % FIELD_ORDER for value in self.verifier_lagrange_coefficients)
        if len(dealer_coefficients) != len(selected_dealers) or len(verifier_coefficients) != len(selected_verifiers):
            raise ValueError("selection coefficient length mismatch")
        object.__setattr__(self, "qualified_dealer_ids", qualified)
        object.__setattr__(self, "selected_dealer_ids", selected_dealers)
        object.__setattr__(self, "valid_verifier_ids", valid_verifiers)
        object.__setattr__(self, "selected_verifier_ids", selected_verifiers)
        object.__setattr__(self, "dealer_lagrange_coefficients", dealer_coefficients)
        object.__setattr__(self, "verifier_lagrange_coefficients", verifier_coefficients)


@dataclass(frozen=True)
class BHK22ShareholderMetrics:
    verifier_public_check_ms: float
    payload_decrypt_ms: float
    reconstruction_ms: float
    aggregate_decrypt_ms: float
    correct_decryption_proof_ms: float
    serialization_ms: float
    total_ms: float
    newly_ignored_verifiers: int
    valid_verifiers: int
    finalize_bytes: int
    correct_decryption_proof_bytes: int
    cpu_seconds: float = 0.0
    verifier_public_check_cpu_seconds: float = 0.0
    correct_decryption_proof_cpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "verifier_public_check_ms", "payload_decrypt_ms", "reconstruction_ms",
            "aggregate_decrypt_ms", "correct_decryption_proof_ms",
            "serialization_ms", "total_ms", "cpu_seconds",
            "verifier_public_check_cpu_seconds",
            "correct_decryption_proof_cpu_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a non-negative number")
        for name in (
            "newly_ignored_verifiers", "valid_verifiers", "finalize_bytes",
            "correct_decryption_proof_bytes",
        ):
            strict_int(getattr(self, name), name=name)


@dataclass(frozen=True)
class BHK22ShareholderGenerationResult:
    public_output: BHK22PublicFinalizeOutput
    successor_result: BHK22SuccessorResult
    selection: BHK22ShareholderSelection
    local_ignored_verifier_ids: Sequence[int]
    newly_ignored_verifier_ids: Sequence[int]
    verifier_failure_reasons: Mapping[int, str]
    metrics: BHK22ShareholderMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.public_output, BHK22PublicFinalizeOutput):
            raise TypeError("shareholder generation requires public finalize output")
        if not isinstance(self.successor_result, BHK22SuccessorResult):
            raise TypeError("shareholder generation requires private successor result")
        if not isinstance(self.selection, BHK22ShareholderSelection):
            raise TypeError("shareholder generation requires a selection result")
        if not isinstance(self.metrics, BHK22ShareholderMetrics):
            raise TypeError("shareholder generation requires shareholder metrics")
        ignored = tuple(sorted(set(self.local_ignored_verifier_ids)))
        newly_ignored = tuple(sorted(set(self.newly_ignored_verifier_ids)))
        reasons = {
            strict_int(verifier, name="failed verifier id"): nonempty_text(
                reason, name="verifier failure reason"
            )
            for verifier, reason in self.verifier_failure_reasons.items()
        }
        if not set(newly_ignored) <= set(ignored):
            raise ValueError("newly ignored verifiers must be locally ignored")
        if tuple(sorted(reasons)) != newly_ignored:
            raise ValueError("newly ignored verifier set/reasons mismatch")
        object.__setattr__(self, "local_ignored_verifier_ids", ignored)
        object.__setattr__(self, "newly_ignored_verifier_ids", newly_ignored)
        object.__setattr__(self, "verifier_failure_reasons", MappingProxyType(reasons))


@dataclass(frozen=True)
class BHK22CarryForwardMetrics:
    verifier_public_check_ms: float
    payload_decrypt_ms: float
    reconstruction_ms: float
    aggregate_ciphertext_ms: float
    private_binding_check_ms: float
    total_ms: float
    newly_ignored_verifiers: int
    valid_verifiers: int
    cpu_seconds: float = 0.0
    verifier_public_check_cpu_seconds: float = 0.0


@dataclass(frozen=True)
class BHK22CarryForwardGenerationResult:
    successor_state: BHK22CarryForwardSuccessorState
    selection: BHK22ShareholderSelection
    local_ignored_verifier_ids: Sequence[int]
    newly_ignored_verifier_ids: Sequence[int]
    verifier_failure_reasons: Mapping[int, str]
    metrics: BHK22CarryForwardMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.successor_state, BHK22CarryForwardSuccessorState):
            raise TypeError("carry-forward result requires private successor state")
        if not isinstance(self.selection, BHK22ShareholderSelection):
            raise TypeError("carry-forward result requires a selection")
        if not isinstance(self.metrics, BHK22CarryForwardMetrics):
            raise TypeError("carry-forward result requires metrics")
        ignored = tuple(sorted(set(self.local_ignored_verifier_ids)))
        newly_ignored = tuple(sorted(set(self.newly_ignored_verifier_ids)))
        reasons = {
            strict_int(verifier, name="failed verifier id"): nonempty_text(
                reason, name="verifier failure reason"
            )
            for verifier, reason in self.verifier_failure_reasons.items()
        }
        if not set(newly_ignored) <= set(ignored):
            raise ValueError("newly ignored verifiers must be locally ignored")
        if tuple(sorted(reasons)) != newly_ignored:
            raise ValueError("newly ignored verifier set/reasons mismatch")
        object.__setattr__(self, "local_ignored_verifier_ids", ignored)
        object.__setattr__(self, "newly_ignored_verifier_ids", newly_ignored)
        object.__setattr__(self, "verifier_failure_reasons", MappingProxyType(reasons))


@dataclass(frozen=True)
class _BHK22ShareholderReconstruction:
    selection: BHK22ShareholderSelection
    successor_share: int
    aggregate_ciphertext: BHK22ElGamalCiphertext
    recipient_secret: int
    public_value_secret: int
    local_ignored_verifier_ids: tuple[int, ...]
    newly_ignored_verifier_ids: tuple[int, ...]
    verifier_failure_reasons: Mapping[int, str]
    verifier_public_check_ms: float
    verifier_public_check_cpu_seconds: float
    payload_decrypt_ms: float
    reconstruction_ms: float
    aggregate_ciphertext_ms: float


class BHK22ShareholderRole:
    def __init__(
        self,
        *,
        context: BHK22PublicContext,
        state: BHK22ShareholderState,
        dealer_batches: Sequence[BHK22DealerBatch],
        verifier_batches: Sequence[BHK22VerifierBatch],
        disqualification: BHK22ShareholderDisqualificationResult,
        deal_phase_qid: str,
        verify_phase_qid: str,
        finalize_phase_qid: str,
        dealer_public_verifier=None,
    ) -> None:
        if not isinstance(context, BHK22PublicContext):
            raise TypeError("shareholder role requires BHK22PublicContext")
        if not isinstance(state, BHK22ShareholderState):
            raise TypeError("shareholder role requires BHK22ShareholderState")
        if not isinstance(disqualification, BHK22ShareholderDisqualificationResult):
            raise TypeError("shareholder role requires disqualification result")
        deal_qid = nonempty_text(deal_phase_qid, name="deal phase qid")
        verify_qid = nonempty_text(verify_phase_qid, name="verify phase qid")
        finalize_qid = nonempty_text(finalize_phase_qid, name="finalize phase qid")
        if disqualification.shareholder_role_id != state.role_id:
            raise ValueError("shareholder/disqualification role mismatch")
        self.context = context
        self.state = state
        self.dealers = _normalize_dealers(context, dealer_batches, deal_phase_qid=deal_qid)
        self.verifiers = _normalize_verifiers(
            context, verifier_batches, verify_phase_qid=verify_qid
        )
        self.disqualification = disqualification
        self.deal_phase_qid = deal_qid
        self.verify_phase_qid = verify_qid
        self.finalize_phase_qid = finalize_qid
        if dealer_public_verifier is not None and not callable(dealer_public_verifier):
            raise TypeError("dealer_public_verifier must be callable or null")
        self.dealer_public_verifier = dealer_public_verifier

    def _reconstruct_for_successor(self) -> _BHK22ShareholderReconstruction:
        context = self.context
        n_r = context.parameters.n_r
        t_r = context.parameters.t_r
        shareholder = self.state.role_id
        recipient_secret = self.state._recipient_secret_for_protocol()
        public_value_secret = self.state._public_value_secret_for_protocol()
        qualified = tuple(self.disqualification.qualified_dealer_ids)
        if len(qualified) <= t_r:
            raise BHK22InsufficientQualifiedDealers("need at least t_r+1 qualified dealers")

        public_started = perf_counter()
        public_cpu_started = process_time()
        public_validity = {}
        for verifier in self.disqualification.remaining_verifier_ids:
            public_validity[verifier] = verify_verifier_batch_public(
                context,
                self.verifiers[verifier],
                self.dealers,
                deal_phase_qid=self.deal_phase_qid,
                expected_verify_phase_qid=self.verify_phase_qid,
                dealer_public_verifier=self.dealer_public_verifier,
            ).accepted
        public_ms = (perf_counter() - public_started) * 1000.0
        public_cpu_seconds = process_time() - public_cpu_started

        decrypt_started = perf_counter()
        valid_payloads = {}
        reasons = {}
        for verifier in self.disqualification.remaining_verifier_ids:
            if not public_validity[verifier]:
                reasons[verifier] = "public-verifier-check"
                continue
            batch = self.verifiers[verifier]
            try:
                payload = decrypt_verifier_shareholder_payload(
                    context,
                    batch,
                    shareholder_role_id=shareholder,
                    shareholder_recipient_secret_key=recipient_secret,
                )
                if not set(qualified) <= set(payload.no_accusation_dealer_ids):
                    raise ValueError("qualified dealer missing from verifier payload")
                challenges = derive_verifier_challenges(
                    root_context_digest=batch.root_context_digest,
                    phase_qid=batch.phase_qid,
                    verifier_role_id=verifier,
                    n_r=n_r,
                    no_accusation_dealer_ids=batch.no_accusation_dealer_ids,
                    shareholder_commitments=[
                        package.vector_commitment
                        for package in batch.shareholder_packages
                    ],
                )
                values_by_dealer = dict(
                    zip(payload.no_accusation_dealer_ids, payload.share_vector)
                )
                rho = sum(
                    challenges[dealer] * values_by_dealer[dealer]
                    for dealer in payload.no_accusation_dealer_ids
                ) % FIELD_ORDER
                expected_r = (
                    G1Point.from_hex(context.crs.generators[shareholder + 1]) * rho
                ).to_hex()
                if expected_r != batch.forwarding_points[shareholder + 1]:
                    raise ValueError("private xi vector does not match R_j,l")
                valid_payloads[verifier] = payload
            except (KeyError, TypeError, ValueError):
                reasons[verifier] = "private-payload-or-R-check"
        payload_ms = (perf_counter() - decrypt_started) * 1000.0
        valid_verifiers = tuple(sorted(valid_payloads))
        if len(valid_verifiers) <= t_r:
            raise BHK22InsufficientValidVerifiers("need at least t_r+1 valid verifiers")

        selected_dealers = qualified[: t_r + 1]
        selected_verifiers = valid_verifiers[: t_r + 1]
        dealer_points = tuple(dealer + 1 for dealer in selected_dealers)
        verifier_points = tuple(verifier + 1 for verifier in selected_verifiers)
        dealer_coefficients = tuple(
            lagrange_coeff_at_zero(dealer_points, point) for point in dealer_points
        )
        verifier_coefficients = tuple(
            lagrange_coeff_at_zero(verifier_points, point) for point in verifier_points
        )
        selection = BHK22ShareholderSelection(
            qualified_dealer_ids=qualified,
            selected_dealer_ids=selected_dealers,
            valid_verifier_ids=valid_verifiers,
            selected_verifier_ids=selected_verifiers,
            dealer_lagrange_coefficients=dealer_coefficients,
            verifier_lagrange_coefficients=verifier_coefficients,
        )

        reconstruction_started = perf_counter()
        dealer_shares = []
        for dealer in selected_dealers:
            values = []
            for verifier in selected_verifiers:
                payload = valid_payloads[verifier]
                offset = payload.no_accusation_dealer_ids.index(dealer)
                values.append(payload.share_vector[offset])
            dealer_shares.append(
                interpolate_at_zero(verifier_points, values)
            )
        successor_share = sum(
            coefficient * share
            for coefficient, share in zip(dealer_coefficients, dealer_shares)
        ) % FIELD_ORDER
        reconstruction_ms = (perf_counter() - reconstruction_started) * 1000.0

        aggregate_started = perf_counter()
        aggregate_ciphertext = combine_public_value_ciphertexts(
            [
                self.dealers[dealer].figure8_ciphertexts[shareholder + 1]
                for dealer in selected_dealers
            ],
            dealer_coefficients,
        )
        aggregate_ms = (perf_counter() - aggregate_started) * 1000.0

        return _BHK22ShareholderReconstruction(
            selection=selection,
            successor_share=successor_share,
            aggregate_ciphertext=aggregate_ciphertext,
            recipient_secret=recipient_secret,
            public_value_secret=public_value_secret,
            local_ignored_verifier_ids=tuple(
                sorted(
                    set(self.disqualification.local_ignored_verifier_ids)
                    | set(reasons)
                )
            ),
            newly_ignored_verifier_ids=tuple(sorted(reasons)),
            verifier_failure_reasons=MappingProxyType(dict(reasons)),
            verifier_public_check_ms=public_ms,
            verifier_public_check_cpu_seconds=public_cpu_seconds,
            payload_decrypt_ms=payload_ms,
            reconstruction_ms=reconstruction_ms,
            aggregate_ciphertext_ms=aggregate_ms,
        )

    def generate_output(self) -> BHK22ShareholderGenerationResult:
        started = perf_counter()
        cpu_started = process_time()
        context = self.context
        n_r = context.parameters.n_r
        shareholder = self.state.role_id
        reconstruction = self._reconstruct_for_successor()
        selection = reconstruction.selection
        qualified = tuple(selection.qualified_dealer_ids)
        selected_dealers = tuple(selection.selected_dealer_ids)
        successor_share = reconstruction.successor_share
        aggregate_ciphertext = reconstruction.aggregate_ciphertext
        recipient_secret = reconstruction.recipient_secret
        public_value_secret = reconstruction.public_value_secret

        aggregate_started = perf_counter()
        public_value = decrypt_public_value(
            aggregate_ciphertext, secret_key=public_value_secret
        )
        expected_public_value = (
            G1Point.from_hex(context.crs.generators[shareholder + 1])
            * successor_share
        ).to_hex()
        if public_value != expected_public_value:
            raise ValueError("decrypted successor public value does not match private share")
        aggregate_ms = (
            reconstruction.aggregate_ciphertext_ms
            + (perf_counter() - aggregate_started) * 1000.0
        )

        proof_context = BHK22ProofContext(
            root_context_digest=context.root_context_digest,
            phase_qid=self.finalize_phase_qid,
            role_type=ROLE_SHAREHOLDER,
            role_id=shareholder,
        )
        proof_started = perf_counter()
        proof_cpu_started = process_time()
        proof = prove_correct_decryption(
            context=proof_context,
            generator_hex=context.crs.generators[shareholder + 1],
            ciphertext=aggregate_ciphertext,
            public_key_hex=context.crs.public_value_keys[shareholder + 1],
            plaintext_hex=public_value,
            secret_key=public_value_secret,
        )
        proof_ms = (perf_counter() - proof_started) * 1000.0
        proof_cpu_seconds = process_time() - proof_cpu_started

        serialization_started = perf_counter()
        public_output = BHK22PublicFinalizeOutput(
            root_context_digest=context.root_context_digest,
            phase_qid=self.finalize_phase_qid,
            n_r=n_r,
            role_id=shareholder,
            network_sender_id=context.committees[ROLE_SHAREHOLDER].role_to_node[
                shareholder
            ],
            qualified_dealer_ids=qualified,
            selected_dealer_ids=selected_dealers,
            aggregate_ciphertext=aggregate_ciphertext,
            public_value=public_value,
            correct_decryption_proof=proof,
        )
        successor_result = BHK22SuccessorResult(
            shareholder_role_id=shareholder,
            evaluation_point=shareholder + 1,
            share=successor_share,
            public_value=public_value,
            root_context_digest=context.root_context_digest,
        )
        finalize_bytes = len(public_output.to_bytes())
        serialization_ms = (perf_counter() - serialization_started) * 1000.0
        consumed = self.state.consume_once()
        if (
            consumed.recipient_secret_key != recipient_secret
            or consumed.public_value_secret_key != public_value_secret
        ):
            raise RuntimeError("shareholder state changed during finalization")

        metrics = BHK22ShareholderMetrics(
            verifier_public_check_ms=reconstruction.verifier_public_check_ms,
            payload_decrypt_ms=reconstruction.payload_decrypt_ms,
            reconstruction_ms=reconstruction.reconstruction_ms,
            aggregate_decrypt_ms=aggregate_ms,
            correct_decryption_proof_ms=proof_ms,
            serialization_ms=serialization_ms,
            total_ms=(perf_counter() - started) * 1000.0,
            newly_ignored_verifiers=len(reconstruction.newly_ignored_verifier_ids),
            valid_verifiers=len(selection.valid_verifier_ids),
            finalize_bytes=finalize_bytes,
            correct_decryption_proof_bytes=proof_size_bytes(proof),
            cpu_seconds=process_time() - cpu_started,
            verifier_public_check_cpu_seconds=(
                reconstruction.verifier_public_check_cpu_seconds
            ),
            correct_decryption_proof_cpu_seconds=proof_cpu_seconds,
        )
        return BHK22ShareholderGenerationResult(
            public_output=public_output,
            successor_result=successor_result,
            selection=selection,
            local_ignored_verifier_ids=reconstruction.local_ignored_verifier_ids,
            newly_ignored_verifier_ids=reconstruction.newly_ignored_verifier_ids,
            verifier_failure_reasons=reconstruction.verifier_failure_reasons,
            metrics=metrics,
        )

    def generate_carry_forward_state(
        self,
        *,
        terminal_phase_qid: str,
    ) -> BHK22CarryForwardGenerationResult:
        """Reconstruct silently and retain the share/key for the next speech."""

        started = perf_counter()
        cpu_started = process_time()
        reconstruction = self._reconstruct_for_successor()
        shareholder = self.state.role_id
        binding_started = perf_counter()
        private_public_value = decrypt_public_value(
            reconstruction.aggregate_ciphertext,
            secret_key=reconstruction.public_value_secret,
        )
        expected_public_value = (
            G1Point.from_hex(self.context.crs.generators[shareholder + 1])
            * reconstruction.successor_share
        ).to_hex()
        if private_public_value != expected_public_value:
            raise ValueError("encrypted successor commitment does not match private share")
        binding_ms = (perf_counter() - binding_started) * 1000.0

        consumed = self.state.consume_once()
        if (
            consumed.recipient_secret_key != reconstruction.recipient_secret
            or consumed.public_value_secret_key != reconstruction.public_value_secret
        ):
            raise RuntimeError("shareholder state changed during carry-forward")
        successor_state = BHK22CarryForwardSuccessorState(
            shareholder_role_id=shareholder,
            evaluation_point=shareholder + 1,
            share=reconstruction.successor_share,
            public_value_secret_key=reconstruction.public_value_secret,
            aggregate_ciphertext=reconstruction.aggregate_ciphertext,
            terminal_phase_qid=terminal_phase_qid,
            root_context_digest=self.context.root_context_digest,
        )
        metrics = BHK22CarryForwardMetrics(
            verifier_public_check_ms=reconstruction.verifier_public_check_ms,
            payload_decrypt_ms=reconstruction.payload_decrypt_ms,
            reconstruction_ms=reconstruction.reconstruction_ms,
            aggregate_ciphertext_ms=reconstruction.aggregate_ciphertext_ms,
            private_binding_check_ms=binding_ms,
            total_ms=(perf_counter() - started) * 1000.0,
            newly_ignored_verifiers=len(reconstruction.newly_ignored_verifier_ids),
            valid_verifiers=len(reconstruction.selection.valid_verifier_ids),
            cpu_seconds=process_time() - cpu_started,
            verifier_public_check_cpu_seconds=(
                reconstruction.verifier_public_check_cpu_seconds
            ),
        )
        return BHK22CarryForwardGenerationResult(
            successor_state=successor_state,
            selection=reconstruction.selection,
            local_ignored_verifier_ids=reconstruction.local_ignored_verifier_ids,
            newly_ignored_verifier_ids=reconstruction.newly_ignored_verifier_ids,
            verifier_failure_reasons=reconstruction.verifier_failure_reasons,
            metrics=metrics,
        )


@dataclass(frozen=True)
class BHK22FinalizePublicVerification:
    accepted: bool
    checks: Mapping[str, bool]
    verify_ms: float
    verify_cpu_seconds: float = 0.0
    correct_decryption_proof_verify_ms: float = 0.0
    correct_decryption_proof_verify_cpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        checks = {str(name): bool(value) for name, value in self.checks.items()}
        if self.accepted != all(checks.values()):
            raise ValueError("finalize verification accepted/check mismatch")
        for name in (
            "verify_ms",
            "verify_cpu_seconds",
            "correct_decryption_proof_verify_ms",
            "correct_decryption_proof_verify_cpu_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ) or value < 0:
                raise ValueError(f"{name} must be a non-negative number")
        object.__setattr__(self, "checks", MappingProxyType(checks))


def verify_finalize_output_public(
    context: BHK22PublicContext,
    output: BHK22PublicFinalizeOutput,
    dealer_batches: Sequence[BHK22DealerBatch],
    *,
    deal_phase_qid: str,
    expected_finalize_phase_qid: str,
    expected_qualified_dealer_ids: Sequence[int],
) -> BHK22FinalizePublicVerification:
    started = perf_counter()
    cpu_started = process_time()
    proof_verify_ms = 0.0
    proof_verify_cpu_seconds = 0.0
    checks = {
        "wire_context": False,
        "qualified_selection": False,
        "aggregate_ciphertext": False,
        "correct_decryption_proof": False,
    }
    try:
        if not isinstance(output, BHK22PublicFinalizeOutput):
            raise TypeError("output must be BHK22PublicFinalizeOutput")
        dealers = _normalize_dealers(
            context, dealer_batches, deal_phase_qid=deal_phase_qid
        )
        n_r = context.parameters.n_r
        shareholder = output.role_id
        qualified = _canonical_ids(
            expected_qualified_dealer_ids,
            n_r=n_r,
            name="expected qualified dealers",
        )
        selected = qualified[: context.parameters.t_r + 1]
        checks["wire_context"] = (
            output.root_context_digest == context.root_context_digest
            and output.phase_qid == expected_finalize_phase_qid
            and output.n_r == n_r
            and output.network_sender_id
            == context.committees[ROLE_SHAREHOLDER].role_to_node[shareholder]
        )
        checks["qualified_selection"] = (
            output.qualified_dealer_ids == qualified
            and output.selected_dealer_ids == selected
            and len(selected) == context.parameters.t_r + 1
        )
        points = tuple(dealer + 1 for dealer in selected)
        coefficients = tuple(
            lagrange_coeff_at_zero(points, point) for point in points
        )
        expected_ciphertext = combine_public_value_ciphertexts(
            [
                dealers[dealer].figure8_ciphertexts[shareholder + 1]
                for dealer in selected
            ],
            coefficients,
        )
        checks["aggregate_ciphertext"] = expected_ciphertext == output.aggregate_ciphertext
        proof_context = BHK22ProofContext(
            root_context_digest=output.root_context_digest,
            phase_qid=output.phase_qid,
            role_type=ROLE_SHAREHOLDER,
            role_id=shareholder,
        )
        proof_started = perf_counter()
        proof_cpu_started = process_time()
        checks["correct_decryption_proof"] = verify_correct_decryption(
            context=proof_context,
            generator_hex=context.crs.generators[shareholder + 1],
            ciphertext=output.aggregate_ciphertext,
            public_key_hex=context.crs.public_value_keys[shareholder + 1],
            plaintext_hex=output.public_value,
            proof=output.correct_decryption_proof,
        )
        proof_verify_ms = (perf_counter() - proof_started) * 1000.0
        proof_verify_cpu_seconds = process_time() - proof_cpu_started
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError):
        pass
    return BHK22FinalizePublicVerification(
        accepted=all(checks.values()),
        checks=checks,
        verify_ms=(perf_counter() - started) * 1000.0,
        verify_cpu_seconds=process_time() - cpu_started,
        correct_decryption_proof_verify_ms=proof_verify_ms,
        correct_decryption_proof_verify_cpu_seconds=proof_verify_cpu_seconds,
    )


@dataclass(frozen=True)
class BHK22PublicCommitmentVector:
    root_context_digest: str
    finalize_phase_qid: str
    qualified_dealer_ids: Sequence[int]
    public_values: Sequence[str]
    commitment_digest: str = ""

    def __post_init__(self) -> None:
        root = digest_hex(self.root_context_digest, name="VC root context digest")
        qid = nonempty_text(self.finalize_phase_qid, name="VC finalize qid")
        values = tuple(
            canonical_point(value, name="successor public value")
            for value in self.public_values
        )
        n_r = len(values)
        if n_r < 1:
            raise ValueError("public commitment vector cannot be empty")
        qualified = _canonical_ids(
            self.qualified_dealer_ids, n_r=n_r, name="VC qualified dealers"
        )
        material = {
            "kind": VC_KIND,
            "version": VC_VERSION,
            "root_context_digest": root,
            "finalize_phase_qid": qid,
            "qualified_dealer_ids": list(qualified),
            "public_values": list(values),
        }
        computed = hashlib.sha256(
            VC_DIGEST_DOMAIN + canonical_json(material)
        ).hexdigest()
        if self.commitment_digest and digest_hex(
            self.commitment_digest, name="commitment_digest"
        ) != computed:
            raise ValueError("public commitment vector digest mismatch")
        object.__setattr__(self, "root_context_digest", root)
        object.__setattr__(self, "finalize_phase_qid", qid)
        object.__setattr__(self, "qualified_dealer_ids", qualified)
        object.__setattr__(self, "public_values", values)
        object.__setattr__(self, "commitment_digest", computed)
        validate_public_object(self.to_obj())

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": VC_KIND,
            "version": VC_VERSION,
            "root_context_digest": self.root_context_digest,
            "finalize_phase_qid": self.finalize_phase_qid,
            "qualified_dealer_ids": list(self.qualified_dealer_ids),
            "public_values": list(self.public_values),
            "commitment_digest": self.commitment_digest,
        }

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22PublicCommitmentVector":
        exact_fields(
            value,
            {
                "kind", "version", "root_context_digest", "finalize_phase_qid",
                "qualified_dealer_ids", "public_values", "commitment_digest",
            },
            name="BHK22 public commitment vector",
        )
        if value["kind"] != VC_KIND or value["version"] != VC_VERSION:
            raise ValueError("unsupported BHK22 public commitment vector")
        return cls(
            root_context_digest=value["root_context_digest"],
            finalize_phase_qid=value["finalize_phase_qid"],
            qualified_dealer_ids=value["qualified_dealer_ids"],
            public_values=value["public_values"],
            commitment_digest=value["commitment_digest"],
        )

    @classmethod
    def from_bytes(cls, data: bytes | str) -> "BHK22PublicCommitmentVector":
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        restored = cls.from_obj(decode_object(raw, name="BHK22 public commitment vector"))
        if not hmac.compare_digest(raw, restored.to_bytes()):
            raise ValueError("BHK22 public commitment vector is not canonical JSON")
        return restored


@dataclass(frozen=True)
class BHK22EncryptedCommitmentVector:
    """Public carry-forward result; plaintext values remain hidden."""

    root_context_digest: str
    terminal_phase_qid: str
    terminal_artifact_digest: str
    qualified_dealer_ids: Sequence[int]
    aggregate_ciphertexts: Sequence[BHK22ElGamalCiphertext]
    commitment_digest: str = ""

    def __post_init__(self) -> None:
        root = digest_hex(self.root_context_digest, name="encrypted VC root digest")
        qid = nonempty_text(self.terminal_phase_qid, name="terminal phase qid")
        artifact = digest_hex(
            self.terminal_artifact_digest, name="terminal artifact digest"
        )
        ciphertexts = tuple(self.aggregate_ciphertexts)
        if not ciphertexts or not all(
            isinstance(item, BHK22ElGamalCiphertext) for item in ciphertexts
        ):
            raise ValueError("encrypted commitment vector requires ciphertexts")
        qualified = _canonical_ids(
            self.qualified_dealer_ids,
            n_r=len(ciphertexts),
            name="encrypted VC qualified dealers",
        )
        material = {
            "kind": CARRY_VC_KIND,
            "version": CARRY_VC_VERSION,
            "root_context_digest": root,
            "terminal_phase_qid": qid,
            "terminal_artifact_digest": artifact,
            "qualified_dealer_ids": list(qualified),
            "aggregate_ciphertexts": [item.to_obj() for item in ciphertexts],
        }
        computed = hashlib.sha256(
            CARRY_VC_DIGEST_DOMAIN + canonical_json(material)
        ).hexdigest()
        if self.commitment_digest and digest_hex(
            self.commitment_digest, name="commitment_digest"
        ) != computed:
            raise ValueError("encrypted commitment vector digest mismatch")
        object.__setattr__(self, "root_context_digest", root)
        object.__setattr__(self, "terminal_phase_qid", qid)
        object.__setattr__(self, "terminal_artifact_digest", artifact)
        object.__setattr__(self, "qualified_dealer_ids", qualified)
        object.__setattr__(self, "aggregate_ciphertexts", ciphertexts)
        object.__setattr__(self, "commitment_digest", computed)
        validate_public_object(self.to_obj())

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": CARRY_VC_KIND,
            "version": CARRY_VC_VERSION,
            "root_context_digest": self.root_context_digest,
            "terminal_phase_qid": self.terminal_phase_qid,
            "terminal_artifact_digest": self.terminal_artifact_digest,
            "qualified_dealer_ids": list(self.qualified_dealer_ids),
            "aggregate_ciphertexts": [
                item.to_obj() for item in self.aggregate_ciphertexts
            ],
            "commitment_digest": self.commitment_digest,
        }

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22EncryptedCommitmentVector":
        exact_fields(
            value,
            {
                "kind", "version", "root_context_digest", "terminal_phase_qid",
                "terminal_artifact_digest", "qualified_dealer_ids",
                "aggregate_ciphertexts", "commitment_digest",
            },
            name="BHK22 encrypted commitment vector",
        )
        if value["kind"] != CARRY_VC_KIND or value["version"] != CARRY_VC_VERSION:
            raise ValueError("unsupported BHK22 encrypted commitment vector")
        return cls(
            root_context_digest=value["root_context_digest"],
            terminal_phase_qid=value["terminal_phase_qid"],
            terminal_artifact_digest=value["terminal_artifact_digest"],
            qualified_dealer_ids=value["qualified_dealer_ids"],
            aggregate_ciphertexts=tuple(
                BHK22ElGamalCiphertext.from_obj(item)
                for item in value["aggregate_ciphertexts"]
            ),
            commitment_digest=value["commitment_digest"],
        )

    @classmethod
    def from_bytes(cls, data: bytes | str) -> "BHK22EncryptedCommitmentVector":
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        restored = cls.from_obj(
            decode_object(raw, name="BHK22 encrypted commitment vector")
        )
        if not hmac.compare_digest(raw, restored.to_bytes()):
            raise ValueError("BHK22 encrypted commitment vector is not canonical JSON")
        return restored


def build_encrypted_commitment_vector(
    context: BHK22PublicContext,
    dealer_batches: Sequence[BHK22DealerBatch],
    *,
    deal_phase_qid: str,
    terminal_phase_qid: str,
    terminal_artifact_digest: str,
    expected_qualified_dealer_ids: Sequence[int],
) -> BHK22EncryptedCommitmentVector:
    dealers = _normalize_dealers(
        context, dealer_batches, deal_phase_qid=deal_phase_qid
    )
    qualified = _canonical_ids(
        expected_qualified_dealer_ids,
        n_r=context.parameters.n_r,
        name="expected qualified dealers",
    )
    selected = qualified[: context.parameters.t_r + 1]
    if len(selected) != context.parameters.t_r + 1:
        raise BHK22InsufficientQualifiedDealers(
            "need at least t_r+1 qualified dealers"
        )
    points = tuple(dealer + 1 for dealer in selected)
    coefficients = tuple(
        lagrange_coeff_at_zero(points, point) for point in points
    )
    ciphertexts = tuple(
        combine_public_value_ciphertexts(
            [
                dealers[dealer].figure8_ciphertexts[shareholder + 1]
                for dealer in selected
            ],
            coefficients,
        )
        for shareholder in range(context.parameters.n_r)
    )
    return BHK22EncryptedCommitmentVector(
        root_context_digest=context.root_context_digest,
        terminal_phase_qid=terminal_phase_qid,
        terminal_artifact_digest=terminal_artifact_digest,
        qualified_dealer_ids=qualified,
        aggregate_ciphertexts=ciphertexts,
    )


def process_finalize_outputs(
    context: BHK22PublicContext,
    outputs: Sequence[BHK22PublicFinalizeOutput],
    dealer_batches: Sequence[BHK22DealerBatch],
    *,
    deal_phase_qid: str,
    finalize_phase_qid: str,
    expected_qualified_dealer_ids: Sequence[int],
) -> BHK22PublicCommitmentVector:
    finalized = tuple(outputs)
    n_r = context.parameters.n_r
    if len(finalized) != n_r or not all(
        isinstance(item, BHK22PublicFinalizeOutput) for item in finalized
    ):
        raise ValueError("need exactly n_r finalize outputs")
    if tuple(item.role_id for item in finalized) != tuple(range(n_r)):
        raise ValueError("finalize outputs must be ordered by shareholder role id")
    qualified = _canonical_ids(
        expected_qualified_dealer_ids,
        n_r=n_r,
        name="expected qualified dealers",
    )
    for output in finalized:
        verification = verify_finalize_output_public(
            context,
            output,
            dealer_batches,
            deal_phase_qid=deal_phase_qid,
            expected_finalize_phase_qid=finalize_phase_qid,
            expected_qualified_dealer_ids=qualified,
        )
        if not verification.accepted:
            raise ValueError(
                f"invalid finalize output for shareholder {output.role_id}: {dict(verification.checks)}"
            )
    return BHK22PublicCommitmentVector(
        root_context_digest=context.root_context_digest,
        finalize_phase_qid=finalize_phase_qid,
        qualified_dealer_ids=qualified,
        public_values=tuple(output.public_value for output in finalized),
    )


__all__ = [
    "BHK22CarryForwardGenerationResult",
    "BHK22CarryForwardMetrics",
    "BHK22EncryptedCommitmentVector",
    "BHK22FinalizePublicVerification",
    "BHK22InsufficientQualifiedDealers",
    "BHK22InsufficientValidVerifiers",
    "BHK22PublicCommitmentVector",
    "BHK22ShareholderGenerationResult",
    "BHK22ShareholderMetrics",
    "BHK22ShareholderRole",
    "BHK22ShareholderSelection",
    "build_encrypted_commitment_vector",
    "process_finalize_outputs",
    "verify_finalize_output_public",
]
