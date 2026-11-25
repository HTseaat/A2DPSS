"""BHK+22 accusation preprocessing, response, and disqualification (Step 5)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from time import perf_counter, process_time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from beaver.baselines.bhk22_context import (
    BHK22PublicContext,
    canonical_json,
    digest_hex,
    nonempty_text,
    strict_int,
)
from beaver.baselines.bhk22_contract import ROLE_RESPONDER
from beaver.baselines.bhk22_crypto import (
    open_vector_coordinate,
    primitive_size_bytes,
    verify_coordinate_opening,
)
from beaver.baselines.bhk22_dealer import (
    decrypt_dealer_responder_state,
    decrypt_dealer_row_with_recovered_epsilon,
)
from beaver.baselines.bhk22_experiment import BHK22ResponderState
from beaver.baselines.bhk22_messages import (
    BHK22DealerBatch,
    BHK22ResponderBatch,
    BHK22ResponderOpening,
    BHK22VerifierBatch,
)
from beaver.baselines.bhk22_verifier import verify_verifier_batch_public
from beaver.g1_backend import FIELD_ORDER
from beaver.pedersen_shamir import interpolate_at, interpolate_at_zero


PREPROCESS_DIGEST_DOMAIN = b"ATCaaS/BHK22/accusation-preprocess/v1\x00"


def _canonical_ids(
    values: Sequence[int], *, n_r: int, name: str, allow_empty: bool = True
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an integer sequence")
    result = tuple(strict_int(value, name=f"{name} entry") for value in values)
    if (not allow_empty and not result) or result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    if any(value >= n_r for value in result):
        raise ValueError(f"{name} contains an out-of-range role id")
    return result


def _canonical_pairs(
    values: Sequence[Sequence[int]], *, n_r: int, name: str
) -> tuple[tuple[int, int], ...]:
    pairs = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise TypeError(f"{name} entries must be dealer/verifier pairs")
        dealer = strict_int(value[0], name=f"{name} dealer")
        verifier = strict_int(value[1], name=f"{name} verifier")
        if dealer >= n_r or verifier >= n_r:
            raise ValueError(f"{name} pair is outside n_r")
        pairs.append((dealer, verifier))
    result = tuple(pairs)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and duplicate-free")
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
        raise ValueError("need exactly n_r canonical dealer batches")
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
        raise ValueError("need exactly n_r canonical verifier batches")
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


@dataclass(frozen=True)
class BHK22AccusationPreprocessing:
    root_context_digest: str
    deal_phase_qid: str
    verify_phase_qid: str
    n_r: int
    t_r: int
    invalid_verifier_ids: Sequence[int]
    over_accusing_verifier_ids: Sequence[int]
    ignored_verifier_ids: Sequence[int]
    pre_disqualified_dealer_ids: Sequence[int]
    remaining_accusations: Sequence[Sequence[int]]
    preprocessing_digest: str = ""

    def __post_init__(self) -> None:
        root = digest_hex(self.root_context_digest, name="preprocess root digest")
        deal_qid = nonempty_text(self.deal_phase_qid, name="preprocess deal qid")
        verify_qid = nonempty_text(self.verify_phase_qid, name="preprocess verify qid")
        n_r = strict_int(self.n_r, name="preprocess n_r", minimum=1)
        t_r = strict_int(self.t_r, name="preprocess t_r")
        invalid = _canonical_ids(
            self.invalid_verifier_ids, n_r=n_r, name="invalid verifiers"
        )
        over = _canonical_ids(
            self.over_accusing_verifier_ids, n_r=n_r, name="over-accusing verifiers"
        )
        ignored = _canonical_ids(
            self.ignored_verifier_ids, n_r=n_r, name="ignored verifiers"
        )
        pre_disqualified = _canonical_ids(
            self.pre_disqualified_dealer_ids,
            n_r=n_r,
            name="pre-disqualified dealers",
        )
        remaining = _canonical_pairs(
            self.remaining_accusations, n_r=n_r, name="remaining accusations"
        )
        if set(ignored) != set(invalid) | set(over):
            raise ValueError("ignored verifier set must equal invalid union over-accusing")
        if any(verifier in ignored for _, verifier in remaining):
            raise ValueError("remaining accusation references ignored verifier")
        if any(dealer in pre_disqualified for dealer, _ in remaining):
            raise ValueError("remaining accusation references pre-disqualified dealer")
        material = {
            "root_context_digest": root,
            "deal_phase_qid": deal_qid,
            "verify_phase_qid": verify_qid,
            "n_r": n_r,
            "t_r": t_r,
            "invalid_verifier_ids": list(invalid),
            "over_accusing_verifier_ids": list(over),
            "ignored_verifier_ids": list(ignored),
            "pre_disqualified_dealer_ids": list(pre_disqualified),
            "remaining_accusations": [list(pair) for pair in remaining],
        }
        computed = hashlib.sha256(
            PREPROCESS_DIGEST_DOMAIN + canonical_json(material)
        ).hexdigest()
        if self.preprocessing_digest and digest_hex(
            self.preprocessing_digest, name="preprocessing_digest"
        ) != computed:
            raise ValueError("accusation preprocessing digest mismatch")
        object.__setattr__(self, "root_context_digest", root)
        object.__setattr__(self, "deal_phase_qid", deal_qid)
        object.__setattr__(self, "verify_phase_qid", verify_qid)
        object.__setattr__(self, "n_r", n_r)
        object.__setattr__(self, "t_r", t_r)
        object.__setattr__(self, "invalid_verifier_ids", invalid)
        object.__setattr__(self, "over_accusing_verifier_ids", over)
        object.__setattr__(self, "ignored_verifier_ids", ignored)
        object.__setattr__(self, "pre_disqualified_dealer_ids", pre_disqualified)
        object.__setattr__(self, "remaining_accusations", remaining)
        object.__setattr__(self, "preprocessing_digest", computed)

    @property
    def response_required(self) -> bool:
        return bool(self.remaining_accusations)

    def to_obj(self) -> dict[str, Any]:
        return {
            "root_context_digest": self.root_context_digest,
            "deal_phase_qid": self.deal_phase_qid,
            "verify_phase_qid": self.verify_phase_qid,
            "n_r": self.n_r,
            "t_r": self.t_r,
            "invalid_verifier_ids": list(self.invalid_verifier_ids),
            "over_accusing_verifier_ids": list(self.over_accusing_verifier_ids),
            "ignored_verifier_ids": list(self.ignored_verifier_ids),
            "pre_disqualified_dealer_ids": list(self.pre_disqualified_dealer_ids),
            "remaining_accusations": [list(pair) for pair in self.remaining_accusations],
            "preprocessing_digest": self.preprocessing_digest,
        }


def preprocess_accusations(
    context: BHK22PublicContext,
    dealer_batches: Sequence[BHK22DealerBatch],
    verifier_batches: Sequence[BHK22VerifierBatch],
    *,
    deal_phase_qid: str,
    verify_phase_qid: str,
    dealer_public_verifier=None,
) -> BHK22AccusationPreprocessing:
    """Apply the paper's public over-accuser/over-accused ordering."""

    deal_qid = nonempty_text(deal_phase_qid, name="deal phase qid")
    verify_qid = nonempty_text(verify_phase_qid, name="verify phase qid")
    dealers = _normalize_dealers(context, dealer_batches, deal_phase_qid=deal_qid)
    verifiers = _normalize_verifiers(
        context, verifier_batches, verify_phase_qid=verify_qid
    )
    t_r = context.parameters.t_r
    invalid = tuple(
        batch.role_id
        for batch in verifiers
        if not verify_verifier_batch_public(
            context,
            batch,
            dealers,
            deal_phase_qid=deal_qid,
            expected_verify_phase_qid=verify_qid,
            dealer_public_verifier=dealer_public_verifier,
        ).accepted
    )
    over = tuple(
        batch.role_id
        for batch in verifiers
        if len(batch.accused_dealer_ids) > t_r
    )
    ignored = tuple(sorted(set(invalid) | set(over)))
    counts = {dealer: 0 for dealer in range(context.parameters.n_r)}
    for batch in verifiers:
        if batch.role_id in ignored:
            continue
        for dealer in batch.accused_dealer_ids:
            counts[dealer] += 1
    pre_disqualified = tuple(
        dealer for dealer, count in counts.items() if count > t_r
    )
    remaining = tuple(
        sorted(
            (dealer, batch.role_id)
            for batch in verifiers
            if batch.role_id not in ignored
            for dealer in batch.accused_dealer_ids
            if dealer not in pre_disqualified
        )
    )
    return BHK22AccusationPreprocessing(
        root_context_digest=context.root_context_digest,
        deal_phase_qid=deal_qid,
        verify_phase_qid=verify_qid,
        n_r=context.parameters.n_r,
        t_r=t_r,
        invalid_verifier_ids=invalid,
        over_accusing_verifier_ids=over,
        ignored_verifier_ids=ignored,
        pre_disqualified_dealer_ids=pre_disqualified,
        remaining_accusations=remaining,
    )


@dataclass(frozen=True)
class BHK22ResponderMetrics:
    future_state_decrypt_ms: float
    opening_ms: float
    serialization_ms: float
    total_ms: float
    requested_accusations: int
    produced_openings: int
    failed_dealer_states: int
    batch_bytes: int
    opening_bytes: int
    cpu_seconds: float = 0.0
    future_state_decrypt_cpu_seconds: float = 0.0
    opening_cpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "future_state_decrypt_ms", "opening_ms", "serialization_ms", "total_ms",
            "cpu_seconds",
            "future_state_decrypt_cpu_seconds", "opening_cpu_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a non-negative number")
        for name in (
            "requested_accusations", "produced_openings", "failed_dealer_states",
            "batch_bytes", "opening_bytes",
        ):
            strict_int(getattr(self, name), name=name)


@dataclass(frozen=True)
class BHK22ResponderGenerationResult:
    batch: BHK22ResponderBatch
    metrics: BHK22ResponderMetrics
    failed_dealer_ids: Sequence[int]

    def __post_init__(self) -> None:
        failed = _canonical_ids(
            self.failed_dealer_ids, n_r=self.batch.n_r, name="failed responder dealers"
        )
        object.__setattr__(self, "failed_dealer_ids", failed)


class BHK22NoResponseRequired(RuntimeError):
    pass


class BHK22ResponderRole:
    def __init__(
        self,
        *,
        context: BHK22PublicContext,
        state: BHK22ResponderState,
        dealer_batches: Sequence[BHK22DealerBatch],
        preprocessing: BHK22AccusationPreprocessing,
        response_phase_qid: str,
    ) -> None:
        if not isinstance(context, BHK22PublicContext):
            raise TypeError("responder role requires BHK22PublicContext")
        if not isinstance(state, BHK22ResponderState):
            raise TypeError("responder role requires BHK22ResponderState")
        if not isinstance(preprocessing, BHK22AccusationPreprocessing):
            raise TypeError("responder role requires accusation preprocessing")
        response_qid = nonempty_text(response_phase_qid, name="response phase qid")
        dealers = _normalize_dealers(
            context,
            dealer_batches,
            deal_phase_qid=preprocessing.deal_phase_qid,
        )
        if (
            preprocessing.root_context_digest != context.root_context_digest
            or preprocessing.n_r != context.parameters.n_r
            or state.role_id >= context.parameters.n_r
        ):
            raise ValueError("responder input/context mismatch")
        self.context = context
        self.state = state
        self.dealer_batches = dealers
        self.preprocessing = preprocessing
        self.response_phase_qid = response_qid

    def generate_batch(self) -> BHK22ResponderGenerationResult:
        if not self.preprocessing.response_required:
            raise BHK22NoResponseRequired("optimistic path has no response phase")
        started = perf_counter()
        cpu_started = process_time()
        responder = self.state.role_id
        secret = self.state._recipient_secret_for_protocol()
        dealer_ids = tuple(sorted({dealer for dealer, _ in self.preprocessing.remaining_accusations}))
        decrypt_started = perf_counter()
        decrypt_cpu_started = process_time()
        states = {}
        failed = []
        for dealer in dealer_ids:
            try:
                states[dealer] = decrypt_dealer_responder_state(
                    self.context,
                    self.dealer_batches[dealer],
                    responder_role_id=responder,
                    responder_recipient_secret_key=secret,
                )
            except (KeyError, TypeError, ValueError):
                failed.append(dealer)
        decrypt_ms = (perf_counter() - decrypt_started) * 1000.0
        decrypt_cpu_seconds = process_time() - decrypt_cpu_started

        opening_started = perf_counter()
        opening_cpu_started = process_time()
        openings = []
        for dealer, verifier in self.preprocessing.remaining_accusations:
            if dealer not in states:
                continue
            openings.append(
                BHK22ResponderOpening(
                    dealer_role_id=dealer,
                    verifier_role_id=verifier,
                    key_share_opening=open_vector_coordinate(
                        states[dealer].key_share_full_opening, verifier
                    ),
                )
            )
        opening_ms = (perf_counter() - opening_started) * 1000.0
        opening_cpu_seconds = process_time() - opening_cpu_started

        serialization_started = perf_counter()
        batch = BHK22ResponderBatch(
            root_context_digest=self.context.root_context_digest,
            phase_qid=self.response_phase_qid,
            n_r=self.context.parameters.n_r,
            role_id=responder,
            network_sender_id=self.context.committees[ROLE_RESPONDER].role_to_node[
                responder
            ],
            openings=openings,
        )
        batch_bytes = len(batch.to_bytes())
        serialization_ms = (perf_counter() - serialization_started) * 1000.0
        consumed = self.state.consume_once()
        if consumed.recipient_secret_key != secret:
            raise RuntimeError("responder state changed during response generation")
        metrics = BHK22ResponderMetrics(
            future_state_decrypt_ms=decrypt_ms,
            opening_ms=opening_ms,
            serialization_ms=serialization_ms,
            total_ms=(perf_counter() - started) * 1000.0,
            requested_accusations=len(self.preprocessing.remaining_accusations),
            produced_openings=len(openings),
            failed_dealer_states=len(failed),
            batch_bytes=batch_bytes,
            opening_bytes=sum(
                primitive_size_bytes(item.key_share_opening) for item in openings
            ),
            cpu_seconds=process_time() - cpu_started,
            future_state_decrypt_cpu_seconds=decrypt_cpu_seconds,
            opening_cpu_seconds=opening_cpu_seconds,
        )
        return BHK22ResponderGenerationResult(
            batch=batch, metrics=metrics, failed_dealer_ids=failed
        )


@dataclass(frozen=True)
class BHK22ResponderPublicVerification:
    accepted: bool
    valid_pairs: Sequence[Sequence[int]]
    invalid_pairs: Sequence[Sequence[int]]
    verify_ms: float
    verify_cpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        n_r = 1 + max(
            [0]
            + [item for pair in self.valid_pairs for item in pair]
            + [item for pair in self.invalid_pairs for item in pair]
        )
        valid = _canonical_pairs(self.valid_pairs, n_r=n_r, name="valid response pairs")
        invalid = _canonical_pairs(self.invalid_pairs, n_r=n_r, name="invalid response pairs")
        if self.accepted != (not invalid):
            raise ValueError("response verification accepted/pair mismatch")
        if self.verify_ms < 0 or self.verify_cpu_seconds < 0:
            raise ValueError("response verification time cannot be negative")
        object.__setattr__(self, "valid_pairs", valid)
        object.__setattr__(self, "invalid_pairs", invalid)


def verify_responder_batch_public(
    context: BHK22PublicContext,
    batch: BHK22ResponderBatch,
    dealer_batches: Sequence[BHK22DealerBatch],
    preprocessing: BHK22AccusationPreprocessing,
    *,
    expected_response_phase_qid: str,
) -> BHK22ResponderPublicVerification:
    started = perf_counter()
    cpu_started = process_time()
    dealers = _normalize_dealers(
        context, dealer_batches, deal_phase_qid=preprocessing.deal_phase_qid
    )
    allowed = set(preprocessing.remaining_accusations)
    valid = []
    invalid = []
    header_valid = (
        batch.root_context_digest == context.root_context_digest
        and batch.phase_qid == expected_response_phase_qid
        and batch.n_r == context.parameters.n_r
        and batch.network_sender_id
        == context.committees[ROLE_RESPONDER].role_to_node[batch.role_id]
    )
    for opening in batch.openings:
        pair = (opening.dealer_role_id, opening.verifier_role_id)
        package = dealers[opening.dealer_role_id].responder_packages[batch.role_id]
        if pair in allowed and verify_coordinate_opening(
            package.key_share_commitment, opening.key_share_opening
        ):
            valid.append(pair)
        else:
            invalid.append(pair)
    if not header_valid:
        invalid.extend(pair for pair in allowed if pair not in invalid)
    valid = tuple(sorted(set(valid) - set(invalid)))
    invalid = tuple(sorted(set(invalid)))
    return BHK22ResponderPublicVerification(
        accepted=header_valid and not invalid,
        valid_pairs=valid,
        invalid_pairs=invalid,
        verify_ms=(perf_counter() - started) * 1000.0,
        verify_cpu_seconds=process_time() - cpu_started,
    )


def scalar_degree_check(
    points: Sequence[int], values: Sequence[int], degree: int
) -> bool:
    try:
        normalized_points = tuple(strict_int(point, name="degree-check point", minimum=1) for point in points)
        normalized_values = tuple(int(value) % FIELD_ORDER for value in values)
        degree = strict_int(degree, name="degree")
        if len(normalized_points) != len(normalized_values) or len(set(normalized_points)) != len(normalized_points):
            return False
        if len(normalized_points) <= degree + 1:
            return bool(normalized_points)
        basis_points = normalized_points[: degree + 1]
        basis_values = normalized_values[: degree + 1]
        return all(
            interpolate_at(basis_points, basis_values, at=point) == value
            for point, value in zip(
                normalized_points[degree + 1 :], normalized_values[degree + 1 :]
            )
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return False


@dataclass(frozen=True)
class BHK22ShareholderDisqualificationResult:
    shareholder_role_id: int
    global_ignored_verifier_ids: Sequence[int]
    local_ignored_verifier_ids: Sequence[int]
    pre_disqualified_dealer_ids: Sequence[int]
    disqualified_dealer_ids: Sequence[int]
    qualified_dealer_ids: Sequence[int]
    remaining_verifier_ids: Sequence[int]
    accusation_outcomes: Mapping[tuple[int, int], str]
    processing_ms: float

    def __post_init__(self) -> None:
        n_r = len(self.qualified_dealer_ids) + len(self.disqualified_dealer_ids)
        shareholder = strict_int(self.shareholder_role_id, name="shareholder role id")
        if shareholder >= n_r:
            raise ValueError("shareholder role id is outside result n_r")
        global_ignored = _canonical_ids(
            self.global_ignored_verifier_ids, n_r=n_r, name="global ignored verifiers"
        )
        local_ignored = _canonical_ids(
            self.local_ignored_verifier_ids, n_r=n_r, name="local ignored verifiers"
        )
        pre = _canonical_ids(
            self.pre_disqualified_dealer_ids, n_r=n_r, name="pre-disqualified dealers"
        )
        disqualified = _canonical_ids(
            self.disqualified_dealer_ids, n_r=n_r, name="disqualified dealers"
        )
        qualified = _canonical_ids(
            self.qualified_dealer_ids, n_r=n_r, name="qualified dealers"
        )
        remaining = _canonical_ids(
            self.remaining_verifier_ids, n_r=n_r, name="remaining verifiers"
        )
        if set(qualified) | set(disqualified) != set(range(n_r)) or set(qualified) & set(disqualified):
            raise ValueError("qualified/disqualified sets must partition dealers")
        if not set(global_ignored) <= set(local_ignored) or not set(pre) <= set(disqualified):
            raise ValueError("local/global preprocessing result mismatch")
        outcomes = {
            (strict_int(pair[0], name="outcome dealer"), strict_int(pair[1], name="outcome verifier")): nonempty_text(status, name="outcome status")
            for pair, status in self.accusation_outcomes.items()
        }
        allowed_outcomes = {
            "insufficient-response",
            "inconsistent-key-shares",
            "dealer-row-invalid",
            "false-accusation",
        }
        if any(status not in allowed_outcomes for status in outcomes.values()):
            raise ValueError("unsupported accusation outcome")
        if isinstance(self.processing_ms, bool) or not isinstance(
            self.processing_ms, (int, float)
        ) or self.processing_ms < 0:
            raise ValueError("processing_ms must be a non-negative number")
        object.__setattr__(self, "global_ignored_verifier_ids", global_ignored)
        object.__setattr__(self, "local_ignored_verifier_ids", local_ignored)
        object.__setattr__(self, "pre_disqualified_dealer_ids", pre)
        object.__setattr__(self, "disqualified_dealer_ids", disqualified)
        object.__setattr__(self, "qualified_dealer_ids", qualified)
        object.__setattr__(self, "remaining_verifier_ids", remaining)
        object.__setattr__(self, "accusation_outcomes", MappingProxyType(outcomes))


def resolve_disqualifications(
    context: BHK22PublicContext,
    dealer_batches: Sequence[BHK22DealerBatch],
    responder_batches: Sequence[BHK22ResponderBatch],
    preprocessing: BHK22AccusationPreprocessing,
    *,
    response_phase_qid: str,
    shareholder_role_id: int,
) -> BHK22ShareholderDisqualificationResult:
    """Resolve remaining accusations using public responder evidence."""

    started = perf_counter()
    dealers = _normalize_dealers(
        context, dealer_batches, deal_phase_qid=preprocessing.deal_phase_qid
    )
    shareholder = strict_int(shareholder_role_id, name="shareholder_role_id")
    if shareholder >= context.parameters.n_r:
        raise ValueError("shareholder role id is outside n_r")
    responses = tuple(responder_batches)
    if preprocessing.response_required:
        if len(responses) != context.parameters.n_r or not all(
            isinstance(item, BHK22ResponderBatch) for item in responses
        ):
            raise ValueError("response path requires exactly n_r responder batches")
        if tuple(item.role_id for item in responses) != tuple(range(context.parameters.n_r)):
            raise ValueError("responder batches must be ordered by distinct role id")
    elif responses:
        raise ValueError("optimistic path must not contain responder batches")

    evidence = {pair: [] for pair in preprocessing.remaining_accusations}
    for batch in responses:
        public = verify_responder_batch_public(
            context,
            batch,
            dealers,
            preprocessing,
            expected_response_phase_qid=response_phase_qid,
        )
        if not public.accepted:
            continue
        for opening in batch.openings:
            pair = (opening.dealer_role_id, opening.verifier_role_id)
            if pair in evidence:
                evidence[pair].append(
                    (batch.role_id + 1, opening.key_share_opening.value)
                )

    threshold = context.parameters.n_r - context.parameters.t_r
    outcomes = {}
    for pair in preprocessing.remaining_accusations:
        dealer, verifier = pair
        shares = sorted(evidence[pair])
        if len(shares) < threshold:
            outcomes[pair] = "insufficient-response"
            continue
        points = tuple(point for point, _ in shares)
        values = tuple(value for _, value in shares)
        if not scalar_degree_check(points, values, context.parameters.t_r):
            outcomes[pair] = "inconsistent-key-shares"
            continue
        epsilon = interpolate_at_zero(
            points[: context.parameters.t_r + 1],
            values[: context.parameters.t_r + 1],
        )
        try:
            decrypt_dealer_row_with_recovered_epsilon(
                context,
                dealers[dealer],
                verifier_role_id=verifier,
                recovered_epsilon=epsilon,
            )
            outcomes[pair] = "false-accusation"
        except (KeyError, TypeError, ValueError):
            outcomes[pair] = "dealer-row-invalid"

    false_accusers = {
        verifier
        for (dealer, verifier), status in outcomes.items()
        if status == "false-accusation"
    }
    local_ignored = tuple(
        sorted(set(preprocessing.ignored_verifier_ids) | false_accusers)
    )
    disqualified = set(preprocessing.pre_disqualified_dealer_ids)
    for (dealer, verifier), status in outcomes.items():
        if verifier in local_ignored:
            continue
        if status != "false-accusation":
            disqualified.add(dealer)
    disqualified_ids = tuple(sorted(disqualified))
    qualified = tuple(
        dealer for dealer in range(context.parameters.n_r) if dealer not in disqualified
    )
    remaining_verifiers = tuple(
        verifier
        for verifier in range(context.parameters.n_r)
        if verifier not in local_ignored
    )
    return BHK22ShareholderDisqualificationResult(
        shareholder_role_id=shareholder,
        global_ignored_verifier_ids=preprocessing.ignored_verifier_ids,
        local_ignored_verifier_ids=local_ignored,
        pre_disqualified_dealer_ids=preprocessing.pre_disqualified_dealer_ids,
        disqualified_dealer_ids=disqualified_ids,
        qualified_dealer_ids=qualified,
        remaining_verifier_ids=remaining_verifiers,
        accusation_outcomes=outcomes,
        processing_ms=(perf_counter() - started) * 1000.0,
    )


__all__ = [
    "BHK22AccusationPreprocessing",
    "BHK22NoResponseRequired",
    "BHK22ResponderGenerationResult",
    "BHK22ResponderMetrics",
    "BHK22ResponderPublicVerification",
    "BHK22ResponderRole",
    "BHK22ShareholderDisqualificationResult",
    "preprocess_accusations",
    "resolve_disqualifications",
    "scalar_degree_check",
    "verify_responder_batch_public",
]
