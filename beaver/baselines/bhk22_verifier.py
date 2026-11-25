"""Node-local BHK+22 Figure 6 verifier role (Step 4)."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from time import perf_counter, process_time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from beaver.baselines.bhk22_context import (
    BHK22PublicContext,
    canonical_json,
    decode_object,
    digest_hex,
    exact_fields,
    nonempty_text,
    strict_int,
)
from beaver.baselines.bhk22_contract import (
    ROLE_SHAREHOLDER,
    ROLE_VERIFIER,
)
from beaver.baselines.bhk22_crypto import (
    BHK22AAD,
    BHK22FullOpening,
    BHK22ProofContext,
    commit_vector,
    primitive_size_bytes,
    proof_size_bytes,
    prove_forwarding_logs,
    recipient_decrypt,
    recipient_encrypt,
    verify_forwarding_logs,
    verify_full_opening,
)
from beaver.baselines.bhk22_dealer import (
    decrypt_dealer_verifier_payload,
    verify_dealer_batch_public,
)
from beaver.baselines.bhk22_experiment import BHK22VerifierState
from beaver.baselines.bhk22_messages import (
    BHK22DealerBatch,
    BHK22VerifierBatch,
    BHK22VerifierShareholderPackage,
)
from beaver.g1_backend import FIELD_ORDER, G1Point, field_random, g1_linear_combination


PRIVATE_PAYLOAD_VERSION = 1
SHAREHOLDER_PAYLOAD_KIND = "atcaas-bhk22-private-verifier-shareholder-payload"
SHAREHOLDER_AAD_KIND = "verifier-shareholder-private-payload"
VERIFIER_CHALLENGE_DOMAIN = b"ATCaaS/BHK22/verifier-challenges/v1\x00"


def _canonical_id_set(values: Sequence[int], *, n_r: int, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an integer sequence")
    result = tuple(strict_int(value, name=f"{name} entry") for value in values)
    if not result or result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be non-empty, sorted, and unique")
    if any(value >= n_r for value in result):
        raise ValueError(f"{name} contains an out-of-range dealer id")
    return result


@dataclass(frozen=True)
class BHK22VerifierShareholderPayload:
    root_context_digest: str
    phase_qid: str
    verifier_role_id: int
    shareholder_role_id: int
    n_r: int
    no_accusation_dealer_ids: Sequence[int]
    vector_full_opening: BHK22FullOpening

    def __post_init__(self) -> None:
        root = digest_hex(self.root_context_digest, name="payload root context digest")
        qid = nonempty_text(self.phase_qid, name="payload phase qid")
        verifier = strict_int(self.verifier_role_id, name="payload verifier_role_id")
        shareholder = strict_int(
            self.shareholder_role_id, name="payload shareholder_role_id"
        )
        n_r = strict_int(self.n_r, name="payload n_r", minimum=1)
        if verifier >= n_r or shareholder >= n_r:
            raise ValueError("payload role id is outside n_r")
        clean = _canonical_id_set(
            self.no_accusation_dealer_ids,
            n_r=n_r,
            name="payload no_accusation_dealer_ids",
        )
        if not isinstance(self.vector_full_opening, BHK22FullOpening):
            raise TypeError("payload requires BHK22FullOpening")
        if len(self.vector_full_opening.values) != len(clean):
            raise ValueError("payload vector length must equal |NA_j|")
        object.__setattr__(self, "root_context_digest", root)
        object.__setattr__(self, "phase_qid", qid)
        object.__setattr__(self, "verifier_role_id", verifier)
        object.__setattr__(self, "shareholder_role_id", shareholder)
        object.__setattr__(self, "n_r", n_r)
        object.__setattr__(self, "no_accusation_dealer_ids", clean)

    @property
    def share_vector(self) -> tuple[int, ...]:
        return tuple(self.vector_full_opening.values)

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": SHAREHOLDER_PAYLOAD_KIND,
            "version": PRIVATE_PAYLOAD_VERSION,
            "root_context_digest": self.root_context_digest,
            "phase_qid": self.phase_qid,
            "verifier_role_id": self.verifier_role_id,
            "shareholder_role_id": self.shareholder_role_id,
            "n_r": self.n_r,
            "no_accusation_dealer_ids": list(self.no_accusation_dealer_ids),
            "vector_full_opening": self.vector_full_opening.to_obj(),
        }

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_obj())

    @classmethod
    def from_obj(
        cls, value: Mapping[str, Any]
    ) -> "BHK22VerifierShareholderPayload":
        exact_fields(
            value,
            {
                "kind", "version", "root_context_digest", "phase_qid",
                "verifier_role_id", "shareholder_role_id", "n_r",
                "no_accusation_dealer_ids", "vector_full_opening",
            },
            name="BHK22 verifier-shareholder payload",
        )
        if value["kind"] != SHAREHOLDER_PAYLOAD_KIND or value["version"] != 1:
            raise ValueError("unsupported BHK22 verifier-shareholder payload")
        return cls(
            root_context_digest=value["root_context_digest"],
            phase_qid=value["phase_qid"],
            verifier_role_id=value["verifier_role_id"],
            shareholder_role_id=value["shareholder_role_id"],
            n_r=value["n_r"],
            no_accusation_dealer_ids=value["no_accusation_dealer_ids"],
            vector_full_opening=BHK22FullOpening.from_obj(
                value["vector_full_opening"]
            ),
        )

    @classmethod
    def from_bytes(
        cls, data: bytes | str
    ) -> "BHK22VerifierShareholderPayload":
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        restored = cls.from_obj(
            decode_object(raw, name="BHK22 verifier-shareholder payload")
        )
        if not hmac.compare_digest(raw, restored.to_bytes()):
            raise ValueError("BHK22 verifier-shareholder payload is not canonical JSON")
        return restored


class BHK22VerifierRandomSource:
    def scalar(self, label: str) -> int:
        nonempty_text(label, name="verifier randomness label")
        return field_random()

    def bytes(self, label: str, length: int) -> bytes:
        nonempty_text(label, name="verifier randomness label")
        return secrets.token_bytes(strict_int(length, name="random byte length", minimum=1))


class BHK22DeterministicVerifierRandomSource(BHK22VerifierRandomSource):
    """Explicit fixture/test-only verifier randomness source."""

    def __init__(self, seed: str) -> None:
        self._seed = nonempty_text(seed, name="deterministic verifier seed").encode("utf-8")

    def _block(self, label: str, counter: int) -> bytes:
        return hashlib.sha256(
            b"ATCaaS/BHK22/verifier-test-randomness/v1\x00"
            + self._seed
            + b"\x00"
            + label.encode("utf-8")
            + counter.to_bytes(4, "big")
        ).digest()

    def scalar(self, label: str) -> int:
        nonempty_text(label, name="verifier randomness label")
        return int.from_bytes(self._block(label, 0), "big") % (FIELD_ORDER - 1) + 1

    def bytes(self, label: str, length: int) -> bytes:
        nonempty_text(label, name="verifier randomness label")
        length = strict_int(length, name="random byte length", minimum=1)
        output = bytearray()
        counter = 0
        while len(output) < length:
            output.extend(self._block(label, counter))
            counter += 1
        return bytes(output[:length])


def derive_verifier_challenges(
    *,
    root_context_digest: str,
    phase_qid: str,
    verifier_role_id: int,
    n_r: int,
    no_accusation_dealer_ids: Sequence[int],
    shareholder_commitments: Sequence[Any],
) -> Mapping[int, int]:
    """Model Fig. 6's ``e <- O(j, h_j, NA_j)`` with explicit context."""

    root = digest_hex(root_context_digest, name="challenge root context digest")
    qid = nonempty_text(phase_qid, name="challenge phase qid")
    verifier = strict_int(verifier_role_id, name="challenge verifier_role_id")
    n_r = strict_int(n_r, name="challenge n_r", minimum=1)
    if verifier >= n_r:
        raise ValueError("challenge verifier id is outside n_r")
    clean = _canonical_id_set(
        no_accusation_dealer_ids, n_r=n_r, name="challenge NA set"
    )
    commitments = tuple(shareholder_commitments)
    if len(commitments) != n_r or not all(
        hasattr(commitment, "to_obj") for commitment in commitments
    ):
        raise ValueError("challenge requires one h commitment per shareholder")
    transcript = canonical_json(
        {
            "root_context_digest": root,
            "phase_qid": qid,
            "verifier_role_id": verifier,
            "n_r": n_r,
            "no_accusation_dealer_ids": list(clean),
            "shareholder_commitments": [item.to_obj() for item in commitments],
        }
    )
    challenges = {}
    for dealer in clean:
        scalar = int.from_bytes(
            hashlib.sha256(
                VERIFIER_CHALLENGE_DOMAIN
                + transcript
                + dealer.to_bytes(4, "big")
            ).digest(),
            "big",
        ) % FIELD_ORDER
        challenges[dealer] = scalar or 1
    return MappingProxyType(challenges)


def _shareholder_aad(
    context: BHK22PublicContext,
    phase_qid: str,
    verifier: int,
    shareholder: int,
) -> BHK22AAD:
    return BHK22AAD(
        root_context_digest=context.root_context_digest,
        phase_qid=phase_qid,
        sender_role=ROLE_VERIFIER,
        sender_role_id=verifier,
        recipient_role=ROLE_SHAREHOLDER,
        recipient_role_id=shareholder,
        payload_kind=SHAREHOLDER_AAD_KIND,
    )


@dataclass(frozen=True)
class BHK22VerifierMetrics:
    public_checks_ms: float
    row_decrypt_ms: float
    commitment_ms: float
    forwarding_proof_ms: float
    shareholder_encryption_ms: float
    serialization_ms: float
    total_ms: float
    public_check_failures: int
    row_decrypt_failures: int
    accusation_count: int
    batch_bytes: int
    shareholder_ciphertext_bytes: int
    commitment_bytes: int
    forwarding_proof_bytes: int
    cpu_seconds: float = 0.0
    public_checks_cpu_seconds: float = 0.0
    forwarding_proof_cpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "public_checks_ms", "row_decrypt_ms", "commitment_ms",
            "forwarding_proof_ms", "shareholder_encryption_ms",
            "serialization_ms", "total_ms", "cpu_seconds",
            "public_checks_cpu_seconds", "forwarding_proof_cpu_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a non-negative number")
        for name in (
            "public_check_failures", "row_decrypt_failures", "accusation_count",
            "batch_bytes", "shareholder_ciphertext_bytes", "commitment_bytes",
            "forwarding_proof_bytes",
        ):
            strict_int(getattr(self, name), name=name)

    def to_obj(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class BHK22VerifierGenerationResult:
    batch: BHK22VerifierBatch
    metrics: BHK22VerifierMetrics
    accusation_reasons: Mapping[int, str]

    def __post_init__(self) -> None:
        if not isinstance(self.batch, BHK22VerifierBatch):
            raise TypeError("verifier result requires BHK22VerifierBatch")
        if not isinstance(self.metrics, BHK22VerifierMetrics):
            raise TypeError("verifier result requires BHK22VerifierMetrics")
        reasons = {
            strict_int(dealer, name="accusation dealer id"): nonempty_text(
                reason, name="accusation reason"
            )
            for dealer, reason in self.accusation_reasons.items()
        }
        if tuple(sorted(reasons)) != self.batch.accused_dealer_ids:
            raise ValueError("accusation reasons do not match public accusation set")
        object.__setattr__(self, "accusation_reasons", MappingProxyType(reasons))


@dataclass(frozen=True)
class BHK22VerifierPublicVerification:
    accepted: bool
    checks: Mapping[str, bool]
    verify_ms: float
    verify_cpu_seconds: float = 0.0
    forwarding_proof_verify_ms: float = 0.0
    forwarding_proof_verify_cpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        checks = {str(name): bool(value) for name, value in self.checks.items()}
        if self.accepted != all(checks.values()):
            raise ValueError("verifier public verification mismatch")
        for value in (
            self.verify_ms,
            self.verify_cpu_seconds,
            self.forwarding_proof_verify_ms,
            self.forwarding_proof_verify_cpu_seconds,
        ):
            if value < 0:
                raise ValueError("verifier public verification time cannot be negative")
        object.__setattr__(self, "checks", MappingProxyType(checks))


def _normalize_dealer_batches(
    context: BHK22PublicContext,
    dealer_batches: Sequence[BHK22DealerBatch],
    *,
    deal_phase_qid: str,
) -> tuple[BHK22DealerBatch, ...]:
    if isinstance(dealer_batches, (str, bytes, bytearray)):
        raise TypeError("dealer_batches must be a sequence")
    batches = tuple(dealer_batches)
    n_r = context.parameters.n_r
    if len(batches) != n_r or not all(
        isinstance(batch, BHK22DealerBatch) for batch in batches
    ):
        raise ValueError("verifier requires exactly n_r dealer batches")
    if tuple(batch.role_id for batch in batches) != tuple(range(n_r)):
        raise ValueError("dealer batches must be canonically ordered by role id")
    if any(
        batch.root_context_digest != context.root_context_digest
        or batch.phase_qid != deal_phase_qid
        or batch.n_r != n_r
        for batch in batches
    ):
        raise ValueError("dealer batch context/qid mismatch")
    return batches


class BHK22VerifierRole:
    """Consume one verifier state and emit one immutable verifier batch."""

    def __init__(
        self,
        *,
        context: BHK22PublicContext,
        state: BHK22VerifierState,
        dealer_batches: Sequence[BHK22DealerBatch],
        deal_phase_qid: str,
        verify_phase_qid: str,
        random_source: BHK22VerifierRandomSource | None = None,
        public_batch_verifier: Callable[[BHK22DealerBatch], bool] | None = None,
    ) -> None:
        if not isinstance(context, BHK22PublicContext):
            raise TypeError("verifier role requires BHK22PublicContext")
        if not isinstance(state, BHK22VerifierState):
            raise TypeError("verifier role requires BHK22VerifierState")
        deal_qid = nonempty_text(deal_phase_qid, name="deal phase qid")
        verify_qid = nonempty_text(verify_phase_qid, name="verify phase qid")
        if state.role_id >= context.parameters.n_r:
            raise ValueError("verifier state role id is outside context")
        if random_source is not None and not isinstance(
            random_source, BHK22VerifierRandomSource
        ):
            raise TypeError("random_source must be BHK22VerifierRandomSource")
        self.context = context
        self.state = state
        self.deal_phase_qid = deal_qid
        self.verify_phase_qid = verify_qid
        self.dealer_batches = _normalize_dealer_batches(
            context, dealer_batches, deal_phase_qid=deal_qid
        )
        self.random_source = random_source or BHK22VerifierRandomSource()
        if public_batch_verifier is not None and not callable(public_batch_verifier):
            raise TypeError("public_batch_verifier must be callable or null")
        self.public_batch_verifier = public_batch_verifier

    def generate_batch(self) -> BHK22VerifierGenerationResult:
        total_started = perf_counter()
        cpu_started = process_time()
        context = self.context
        n_r = context.parameters.n_r
        verifier = self.state.role_id
        recipient_secret = self.state._recipient_secret_for_protocol()
        reasons = {}
        clean_rows = {}

        public_started = perf_counter()
        public_cpu_started = process_time()
        public_results = {
            batch.role_id: (
                bool(self.public_batch_verifier(batch))
                if self.public_batch_verifier is not None
                else verify_dealer_batch_public(
                    context, batch, expected_phase_qid=self.deal_phase_qid
                )
            )
            for batch in self.dealer_batches
        }
        public_checks_ms = (perf_counter() - public_started) * 1000.0
        public_checks_cpu_seconds = process_time() - public_cpu_started
        for dealer, result in public_results.items():
            if not (result if isinstance(result, bool) else result.accepted):
                reasons[dealer] = "public-check"

        decrypt_started = perf_counter()
        for batch in self.dealer_batches:
            dealer = batch.role_id
            if dealer in reasons:
                continue
            try:
                clean_rows[dealer] = decrypt_dealer_verifier_payload(
                    context,
                    batch,
                    verifier_role_id=verifier,
                    verifier_recipient_secret_key=recipient_secret,
                ).row
            except (KeyError, TypeError, ValueError):
                reasons[dealer] = "row-decrypt-or-check"
        row_decrypt_ms = (perf_counter() - decrypt_started) * 1000.0

        accused = tuple(sorted(reasons))
        clean = tuple(dealer for dealer in range(n_r) if dealer not in reasons)
        if not clean:
            raise ValueError("verifier has no non-accused dealer to forward")

        commitment_started = perf_counter()
        commitment_records = []
        for shareholder in range(n_r):
            vector = tuple(clean_rows[dealer][shareholder + 1] for dealer in clean)
            commitment, full_opening = commit_vector(
                vector,
                salts=tuple(
                    self.random_source.bytes(
                        f"verifier:{verifier}:h-salt:{shareholder}:{offset}", 32
                    )
                    for offset in range(len(clean))
                ),
            )
            commitment_records.append((shareholder, commitment, full_opening))
        commitment_ms = (perf_counter() - commitment_started) * 1000.0

        challenges = derive_verifier_challenges(
            root_context_digest=context.root_context_digest,
            phase_qid=self.verify_phase_qid,
            verifier_role_id=verifier,
            n_r=n_r,
            no_accusation_dealer_ids=clean,
            shareholder_commitments=[item[1] for item in commitment_records],
        )
        rho = tuple(
            sum(clean_rows[dealer][coordinate] * challenges[dealer] for dealer in clean)
            % FIELD_ORDER
            for coordinate in range(n_r + 1)
        )
        forwarding_points = tuple(
            (G1Point.from_hex(context.crs.generators[index]) * rho[index]).to_hex()
            for index in range(n_r + 1)
        )
        proof_context = BHK22ProofContext(
            root_context_digest=context.root_context_digest,
            phase_qid=self.verify_phase_qid,
            role_type=ROLE_VERIFIER,
            role_id=verifier,
        )
        proof_started = perf_counter()
        proof_cpu_started = process_time()
        forwarding_proof = prove_forwarding_logs(
            context=proof_context,
            bases=context.crs.generators,
            points=forwarding_points,
            logs=rho,
        )
        forwarding_proof_ms = (perf_counter() - proof_started) * 1000.0
        forwarding_proof_cpu_seconds = process_time() - proof_cpu_started

        encryption_started = perf_counter()
        shareholder_catalog = context.recipient_key_catalogs[ROLE_SHAREHOLDER]
        shareholder_packages = []
        for shareholder, commitment, full_opening in commitment_records:
            payload = BHK22VerifierShareholderPayload(
                root_context_digest=context.root_context_digest,
                phase_qid=self.verify_phase_qid,
                verifier_role_id=verifier,
                shareholder_role_id=shareholder,
                n_r=n_r,
                no_accusation_dealer_ids=clean,
                vector_full_opening=full_opening,
            )
            ciphertext = recipient_encrypt(
                key_base_hex=shareholder_catalog.key_base_hex,
                recipient_public_key_hex=shareholder_catalog.public_keys[shareholder],
                plaintext=payload.to_bytes(),
                aad=_shareholder_aad(
                    context, self.verify_phase_qid, verifier, shareholder
                ),
                purpose="shareholder-private-payload",
                ephemeral_secret=self.random_source.scalar(
                    f"verifier:{verifier}:shareholder-ephemeral:{shareholder}"
                ),
                nonce=self.random_source.bytes(
                    f"verifier:{verifier}:shareholder-nonce:{shareholder}", 12
                ),
            )
            shareholder_packages.append(
                BHK22VerifierShareholderPackage(
                    shareholder_role_id=shareholder,
                    vector_commitment=commitment,
                    encrypted_payload=ciphertext,
                )
            )
        shareholder_encryption_ms = (perf_counter() - encryption_started) * 1000.0

        serialization_started = perf_counter()
        batch = BHK22VerifierBatch(
            root_context_digest=context.root_context_digest,
            phase_qid=self.verify_phase_qid,
            n_r=n_r,
            role_id=verifier,
            network_sender_id=context.committees[ROLE_VERIFIER].role_to_node[verifier],
            accused_dealer_ids=accused,
            no_accusation_dealer_ids=clean,
            shareholder_packages=shareholder_packages,
            forwarding_points=forwarding_points,
            forwarding_proof=forwarding_proof,
        )
        batch_bytes = len(batch.to_bytes())
        serialization_ms = (perf_counter() - serialization_started) * 1000.0

        consumed = self.state.consume_once()
        if consumed.recipient_secret_key != recipient_secret or consumed.role_id != verifier:
            raise RuntimeError("verifier state changed during batch generation")

        metrics = BHK22VerifierMetrics(
            public_checks_ms=public_checks_ms,
            row_decrypt_ms=row_decrypt_ms,
            commitment_ms=commitment_ms,
            forwarding_proof_ms=forwarding_proof_ms,
            shareholder_encryption_ms=shareholder_encryption_ms,
            serialization_ms=serialization_ms,
            total_ms=(perf_counter() - total_started) * 1000.0,
            public_check_failures=sum(
                not (result if isinstance(result, bool) else result.accepted)
                for result in public_results.values()
            ),
            row_decrypt_failures=sum(
                reason == "row-decrypt-or-check" for reason in reasons.values()
            ),
            accusation_count=len(accused),
            batch_bytes=batch_bytes,
            shareholder_ciphertext_bytes=sum(
                primitive_size_bytes(item.encrypted_payload)
                for item in shareholder_packages
            ),
            commitment_bytes=sum(
                primitive_size_bytes(item.vector_commitment)
                for item in shareholder_packages
            ),
            forwarding_proof_bytes=proof_size_bytes(forwarding_proof),
            cpu_seconds=process_time() - cpu_started,
            public_checks_cpu_seconds=public_checks_cpu_seconds,
            forwarding_proof_cpu_seconds=forwarding_proof_cpu_seconds,
        )
        return BHK22VerifierGenerationResult(
            batch=batch,
            metrics=metrics,
            accusation_reasons=reasons,
        )


def verify_verifier_batch_public(
    context: BHK22PublicContext,
    batch: BHK22VerifierBatch,
    dealer_batches: Sequence[BHK22DealerBatch],
    *,
    deal_phase_qid: str,
    expected_verify_phase_qid: str | None = None,
    dealer_public_verifier: Callable[[BHK22DealerBatch], bool] | None = None,
) -> BHK22VerifierPublicVerification:
    """Verify public h/R/NIZK and the Figure 6 row-sum relation."""

    started = perf_counter()
    cpu_started = process_time()
    forwarding_proof_verify_ms = 0.0
    forwarding_proof_verify_cpu_seconds = 0.0
    checks = {
        "wire_context": False,
        "dealer_public_inputs": False,
        "forwarding_proof": False,
        "row_sum_relation": False,
    }
    try:
        if not isinstance(context, BHK22PublicContext) or not isinstance(
            batch, BHK22VerifierBatch
        ):
            raise TypeError("wrong verifier public verification input")
        dealers = _normalize_dealer_batches(
            context, dealer_batches, deal_phase_qid=deal_phase_qid
        )
        verifier = batch.role_id
        checks["wire_context"] = (
            batch.root_context_digest == context.root_context_digest
            and batch.n_r == context.parameters.n_r
            and batch.network_sender_id
            == context.committees[ROLE_VERIFIER].role_to_node[verifier]
            and (
                expected_verify_phase_qid is None
                or batch.phase_qid == expected_verify_phase_qid
            )
        )
        dealer_by_id = {dealer.role_id: dealer for dealer in dealers}
        checks["dealer_public_inputs"] = all(
            (
                bool(dealer_public_verifier(dealer_by_id[dealer]))
                if dealer_public_verifier is not None
                else verify_dealer_batch_public(
                    context,
                    dealer_by_id[dealer],
                    expected_phase_qid=deal_phase_qid,
                ).accepted
            )
            for dealer in batch.no_accusation_dealer_ids
        )
        challenges = derive_verifier_challenges(
            root_context_digest=batch.root_context_digest,
            phase_qid=batch.phase_qid,
            verifier_role_id=verifier,
            n_r=batch.n_r,
            no_accusation_dealer_ids=batch.no_accusation_dealer_ids,
            shareholder_commitments=[
                package.vector_commitment for package in batch.shareholder_packages
            ],
        )
        proof_context = BHK22ProofContext(
            root_context_digest=batch.root_context_digest,
            phase_qid=batch.phase_qid,
            role_type=ROLE_VERIFIER,
            role_id=verifier,
        )
        proof_started = perf_counter()
        proof_cpu_started = process_time()
        checks["forwarding_proof"] = verify_forwarding_logs(
            context=proof_context,
            bases=context.crs.generators,
            points=batch.forwarding_points,
            proof=batch.forwarding_proof,
        )
        forwarding_proof_verify_ms = (perf_counter() - proof_started) * 1000.0
        forwarding_proof_verify_cpu_seconds = process_time() - proof_cpu_started
        left = g1_linear_combination(
            batch.forwarding_points, [1] * (batch.n_r + 1)
        ).to_hex()
        right = g1_linear_combination(
            [
                dealer_by_id[dealer].verifier_columns[verifier].column_commitment
                for dealer in batch.no_accusation_dealer_ids
            ],
            [challenges[dealer] for dealer in batch.no_accusation_dealer_ids],
        ).to_hex()
        checks["row_sum_relation"] = left == right
    except (KeyError, TypeError, ValueError):
        pass
    return BHK22VerifierPublicVerification(
        accepted=all(checks.values()),
        checks=checks,
        verify_ms=(perf_counter() - started) * 1000.0,
        verify_cpu_seconds=process_time() - cpu_started,
        forwarding_proof_verify_ms=forwarding_proof_verify_ms,
        forwarding_proof_verify_cpu_seconds=forwarding_proof_verify_cpu_seconds,
    )


def decrypt_verifier_shareholder_payload(
    context: BHK22PublicContext,
    batch: BHK22VerifierBatch,
    *,
    shareholder_role_id: int,
    shareholder_recipient_secret_key: int,
) -> BHK22VerifierShareholderPayload:
    """Decrypt one target payload and verify its h_j,l full opening."""

    shareholder = strict_int(shareholder_role_id, name="shareholder_role_id")
    if batch.root_context_digest != context.root_context_digest or batch.n_r != context.parameters.n_r:
        raise ValueError("verifier batch/context mismatch")
    if shareholder >= batch.n_r:
        raise ValueError("shareholder role id is outside n_r")
    package = batch.shareholder_packages[shareholder]
    if package.shareholder_role_id != shareholder:
        raise ValueError("shareholder package index mismatch")
    catalog = context.recipient_key_catalogs[ROLE_SHAREHOLDER]
    plaintext = recipient_decrypt(
        key_base_hex=catalog.key_base_hex,
        recipient_secret_key=shareholder_recipient_secret_key,
        ciphertext=package.encrypted_payload,
        aad=_shareholder_aad(context, batch.phase_qid, batch.role_id, shareholder),
        purpose="shareholder-private-payload",
    )
    payload = BHK22VerifierShareholderPayload.from_bytes(plaintext)
    expected_binding = (
        context.root_context_digest,
        batch.phase_qid,
        batch.role_id,
        shareholder,
        batch.n_r,
        batch.no_accusation_dealer_ids,
    )
    actual_binding = (
        payload.root_context_digest,
        payload.phase_qid,
        payload.verifier_role_id,
        payload.shareholder_role_id,
        payload.n_r,
        payload.no_accusation_dealer_ids,
    )
    if actual_binding != expected_binding:
        raise ValueError("verifier-shareholder payload binding mismatch")
    if not verify_full_opening(
        package.vector_commitment, payload.vector_full_opening
    ):
        raise ValueError("shareholder payload does not open h_j,l")
    return payload


__all__ = [
    "BHK22DeterministicVerifierRandomSource",
    "BHK22VerifierGenerationResult",
    "BHK22VerifierMetrics",
    "BHK22VerifierPublicVerification",
    "BHK22VerifierRandomSource",
    "BHK22VerifierRole",
    "BHK22VerifierShareholderPayload",
    "decrypt_verifier_shareholder_payload",
    "derive_verifier_challenges",
    "verify_verifier_batch_public",
]
