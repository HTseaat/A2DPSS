"""Node-local BHK+22 Figure 6/7/8 dealer role (Step 3)."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from time import perf_counter, process_time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from beaver.baselines.bhk22_context import (
    BHK22PublicContext,
    canonical_json,
    canonical_point,
    canonical_scalar,
    decode_object,
    digest_hex,
    exact_fields,
    nonempty_text,
    strict_int,
)
from beaver.baselines.bhk22_contract import (
    ROLE_DEALER,
    ROLE_RESPONDER,
    ROLE_VERIFIER,
)
from beaver.baselines.bhk22_crypto import (
    BHK22AAD,
    BHK22AEADCiphertext,
    BHK22FullOpening,
    BHK22ProofContext,
    aead_decrypt,
    aead_encrypt,
    commit_vector,
    derive_aead_key,
    encrypt_public_value,
    primitive_size_bytes,
    proof_size_bytes,
    prove_figure7,
    prove_figure8_randomness,
    recipient_decrypt,
    recipient_encrypt,
    verify_figure7,
    verify_figure8_randomness,
    verify_full_opening,
)
from beaver.baselines.bhk22_experiment import BHK22DealerState
from beaver.baselines.bhk22_messages import (
    BHK22DealerBatch,
    BHK22DealerResponderPackage,
    BHK22DealerVerifierColumn,
)
from beaver.g1_backend import FIELD_ORDER, G1Point, field_random, g1_linear_combination
from beaver.pedersen_shamir import eval_poly, lagrange_coeff, pedersen_deg_check


PRIVATE_PAYLOAD_VERSION = 1
VERIFIER_ROW_KIND = "atcaas-bhk22-private-verifier-row"
VERIFIER_KEY_KIND = "atcaas-bhk22-private-verifier-key"
RESPONDER_STATE_KIND = "atcaas-bhk22-private-responder-future-state"

ROW_KDF_PURPOSE = "dealer-verifier-row"
ROW_PAYLOAD_KIND = "dealer-verifier-row"
VERIFIER_KEY_PAYLOAD_KIND = "dealer-verifier-key"
RESPONDER_STATE_PAYLOAD_KIND = "responder-future-state"


def _canonical_private_bytes(value: Any) -> bytes:
    return canonical_json(value.to_obj())


def _parse_private_payload(data: bytes | str, cls, *, name: str):
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    restored = cls.from_obj(decode_object(raw, name=name))
    if not hmac.compare_digest(raw, _canonical_private_bytes(restored)):
        raise ValueError(f"{name} is not canonical JSON")
    return restored


def _scalar_sequence(values: Sequence[Any], *, name: str, length: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a scalar sequence")
    normalized = tuple(
        canonical_scalar(value, name=f"{name} entry") for value in values
    )
    if len(normalized) != length:
        raise ValueError(f"{name} must contain exactly {length} scalars")
    return normalized


@dataclass(frozen=True)
class BHK22VerifierRowPayload:
    root_context_digest: str
    phase_qid: str
    dealer_role_id: int
    verifier_role_id: int
    n_r: int
    row: Sequence[int]

    def __post_init__(self) -> None:
        root = digest_hex(self.root_context_digest, name="row root context digest")
        qid = nonempty_text(self.phase_qid, name="row phase qid")
        dealer = strict_int(self.dealer_role_id, name="row dealer_role_id")
        verifier = strict_int(self.verifier_role_id, name="row verifier_role_id")
        n_r = strict_int(self.n_r, name="row n_r", minimum=1)
        if dealer >= n_r or verifier >= n_r:
            raise ValueError("row role id is outside n_r")
        row = _scalar_sequence(self.row, name="verifier row", length=n_r + 1)
        object.__setattr__(self, "root_context_digest", root)
        object.__setattr__(self, "phase_qid", qid)
        object.__setattr__(self, "dealer_role_id", dealer)
        object.__setattr__(self, "verifier_role_id", verifier)
        object.__setattr__(self, "n_r", n_r)
        object.__setattr__(self, "row", row)

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": VERIFIER_ROW_KIND,
            "version": PRIVATE_PAYLOAD_VERSION,
            "root_context_digest": self.root_context_digest,
            "phase_qid": self.phase_qid,
            "dealer_role_id": self.dealer_role_id,
            "verifier_role_id": self.verifier_role_id,
            "n_r": self.n_r,
            "row": [str(value) for value in self.row],
        }

    def to_bytes(self) -> bytes:
        return _canonical_private_bytes(self)

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22VerifierRowPayload":
        exact_fields(
            value,
            {
                "kind", "version", "root_context_digest", "phase_qid",
                "dealer_role_id", "verifier_role_id", "n_r", "row",
            },
            name="BHK22 verifier-row payload",
        )
        if value["kind"] != VERIFIER_ROW_KIND or value["version"] != 1:
            raise ValueError("unsupported BHK22 verifier-row payload")
        return cls(
            root_context_digest=value["root_context_digest"],
            phase_qid=value["phase_qid"],
            dealer_role_id=value["dealer_role_id"],
            verifier_role_id=value["verifier_role_id"],
            n_r=value["n_r"],
            row=value["row"],
        )

    @classmethod
    def from_bytes(cls, data: bytes | str) -> "BHK22VerifierRowPayload":
        return _parse_private_payload(data, cls, name="BHK22 verifier-row payload")


@dataclass(frozen=True)
class BHK22VerifierKeyPayload:
    root_context_digest: str
    phase_qid: str
    dealer_role_id: int
    verifier_role_id: int
    n_r: int
    epsilon: int
    encrypted_row_digest: str

    def __post_init__(self) -> None:
        root = digest_hex(self.root_context_digest, name="key root context digest")
        qid = nonempty_text(self.phase_qid, name="key phase qid")
        dealer = strict_int(self.dealer_role_id, name="key dealer_role_id")
        verifier = strict_int(self.verifier_role_id, name="key verifier_role_id")
        n_r = strict_int(self.n_r, name="key n_r", minimum=1)
        if dealer >= n_r or verifier >= n_r:
            raise ValueError("key payload role id is outside n_r")
        epsilon = canonical_scalar(self.epsilon, name="epsilon", nonzero=True)
        row_digest = digest_hex(
            self.encrypted_row_digest, name="encrypted_row_digest"
        )
        object.__setattr__(self, "root_context_digest", root)
        object.__setattr__(self, "phase_qid", qid)
        object.__setattr__(self, "dealer_role_id", dealer)
        object.__setattr__(self, "verifier_role_id", verifier)
        object.__setattr__(self, "n_r", n_r)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "encrypted_row_digest", row_digest)

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": VERIFIER_KEY_KIND,
            "version": PRIVATE_PAYLOAD_VERSION,
            "root_context_digest": self.root_context_digest,
            "phase_qid": self.phase_qid,
            "dealer_role_id": self.dealer_role_id,
            "verifier_role_id": self.verifier_role_id,
            "n_r": self.n_r,
            "epsilon": str(self.epsilon),
            "encrypted_row_digest": self.encrypted_row_digest,
        }

    def to_bytes(self) -> bytes:
        return _canonical_private_bytes(self)

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22VerifierKeyPayload":
        exact_fields(
            value,
            {
                "kind", "version", "root_context_digest", "phase_qid",
                "dealer_role_id", "verifier_role_id", "n_r", "epsilon",
                "encrypted_row_digest",
            },
            name="BHK22 verifier-key payload",
        )
        if value["kind"] != VERIFIER_KEY_KIND or value["version"] != 1:
            raise ValueError("unsupported BHK22 verifier-key payload")
        return cls(
            root_context_digest=value["root_context_digest"],
            phase_qid=value["phase_qid"],
            dealer_role_id=value["dealer_role_id"],
            verifier_role_id=value["verifier_role_id"],
            n_r=value["n_r"],
            epsilon=value["epsilon"],
            encrypted_row_digest=value["encrypted_row_digest"],
        )

    @classmethod
    def from_bytes(cls, data: bytes | str) -> "BHK22VerifierKeyPayload":
        return _parse_private_payload(data, cls, name="BHK22 verifier-key payload")


@dataclass(frozen=True)
class BHK22ResponderFutureState:
    root_context_digest: str
    phase_qid: str
    dealer_role_id: int
    responder_role_id: int
    n_r: int
    key_share_full_opening: BHK22FullOpening

    def __post_init__(self) -> None:
        root = digest_hex(self.root_context_digest, name="responder root context digest")
        qid = nonempty_text(self.phase_qid, name="responder phase qid")
        dealer = strict_int(self.dealer_role_id, name="state dealer_role_id")
        responder = strict_int(self.responder_role_id, name="state responder_role_id")
        n_r = strict_int(self.n_r, name="state n_r", minimum=1)
        if dealer >= n_r or responder >= n_r:
            raise ValueError("responder state role id is outside n_r")
        if not isinstance(self.key_share_full_opening, BHK22FullOpening):
            raise TypeError("responder state requires BHK22FullOpening")
        if len(self.key_share_full_opening.values) != n_r:
            raise ValueError("responder key-share vector must have length n_r")
        object.__setattr__(self, "root_context_digest", root)
        object.__setattr__(self, "phase_qid", qid)
        object.__setattr__(self, "dealer_role_id", dealer)
        object.__setattr__(self, "responder_role_id", responder)
        object.__setattr__(self, "n_r", n_r)

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": RESPONDER_STATE_KIND,
            "version": PRIVATE_PAYLOAD_VERSION,
            "root_context_digest": self.root_context_digest,
            "phase_qid": self.phase_qid,
            "dealer_role_id": self.dealer_role_id,
            "responder_role_id": self.responder_role_id,
            "n_r": self.n_r,
            "key_share_full_opening": self.key_share_full_opening.to_obj(),
        }

    def to_bytes(self) -> bytes:
        return _canonical_private_bytes(self)

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "BHK22ResponderFutureState":
        exact_fields(
            value,
            {
                "kind", "version", "root_context_digest", "phase_qid",
                "dealer_role_id", "responder_role_id", "n_r",
                "key_share_full_opening",
            },
            name="BHK22 responder future state",
        )
        if value["kind"] != RESPONDER_STATE_KIND or value["version"] != 1:
            raise ValueError("unsupported BHK22 responder future state")
        return cls(
            root_context_digest=value["root_context_digest"],
            phase_qid=value["phase_qid"],
            dealer_role_id=value["dealer_role_id"],
            responder_role_id=value["responder_role_id"],
            n_r=value["n_r"],
            key_share_full_opening=BHK22FullOpening.from_obj(
                value["key_share_full_opening"]
            ),
        )

    @classmethod
    def from_bytes(cls, data: bytes | str) -> "BHK22ResponderFutureState":
        return _parse_private_payload(data, cls, name="BHK22 responder future state")


class BHK22DealerRandomSource:
    """Random-source interface used by the local dealer role."""

    def scalar(self, label: str) -> int:
        nonempty_text(label, name="dealer randomness label")
        return field_random()

    def bytes(self, label: str, length: int) -> bytes:
        nonempty_text(label, name="dealer randomness label")
        return secrets.token_bytes(strict_int(length, name="random byte length", minimum=1))


class BHK22DeterministicDealerRandomSource(BHK22DealerRandomSource):
    """Fixture/test-only deterministic source; never selected implicitly."""

    def __init__(self, seed: str) -> None:
        self._seed = nonempty_text(seed, name="deterministic dealer seed").encode("utf-8")

    def _block(self, label: str, counter: int) -> bytes:
        return hashlib.sha256(
            b"ATCaaS/BHK22/dealer-test-randomness/v1\x00"
            + self._seed
            + b"\x00"
            + label.encode("utf-8")
            + counter.to_bytes(4, "big")
        ).digest()

    def scalar(self, label: str) -> int:
        nonempty_text(label, name="dealer randomness label")
        return int.from_bytes(self._block(label, 0), "big") % (FIELD_ORDER - 1) + 1

    def bytes(self, label: str, length: int) -> bytes:
        nonempty_text(label, name="dealer randomness label")
        length = strict_int(length, name="random byte length", minimum=1)
        output = bytearray()
        counter = 0
        while len(output) < length:
            output.extend(self._block(label, counter))
            counter += 1
        return bytes(output[:length])


@dataclass(frozen=True)
class BHK22DealerMetrics:
    sharing_ms: float
    commitment_ms: float
    row_encryption_ms: float
    responder_encryption_ms: float
    figure8_encryption_ms: float
    figure7_proof_ms: float
    figure8_proof_ms: float
    serialization_ms: float
    total_ms: float
    batch_bytes: int
    encrypted_row_bytes: int
    verifier_key_ciphertext_bytes: int
    responder_ciphertext_bytes: int
    figure8_ciphertext_bytes: int
    figure7_proof_bytes: int
    figure8_proof_bytes: int
    cpu_seconds: float = 0.0
    figure7_proof_cpu_seconds: float = 0.0
    figure8_proof_cpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "sharing_ms", "commitment_ms", "row_encryption_ms",
            "responder_encryption_ms", "figure8_encryption_ms",
            "figure7_proof_ms", "figure8_proof_ms", "serialization_ms", "total_ms",
            "cpu_seconds", "figure7_proof_cpu_seconds",
            "figure8_proof_cpu_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a non-negative number")
        for name in (
            "batch_bytes", "encrypted_row_bytes", "verifier_key_ciphertext_bytes",
            "responder_ciphertext_bytes", "figure8_ciphertext_bytes",
            "figure7_proof_bytes", "figure8_proof_bytes",
        ):
            strict_int(getattr(self, name), name=name)

    def to_obj(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class BHK22DealerGenerationResult:
    batch: BHK22DealerBatch
    metrics: BHK22DealerMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.batch, BHK22DealerBatch):
            raise TypeError("dealer generation result requires BHK22DealerBatch")
        if not isinstance(self.metrics, BHK22DealerMetrics):
            raise TypeError("dealer generation result requires BHK22DealerMetrics")


@dataclass(frozen=True)
class BHK22DealerPublicVerification:
    accepted: bool
    checks: Mapping[str, bool]
    verify_ms: float
    verify_cpu_seconds: float = 0.0
    figure7_proof_verify_ms: float = 0.0
    figure7_proof_verify_cpu_seconds: float = 0.0
    figure8_proof_verify_ms: float = 0.0
    figure8_proof_verify_cpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        checks = {str(name): bool(value) for name, value in self.checks.items()}
        if self.accepted != all(checks.values()):
            raise ValueError("dealer verification accepted/check mismatch")
        for value in (
            self.verify_ms,
            self.verify_cpu_seconds,
            self.figure7_proof_verify_ms,
            self.figure7_proof_verify_cpu_seconds,
            self.figure8_proof_verify_ms,
            self.figure8_proof_verify_cpu_seconds,
        ):
            if value < 0:
                raise ValueError("dealer verification time cannot be negative")
        object.__setattr__(self, "checks", MappingProxyType(checks))


def _row_aad(context: BHK22PublicContext, phase_qid: str, dealer: int, verifier: int) -> BHK22AAD:
    return BHK22AAD(
        root_context_digest=context.root_context_digest,
        phase_qid=phase_qid,
        sender_role=ROLE_DEALER,
        sender_role_id=dealer,
        recipient_role=ROLE_VERIFIER,
        recipient_role_id=verifier,
        payload_kind=ROW_PAYLOAD_KIND,
    )


def _verifier_key_aad(
    context: BHK22PublicContext, phase_qid: str, dealer: int, verifier: int
) -> BHK22AAD:
    return BHK22AAD(
        root_context_digest=context.root_context_digest,
        phase_qid=phase_qid,
        sender_role=ROLE_DEALER,
        sender_role_id=dealer,
        recipient_role=ROLE_VERIFIER,
        recipient_role_id=verifier,
        payload_kind=VERIFIER_KEY_PAYLOAD_KIND,
    )


def _responder_aad(
    context: BHK22PublicContext, phase_qid: str, dealer: int, responder: int
) -> BHK22AAD:
    return BHK22AAD(
        root_context_digest=context.root_context_digest,
        phase_qid=phase_qid,
        sender_role=ROLE_DEALER,
        sender_role_id=dealer,
        recipient_role=ROLE_RESPONDER,
        recipient_role_id=responder,
        payload_kind=RESPONDER_STATE_PAYLOAD_KIND,
    )


def _aead_digest(ciphertext: BHK22AEADCiphertext) -> str:
    return hashlib.sha256(canonical_json(ciphertext.to_obj())).hexdigest()


def _sample_poly(
    constant: int,
    degree: int,
    random_source: BHK22DealerRandomSource,
    label: str,
) -> tuple[int, ...]:
    return (canonical_scalar(constant, name="polynomial constant"),) + tuple(
        random_source.scalar(f"{label}:coefficient:{power}")
        for power in range(1, degree + 1)
    )


class BHK22DealerRole:
    """Generate exactly one Figure 6/7/8 dealer batch from local state."""

    def __init__(
        self,
        *,
        context: BHK22PublicContext,
        state: BHK22DealerState,
        phase_qid: str,
        random_source: BHK22DealerRandomSource | None = None,
        application_figure7_share_base_hex: str | None = None,
        application_figure7_share_point_hex: str | None = None,
    ) -> None:
        if not isinstance(context, BHK22PublicContext):
            raise TypeError("dealer role requires BHK22PublicContext")
        if not isinstance(state, BHK22DealerState):
            raise TypeError("dealer role requires BHK22DealerState")
        qid = nonempty_text(phase_qid, name="dealer phase qid")
        n_r = context.parameters.n_r
        if state.role_id >= n_r:
            raise ValueError("dealer state role id is outside context")
        expected_node = context.committees[ROLE_DEALER].role_to_node[state.role_id]
        if expected_node >= context.parameters.M:
            raise ValueError("dealer role map points outside M")
        if random_source is not None and not isinstance(
            random_source, BHK22DealerRandomSource
        ):
            raise TypeError("random_source must be BHK22DealerRandomSource")
        application_binding = (
            application_figure7_share_base_hex,
            application_figure7_share_point_hex,
        )
        if any(item is not None for item in application_binding):
            if not all(item is not None for item in application_binding):
                raise ValueError("application Figure-7 binding requires base and point")
            application_binding = (
                canonical_point(
                    application_figure7_share_base_hex,
                    name="application Figure-7 share base",
                ),
                canonical_point(
                    application_figure7_share_point_hex,
                    name="application Figure-7 share point",
                ),
            )
        else:
            application_binding = None
        if context.encrypted_source is not None and application_binding is None:
            raise ValueError("encrypted source requires an application Figure-7 binding")
        self.context = context
        self.state = state
        self.phase_qid = qid
        self.random_source = random_source or BHK22DealerRandomSource()
        self.application_figure7_binding = application_binding

    def generate_batch(self) -> BHK22DealerGenerationResult:
        total_started = perf_counter()
        cpu_started = process_time()
        context = self.context
        n_r = context.parameters.n_r
        degree = context.parameters.t_r
        dealer = self.state.role_id
        source_share = self.state._source_share_for_protocol()
        if self.application_figure7_binding is None:
            figure7_share_base = context.crs.generators[dealer + 1]
            figure7_share_point = context.source_public_values[dealer]
            expected_source = (
                G1Point.from_hex(figure7_share_base) * source_share
            ).to_hex()
            if expected_source != figure7_share_point:
                raise ValueError("dealer local source share does not match public context")
        else:
            figure7_share_base, figure7_share_point = (
                self.application_figure7_binding
            )
            if (
                G1Point.from_hex(figure7_share_base) * source_share
            ).to_hex() != figure7_share_point:
                raise ValueError("dealer local share does not match application Figure-7 vk")

        sharing_started = perf_counter()
        first_poly = _sample_poly(
            source_share, degree, self.random_source, f"dealer:{dealer}:first-level"
        )
        first_level = tuple(
            eval_poly(first_poly, point) for point in range(1, n_r + 1)
        )
        dummy = self.random_source.scalar(f"dealer:{dealer}:dummy:0")
        share_vector = (dummy,) + first_level
        second_polys = tuple(
            _sample_poly(
                share_vector[index],
                degree,
                self.random_source,
                f"dealer:{dealer}:second-level:{index}",
            )
            for index in range(n_r + 1)
        )
        rows = tuple(
            tuple(eval_poly(second_polys[index], verifier + 1) for index in range(n_r + 1))
            for verifier in range(n_r)
        )
        epsilon_polys = tuple(
            _sample_poly(
                self.random_source.scalar(f"dealer:{dealer}:epsilon:{verifier}"),
                degree,
                self.random_source,
                f"dealer:{dealer}:epsilon-sharing:{verifier}",
            )
            for verifier in range(n_r)
        )
        epsilons = tuple(poly[0] for poly in epsilon_polys)
        epsilon_shares = tuple(
            tuple(eval_poly(poly, responder + 1) for responder in range(n_r))
            for poly in epsilon_polys
        )
        sharing_ms = (perf_counter() - sharing_started) * 1000.0

        commitment_started = perf_counter()
        column_commitments = tuple(
            g1_linear_combination(context.crs.generators, row).to_hex()
            for row in rows
        )
        c_i0 = g1_linear_combination(context.crs.generators, share_vector).to_hex()
        commitment_ms = (perf_counter() - commitment_started) * 1000.0

        row_started = perf_counter()
        verifier_catalog = context.recipient_key_catalogs[ROLE_VERIFIER]
        verifier_columns = []
        for verifier in range(n_r):
            row_payload = BHK22VerifierRowPayload(
                root_context_digest=context.root_context_digest,
                phase_qid=self.phase_qid,
                dealer_role_id=dealer,
                verifier_role_id=verifier,
                n_r=n_r,
                row=rows[verifier],
            )
            row_aad = _row_aad(context, self.phase_qid, dealer, verifier)
            row_key = derive_aead_key(
                epsilons[verifier].to_bytes(32, "big"),
                root_context_digest=context.root_context_digest,
                purpose=ROW_KDF_PURPOSE,
                binding=row_aad.to_bytes(),
            )
            encrypted_row = aead_encrypt(
                row_key,
                row_payload.to_bytes(),
                aad=row_aad,
                nonce=self.random_source.bytes(
                    f"dealer:{dealer}:row-nonce:{verifier}", 12
                ),
            )
            key_payload = BHK22VerifierKeyPayload(
                root_context_digest=context.root_context_digest,
                phase_qid=self.phase_qid,
                dealer_role_id=dealer,
                verifier_role_id=verifier,
                n_r=n_r,
                epsilon=epsilons[verifier],
                encrypted_row_digest=_aead_digest(encrypted_row),
            )
            encrypted_key = recipient_encrypt(
                key_base_hex=verifier_catalog.key_base_hex,
                recipient_public_key_hex=verifier_catalog.public_keys[verifier],
                plaintext=key_payload.to_bytes(),
                aad=_verifier_key_aad(context, self.phase_qid, dealer, verifier),
                purpose="verifier-key",
                ephemeral_secret=self.random_source.scalar(
                    f"dealer:{dealer}:verifier-key-ephemeral:{verifier}"
                ),
                nonce=self.random_source.bytes(
                    f"dealer:{dealer}:verifier-key-nonce:{verifier}", 12
                ),
            )
            verifier_columns.append(
                BHK22DealerVerifierColumn(
                    verifier_role_id=verifier,
                    column_commitment=column_commitments[verifier],
                    encrypted_row=encrypted_row,
                    encrypted_row_key=encrypted_key,
                )
            )
        row_encryption_ms = (perf_counter() - row_started) * 1000.0

        responder_started = perf_counter()
        responder_catalog = context.recipient_key_catalogs[ROLE_RESPONDER]
        responder_packages = []
        for responder in range(n_r):
            key_share_vector = tuple(
                epsilon_shares[verifier][responder] for verifier in range(n_r)
            )
            gamma, full_opening = commit_vector(
                key_share_vector,
                salts=tuple(
                    self.random_source.bytes(
                        f"dealer:{dealer}:gamma-salt:{responder}:{verifier}", 32
                    )
                    for verifier in range(n_r)
                ),
            )
            future_state = BHK22ResponderFutureState(
                root_context_digest=context.root_context_digest,
                phase_qid=self.phase_qid,
                dealer_role_id=dealer,
                responder_role_id=responder,
                n_r=n_r,
                key_share_full_opening=full_opening,
            )
            encrypted_state = recipient_encrypt(
                key_base_hex=responder_catalog.key_base_hex,
                recipient_public_key_hex=responder_catalog.public_keys[responder],
                plaintext=future_state.to_bytes(),
                aad=_responder_aad(context, self.phase_qid, dealer, responder),
                purpose="responder-future-state",
                ephemeral_secret=self.random_source.scalar(
                    f"dealer:{dealer}:responder-ephemeral:{responder}"
                ),
                nonce=self.random_source.bytes(
                    f"dealer:{dealer}:responder-nonce:{responder}", 12
                ),
            )
            responder_packages.append(
                BHK22DealerResponderPackage(
                    responder_role_id=responder,
                    key_share_commitment=gamma,
                    encrypted_future_state=encrypted_state,
                )
            )
        responder_encryption_ms = (perf_counter() - responder_started) * 1000.0

        figure8_started = perf_counter()
        figure8_ciphertexts = []
        figure8_randomness = []
        for index, share in enumerate(share_vector):
            randomness = self.random_source.scalar(
                f"dealer:{dealer}:figure8-randomness:{index}"
            )
            ciphertext, normalized_randomness = encrypt_public_value(
                generator_hex=context.crs.generators[index],
                public_key_hex=context.crs.public_value_keys[index],
                plaintext_hex=(G1Point.from_hex(context.crs.generators[index]) * share).to_hex(),
                randomness=randomness,
            )
            figure8_ciphertexts.append(ciphertext)
            figure8_randomness.append(normalized_randomness)
        c_prime_i0 = g1_linear_combination(
            [item.Y for item in figure8_ciphertexts], [1] * (n_r + 1)
        ).to_hex()
        delta_i = (
            G1Point.from_hex(c_prime_i0) - G1Point.from_hex(c_i0)
        ).to_hex()
        figure8_encryption_ms = (perf_counter() - figure8_started) * 1000.0

        proof_context = BHK22ProofContext(
            root_context_digest=context.root_context_digest,
            phase_qid=self.phase_qid,
            role_type=ROLE_DEALER,
            role_id=dealer,
        )
        figure8_proof_started = perf_counter()
        figure8_proof_cpu_started = process_time()
        figure8_proof = prove_figure8_randomness(
            context=proof_context,
            delta_hex=delta_i,
            ciphertexts=figure8_ciphertexts,
            h_bases=context.crs.public_value_keys,
            g_bases=context.crs.generators,
            randomness=figure8_randomness,
        )
        figure8_proof_ms = (perf_counter() - figure8_proof_started) * 1000.0
        figure8_proof_cpu_seconds = process_time() - figure8_proof_cpu_started

        figure7_proof_started = perf_counter()
        figure7_proof_cpu_started = process_time()
        figure7_proof = prove_figure7(
            context=proof_context,
            anchor_base_hex=context.crs.generators[0],
            share_base_hex=figure7_share_base,
            eval_bases=context.crs.generators[1:],
            share_point_hex=figure7_share_point,
            aggregate_commitment_hex=c_i0,
            polynomial_coeffs=first_poly,
            anchor_scalar=dummy,
            eval_points=tuple(range(1, n_r + 1)),
        )
        figure7_proof_ms = (perf_counter() - figure7_proof_started) * 1000.0
        figure7_proof_cpu_seconds = process_time() - figure7_proof_cpu_started

        serialization_started = perf_counter()
        batch = BHK22DealerBatch(
            root_context_digest=context.root_context_digest,
            phase_qid=self.phase_qid,
            n_r=n_r,
            role_id=dealer,
            network_sender_id=context.committees[ROLE_DEALER].role_to_node[dealer],
            verifier_columns=verifier_columns,
            responder_packages=responder_packages,
            figure8_ciphertexts=figure8_ciphertexts,
            c_i0=c_i0,
            c_prime_i0=c_prime_i0,
            delta_i=delta_i,
            figure7_proof=figure7_proof,
            figure8_proof=figure8_proof,
        )
        batch_bytes = len(batch.to_bytes())
        serialization_ms = (perf_counter() - serialization_started) * 1000.0

        consumed = self.state.consume_once()
        if consumed.source_share != source_share or consumed.role_id != dealer:
            raise RuntimeError("dealer state changed during batch generation")

        metrics = BHK22DealerMetrics(
            sharing_ms=sharing_ms,
            commitment_ms=commitment_ms,
            row_encryption_ms=row_encryption_ms,
            responder_encryption_ms=responder_encryption_ms,
            figure8_encryption_ms=figure8_encryption_ms,
            figure7_proof_ms=figure7_proof_ms,
            figure8_proof_ms=figure8_proof_ms,
            serialization_ms=serialization_ms,
            total_ms=(perf_counter() - total_started) * 1000.0,
            batch_bytes=batch_bytes,
            encrypted_row_bytes=sum(
                primitive_size_bytes(item.encrypted_row) for item in verifier_columns
            ),
            verifier_key_ciphertext_bytes=sum(
                primitive_size_bytes(item.encrypted_row_key) for item in verifier_columns
            ),
            responder_ciphertext_bytes=sum(
                primitive_size_bytes(item.encrypted_future_state)
                for item in responder_packages
            ),
            figure8_ciphertext_bytes=sum(
                primitive_size_bytes(item) for item in figure8_ciphertexts
            ),
            figure7_proof_bytes=proof_size_bytes(figure7_proof),
            figure8_proof_bytes=proof_size_bytes(figure8_proof),
            cpu_seconds=process_time() - cpu_started,
            figure7_proof_cpu_seconds=figure7_proof_cpu_seconds,
            figure8_proof_cpu_seconds=figure8_proof_cpu_seconds,
        )
        return BHK22DealerGenerationResult(batch=batch, metrics=metrics)


def verify_dealer_batch_public(
    context: BHK22PublicContext,
    batch: BHK22DealerBatch,
    *,
    expected_phase_qid: str | None = None,
) -> BHK22DealerPublicVerification:
    """Independently verify all public Figure 6/7/8 dealer relations."""

    started = perf_counter()
    cpu_started = process_time()
    figure7_proof_verify_ms = 0.0
    figure7_proof_verify_cpu_seconds = 0.0
    figure8_proof_verify_ms = 0.0
    figure8_proof_verify_cpu_seconds = 0.0
    checks = {
        "wire_context": False,
        "column_degree": False,
        "column_zero": False,
        "shifted_column": False,
        "delta": False,
        "figure7_proof": False,
        "figure8_proof": False,
    }
    try:
        if not isinstance(context, BHK22PublicContext) or not isinstance(
            batch, BHK22DealerBatch
        ):
            raise TypeError("wrong public dealer verifier input")
        if context.encrypted_source is not None:
            raise ValueError("encrypted source requires the operation dealer verifier")
        dealer = batch.role_id
        checks["wire_context"] = (
            batch.root_context_digest == context.root_context_digest
            and batch.n_r == context.parameters.n_r
            and dealer < batch.n_r
            and batch.network_sender_id
            == context.committees[ROLE_DEALER].role_to_node[dealer]
            and (expected_phase_qid is None or batch.phase_qid == expected_phase_qid)
        )
        commitments = tuple(item.column_commitment for item in batch.verifier_columns)
        evaluation_points = tuple(range(1, batch.n_r + 1))
        checks["column_degree"] = pedersen_deg_check(
            commitments,
            context.parameters.t_r,
            eval_points=evaluation_points,
        )
        basis_points = evaluation_points[: context.parameters.t_r + 1]
        coefficients = tuple(
            lagrange_coeff(basis_points, point, at=0) for point in basis_points
        )
        interpolated_zero = g1_linear_combination(
            commitments[: context.parameters.t_r + 1], coefficients
        ).to_hex()
        checks["column_zero"] = interpolated_zero == batch.c_i0
        summed_y = g1_linear_combination(
            [item.Y for item in batch.figure8_ciphertexts], [1] * (batch.n_r + 1)
        ).to_hex()
        checks["shifted_column"] = summed_y == batch.c_prime_i0
        checks["delta"] = (
            G1Point.from_hex(batch.c_prime_i0) - G1Point.from_hex(batch.c_i0)
        ).to_hex() == batch.delta_i
        proof_context = BHK22ProofContext(
            root_context_digest=batch.root_context_digest,
            phase_qid=batch.phase_qid,
            role_type=ROLE_DEALER,
            role_id=dealer,
        )
        proof_started = perf_counter()
        proof_cpu_started = process_time()
        checks["figure7_proof"] = verify_figure7(
            context=proof_context,
            anchor_base_hex=context.crs.generators[0],
            share_base_hex=context.crs.generators[dealer + 1],
            eval_bases=context.crs.generators[1:],
            share_point_hex=context.source_public_values[dealer],
            aggregate_commitment_hex=batch.c_i0,
            degree=context.parameters.t_r,
            eval_points=evaluation_points,
            proof=batch.figure7_proof,
        )
        figure7_proof_verify_ms = (perf_counter() - proof_started) * 1000.0
        figure7_proof_verify_cpu_seconds = process_time() - proof_cpu_started
        proof_started = perf_counter()
        proof_cpu_started = process_time()
        checks["figure8_proof"] = verify_figure8_randomness(
            context=proof_context,
            delta_hex=batch.delta_i,
            ciphertexts=batch.figure8_ciphertexts,
            h_bases=context.crs.public_value_keys,
            g_bases=context.crs.generators,
            proof=batch.figure8_proof,
        )
        figure8_proof_verify_ms = (perf_counter() - proof_started) * 1000.0
        figure8_proof_verify_cpu_seconds = process_time() - proof_cpu_started
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass
    return BHK22DealerPublicVerification(
        accepted=all(checks.values()),
        checks=checks,
        verify_ms=(perf_counter() - started) * 1000.0,
        verify_cpu_seconds=process_time() - cpu_started,
        figure7_proof_verify_ms=figure7_proof_verify_ms,
        figure7_proof_verify_cpu_seconds=figure7_proof_verify_cpu_seconds,
        figure8_proof_verify_ms=figure8_proof_verify_ms,
        figure8_proof_verify_cpu_seconds=figure8_proof_verify_cpu_seconds,
    )


def decrypt_dealer_verifier_payload(
    context: BHK22PublicContext,
    batch: BHK22DealerBatch,
    *,
    verifier_role_id: int,
    verifier_recipient_secret_key: int,
) -> BHK22VerifierRowPayload:
    """Decrypt and bind-check one dealer row without producing an accusation."""

    verifier = strict_int(verifier_role_id, name="verifier_role_id")
    if batch.root_context_digest != context.root_context_digest or batch.n_r != context.parameters.n_r:
        raise ValueError("dealer batch/context mismatch")
    if verifier >= batch.n_r:
        raise ValueError("verifier role id is outside n_r")
    column = batch.verifier_columns[verifier]
    if column.verifier_role_id != verifier:
        raise ValueError("dealer verifier column index mismatch")
    catalog = context.recipient_key_catalogs[ROLE_VERIFIER]
    key_plaintext = recipient_decrypt(
        key_base_hex=catalog.key_base_hex,
        recipient_secret_key=verifier_recipient_secret_key,
        ciphertext=column.encrypted_row_key,
        aad=_verifier_key_aad(context, batch.phase_qid, batch.role_id, verifier),
        purpose="verifier-key",
    )
    key_payload = BHK22VerifierKeyPayload.from_bytes(key_plaintext)
    expected_binding = (
        context.root_context_digest,
        batch.phase_qid,
        batch.role_id,
        verifier,
        batch.n_r,
    )
    actual_binding = (
        key_payload.root_context_digest,
        key_payload.phase_qid,
        key_payload.dealer_role_id,
        key_payload.verifier_role_id,
        key_payload.n_r,
    )
    if actual_binding != expected_binding:
        raise ValueError("verifier-key private payload binding mismatch")
    if key_payload.encrypted_row_digest != _aead_digest(column.encrypted_row):
        raise ValueError("verifier key is not bound to encrypted row")
    return decrypt_dealer_row_with_recovered_epsilon(
        context,
        batch,
        verifier_role_id=verifier,
        recovered_epsilon=key_payload.epsilon,
    )


def decrypt_dealer_row_with_recovered_epsilon(
    context: BHK22PublicContext,
    batch: BHK22DealerBatch,
    *,
    verifier_role_id: int,
    recovered_epsilon: int,
) -> BHK22VerifierRowPayload:
    """Authenticate one E_i,j using epsilon reconstructed by responders."""

    verifier = strict_int(verifier_role_id, name="verifier_role_id")
    epsilon = canonical_scalar(
        recovered_epsilon, name="recovered epsilon", nonzero=True
    )
    if batch.root_context_digest != context.root_context_digest or batch.n_r != context.parameters.n_r:
        raise ValueError("dealer batch/context mismatch")
    if verifier >= batch.n_r:
        raise ValueError("verifier role id is outside n_r")
    column = batch.verifier_columns[verifier]
    if column.verifier_role_id != verifier:
        raise ValueError("dealer verifier column index mismatch")
    row_aad = _row_aad(context, batch.phase_qid, batch.role_id, verifier)
    row_key = derive_aead_key(
        epsilon.to_bytes(32, "big"),
        root_context_digest=context.root_context_digest,
        purpose=ROW_KDF_PURPOSE,
        binding=row_aad.to_bytes(),
    )
    row_payload = BHK22VerifierRowPayload.from_bytes(
        aead_decrypt(row_key, column.encrypted_row, aad=row_aad)
    )
    expected_binding = (
        context.root_context_digest,
        batch.phase_qid,
        batch.role_id,
        verifier,
        batch.n_r,
    )
    row_binding = (
        row_payload.root_context_digest,
        row_payload.phase_qid,
        row_payload.dealer_role_id,
        row_payload.verifier_role_id,
        row_payload.n_r,
    )
    if row_binding != expected_binding:
        raise ValueError("verifier-row private payload binding mismatch")
    expected_commitment = g1_linear_combination(
        context.crs.generators, row_payload.row
    ).to_hex()
    if expected_commitment != column.column_commitment:
        raise ValueError("decrypted verifier row does not match C_i,j")
    return row_payload


def decrypt_dealer_responder_state(
    context: BHK22PublicContext,
    batch: BHK22DealerBatch,
    *,
    responder_role_id: int,
    responder_recipient_secret_key: int,
) -> BHK22ResponderFutureState:
    """Decrypt and commitment-check responder future state without responding."""

    responder = strict_int(responder_role_id, name="responder_role_id")
    if batch.root_context_digest != context.root_context_digest or batch.n_r != context.parameters.n_r:
        raise ValueError("dealer batch/context mismatch")
    if responder >= batch.n_r:
        raise ValueError("responder role id is outside n_r")
    package = batch.responder_packages[responder]
    if package.responder_role_id != responder:
        raise ValueError("dealer responder package index mismatch")
    catalog = context.recipient_key_catalogs[ROLE_RESPONDER]
    plaintext = recipient_decrypt(
        key_base_hex=catalog.key_base_hex,
        recipient_secret_key=responder_recipient_secret_key,
        ciphertext=package.encrypted_future_state,
        aad=_responder_aad(context, batch.phase_qid, batch.role_id, responder),
        purpose="responder-future-state",
    )
    state = BHK22ResponderFutureState.from_bytes(plaintext)
    expected_binding = (
        context.root_context_digest,
        batch.phase_qid,
        batch.role_id,
        responder,
        batch.n_r,
    )
    actual_binding = (
        state.root_context_digest,
        state.phase_qid,
        state.dealer_role_id,
        state.responder_role_id,
        state.n_r,
    )
    if actual_binding != expected_binding:
        raise ValueError("responder future-state binding mismatch")
    if not verify_full_opening(
        package.key_share_commitment, state.key_share_full_opening
    ):
        raise ValueError("responder future state does not open gamma_i,k")
    return state


__all__ = [
    "BHK22DealerGenerationResult",
    "BHK22DealerMetrics",
    "BHK22DealerPublicVerification",
    "BHK22DealerRandomSource",
    "BHK22DealerRole",
    "BHK22DeterministicDealerRandomSource",
    "BHK22ResponderFutureState",
    "BHK22VerifierKeyPayload",
    "BHK22VerifierRowPayload",
    "decrypt_dealer_responder_state",
    "decrypt_dealer_row_with_recovered_epsilon",
    "decrypt_dealer_verifier_payload",
    "verify_dealer_batch_public",
]
