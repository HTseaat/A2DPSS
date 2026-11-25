"""Frozen Step-0 contract for the formal BHK+22 + DAG-TOB baseline.

This module deliberately contains no protocol engine, wire implementation, or
cryptographic prover.  It is the executable boundary that later BHK22 modules
must import instead of the legacy ``benhamouda_interactive_handover``
prototype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from beaver.atcaas_params import ATCaaSProtocolParameters


BHK22_PROTOCOL_KIND = "bhk22-basic-fig6-7-8-dual-completion"
BHK22_PROTOCOL_VERSION = 2
BHK22_QID_KIND = "bhk22-dag-handover"
BHK22_CONTRACT_SCHEMA = "atcaas-bhk22-step0-contract/v2"

PATH_OPTIMISTIC = "optimistic"
PATH_ONE_ACCUSATION = "one-accusation"
PATH_MODES = (PATH_OPTIMISTIC, PATH_ONE_ACCUSATION)

COMPLETION_ENCRYPTED_CARRY_FORWARD = "encrypted-carry-forward"
COMPLETION_PLAINTEXT_FINALIZE = "plaintext-finalize"
COMPLETION_MODES = (
    COMPLETION_ENCRYPTED_CARRY_FORWARD,
    COMPLETION_PLAINTEXT_FINALIZE,
)
DEFAULT_COMPLETION_MODE = COMPLETION_ENCRYPTED_CARRY_FORWARD

PHASE_DEAL = "deal"
PHASE_VERIFY = "verify"
PHASE_RESPONSE = "response"
PHASE_FINALIZE = "finalize"
PHASES = (PHASE_DEAL, PHASE_VERIFY, PHASE_RESPONSE, PHASE_FINALIZE)

ROLE_DEALER = "dealer"
ROLE_VERIFIER = "verifier"
ROLE_RESPONDER = "responder"
ROLE_SHAREHOLDER = "shareholder"
ROLE_TYPES = (ROLE_DEALER, ROLE_VERIFIER, ROLE_RESPONDER, ROLE_SHAREHOLDER)

PROTOCOL_FIGURES = (6, 7, 8)
STRICT_CHAINING_SUPPORTED = True
RESPONSIVE_ROLE_PROFILE = True

PATH_PHASES = {
    PATH_OPTIMISTIC: (PHASE_DEAL, PHASE_VERIFY, PHASE_FINALIZE),
    PATH_ONE_ACCUSATION: (
        PHASE_DEAL,
        PHASE_VERIFY,
        PHASE_RESPONSE,
        PHASE_FINALIZE,
    ),
}
PATH_BLOCK_COUNTS = {
    path_mode: len(phases) for path_mode, phases in PATH_PHASES.items()
}
CARRY_FORWARD_PATH_PHASES = {
    PATH_OPTIMISTIC: (PHASE_DEAL, PHASE_VERIFY),
    PATH_ONE_ACCUSATION: (
        PHASE_DEAL,
        PHASE_VERIFY,
        PHASE_RESPONSE,
    ),
}

# This is the first correctness/smoke profile, not a performance-scan default.
FIRST_EXPERIMENT_PARAMETERS = ATCaaSProtocolParameters(
    M=5,
    n_r=5,
    t_r=2,
    n_a=5,
    t_a=1,
    m_a=5,
    quorum_h=4,
    coin_c=5,
    coin_t=1,
    coin_k=4,
)


@dataclass(frozen=True)
class BHK22AlgorithmSuite:
    """Algorithm identifiers frozen before wire and crypto implementation."""

    canonical_encoding: str = "canonical-json-utf8-v1"
    group: str = "bls12-381-g1-amcl-canonical-v1"
    scalar_field: str = "bls12-381-fr-canonical-decimal-v1"
    generator_derivation: str = "hash-to-g1-domain-separated-v1"
    vector_commitment: str = "salted-merkle-sha256-coordinate-vector-v1"
    kdf: str = "hkdf-sha256-v1"
    aead: str = "aes-256-gcm-96bit-nonce-128bit-tag-v1"
    recipient_pke: str = "g1-dh-hkdf-sha256-aes-256-gcm-v1"
    public_value_encryption: str = "elgamal-bls12-381-g1-v1"
    figure7_proof: str = "bhk22-figure7-fs-sha256-g1-v1"
    figure8_proof: str = "bhk22-figure8-randomness-fs-sha256-g1-v1"
    verifier_forwarding_proof: str = "bhk22-figure6-forwarding-fs-sha256-g1-v1"
    correct_decryption_proof: str = "bhk22-elgamal-correct-decryption-fs-sha256-g1-v1"

    def to_obj(self) -> dict[str, str]:
        return {
            name: str(getattr(self, name))
            for name in self.__dataclass_fields__
        }


ALGORITHM_SUITE = BHK22AlgorithmSuite()


@dataclass(frozen=True)
class BHK22IndexContract:
    """Exact zero-based wire IDs and one-based paper evaluation points."""

    n_r: int

    def __post_init__(self) -> None:
        if isinstance(self.n_r, bool) or not isinstance(self.n_r, int):
            raise TypeError("n_r must be an integer")
        if self.n_r < 1:
            raise ValueError("n_r must be positive")

    @property
    def role_ids(self) -> tuple[int, ...]:
        return tuple(range(self.n_r))

    @property
    def evaluation_points(self) -> tuple[int, ...]:
        return tuple(range(1, self.n_r + 1))

    @property
    def extended_share_indices(self) -> tuple[int, ...]:
        """Paper's ``[[n]] = {0,...,n}``, including the dummy index 0."""

        return tuple(range(self.n_r + 1))

    @property
    def vector_lengths(self) -> dict[str, int]:
        return {
            "source_public_values": self.n_r,
            "dealer_column_checks": self.n_r,
            "dealer_verifier_rows": self.n_r + 1,
            "responder_key_share_vectors": self.n_r,
            "figure8_ciphertexts": self.n_r + 1,
            "verifier_public_h": self.n_r,
            "verifier_public_R": self.n_r + 1,
            "successor_private_shares": self.n_r,
            "successor_public_values": self.n_r,
        }

    def to_obj(self) -> dict[str, Any]:
        return {
            "wire_role_ids": list(self.role_ids),
            "paper_evaluation_points": list(self.evaluation_points),
            "extended_share_indices": list(self.extended_share_indices),
            "vector_lengths": self.vector_lengths,
        }


PUBLIC_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "rsk",
        "k_l",
        "share",
        "share_blind",
        "old_share",
        "new_share",
        "secret",
        "secret_key",
        "symmetric_key",
        "decrypted_row",
        "polynomial_coefficients",
        "opening_randomness",
        "proof_witness",
    }
)


def path_phases(
    path_mode: str,
    completion_mode: str = DEFAULT_COMPLETION_MODE,
) -> tuple[str, ...]:
    try:
        normalized_path = str(path_mode)
        normalized_completion = str(completion_mode)
        if normalized_completion == COMPLETION_PLAINTEXT_FINALIZE:
            return PATH_PHASES[normalized_path]
        if normalized_completion == COMPLETION_ENCRYPTED_CARRY_FORWARD:
            return CARRY_FORWARD_PATH_PHASES[normalized_path]
        raise KeyError(normalized_completion)
    except KeyError as exc:
        raise ValueError(
            "unsupported BHK22 path/completion mode: "
            f"{path_mode!r}/{completion_mode!r}"
        ) from exc


def expected_application_blocks(
    path_mode: str,
    completion_mode: str = DEFAULT_COMPLETION_MODE,
) -> int:
    return len(path_phases(path_mode, completion_mode))


def validate_bhk22_parameters(
    parameters: ATCaaSProtocolParameters,
    *,
    require_first_experiment_profile: bool = False,
) -> ATCaaSProtocolParameters:
    """Fail before networking unless resharing/agreement/coin parameters fit."""

    if not isinstance(parameters, ATCaaSProtocolParameters):
        raise TypeError("parameters must use ATCaaSProtocolParameters")
    if parameters.n_r < 2 * parameters.t_r + 1:
        raise ValueError("BHK22 requires n_r >= 2*t_r + 1")
    if require_first_experiment_profile and parameters != FIRST_EXPERIMENT_PARAMETERS:
        raise ValueError("parameters do not match the frozen first BHK22 profile")
    return parameters


def _walk_public_value(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"public object key at {path} must be a string")
            normalized = key.strip().lower()
            if normalized in PUBLIC_FORBIDDEN_FIELD_NAMES:
                raise ValueError(f"private field {key!r} is forbidden at {path}")
            _walk_public_value(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_public_value(item, path=f"{path}[{index}]")


def validate_public_object(value: Mapping[str, Any]) -> None:
    """Reject explicitly private fields at every depth of a public object."""

    if not isinstance(value, Mapping):
        raise TypeError("public object must be a mapping")
    _walk_public_value(value, path="$public")


CRS_METADATA_KIND = "bhk22-crs-construction-metadata"
CRS_METADATA_VERSION = 1
_CRS_FIELDS = {
    "kind",
    "version",
    "n_r",
    "generator_count",
    "shareholder_key_count",
    "generator_derivation",
    "h0_derivation",
    "h0_secret_present",
    "known_scalar_relations",
}


def validate_crs_construction_metadata(
    value: Mapping[str, Any],
    *,
    n_r: int,
) -> None:
    """Validate Step-0 CRS provenance metadata; Step 1 validates real points."""

    if not isinstance(value, Mapping) or set(value) != _CRS_FIELDS:
        raise ValueError("CRS construction metadata fields are not canonical")
    if value["kind"] != CRS_METADATA_KIND or value["version"] != CRS_METADATA_VERSION:
        raise ValueError("unsupported BHK22 CRS construction metadata")
    if value["n_r"] != n_r:
        raise ValueError("CRS n_r does not match the protocol parameters")
    if value["generator_count"] != n_r + 1:
        raise ValueError("CRS must contain G_0 through G_n")
    if value["shareholder_key_count"] != n_r:
        raise ValueError("CRS must contain exactly n shareholder public keys")
    if value["generator_derivation"] != ALGORITHM_SUITE.generator_derivation:
        raise ValueError("BHK22 bases require domain-separated hash-to-G1")
    if value["h0_derivation"] != "public-random-hash-to-g1-no-witness-v1":
        raise ValueError("H_0 must be a public random point with no witness")
    if value["h0_secret_present"] is not False:
        raise ValueError("H_0 secret material is forbidden")
    relations = value["known_scalar_relations"]
    if not isinstance(relations, list) or relations:
        raise ValueError("known scalar relations between CRS bases are forbidden")


def make_crs_contract_metadata(n_r: int) -> dict[str, Any]:
    """Return the only Step-0 metadata shape accepted for a future CRS."""

    index = BHK22IndexContract(n_r)
    return {
        "kind": CRS_METADATA_KIND,
        "version": CRS_METADATA_VERSION,
        "n_r": index.n_r,
        "generator_count": index.n_r + 1,
        "shareholder_key_count": index.n_r,
        "generator_derivation": ALGORITHM_SUITE.generator_derivation,
        "h0_derivation": "public-random-hash-to-g1-no-witness-v1",
        "h0_secret_present": False,
        "known_scalar_relations": [],
    }


__all__ = [
    "ALGORITHM_SUITE",
    "BHK22AlgorithmSuite",
    "BHK22IndexContract",
    "BHK22_CONTRACT_SCHEMA",
    "BHK22_PROTOCOL_KIND",
    "BHK22_PROTOCOL_VERSION",
    "BHK22_QID_KIND",
    "CARRY_FORWARD_PATH_PHASES",
    "COMPLETION_ENCRYPTED_CARRY_FORWARD",
    "COMPLETION_MODES",
    "COMPLETION_PLAINTEXT_FINALIZE",
    "CRS_METADATA_KIND",
    "CRS_METADATA_VERSION",
    "DEFAULT_COMPLETION_MODE",
    "FIRST_EXPERIMENT_PARAMETERS",
    "PATH_BLOCK_COUNTS",
    "PATH_MODES",
    "PATH_ONE_ACCUSATION",
    "PATH_OPTIMISTIC",
    "PATH_PHASES",
    "PHASES",
    "PROTOCOL_FIGURES",
    "PUBLIC_FORBIDDEN_FIELD_NAMES",
    "RESPONSIVE_ROLE_PROFILE",
    "ROLE_TYPES",
    "STRICT_CHAINING_SUPPORTED",
    "expected_application_blocks",
    "make_crs_contract_metadata",
    "path_phases",
    "validate_bhk22_parameters",
    "validate_crs_construction_metadata",
    "validate_public_object",
]
