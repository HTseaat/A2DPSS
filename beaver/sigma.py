"""Small Fiat-Shamir/Sigma helpers for PVSS/DPSS proofs."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

from beaver.g1_backend import (
    FIELD_ORDER,
    G1Point,
    field_add,
    field_mul,
    field_normalize,
    field_random,
    g1_linear_combination,
    g1_verify_linear_relations,
)


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def hash_to_scalar(label: str, value, *, nonzero: bool = False) -> int:
    digest = hashlib.sha256(label.encode("utf-8") + b":" + canonical_json(value)).digest()
    scalar = int.from_bytes(digest, byteorder="big") % FIELD_ORDER
    if nonzero and scalar == 0:
        return 1
    return scalar


def _normalize_point(value: str | G1Point) -> str:
    return value.to_hex() if isinstance(value, G1Point) else G1Point.from_hex(value).to_hex()


def _normalize_relations(relations: Sequence[dict]) -> list[dict]:
    normalized = []
    for relation in relations:
        terms = []
        for term in relation["terms"]:
            terms.append(
                {
                    "secret": str(term["secret"]),
                    "base": _normalize_point(term["base"]),
                    "coeff": field_normalize(term.get("coeff", 1)),
                }
            )
        normalized.append(
            {
                "name": str(relation.get("name", len(normalized))),
                "target": _normalize_point(relation["target"]),
                "terms": terms,
            }
        )
    return normalized


def _relation_commitment(relation: dict, values: Mapping[str, int]) -> G1Point:
    points = []
    scalars = []
    for term in relation["terms"]:
        points.append(term["base"])
        scalars.append(field_mul(values[term["secret"]], term["coeff"]))
    return g1_linear_combination(points, scalars)


def _challenge(statement: dict, relations: Sequence[dict], announcements: Sequence[str]) -> int:
    return hash_to_scalar(
        "sigma-linear-representation",
        {
            "statement": statement,
            "relations": relations,
            "announcements": list(announcements),
        },
        nonzero=True,
    )


def prove_linear_relations(
    statement: dict,
    relations: Sequence[dict],
    secrets: Mapping[str, int],
) -> dict:
    """Prove several linear G1 relations over a shared secret vector.

    Each relation has shape ``target = sum coeff_i * secret_i * base_i``.
    """

    normalized_relations = _normalize_relations(relations)
    secret_names = sorted({term["secret"] for relation in normalized_relations for term in relation["terms"]})
    missing = [name for name in secret_names if name not in secrets]
    if missing:
        raise ValueError(f"missing secrets for Sigma proof: {missing}")

    blindings = {name: field_random() for name in secret_names}
    announcements = [
        _relation_commitment(relation, blindings).to_hex()
        for relation in normalized_relations
    ]
    challenge = _challenge(statement, normalized_relations, announcements)
    responses = {
        name: field_add(blindings[name], field_mul(challenge, secrets[name]))
        for name in secret_names
    }
    return {
        "version": 1,
        "announcements": announcements,
        "responses": responses,
    }


def verify_linear_relations(statement: dict, relations: Sequence[dict], proof: dict) -> bool:
    try:
        if proof.get("version") != 1:
            return False
        normalized_relations = _normalize_relations(relations)
        announcements = [_normalize_point(point) for point in proof["announcements"]]
        if len(announcements) != len(normalized_relations):
            return False
        responses = {str(name): field_normalize(value) for name, value in proof["responses"].items()}
        secret_names = sorted({term["secret"] for relation in normalized_relations for term in relation["terms"]})
        if set(responses) != set(secret_names):
            return False

        challenge = _challenge(statement, normalized_relations, announcements)
        return g1_verify_linear_relations(
            {
                "challenge": format(challenge, "x"),
                "relations": [
                    {
                        "announcement": announcement,
                        "target": relation["target"],
                        "points": [term["base"] for term in relation["terms"]],
                        "scalars": [
                            format(
                                field_mul(
                                    responses[term["secret"]],
                                    term["coeff"],
                                ),
                                "x",
                            )
                            for term in relation["terms"]
                        ],
                    }
                    for relation, announcement in zip(
                        normalized_relations,
                        announcements,
                    )
                ],
            }
        )
    except (KeyError, TypeError, ValueError):
        return False
