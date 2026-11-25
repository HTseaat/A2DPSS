"""Sigma/Fiat-Shamir proofs for the Benhamouda [3] baseline.

The Go reference implementation in ``go-yosovss/protocols/resharing`` uses
batched DL proofs, double-DLEQ proofs, and verifier forwarding proofs.  This
module keeps the same proof semantics while expressing them over the G1/scalar
stack already used by ATCaaS.
"""

from __future__ import annotations

from time import perf_counter
from typing import Mapping, Sequence

from beaver.g1_backend import FIELD_ORDER, G1Point, field_normalize, g1_linear_combination
from beaver.sigma import canonical_json, prove_linear_relations, verify_linear_relations


PROOF_VERSION = 1


def _normalize_point(value: str | G1Point) -> str:
    return value.to_hex() if isinstance(value, G1Point) else G1Point.from_hex(value).to_hex()


def _normalize_points(values: Sequence[str | G1Point]) -> tuple[str, ...]:
    return tuple(_normalize_point(value) for value in values)


def _normalize_scalars(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(field_normalize(value) for value in values)


def _check_same_length(label: str, *values: Sequence) -> int:
    lengths = [len(value) for value in values]
    if not lengths or lengths[0] == 0:
        raise ValueError(f"{label} must be non-empty")
    if any(length != lengths[0] for length in lengths):
        raise ValueError(f"{label} inputs must have the same length")
    return lengths[0]


def proof_size_bytes(proof: Mapping) -> int:
    """Return canonical serialized proof size for experiment accounting."""

    return len(canonical_json(dict(proof)))


def timed_prove(prover, *args, **kwargs) -> tuple[dict, dict]:
    """Run a proof generator and return ``(proof, metrics)``."""

    start = perf_counter()
    proof = prover(*args, **kwargs)
    elapsed_ms = (perf_counter() - start) * 1000.0
    return proof, {
        "proof_gen_ms": elapsed_ms,
        "proof_bytes": proof_size_bytes(proof),
    }


def timed_verify(verifier, *args, **kwargs) -> tuple[bool, dict]:
    """Run a verifier and return ``(accepted, metrics)``."""

    start = perf_counter()
    accepted = bool(verifier(*args, **kwargs))
    elapsed_ms = (perf_counter() - start) * 1000.0
    return accepted, {
        "proof_verify_ms": elapsed_ms,
    }


def _wrap(kind: str, sigma_proof: Mapping) -> dict:
    return {
        "version": PROOF_VERSION,
        "kind": kind,
        "sigma": dict(sigma_proof),
    }


def _unwrap(kind: str, proof: Mapping) -> dict:
    if proof.get("version") != PROOF_VERSION:
        raise ValueError("unsupported Benhamouda proof version")
    if proof.get("kind") != kind:
        raise ValueError(f"expected proof kind {kind!r}")
    return dict(proof["sigma"])


def _eval_power(point: int, degree: int) -> int:
    return pow(field_normalize(point), int(degree), FIELD_ORDER)


def prove_discrete_logs(
    bases: Sequence[str | G1Point],
    points: Sequence[str | G1Point],
    logs: Sequence[int],
    *,
    label: str = "benhamouda-dl",
) -> dict:
    """Prove knowledge of ``logs[i]`` such that ``points[i] = logs[i] * bases[i]``."""

    n = _check_same_length("discrete-log proof", bases, points, logs)
    norm_bases = _normalize_points(bases)
    norm_points = _normalize_points(points)
    norm_logs = _normalize_scalars(logs)
    statement = {
        "kind": label,
        "bases": list(norm_bases),
        "points": list(norm_points),
    }
    relations = [
        {
            "name": f"dl:{idx}",
            "target": norm_points[idx],
            "terms": [
                {
                    "secret": f"x:{idx}",
                    "base": norm_bases[idx],
                }
            ],
        }
        for idx in range(n)
    ]
    secrets = {
        f"x:{idx}": norm_logs[idx]
        for idx in range(n)
    }
    return _wrap("dl", prove_linear_relations(statement, relations, secrets))


def verify_discrete_logs(
    bases: Sequence[str | G1Point],
    points: Sequence[str | G1Point],
    proof: Mapping,
    *,
    label: str = "benhamouda-dl",
) -> bool:
    try:
        n = _check_same_length("discrete-log proof", bases, points)
        norm_bases = _normalize_points(bases)
        norm_points = _normalize_points(points)
        statement = {
            "kind": label,
            "bases": list(norm_bases),
            "points": list(norm_points),
        }
        relations = [
            {
                "name": f"dl:{idx}",
                "target": norm_points[idx],
                "terms": [
                    {
                        "secret": f"x:{idx}",
                        "base": norm_bases[idx],
                    }
                ],
            }
            for idx in range(n)
        ]
        return verify_linear_relations(statement, relations, _unwrap("dl", proof))
    except (KeyError, TypeError, ValueError):
        return False


def prove_double_dleq(
    *,
    g_hex: str | G1Point,
    h_hex: str | G1Point,
    left_bases: Sequence[str | G1Point],
    right_bases: Sequence[str | G1Point],
    z_points: Sequence[str | G1Point],
    z_prime_points: Sequence[str | G1Point],
    x_values: Sequence[int],
    y_values: Sequence[int],
    label: str = "benhamouda-double-dleq",
) -> dict:
    """Prove the Go implementation's dealer-side double-DLEQ relation.

    For every index ``i``:
    ``z_i = x_i*g + y_i*h`` and
    ``z_prime_i = x_i*left_bases_i + y_i*right_bases_i``.
    """

    n = _check_same_length(
        "double-DLEQ proof",
        left_bases,
        right_bases,
        z_points,
        z_prime_points,
        x_values,
        y_values,
    )
    g = _normalize_point(g_hex)
    h = _normalize_point(h_hex)
    left = _normalize_points(left_bases)
    right = _normalize_points(right_bases)
    z = _normalize_points(z_points)
    z_prime = _normalize_points(z_prime_points)
    xs = _normalize_scalars(x_values)
    ys = _normalize_scalars(y_values)
    statement = {
        "kind": label,
        "g": g,
        "h": h,
        "left_bases": list(left),
        "right_bases": list(right),
        "z_points": list(z),
        "z_prime_points": list(z_prime),
    }
    relations = []
    for idx in range(n):
        relations.append(
            {
                "name": f"z:{idx}",
                "target": z[idx],
                "terms": [
                    {"secret": f"x:{idx}", "base": g},
                    {"secret": f"y:{idx}", "base": h},
                ],
            }
        )
        relations.append(
            {
                "name": f"zprime:{idx}",
                "target": z_prime[idx],
                "terms": [
                    {"secret": f"x:{idx}", "base": left[idx]},
                    {"secret": f"y:{idx}", "base": right[idx]},
                ],
            }
        )
    secrets = {}
    for idx in range(n):
        secrets[f"x:{idx}"] = xs[idx]
        secrets[f"y:{idx}"] = ys[idx]
    return _wrap("double-dleq", prove_linear_relations(statement, relations, secrets))


def verify_double_dleq(
    *,
    g_hex: str | G1Point,
    h_hex: str | G1Point,
    left_bases: Sequence[str | G1Point],
    right_bases: Sequence[str | G1Point],
    z_points: Sequence[str | G1Point],
    z_prime_points: Sequence[str | G1Point],
    proof: Mapping,
    label: str = "benhamouda-double-dleq",
) -> bool:
    try:
        n = _check_same_length(
            "double-DLEQ proof",
            left_bases,
            right_bases,
            z_points,
            z_prime_points,
        )
        g = _normalize_point(g_hex)
        h = _normalize_point(h_hex)
        left = _normalize_points(left_bases)
        right = _normalize_points(right_bases)
        z = _normalize_points(z_points)
        z_prime = _normalize_points(z_prime_points)
        statement = {
            "kind": label,
            "g": g,
            "h": h,
            "left_bases": list(left),
            "right_bases": list(right),
            "z_points": list(z),
            "z_prime_points": list(z_prime),
        }
        relations = []
        for idx in range(n):
            relations.append(
                {
                    "name": f"z:{idx}",
                    "target": z[idx],
                    "terms": [
                        {"secret": f"x:{idx}", "base": g},
                        {"secret": f"y:{idx}", "base": h},
                    ],
                }
            )
            relations.append(
                {
                    "name": f"zprime:{idx}",
                    "target": z_prime[idx],
                    "terms": [
                        {"secret": f"x:{idx}", "base": left[idx]},
                        {"secret": f"y:{idx}", "base": right[idx]},
                    ],
                }
            )
        return verify_linear_relations(
            statement,
            relations,
            _unwrap("double-dleq", proof),
        )
    except (KeyError, TypeError, ValueError):
        return False


def prove_resharing_consistency(
    *,
    anchor_base_hex: str | G1Point,
    share_base_hex: str | G1Point,
    eval_bases: Sequence[str | G1Point],
    share_point_hex: str | G1Point,
    aggregate_commitment_hex: str | G1Point,
    polynomial_coeffs: Sequence[int],
    anchor_scalar: int,
    eval_points: Sequence[int] | None = None,
    label: str = "benhamouda-resharing",
) -> dict:
    """Fig. 7-style proof for a resharing polynomial.

    It proves knowledge of ``f`` and ``anchor_scalar`` such that
    ``share_point = f(0) * share_base`` and
    ``aggregate_commitment = anchor_scalar*anchor_base + sum_l f(l)*eval_bases[l]``.
    """

    degree = len(polynomial_coeffs) - 1
    if degree < 0:
        raise ValueError("polynomial_coeffs must be non-empty")
    bases = _normalize_points(eval_bases)
    points = tuple(range(1, len(bases) + 1)) if eval_points is None else tuple(int(x) for x in eval_points)
    _check_same_length("resharing eval bases", bases, points)
    anchor_base = _normalize_point(anchor_base_hex)
    share_base = _normalize_point(share_base_hex)
    share_point = _normalize_point(share_point_hex)
    aggregate = _normalize_point(aggregate_commitment_hex)
    coeffs = _normalize_scalars(polynomial_coeffs)
    statement = {
        "kind": label,
        "degree": degree,
        "anchor_base": anchor_base,
        "share_base": share_base,
        "eval_bases": list(bases),
        "eval_points": list(points),
        "share_point": share_point,
        "aggregate_commitment": aggregate,
    }
    relations = [
        {
            "name": "share-at-zero",
            "target": share_point,
            "terms": [
                {"secret": "f:0", "base": share_base},
            ],
        },
        {
            "name": "aggregate-commitment",
            "target": aggregate,
            "terms": [{"secret": "anchor", "base": anchor_base}]
            + [
                {
                    "secret": f"f:{power}",
                    "base": bases[idx],
                    "coeff": _eval_power(point, power),
                }
                for idx, point in enumerate(points)
                for power in range(degree + 1)
            ],
        },
    ]
    secrets = {"anchor": field_normalize(anchor_scalar)}
    for idx, coeff in enumerate(coeffs):
        secrets[f"f:{idx}"] = coeff
    return _wrap("resharing", prove_linear_relations(statement, relations, secrets))


def verify_resharing_consistency(
    *,
    anchor_base_hex: str | G1Point,
    share_base_hex: str | G1Point,
    eval_bases: Sequence[str | G1Point],
    share_point_hex: str | G1Point,
    aggregate_commitment_hex: str | G1Point,
    degree: int,
    proof: Mapping,
    eval_points: Sequence[int] | None = None,
    label: str = "benhamouda-resharing",
) -> bool:
    try:
        degree = int(degree)
        if degree < 0:
            return False
        bases = _normalize_points(eval_bases)
        points = tuple(range(1, len(bases) + 1)) if eval_points is None else tuple(int(x) for x in eval_points)
        _check_same_length("resharing eval bases", bases, points)
        anchor_base = _normalize_point(anchor_base_hex)
        share_base = _normalize_point(share_base_hex)
        share_point = _normalize_point(share_point_hex)
        aggregate = _normalize_point(aggregate_commitment_hex)
        statement = {
            "kind": label,
            "degree": degree,
            "anchor_base": anchor_base,
            "share_base": share_base,
            "eval_bases": list(bases),
            "eval_points": list(points),
            "share_point": share_point,
            "aggregate_commitment": aggregate,
        }
        relations = [
            {
                "name": "share-at-zero",
                "target": share_point,
                "terms": [
                    {"secret": "f:0", "base": share_base},
                ],
            },
            {
                "name": "aggregate-commitment",
                "target": aggregate,
                "terms": [{"secret": "anchor", "base": anchor_base}]
                + [
                    {
                        "secret": f"f:{power}",
                        "base": bases[idx],
                        "coeff": _eval_power(point, power),
                    }
                    for idx, point in enumerate(points)
                    for power in range(degree + 1)
                ],
            },
        ]
        return verify_linear_relations(
            statement,
            relations,
            _unwrap("resharing", proof),
        )
    except (KeyError, TypeError, ValueError):
        return False


def prove_randomness_consistency(
    *,
    delta_hex: str | G1Point,
    x_points: Sequence[str | G1Point],
    h_bases: Sequence[str | G1Point],
    g_bases: Sequence[str | G1Point],
    randomness: Sequence[int],
    label: str = "benhamouda-randomness",
) -> dict:
    """Fig. 8 dealer-side proof that ``Delta`` and ``X_l`` use the same randomness."""

    n = _check_same_length("randomness consistency", x_points, h_bases, g_bases, randomness)
    delta = _normalize_point(delta_hex)
    xs = _normalize_points(x_points)
    hs = _normalize_points(h_bases)
    gs = _normalize_points(g_bases)
    rs = _normalize_scalars(randomness)
    statement = {
        "kind": label,
        "delta": delta,
        "x_points": list(xs),
        "h_bases": list(hs),
        "g_bases": list(gs),
    }
    relations = [
        {
            "name": "delta",
            "target": delta,
            "terms": [
                {"secret": f"r:{idx}", "base": hs[idx]}
                for idx in range(n)
            ],
        }
    ] + [
        {
            "name": f"x:{idx}",
            "target": xs[idx],
            "terms": [
                {"secret": f"r:{idx}", "base": gs[idx]},
            ],
        }
        for idx in range(n)
    ]
    secrets = {
        f"r:{idx}": rs[idx]
        for idx in range(n)
    }
    return _wrap("randomness", prove_linear_relations(statement, relations, secrets))


def verify_randomness_consistency(
    *,
    delta_hex: str | G1Point,
    x_points: Sequence[str | G1Point],
    h_bases: Sequence[str | G1Point],
    g_bases: Sequence[str | G1Point],
    proof: Mapping,
    label: str = "benhamouda-randomness",
) -> bool:
    try:
        n = _check_same_length("randomness consistency", x_points, h_bases, g_bases)
        delta = _normalize_point(delta_hex)
        xs = _normalize_points(x_points)
        hs = _normalize_points(h_bases)
        gs = _normalize_points(g_bases)
        statement = {
            "kind": label,
            "delta": delta,
            "x_points": list(xs),
            "h_bases": list(hs),
            "g_bases": list(gs),
        }
        relations = [
            {
                "name": "delta",
                "target": delta,
                "terms": [
                    {"secret": f"r:{idx}", "base": hs[idx]}
                    for idx in range(n)
                ],
            }
        ] + [
            {
                "name": f"x:{idx}",
                "target": xs[idx],
                "terms": [
                    {"secret": f"r:{idx}", "base": gs[idx]},
                ],
            }
            for idx in range(n)
        ]
        return verify_linear_relations(
            statement,
            relations,
            _unwrap("randomness", proof),
        )
    except (KeyError, TypeError, ValueError):
        return False


def prove_decryption_correctness(
    *,
    key_base_hex: str | G1Point,
    ciphertext_c1_hex: str | G1Point,
    ciphertext_c2_hex: str | G1Point,
    public_key_hex: str | G1Point,
    plaintext_hex: str | G1Point,
    secret_key: int,
    label: str = "benhamouda-decryption",
) -> dict:
    """Prove correct ElGamal-in-group decryption for one ciphertext."""

    key_base = _normalize_point(key_base_hex)
    c1 = _normalize_point(ciphertext_c1_hex)
    c2 = G1Point.from_hex(_normalize_point(ciphertext_c2_hex))
    public_key = _normalize_point(public_key_hex)
    plaintext = G1Point.from_hex(_normalize_point(plaintext_hex))
    masked_plaintext = (c2 - plaintext).to_hex()
    statement = {
        "kind": label,
        "key_base": key_base,
        "ciphertext_c1": c1,
        "ciphertext_c2": c2.to_hex(),
        "public_key": public_key,
        "plaintext": plaintext.to_hex(),
        "masked_plaintext": masked_plaintext,
    }
    relations = [
        {
            "name": "public-key",
            "target": public_key,
            "terms": [
                {"secret": "sk", "base": key_base},
            ],
        },
        {
            "name": "decryption",
            "target": masked_plaintext,
            "terms": [
                {"secret": "sk", "base": c1},
            ],
        },
    ]
    return _wrap(
        "decryption",
        prove_linear_relations(
            statement,
            relations,
            {"sk": field_normalize(secret_key)},
        ),
    )


def verify_decryption_correctness(
    *,
    key_base_hex: str | G1Point,
    ciphertext_c1_hex: str | G1Point,
    ciphertext_c2_hex: str | G1Point,
    public_key_hex: str | G1Point,
    plaintext_hex: str | G1Point,
    proof: Mapping,
    label: str = "benhamouda-decryption",
) -> bool:
    try:
        key_base = _normalize_point(key_base_hex)
        c1 = _normalize_point(ciphertext_c1_hex)
        c2 = G1Point.from_hex(_normalize_point(ciphertext_c2_hex))
        public_key = _normalize_point(public_key_hex)
        plaintext = G1Point.from_hex(_normalize_point(plaintext_hex))
        masked_plaintext = (c2 - plaintext).to_hex()
        statement = {
            "kind": label,
            "key_base": key_base,
            "ciphertext_c1": c1,
            "ciphertext_c2": c2.to_hex(),
            "public_key": public_key,
            "plaintext": plaintext.to_hex(),
            "masked_plaintext": masked_plaintext,
        }
        relations = [
            {
                "name": "public-key",
                "target": public_key,
                "terms": [
                    {"secret": "sk", "base": key_base},
                ],
            },
            {
                "name": "decryption",
                "target": masked_plaintext,
                "terms": [
                    {"secret": "sk", "base": c1},
                ],
            },
        ]
        return verify_linear_relations(
            statement,
            relations,
            _unwrap("decryption", proof),
        )
    except (KeyError, TypeError, ValueError):
        return False


def point_linear_combination(
    points: Sequence[str | G1Point],
    scalars: Sequence[int],
) -> str:
    """Small public helper used by A1/A2 smoke fixtures."""

    return g1_linear_combination(
        list(_normalize_points(points)),
        list(_normalize_scalars(scalars)),
    ).to_hex()
