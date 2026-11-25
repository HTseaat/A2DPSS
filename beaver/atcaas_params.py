"""Canonical Phase-6 ATCaaS protocol parameter schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


PARAMETER_SCHEMA = "atcaas-protocol-parameters/v2"


def _plain_int(value, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class ATCaaSProtocolParameters:
    """Separate resharing, agreement-sortition, and common-coin parameters.

    ``t_a`` is the paper's corrupted-role bound. ``agreement_t_impl`` is only
    the complement required by legacy fixed-size RBC constructors.
    """

    M: int
    n_r: int
    t_r: int
    n_a: int
    t_a: int
    m_a: int
    quorum_h: int
    coin_c: int
    coin_t: int
    coin_k: int

    def __post_init__(self) -> None:
        M = _plain_int(self.M, name="M", minimum=1)
        n_r = _plain_int(self.n_r, name="n_r", minimum=1)
        t_r = _plain_int(self.t_r, name="t_r")
        n_a = _plain_int(self.n_a, name="n_a", minimum=1)
        t_a = _plain_int(self.t_a, name="t_a")
        m_a = _plain_int(self.m_a, name="m_a", minimum=1)
        quorum_h = _plain_int(self.quorum_h, name="quorum_h", minimum=1)
        coin_c = _plain_int(self.coin_c, name="coin_c", minimum=1)
        coin_t = _plain_int(self.coin_t, name="coin_t")
        coin_k = _plain_int(self.coin_k, name="coin_k", minimum=1)

        if n_r > M:
            raise ValueError("n_r must not exceed M")
        if 2 * t_r >= n_r:
            raise ValueError("resharing requires t_r < n_r/2")
        if n_a > M:
            raise ValueError("n_a must not exceed M")
        if 3 * t_a >= n_a:
            raise ValueError("agreement requires t_a < n_a/3")
        if not n_a <= m_a <= M:
            raise ValueError("m_a must satisfy n_a <= m_a <= M")
        if quorum_h != n_a - t_a:
            raise ValueError("quorum_h must equal n_a - t_a")
        if quorum_h > m_a:
            raise ValueError("quorum_h must not exceed m_a")
        # Any two h-quorums in an actual committee of size at most m_a must
        # intersect in more roles than the adversary can control.
        if 2 * quorum_h - m_a <= t_a:
            raise ValueError(
                "agreement quorum intersection requires "
                "2*quorum_h - m_a > t_a"
            )
        if coin_c > M:
            raise ValueError("coin_c must not exceed M")
        if coin_t >= coin_c:
            raise ValueError("coin_t must be smaller than coin_c")
        if coin_k <= coin_t:
            raise ValueError("common coin requires coin_t < coin_k")
        if coin_k > coin_c - coin_t:
            raise ValueError("common coin requires coin_k <= coin_c - coin_t")

    @property
    def expected_committee_size(self) -> int:
        return self.n_a

    @property
    def committee_m(self) -> int:
        return self.m_a

    @property
    def agreement_t_impl(self) -> int:
        return self.m_a - self.quorum_h

    def to_config_obj(self) -> dict:
        return {
            "schema": PARAMETER_SCHEMA,
            "M": self.M,
            "n_r": self.n_r,
            "t_r": self.t_r,
            "n_a": self.n_a,
            "t_a": self.t_a,
            "m_a": self.m_a,
            "quorum_h": self.quorum_h,
            "agreement_t_impl": self.agreement_t_impl,
            "coin_c": self.coin_c,
            "coin_t": self.coin_t,
            "coin_k": self.coin_k,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping,
        *,
        extras: Optional[Mapping] = None,
        require_schema: bool = True,
    ) -> "ATCaaSProtocolParameters":
        if not isinstance(value, Mapping):
            raise TypeError("ATCaaS config must be a mapping")
        raw = value.get("atcaas_parameters")
        if raw is not None:
            if not isinstance(raw, Mapping):
                raise TypeError("atcaas_parameters must be a mapping")
            if raw.get("schema") != PARAMETER_SCHEMA:
                raise ValueError("unsupported ATCaaS parameter schema")
            params = cls(
                M=raw["M"],
                n_r=raw["n_r"],
                t_r=raw["t_r"],
                n_a=raw["n_a"],
                t_a=raw["t_a"],
                m_a=raw["m_a"],
                quorum_h=raw["quorum_h"],
                coin_c=raw["coin_c"],
                coin_t=raw["coin_t"],
                coin_k=raw["coin_k"],
            )
            if raw.get("agreement_t_impl") != params.agreement_t_impl:
                raise ValueError("agreement_t_impl does not match m_a-quorum_h")
            if value.get("M", value.get("N")) != params.M:
                raise ValueError("top-level M does not match atcaas_parameters")
            return params
        if require_schema:
            raise ValueError("atcaas_parameters schema is required")

        extra = dict(extras or value.get("extra") or {})
        dpss = dict(extra.get("dpss") or extra.get("dpss_smoke") or {})
        M = int(value.get("M", value["N"]))
        n_a = int(
            value.get("n_a", value.get("expected_committee_size", value.get("n", M)))
        )
        m_a = int(value.get("m_a", value.get("committee_m", n_a)))
        quorum_h = int(value.get("quorum_h", n_a - int(value.get("t", 0))))
        t_a = int(value.get("t_a", n_a - quorum_h))
        t_r = int(value.get("t_r", dpss.get("service_t", value.get("t", 0))))
        n_r = int(
            value.get(
                "n_r",
                len(dpss.get("source_committee", ()))
                or min(M, max(t_r + 2, 3)),
            )
        )
        coin_committee = tuple(extra.get("coin_committee", range(M)))
        coin_c = int(value.get("coin_c", len(coin_committee)))
        coin_k = int(
            value.get("coin_k", extra.get("coin_threshold", max(1, coin_c - t_a)))
        )
        # Legacy, schema-less configurations did not declare a coin corruption
        # bound.  Preserve their old threshold while exposing the strongest
        # bound that threshold can actually support.  Versioned experiment
        # configurations must carry coin_t explicitly.
        coin_t = int(
            value.get("coin_t", min(coin_k - 1, coin_c - coin_k))
        )
        return cls(
            M=M,
            n_r=n_r,
            t_r=t_r,
            n_a=n_a,
            t_a=t_a,
            m_a=m_a,
            quorum_h=quorum_h,
            coin_c=coin_c,
            coin_t=coin_t,
            coin_k=coin_k,
        )


__all__ = ["PARAMETER_SCHEMA", "ATCaaSProtocolParameters"]
