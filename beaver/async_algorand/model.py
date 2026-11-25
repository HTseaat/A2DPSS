"""Canonical wire values and finite committee threshold checks."""
from dataclasses import dataclass
import hashlib
import json


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate wire field")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("nonfinite wire value")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


@dataclass(frozen=True)
class Parameters:
    population: int
    m: int
    h: int
    probability: float = 1.0
    mode: str = "bernoulli"
    seed: str = ""
    max_views: int = 16

    def __post_init__(self):
        if any(type(x) is not int for x in
               (self.population, self.m, self.h, self.max_views)):
            raise ValueError("committee bounds must be integers")
        if not 1 <= self.h <= self.m <= self.population or self.max_views < 1:
            raise ValueError("invalid committee bounds")
        if 3 * self.h <= 2 * self.m:
            raise ValueError("robust quorum intersection requires 3h > 2m")
        if 2 * self.h - self.m < self.reconstruct:
            raise ValueError("rounded AVSS threshold exceeds honest quorum intersection")
        if self.mode not in ("bernoulli", "fixed-size"):
            raise ValueError("unknown committee selection mode")
        if not 0 < self.probability <= 1:
            raise ValueError("invalid selection probability")

    @property
    def degree(self):
        return (self.h + 1) // 2

    @property
    def reconstruct(self):
        return self.degree + 1

    def role(self, view, name):
        if type(view) is not int or not 1 <= view <= self.max_views:
            raise ValueError("view outside configured experiment limit")
        return "v%d/%s" % (view, name)


def node_set(value, population, minimum=0):
    if not isinstance(value, list) or not minimum <= len(value) <= population:
        return False
    return (all(type(i) is int and 0 <= i < population for i in value)
            and value == sorted(set(value)))
