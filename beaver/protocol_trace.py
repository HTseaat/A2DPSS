"""Replayable ideal-coin and VRF committee-role traces for TOB experiments."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping, Sequence, Tuple


class IdealCoin(ABC):
    """Ideal common-coin interface; no setup or communication is modelled."""

    @abstractmethod
    def get(self, index: int) -> int:
        raise NotImplementedError

    def get_bytes(self, index: int, length: int = 32) -> bytes:
        value = self.get(index)
        if value < 0:
            raise ValueError("ideal coin values must be non-negative")
        return int(value).to_bytes(length, byteorder="big", signed=False)


class DeterministicIdealCoin(IdealCoin):
    """Seeded deterministic adapter for the ideal common-coin interface."""

    def __init__(self, seed: str):
        self.seed = str(seed)

    def get(self, index: int) -> int:
        material = f"{self.seed}:ideal-coin:{int(index)}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest(), "big")


class RecordedIdealCoin(IdealCoin):
    """Strict replay of pre-generated ideal-coin outputs by logical index."""

    def __init__(self, values: Mapping[int, int]):
        self.values = {int(index): int(value) for index, value in values.items()}
        if any(index <= 0 for index in self.values):
            raise ValueError("coin indices must be positive")
        if any(value < 0 for value in self.values.values()):
            raise ValueError("coin values must be non-negative")

    def get(self, index: int) -> int:
        try:
            return self.values[int(index)]
        except KeyError as exc:
            raise KeyError(f"ideal coin trace has no index {int(index)}") from exc


class CommitteeRoleTrace(ABC):
    """Public logical role namespace for each committee-selection index."""

    @abstractmethod
    def roles(self, index: int) -> Tuple[int, ...]:
        raise NotImplementedError


class AllMembersCommitteeTrace(CommitteeRoleTrace):
    """All-role fixture used only when every role is VRF-selected."""

    def __init__(self, M: int):
        if int(M) <= 0:
            raise ValueError("M must be positive")
        self.M = int(M)

    def roles(self, index: int) -> Tuple[int, ...]:
        if int(index) <= 0:
            raise ValueError("committee index must be positive")
        return tuple(range(self.M))


class RecordedCommitteeRoleTrace(CommitteeRoleTrace):
    """Strict replay of pre-generated VRF-selected logical role indices."""

    def __init__(self, roles_by_index: Mapping[int, Sequence[int]]):
        normalized = {}
        for index, roles in roles_by_index.items():
            index = int(index)
            ordered = tuple(int(role) for role in roles)
            if index <= 0:
                raise ValueError("committee indices must be positive")
            if not ordered:
                raise ValueError(f"committee {index} cannot be empty")
            if any(role < 0 for role in ordered):
                raise ValueError("committee roles must be non-negative")
            if len(set(ordered)) != len(ordered):
                raise ValueError(f"committee {index} contains duplicate roles")
            normalized[index] = ordered
        self.roles_by_index = normalized

    def roles(self, index: int) -> Tuple[int, ...]:
        try:
            return self.roles_by_index[int(index)]
        except KeyError as exc:
            raise KeyError(f"committee trace has no index {int(index)}") from exc


class CommitteeRoleTraceArtifact:
    """Role-only replay artifact, deliberately independent of common coin.

    Formal experiments can replay the public VRF-selected role namespace while
    continuing to invoke an online threshold common coin.  Keeping this schema
    separate prevents merely loading a role fixture from changing coin mode.
    """

    VERSION = 1

    def __init__(
        self,
        committee_roles_by_index: Mapping[int, Sequence[int]],
    ) -> None:
        self.committees = RecordedCommitteeRoleTrace(
            committee_roles_by_index
        )
        if not self.committees.roles_by_index:
            raise ValueError("committee role trace cannot be empty")

    @classmethod
    def from_dict(cls, data: Mapping) -> "CommitteeRoleTraceArtifact":
        if not isinstance(data, Mapping):
            raise ValueError("committee role trace must be a JSON object")
        version = data.get("version", cls.VERSION)
        if isinstance(version, bool) or int(version) != cls.VERSION:
            raise ValueError(
                f"unsupported committee role trace version {version!r}; "
                f"expected {cls.VERSION}"
            )
        if "coin_by_index" in data:
            raise ValueError(
                "role-only trace must not contain ideal-coin outputs"
            )
        try:
            roles = data["committee_roles_by_index"]
        except KeyError as exc:
            raise ValueError(
                "role-only trace requires committee_roles_by_index"
            ) from exc
        if not isinstance(roles, Mapping):
            raise TypeError("committee_roles_by_index must be a mapping")
        return cls(roles)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
    ) -> "CommitteeRoleTraceArtifact":
        with Path(path).open(encoding="utf-8") as fp:
            data = json.load(fp)
        return cls.from_dict(data)

    def validate_population(self, M: int) -> None:
        M = int(M)
        if M <= 0:
            raise ValueError("M must be positive")
        invalid = sorted(
            {
                role
                for roles in self.committees.roles_by_index.values()
                for role in roles
                if role >= M
            }
        )
        if invalid:
            raise ValueError(
                "committee role trace contains roles outside population "
                f"M={M}: {invalid}"
            )

    def to_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "committee_roles_by_index": {
                str(index): list(roles)
                for index, roles in sorted(
                    self.committees.roles_by_index.items()
                )
            },
        }

    def digest(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


class ProtocolReplayTrace:
    """One artifact shared by DAG-TOB and ACS-TOB experiment runners.

    Logical index ``k`` is a DAG wave for DAG-TOB.  ACS-TOB reserves a disjoint
    window for every batch and maps each SelectBlock attempt into that window.
    Both protocols consume ``coin_by_index[k]`` and the ordered VRF role
    namespace ``committee_roles_by_index[k]`` at the resulting logical index.
    """

    def __init__(
        self,
        coin_by_index: Mapping[int, int],
        committee_roles_by_index: Mapping[int, Sequence[int]],
    ):
        self.coin = RecordedIdealCoin(coin_by_index)
        self.committees = RecordedCommitteeRoleTrace(committee_roles_by_index)

    @classmethod
    def from_dict(cls, data: Mapping) -> "ProtocolReplayTrace":
        return cls(
            data.get("coin_by_index", {}),
            data.get("committee_roles_by_index", {}),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ProtocolReplayTrace":
        with Path(path).open(encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            raise ValueError("protocol replay trace must be a JSON object")
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return {
            "coin_by_index": {
                str(index): value
                for index, value in sorted(self.coin.values.items())
            },
            "committee_roles_by_index": {
                str(index): list(roles)
                for index, roles in sorted(
                    self.committees.roles_by_index.items()
                )
            },
        }

    def digest(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
