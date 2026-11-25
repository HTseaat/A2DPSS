"""Deterministic single- and multi-VPS RU22 DAG-TOB fixtures."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from beaver.baselines.ru22_dpss import (
    RU22Context,
    RU22HolderState,
    RU22InitialCertificate,
    RU22LocalVPSShare,
    RU22VPS,
)
from beaver.chunk_elgamal import (
    decrypt_elgamal_ciphertext,
    derive_public_key,
)
from beaver.g1_backend import FIELD_ORDER, G1Point
from beaver.pedersen_shamir import eval_poly, pedersen_commit


MANIFEST_KIND = "atcaas-ru22-dag-experiment-manifest"
MANIFEST_VERSION = 2
LOCAL_SHARD_KIND = "atcaas-ru22-dag-local-shard"
LOCAL_SHARD_VERSION = 1
FIXTURE_MODE_SINGLE_VPS = "single-vps"
FIXTURE_MODE_WORST_CASE_MULTI_VPS = "worst-case-multi-vps"

_MANIFEST_DOMAIN = b"ATCaaS/RU22/dag-manifest/v2\x00"
_SHARD_DOMAIN = b"ATCaaS/RU22/dag-local-shard/v1\x00"
_FIXTURE_SCALAR_DOMAIN = b"ATCaaS/RU22/dag-fixture-scalar/v1\x00"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _exact_fields(value: Mapping, expected: set[str], *, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} fields are not canonical")


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be lowercase sha256 hex")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be lowercase sha256 hex") from exc
    if raw.hex() != value:
        raise ValueError(f"{name} must be lowercase sha256 hex")
    return value


def _scalar(value: Any, *, name: str, nonzero: bool) -> int:
    if isinstance(value, str):
        if not value.isdigit() or str(int(value)) != value:
            raise ValueError(f"{name} must be a canonical decimal scalar")
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a scalar")
    minimum = 1 if nonzero else 0
    if not minimum <= value < FIELD_ORDER:
        raise ValueError(f"{name} is outside the scalar field")
    return value


def _derive_scalar(seed: str, label: str, *, nonzero: bool = True) -> int:
    counter = 0
    while True:
        digest = hashlib.sha256(
            _FIXTURE_SCALAR_DOMAIN
            + seed.encode("utf-8")
            + b"\x00"
            + label.encode("utf-8")
            + b"\x00"
            + str(counter).encode("ascii")
        ).digest()
        value = int.from_bytes(digest, "big") % FIELD_ORDER
        if value != 0 or not nonzero:
            return value
        counter += 1


def _raw_g1(point: Mapping) -> str:
    return (
        b"\x04"
        + int(point["X"]).to_bytes(48, byteorder="big")
        + int(point["Y"]).to_bytes(48, byteorder="big")
    ).hex()


def _generators_from_config(config_path: str | Path) -> tuple[str, str]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    srs = json.loads(
        base64.b64decode(config["extra"]["SRS"]).decode("utf-8")
    )
    return _raw_g1(srs["Pk"]["G1_g"][0]), _raw_g1(
        srs["Pk"]["G1_h"][0]
    )


def _encrypt_commitment(
    key_base_hex: str,
    public_key_hex: str,
    commitment_hex: str,
    randomness: int,
) -> dict[str, str]:
    return {
        "C1": (
            G1Point.from_hex(key_base_hex) * randomness
        ).to_hex(),
        "C2": (
            G1Point.from_hex(commitment_hex)
            + G1Point.from_hex(public_key_hex) * randomness
        ).to_hex(),
    }


@dataclass(frozen=True)
class RU22DAGExperimentManifest:
    trial_id: str
    trial_epoch: str
    fixture_seed: str
    context: RU22Context
    parent_vps_catalog: Sequence[RU22VPS]
    source_role_nodes: Mapping[int, int]
    source_vps_views: Mapping[int, Sequence[str]]
    expected_successor_nodes: Sequence[int]
    initial_secret_commitment: str
    dag_m: int
    dag_t: int
    fixture_mode: str = FIXTURE_MODE_SINGLE_VPS
    fixture_only_deterministic_keys: bool = True
    not_for_production: bool = True

    def __post_init__(self) -> None:
        trial_id = _text(self.trial_id, name="trial_id")
        trial_epoch = _text(self.trial_epoch, name="trial_epoch")
        seed = _text(self.fixture_seed, name="fixture_seed")
        if not isinstance(self.context, RU22Context):
            raise TypeError("context must be an RU22Context")
        context = RU22Context.from_bytes(self.context.to_bytes())
        if self.fixture_mode not in {
            FIXTURE_MODE_SINGLE_VPS,
            FIXTURE_MODE_WORST_CASE_MULTI_VPS,
        }:
            raise ValueError("unsupported RU22 fixture mode")
        if self.fixture_only_deterministic_keys is not True:
            raise ValueError("fixture-only deterministic key flag is required")
        if self.not_for_production is not True:
            raise ValueError("not_for_production flag is required")
        if context.resharing_n != 2 * context.resharing_threshold + 1:
            raise ValueError("RU22 fixture requires n_r = 2*t_r + 1")

        catalog = tuple(
            RU22VPS.from_bytes(vps.to_bytes())
            for vps in self.parent_vps_catalog
        )
        if len({vps.vps_id for vps in catalog}) != len(catalog):
            raise ValueError("parent VPS catalog contains duplicate ids")
        for parent in catalog:
            if (
                parent.epoch != context.source_epoch
                or parent.committee != context.source_committee
                or len(parent.encrypted_commitments)
                != context.resharing_n
            ):
                raise ValueError("parent VPS does not match RU22 context")

        if not isinstance(self.source_role_nodes, Mapping):
            raise TypeError("source_role_nodes must be a mapping")
        source_role_nodes = {
            _integer(role, name="source role"): _integer(
                node, name="source node"
            )
            for role, node in self.source_role_nodes.items()
        }
        if set(source_role_nodes) != set(context.source_role_indices):
            raise ValueError("source role/node map does not match context")
        if len(set(source_role_nodes.values())) != len(source_role_nodes):
            raise ValueError("source role/node map must be one-to-one")
        if not isinstance(self.source_vps_views, Mapping):
            raise TypeError("source_vps_views must be a mapping")
        catalog_ids = {vps.vps_id for vps in catalog}
        source_vps_views = {}
        for role, raw_view in self.source_vps_views.items():
            role = _integer(role, name="source VPS view role")
            if not isinstance(raw_view, (list, tuple)):
                raise TypeError("source VPS view must be a sequence")
            view = tuple(
                _digest(vps_id, name="source VPS view id")
                for vps_id in raw_view
            )
            if (
                not view
                or len(view) > context.resharing_n
                or len(set(view)) != len(view)
                or not set(view) <= catalog_ids
            ):
                raise ValueError("source VPS view is invalid")
            source_vps_views[role] = view
        if set(source_vps_views) != set(source_role_nodes):
            raise ValueError("source VPS views do not match source roles")
        intersection = set.intersection(
            *(set(view) for view in source_vps_views.values())
        )
        if self.fixture_mode == FIXTURE_MODE_SINGLE_VPS:
            if len(catalog) != 1 or any(
                view != (catalog[0].vps_id,)
                for view in source_vps_views.values()
            ):
                raise ValueError(
                    "single-vps fixture requires one shared parent VPS"
                )
        else:
            if (
                len(catalog) != 2 * context.resharing_n
                or any(
                    len(view) != context.resharing_n
                    for view in source_vps_views.values()
                )
                or len({view for view in source_vps_views.values()})
                != context.resharing_n
                or set().union(
                    *(set(view) for view in source_vps_views.values())
                )
                != catalog_ids
                or len(intersection) != 1
                or next(iter(intersection), None) != min(catalog_ids)
            ):
                raise ValueError(
                    "worst-case multi-VPS fixture shape is invalid"
                )
        successors = tuple(
            _integer(node, name="successor node")
            for node in self.expected_successor_nodes
        )
        if len(set(successors)) != len(successors) or set(successors) != set(
            context.successor_role_indices
        ):
            raise ValueError("successor node set does not match context")
        commitment = G1Point.from_hex(
            self.initial_secret_commitment
        ).to_hex()
        dag_m = _integer(self.dag_m, name="dag_m", minimum=1)
        dag_t = _integer(self.dag_t, name="dag_t")
        if dag_t >= dag_m:
            raise ValueError("dag_t must be smaller than dag_m")
        if any(node >= dag_m for node in source_role_nodes.values()) or any(
            node >= dag_m for node in successors
        ):
            raise ValueError("RU22 role node is outside the DAG network")

        for name, value in (
            ("trial_id", trial_id),
            ("trial_epoch", trial_epoch),
            ("fixture_seed", seed),
            ("context", context),
            ("parent_vps_catalog", catalog),
            ("source_role_nodes", source_role_nodes),
            ("source_vps_views", source_vps_views),
            ("expected_successor_nodes", successors),
            ("initial_secret_commitment", commitment),
            ("dag_m", dag_m),
            ("dag_t", dag_t),
        ):
            object.__setattr__(self, name, value)

    def digest_material(self) -> dict[str, Any]:
        return {
            "kind": MANIFEST_KIND,
            "version": MANIFEST_VERSION,
            "trial_id": self.trial_id,
            "trial_epoch": self.trial_epoch,
            "fixture_seed": self.fixture_seed,
            "fixture_mode": self.fixture_mode,
            "fixture_only_deterministic_keys": True,
            "not_for_production": True,
            "context": self.context.to_obj(),
            "parent_vps_catalog": [
                vps.to_obj() for vps in self.parent_vps_catalog
            ],
            "source_role_nodes": {
                str(role): node
                for role, node in sorted(self.source_role_nodes.items())
            },
            "source_vps_views": {
                str(role): list(view)
                for role, view in sorted(self.source_vps_views.items())
            },
            "expected_successor_nodes": list(
                self.expected_successor_nodes
            ),
            "initial_secret_commitment": self.initial_secret_commitment,
            "dag_parameters": {"M": self.dag_m, "t_a": self.dag_t},
        }

    def digest(self) -> str:
        return hashlib.sha256(
            _MANIFEST_DOMAIN + _canonical_json(self.digest_material())
        ).hexdigest()

    def to_obj(self) -> dict[str, Any]:
        obj = self.digest_material()
        obj["manifest_digest"] = self.digest()
        return obj

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "RU22DAGExperimentManifest":
        _exact_fields(
            value,
            {
                "kind",
                "version",
                "trial_id",
                "trial_epoch",
                "fixture_seed",
                "fixture_mode",
                "fixture_only_deterministic_keys",
                "not_for_production",
                "context",
                "parent_vps_catalog",
                "source_role_nodes",
                "source_vps_views",
                "expected_successor_nodes",
                "initial_secret_commitment",
                "dag_parameters",
                "manifest_digest",
            },
            name="RU22 DAG manifest",
        )
        if (
            value["kind"] != MANIFEST_KIND
            or value["version"] != MANIFEST_VERSION
        ):
            raise ValueError("unsupported RU22 DAG manifest")
        _exact_fields(
            value["dag_parameters"], {"M", "t_a"}, name="DAG parameters"
        )
        supplied = _digest(
            value["manifest_digest"], name="manifest_digest"
        )
        manifest = cls(
            trial_id=value["trial_id"],
            trial_epoch=value["trial_epoch"],
            fixture_seed=value["fixture_seed"],
            fixture_mode=value["fixture_mode"],
            fixture_only_deterministic_keys=value[
                "fixture_only_deterministic_keys"
            ],
            not_for_production=value["not_for_production"],
            context=RU22Context.from_obj(value["context"]),
            parent_vps_catalog=tuple(
                RU22VPS.from_obj(item)
                for item in value["parent_vps_catalog"]
            ),
            source_role_nodes={
                int(role): node
                for role, node in value["source_role_nodes"].items()
            },
            source_vps_views={
                int(role): tuple(view)
                for role, view in value["source_vps_views"].items()
            },
            expected_successor_nodes=tuple(
                value["expected_successor_nodes"]
            ),
            initial_secret_commitment=value[
                "initial_secret_commitment"
            ],
            dag_m=value["dag_parameters"]["M"],
            dag_t=value["dag_parameters"]["t_a"],
        )
        if manifest.digest() != supplied:
            raise ValueError("RU22 manifest digest mismatch")
        return manifest

    @classmethod
    def from_file(cls, path: str | Path) -> "RU22DAGExperimentManifest":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_obj(json.load(handle))

    @property
    def common_parent_vps_id(self) -> str:
        intersection = set.intersection(
            *(set(view) for view in self.source_vps_views.values())
        )
        if len(intersection) != 1:
            raise ValueError("RU22 manifest does not have one common parent")
        return next(iter(intersection))


@dataclass(frozen=True)
class RU22SourceRoleShard:
    holder_role_id: int
    holder_eval_point: int
    recipient_index: int
    recipient_secret_key: int
    local_vps_shares: Sequence[RU22LocalVPSShare]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "holder_role_id",
            _integer(self.holder_role_id, name="holder_role_id"),
        )
        object.__setattr__(
            self,
            "holder_eval_point",
            _integer(
                self.holder_eval_point,
                name="holder_eval_point",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "recipient_index",
            _integer(self.recipient_index, name="recipient_index"),
        )
        object.__setattr__(
            self,
            "recipient_secret_key",
            _scalar(
                self.recipient_secret_key,
                name="source recipient secret key",
                nonzero=True,
            ),
        )
        object.__setattr__(
            self,
            "local_vps_shares",
            tuple(
                RU22LocalVPSShare(
                    vps=item.vps,
                    share_s=item.share_s,
                    share_s_hat=item.share_s_hat,
                )
                for item in self.local_vps_shares
            ),
        )

    def to_obj(self) -> dict[str, Any]:
        return {
            "holder_role_id": self.holder_role_id,
            "holder_eval_point": self.holder_eval_point,
            "recipient_index": self.recipient_index,
            "recipient_secret_key": str(self.recipient_secret_key),
            "local_vps_shares": [
                {
                    "vps": item.vps.to_obj(),
                    "share_s": str(item.share_s),
                    "share_s_hat": str(item.share_s_hat),
                }
                for item in self.local_vps_shares
            ],
        }

    @classmethod
    def from_obj(cls, value: Mapping[str, Any]) -> "RU22SourceRoleShard":
        _exact_fields(
            value,
            {
                "holder_role_id",
                "holder_eval_point",
                "recipient_index",
                "recipient_secret_key",
                "local_vps_shares",
            },
            name="RU22 source shard",
        )
        shares = []
        for item in value["local_vps_shares"]:
            _exact_fields(
                item, {"vps", "share_s", "share_s_hat"}, name="local VPS share"
            )
            shares.append(
                RU22LocalVPSShare(
                    vps=RU22VPS.from_obj(item["vps"]),
                    share_s=_scalar(
                        item["share_s"], name="share_s", nonzero=False
                    ),
                    share_s_hat=_scalar(
                        item["share_s_hat"],
                        name="share_s_hat",
                        nonzero=False,
                    ),
                )
            )
        return cls(
            holder_role_id=value["holder_role_id"],
            holder_eval_point=value["holder_eval_point"],
            recipient_index=value["recipient_index"],
            recipient_secret_key=_scalar(
                value["recipient_secret_key"],
                name="source recipient secret key",
                nonzero=True,
            ),
            local_vps_shares=shares,
        )

    def holder_state(self, resharing_n: int) -> RU22HolderState:
        return RU22HolderState(
            holder_role_id=self.holder_role_id,
            holder_eval_point=self.holder_eval_point,
            recipient_index=self.recipient_index,
            recipient_secret_key=self.recipient_secret_key,
            resharing_n=resharing_n,
            local_vps_shares=self.local_vps_shares,
        )


@dataclass(frozen=True)
class RU22DAGLocalShard:
    node_id: int
    manifest_digest: str
    source: RU22SourceRoleShard | None
    successor_recipient_secret_key: int | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "node_id", _integer(self.node_id, name="node_id")
        )
        object.__setattr__(
            self,
            "manifest_digest",
            _digest(self.manifest_digest, name="manifest_digest"),
        )
        if self.source is not None and not isinstance(
            self.source, RU22SourceRoleShard
        ):
            raise TypeError("source must be an RU22SourceRoleShard")
        if self.successor_recipient_secret_key is not None:
            object.__setattr__(
                self,
                "successor_recipient_secret_key",
                _scalar(
                    self.successor_recipient_secret_key,
                    name="successor recipient secret key",
                    nonzero=True,
                ),
            )

    def digest_material(self) -> dict[str, Any]:
        return {
            "kind": LOCAL_SHARD_KIND,
            "version": LOCAL_SHARD_VERSION,
            "node_id": self.node_id,
            "manifest_digest": self.manifest_digest,
            "source": self.source.to_obj() if self.source else None,
            "successor": (
                {
                    "recipient_secret_key": str(
                        self.successor_recipient_secret_key
                    )
                }
                if self.successor_recipient_secret_key is not None
                else None
            ),
        }

    def digest(self) -> str:
        return hashlib.sha256(
            _SHARD_DOMAIN + _canonical_json(self.digest_material())
        ).hexdigest()

    def to_obj(self) -> dict[str, Any]:
        obj = self.digest_material()
        obj["shard_digest"] = self.digest()
        return obj

    @classmethod
    def from_obj(
        cls,
        value: Mapping[str, Any],
        manifest: RU22DAGExperimentManifest,
    ) -> "RU22DAGLocalShard":
        _exact_fields(
            value,
            {
                "kind",
                "version",
                "node_id",
                "manifest_digest",
                "source",
                "successor",
                "shard_digest",
            },
            name="RU22 local shard",
        )
        if value["kind"] != LOCAL_SHARD_KIND or value["version"] != 1:
            raise ValueError("unsupported RU22 local shard")
        successor = value["successor"]
        if successor is not None:
            _exact_fields(
                successor,
                {"recipient_secret_key"},
                name="RU22 successor shard",
            )
        supplied = _digest(value["shard_digest"], name="shard_digest")
        shard = cls(
            node_id=value["node_id"],
            manifest_digest=value["manifest_digest"],
            source=(
                RU22SourceRoleShard.from_obj(value["source"])
                if value["source"] is not None
                else None
            ),
            successor_recipient_secret_key=(
                _scalar(
                    successor["recipient_secret_key"],
                    name="successor recipient secret key",
                    nonzero=True,
                )
                if successor is not None
                else None
            ),
        )
        if shard.digest() != supplied:
            raise ValueError("RU22 local shard digest mismatch")
        shard.validate(manifest)
        return shard

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        manifest: RU22DAGExperimentManifest,
    ) -> "RU22DAGLocalShard":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_obj(json.load(handle), manifest)

    def validate(self, manifest: RU22DAGExperimentManifest) -> None:
        if self.manifest_digest != manifest.digest():
            raise ValueError("local shard manifest digest mismatch")
        context = manifest.context
        if self.source is not None:
            source = self.source
            if manifest.source_role_nodes.get(source.holder_role_id) != self.node_id:
                raise ValueError("source role is assigned to another node")
            if context.holder_eval_points.get(
                source.holder_role_id
            ) != source.holder_eval_point:
                raise ValueError("source holder evaluation point mismatch")
            if context.source_role_indices.get(
                source.holder_role_id
            ) != source.recipient_index:
                raise ValueError("source recipient index mismatch")
            if derive_public_key(
                source.recipient_secret_key,
                context.source_key_base_hex,
            ) != context.source_rpk_list[source.recipient_index]:
                raise ValueError("source recipient secret key mismatch")
            catalog = {
                vps.vps_id: vps for vps in manifest.parent_vps_catalog
            }
            actual_view = tuple(
                item.vps.vps_id for item in source.local_vps_shares
            )
            if actual_view != manifest.source_vps_views[
                source.holder_role_id
            ]:
                raise ValueError("local VPS list does not match manifest view")
            for item in source.local_vps_shares:
                if catalog.get(item.vps.vps_id) != item.vps:
                    raise ValueError("local VPS is outside public catalog")
                decrypted = decrypt_elgamal_ciphertext(
                    source.recipient_secret_key,
                    item.vps.encrypted_commitments[
                        source.recipient_index
                    ].to_obj(),
                )
                claimed = pedersen_commit(
                    context.g_hex,
                    context.h_hex,
                    item.share_s,
                    item.share_s_hat,
                )
                if G1Point.from_hex(decrypted) != G1Point.from_hex(claimed):
                    raise ValueError("local share does not open parent VPS")
        successor_index = context.successor_role_indices.get(self.node_id)
        if successor_index is None:
            if self.successor_recipient_secret_key is not None:
                raise ValueError("non-successor shard contains a secret key")
        else:
            if self.successor_recipient_secret_key is None:
                raise ValueError("successor shard is missing its secret key")
            if derive_public_key(
                self.successor_recipient_secret_key,
                context.successor_key_base_hex,
            ) != context.successor_rpk_list[successor_index]:
                raise ValueError("successor recipient secret key mismatch")

    def holder_state(
        self, manifest: RU22DAGExperimentManifest
    ) -> RU22HolderState | None:
        self.validate(manifest)
        if self.source is None:
            return None
        return self.source.holder_state(manifest.context.resharing_n)


@dataclass(frozen=True)
class RU22DAGExperimentFixture:
    manifest: RU22DAGExperimentManifest
    shards: Sequence[RU22DAGLocalShard]
    maintained_secret: int
    maintained_blind: int

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, RU22DAGExperimentManifest):
            raise TypeError("manifest must be an RU22DAGExperimentManifest")
        shards = tuple(sorted(self.shards, key=lambda item: item.node_id))
        if len({shard.node_id for shard in shards}) != len(shards):
            raise ValueError("fixture contains duplicate node shards")
        expected_nodes = set(range(self.manifest.dag_m))
        if {shard.node_id for shard in shards} != expected_nodes:
            raise ValueError("fixture shard node set is incomplete")
        for shard in shards:
            shard.validate(self.manifest)
        secret = _scalar(
            self.maintained_secret,
            name="maintained_secret",
            nonzero=False,
        )
        blind = _scalar(
            self.maintained_blind,
            name="maintained_blind",
            nonzero=False,
        )
        expected_commitment = pedersen_commit(
            self.manifest.context.g_hex,
            self.manifest.context.h_hex,
            secret,
            blind,
        )
        if G1Point.from_hex(expected_commitment) != G1Point.from_hex(
            self.manifest.initial_secret_commitment
        ):
            raise ValueError("private fixture secret does not match manifest")
        object.__setattr__(self, "shards", shards)
        object.__setattr__(self, "maintained_secret", secret)
        object.__setattr__(self, "maintained_blind", blind)

    def shard_for(self, node_id: int) -> RU22DAGLocalShard:
        for shard in self.shards:
            if shard.node_id == int(node_id):
                return shard
        raise KeyError(f"node {node_id} is not in the RU22 fixture")


_MULTI_VPS_ALIAS_VIEWS = {
    0: ("common", "a", "b", "c", "d"),
    1: ("common", "b", "d", "e", "f"),
    2: ("common", "c", "e", "g", "h"),
    3: ("common", "f", "g", "h", "i"),
    4: ("common", "a", "e", "f", "i"),
}


def _multi_vps_alias_layout(
    n_r: int,
) -> tuple[tuple[str, ...], dict[int, tuple[str, ...]]]:
    if n_r == 5:
        aliases = (
            "common", "a", "b", "c", "d",
            "e", "f", "g", "h", "i",
        )
        return aliases, dict(_MULTI_VPS_ALIAS_VIEWS)

    aliases = ("common",) + tuple(
        f"vps-{index}" for index in range(1, 2 * n_r)
    )
    non_common = aliases[1:]
    width = n_r - 1
    views = {}
    for role in range(n_r):
        start = (2 * role) % len(non_common)
        views[role] = ("common",) + tuple(
            non_common[(start + offset) % len(non_common)]
            for offset in range(width)
        )
    return aliases, views


def _build_vps_fixture(
    *,
    config_path: str | Path,
    trial_id: str,
    trial_epoch: str,
    seed: str,
    n_r: int,
    t_r: int,
    dag_m: int,
    dag_t: int,
    fixture_mode: str,
    maintained_secret: int | None = None,
    maintained_blind: int | None = None,
    source_polynomial_coefficients: Sequence[int] | None = None,
    blind_polynomial_coefficients: Sequence[int] | None = None,
) -> RU22DAGExperimentFixture:
    n_r = _integer(n_r, name="n_r", minimum=1)
    t_r = _integer(t_r, name="t_r")
    if n_r != 2 * t_r + 1:
        raise ValueError("RU22 fixture requires n_r = 2*t_r + 1")
    if fixture_mode not in {
        FIXTURE_MODE_SINGLE_VPS,
        FIXTURE_MODE_WORST_CASE_MULTI_VPS,
    }:
        raise ValueError("unsupported RU22 fixture mode")
    seed = _text(seed, name="seed")
    g_hex, h_hex = _generators_from_config(config_path)
    source_roles = tuple(range(n_r))
    successor_nodes = tuple(range(n_r))
    source_base = (
        G1Point.from_hex(g_hex)
        * _derive_scalar(seed, "source-key-base")
    ).to_hex()
    successor_base = (
        G1Point.from_hex(g_hex)
        * _derive_scalar(seed, "successor-key-base")
    ).to_hex()
    source_rsks = tuple(
        _derive_scalar(seed, f"source-rsk:{role}")
        for role in source_roles
    )
    successor_rsks = tuple(
        _derive_scalar(seed, f"successor-rsk:{node}")
        for node in successor_nodes
    )
    context = RU22Context(
        source_epoch=1,
        target_epoch=2,
        sid=f"ru22-dag-{trial_id}",
        source_committee=f"ru22-source-{trial_epoch}",
        successor_committee=f"ru22-successor-{trial_epoch}",
        transfer_seq=0,
        g_hex=g_hex,
        h_hex=h_hex,
        source_key_base_hex=source_base,
        successor_key_base_hex=successor_base,
        source_rpk_list=tuple(
            derive_public_key(secret_key, source_base)
            for secret_key in source_rsks
        ),
        successor_rpk_list=tuple(
            derive_public_key(secret_key, successor_base)
            for secret_key in successor_rsks
        ),
        resharing_n=n_r,
        resharing_threshold=t_r,
        agreement_n=dag_m,
        agreement_threshold=dag_t,
        holder_eval_points={role: role + 1 for role in source_roles},
        source_role_indices={role: role for role in source_roles},
        successor_role_indices={node: node for node in successor_nodes},
    )
    secret = (
        _derive_scalar(seed, "maintained-secret", nonzero=False)
        if maintained_secret is None
        else int(maintained_secret) % FIELD_ORDER
    )
    blind = (
        _derive_scalar(seed, "maintained-blind", nonzero=False)
        if maintained_blind is None
        else int(maintained_blind) % FIELD_ORDER
    )
    if source_polynomial_coefficients is not None:
        source_polynomial_coefficients = tuple(
            int(value) % FIELD_ORDER
            for value in source_polynomial_coefficients
        )
        if (
            len(source_polynomial_coefficients) != t_r + 1
            or source_polynomial_coefficients[0] != secret
        ):
            raise ValueError(
                "source polynomial must contain t_r+1 coefficients "
                "with maintained_secret as its constant"
            )
    if blind_polynomial_coefficients is not None:
        blind_polynomial_coefficients = tuple(
            int(value) % FIELD_ORDER
            for value in blind_polynomial_coefficients
        )
        if (
            len(blind_polynomial_coefficients) != t_r + 1
            or blind_polynomial_coefficients[0] != blind
        ):
            raise ValueError(
                "blind polynomial must contain t_r+1 coefficients "
                "with maintained_blind as its constant"
            )
    candidate_count = (
        1 if fixture_mode == FIXTURE_MODE_SINGLE_VPS else 2 * n_r
    )
    candidates = []
    candidate_shares = {}
    for candidate_index in range(candidate_count):
        label = f"candidate:{candidate_index}"
        secret_poly = source_polynomial_coefficients or (
            (secret,)
            + tuple(
                _derive_scalar(
                    seed,
                    f"{label}:secret-coefficient:{index}",
                    nonzero=False,
                )
                for index in range(1, t_r + 1)
            )
        )
        blind_poly = blind_polynomial_coefficients or (
            (blind,)
            + tuple(
                _derive_scalar(
                    seed,
                    f"{label}:blind-coefficient:{index}",
                    nonzero=False,
                )
                for index in range(1, t_r + 1)
            )
        )
        share_pairs = tuple(
            (
                eval_poly(secret_poly, point),
                eval_poly(blind_poly, point),
            )
            for point in range(1, n_r + 1)
        )
        commitments = tuple(
            pedersen_commit(g_hex, h_hex, share, share_hat)
            for share, share_hat in share_pairs
        )
        parent = RU22VPS(
            epoch=context.source_epoch,
            committee=context.source_committee,
            committee_context_digest=hashlib.sha256(
                b"ATCaaS/RU22/source-state/v1\x00"
                + seed.encode("utf-8")
                + b"\x00"
                + label.encode("ascii")
            ).hexdigest(),
            encrypted_commitments=tuple(
                _encrypt_commitment(
                    source_base,
                    context.source_rpk_list[index],
                    commitments[index],
                    _derive_scalar(
                        seed, f"{label}:parent-encryption:{index}"
                    ),
                )
                for index in range(n_r)
            ),
            certificate=RU22InitialCertificate(
                hashlib.sha256(
                    b"ATCaaS/RU22/initial-certificate/v1\x00"
                    + seed.encode("utf-8")
                    + b"\x00"
                    + label.encode("ascii")
                ).hexdigest()
            ),
        )
        candidates.append(parent)
        candidate_shares[parent.vps_id] = share_pairs

    catalog = tuple(sorted(candidates, key=lambda item: item.vps_id))
    if fixture_mode == FIXTURE_MODE_SINGLE_VPS:
        alias_to_parent = {"common": catalog[0]}
        alias_views = {
            role: ("common",) for role in source_roles
        }
    else:
        aliases, alias_views = _multi_vps_alias_layout(n_r)
        alias_to_parent = dict(zip(aliases, catalog))
    source_vps_views = {
        role: tuple(
            alias_to_parent[alias].vps_id
            for alias in alias_views[role]
        )
        for role in source_roles
    }
    manifest = RU22DAGExperimentManifest(
        trial_id=trial_id,
        trial_epoch=trial_epoch,
        fixture_seed=seed,
        context=context,
        parent_vps_catalog=catalog,
        source_role_nodes={role: role for role in source_roles},
        source_vps_views=source_vps_views,
        expected_successor_nodes=successor_nodes,
        initial_secret_commitment=pedersen_commit(
            g_hex, h_hex, secret, blind
        ),
        dag_m=dag_m,
        dag_t=dag_t,
        fixture_mode=fixture_mode,
    )
    shards = tuple(
        RU22DAGLocalShard(
            node_id=node,
            manifest_digest=manifest.digest(),
            source=(
                RU22SourceRoleShard(
                    holder_role_id=node,
                    holder_eval_point=node + 1,
                    recipient_index=node,
                    recipient_secret_key=source_rsks[node],
                    local_vps_shares=tuple(
                        RU22LocalVPSShare(
                            vps=alias_to_parent[alias],
                            share_s=candidate_shares[
                                alias_to_parent[alias].vps_id
                            ][node][0],
                            share_s_hat=candidate_shares[
                                alias_to_parent[alias].vps_id
                            ][node][1],
                        )
                        for alias in alias_views[node]
                    ),
                )
                if node < n_r
                else None
            ),
            successor_recipient_secret_key=(
                successor_rsks[node] if node < n_r else None
            ),
        )
        for node in range(dag_m)
    )
    return RU22DAGExperimentFixture(
        manifest=manifest,
        shards=shards,
        maintained_secret=secret,
        maintained_blind=blind,
    )


def build_single_vps_fixture(
    *,
    config_path: str | Path,
    trial_id: str,
    trial_epoch: str,
    seed: str,
    n_r: int = 5,
    t_r: int = 2,
    dag_m: int = 5,
    dag_t: int = 1,
    maintained_secret: int | None = None,
    maintained_blind: int | None = None,
    source_polynomial_coefficients: Sequence[int] | None = None,
    blind_polynomial_coefficients: Sequence[int] | None = None,
) -> RU22DAGExperimentFixture:
    return _build_vps_fixture(
        config_path=config_path,
        trial_id=trial_id,
        trial_epoch=trial_epoch,
        seed=seed,
        n_r=n_r,
        t_r=t_r,
        dag_m=dag_m,
        dag_t=dag_t,
        fixture_mode=FIXTURE_MODE_SINGLE_VPS,
        maintained_secret=maintained_secret,
        maintained_blind=maintained_blind,
        source_polynomial_coefficients=source_polynomial_coefficients,
        blind_polynomial_coefficients=blind_polynomial_coefficients,
    )


def build_worst_case_multi_vps_fixture(
    *,
    config_path: str | Path,
    trial_id: str,
    trial_epoch: str,
    seed: str,
    n_r: int = 5,
    t_r: int = 2,
    dag_m: int = 5,
    dag_t: int = 1,
    maintained_secret: int | None = None,
    maintained_blind: int | None = None,
    source_polynomial_coefficients: Sequence[int] | None = None,
    blind_polynomial_coefficients: Sequence[int] | None = None,
) -> RU22DAGExperimentFixture:
    return _build_vps_fixture(
        config_path=config_path,
        trial_id=trial_id,
        trial_epoch=trial_epoch,
        seed=seed,
        n_r=n_r,
        t_r=t_r,
        dag_m=dag_m,
        dag_t=dag_t,
        fixture_mode=FIXTURE_MODE_WORST_CASE_MULTI_VPS,
        maintained_secret=maintained_secret,
        maintained_blind=maintained_blind,
        source_polynomial_coefficients=source_polynomial_coefficients,
        blind_polynomial_coefficients=blind_polynomial_coefficients,
    )


__all__ = [
    "FIXTURE_MODE_SINGLE_VPS",
    "FIXTURE_MODE_WORST_CASE_MULTI_VPS",
    "LOCAL_SHARD_KIND",
    "LOCAL_SHARD_VERSION",
    "MANIFEST_KIND",
    "MANIFEST_VERSION",
    "RU22DAGExperimentFixture",
    "RU22DAGExperimentManifest",
    "RU22DAGLocalShard",
    "RU22SourceRoleShard",
    "build_single_vps_fixture",
    "build_worst_case_multi_vps_fixture",
]
