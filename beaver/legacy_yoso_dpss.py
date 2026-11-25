"""Legacy transcript-only YOSO DPSS implementation.

This module implements the resharing-only protocol from an older paper draft:
holders broadcast a bare PVSS transcript and the justifier checks only PVSS and
the old-sharing anchor.  Although it shares resharing primitives with A²DPSS,
it does *not* implement the formal Algorithm-4 wire model, qid, finalizer, or
lifecycle.  Production Algorithm 4 uses :mod:`beaver.a2dpss`; operation-coupled
Algorithm 5 uses :mod:`beaver.ocdpss`.

Import this module explicitly only for historical smokes and downstream
compatibility.  A separate ``dpss`` qid namespace is intentionally preserved
so a transcript-only payload can never be dispatched to either the formal
``a2dpss`` or ``ocdpss`` justifier by mistake.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from beaver.chunk_elgamal import derive_public_key
from beaver.dag_tob import Justifier, PoolEntry, Qid
from beaver.pedersen_shamir import batch_pedersen_commit, eval_poly, sample_poly
from beaver.pvss import (
    PVSSTranscript,
    pvss_deal,
    pvss_deliver,
    pvss_interpolate,
    pvss_verify,
    pvss_verify_anchor,
)


LEGACY_DPSS_COMPATIBILITY = True
LEGACY_DPSS_QID_KIND = "dpss"
# Backward-compatible alias. New code should spell the legacy status explicitly.
DPSS_QID_KIND = LEGACY_DPSS_QID_KIND
DPSSContextKey = Tuple[int, str, str, str, int]


@dataclass(frozen=True)
class DPSSGenesisShares:
    """Local testing fixture for an already-existing old sharing."""

    secret: int
    secret_hat: int
    service_t: int
    holder_eval_points: Mapping[int, int]
    holder_shares: Mapping[int, Tuple[int, int]]
    old_commitment_vector: Sequence[str]

    def share_for(self, holder_id: int) -> Tuple[int, int]:
        return self.holder_shares[int(holder_id)]


@dataclass(frozen=True)
class DPSSResharingResult:
    qid: str
    mid: str
    transcript: PVSSTranscript


@dataclass(frozen=True)
class DPSSReconstructionResult:
    key: DPSSContextKey
    qid: str
    my_node_id: int
    my_role_idx: int
    share_s: int
    share_s_hat: int
    dealer_eval_points: Tuple[int, ...]


@dataclass(frozen=True)
class DPSSLocalConfig:
    ctx: DPSSContext
    source_committee: Tuple[int, ...]
    successor_committee: Tuple[int, ...]
    holder_shares: Mapping[int, Tuple[int, int]]
    my_holder_share: Optional[Tuple[int, int]]
    my_holder_eval_point: Optional[int]
    my_successor_rsk: Optional[int]
    my_successor_role_idx: Optional[int]

    @property
    def am_holder(self) -> bool:
        return self.my_holder_share is not None

    @property
    def am_successor(self) -> bool:
        return self.my_successor_rsk is not None


@dataclass(frozen=True)
class DPSSContext:
    """Static metadata needed to verify one DPSS transfer qid.

    ``old_commitment_vector`` is explicit in the first implementation.  Later,
    once DPSS entries are wired into the ledger state machine, it can be
    derived locally from the accepted previous sharing transcript.
    """

    epoch: int
    sid: str
    source_committee: str
    successor_committee: str
    transfer_seq: int
    g_hex: str
    h_hex: str
    successor_rpk_list: Sequence[str]
    service_t: int
    old_commitment_vector: Sequence[str]
    holder_eval_points: Mapping[int, int]
    successor_role_indices: Mapping[int, int]
    key_base: str = ""   # ElGamal key base for encrypting to successor rpks (P4 §5.1); empty -> g_hex

    def __post_init__(self) -> None:
        epoch = int(self.epoch)
        transfer_seq = int(self.transfer_seq)
        service_t = int(self.service_t)
        rpk_list = [str(item) for item in self.successor_rpk_list]
        old_vector = [str(item) for item in self.old_commitment_vector]
        holder_points = {
            int(node_id): int(eval_point)
            for node_id, eval_point in self.holder_eval_points.items()
        }
        role_indices = {
            int(node_id): int(role_idx)
            for node_id, role_idx in self.successor_role_indices.items()
        }

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if transfer_seq < 0:
            raise ValueError("transfer_seq must be non-negative")
        if not str(self.sid):
            raise ValueError("sid must be non-empty")
        if not str(self.source_committee):
            raise ValueError("source_committee must be non-empty")
        if not str(self.successor_committee):
            raise ValueError("successor_committee must be non-empty")
        if service_t < 0:
            raise ValueError("service_t must be non-negative")
        if len(rpk_list) < service_t + 1:
            raise ValueError("successor_rpk_list must contain at least service_t+1 keys")
        if not old_vector:
            raise ValueError("old_commitment_vector must be non-empty")
        if not holder_points:
            raise ValueError("holder_eval_points must be non-empty")
        if not role_indices:
            raise ValueError("successor_role_indices must be non-empty")

        max_old_idx = max(holder_points.values())
        if max_old_idx >= len(old_vector):
            raise ValueError("old_commitment_vector does not cover holder eval points")
        for node_id, role_idx in role_indices.items():
            if role_idx < 0 or role_idx >= len(rpk_list):
                raise ValueError(f"successor role index out of range for node {node_id}")

        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "sid", str(self.sid))
        object.__setattr__(self, "source_committee", str(self.source_committee))
        object.__setattr__(self, "successor_committee", str(self.successor_committee))
        object.__setattr__(self, "transfer_seq", transfer_seq)
        object.__setattr__(self, "g_hex", str(self.g_hex))
        object.__setattr__(self, "h_hex", str(self.h_hex))
        object.__setattr__(self, "successor_rpk_list", tuple(rpk_list))
        object.__setattr__(self, "service_t", service_t)
        object.__setattr__(self, "old_commitment_vector", tuple(old_vector))
        object.__setattr__(self, "holder_eval_points", dict(holder_points))
        object.__setattr__(self, "successor_role_indices", dict(role_indices))
        object.__setattr__(self, "key_base", str(self.key_base) or str(self.g_hex))

    @property
    def key(self) -> DPSSContextKey:
        return (
            self.epoch,
            self.sid,
            self.source_committee,
            self.successor_committee,
            self.transfer_seq,
        )

    def build_qid(self) -> Qid:
        return build_dpss_qid(
            epoch=self.epoch,
            sid=self.sid,
            source_committee=self.source_committee,
            successor_committee=self.successor_committee,
            transfer_seq=self.transfer_seq,
        )

    def encode_qid(self) -> str:
        return self.build_qid().encode()

    def to_obj(self) -> dict:
        return {
            "epoch": self.epoch,
            "sid": self.sid,
            "source_committee": self.source_committee,
            "successor_committee": self.successor_committee,
            "transfer_seq": self.transfer_seq,
            "g_hex": self.g_hex,
            "h_hex": self.h_hex,
            "key_base": self.key_base,
            "successor_rpk_list": list(self.successor_rpk_list),
            "service_t": self.service_t,
            "old_commitment_vector": list(self.old_commitment_vector),
            "holder_eval_points": {
                str(node_id): eval_point
                for node_id, eval_point in sorted(self.holder_eval_points.items())
            },
            "successor_role_indices": {
                str(node_id): role_idx
                for node_id, role_idx in sorted(self.successor_role_indices.items())
            },
        }

    @staticmethod
    def from_obj(data: Mapping) -> "DPSSContext":
        return DPSSContext(
            epoch=int(data["epoch"]),
            sid=str(data["sid"]),
            source_committee=str(data["source_committee"]),
            successor_committee=str(data["successor_committee"]),
            transfer_seq=int(data.get("transfer_seq", 0)),
            g_hex=str(data["g_hex"]),
            h_hex=str(data["h_hex"]),
            key_base=str(data.get("key_base", "")),
            successor_rpk_list=list(data["successor_rpk_list"]),
            service_t=int(data["service_t"]),
            old_commitment_vector=list(data["old_commitment_vector"]),
            holder_eval_points={
                int(node_id): int(eval_point)
                for node_id, eval_point in dict(data["holder_eval_points"]).items()
            },
            successor_role_indices={
                int(node_id): int(role_idx)
                for node_id, role_idx in dict(data["successor_role_indices"]).items()
            },
        )


def build_dpss_qid(
    epoch: int,
    sid: str,
    source_committee: str,
    successor_committee: str,
    transfer_seq: int = 0,
) -> Qid:
    """Build a legacy transcript-only DPSS qid.

    Formal Algorithm-4 contexts use ``build_a2dpss_qid``; Algorithm-5
    operation-coupled contexts use ``build_ocdpss_qid``.
    """

    return Qid.build(
        DPSS_QID_KIND,
        epoch=str(int(epoch)),
        sid=str(sid),
        source_committee=str(source_committee),
        successor_committee=str(successor_committee),
        transfer_seq=str(int(transfer_seq)),
    )


def encode_dpss_qid(
    epoch: int,
    sid: str,
    source_committee: str,
    successor_committee: str,
    transfer_seq: int = 0,
) -> str:
    return build_dpss_qid(
        epoch,
        sid,
        source_committee,
        successor_committee,
        transfer_seq,
    ).encode()


def is_legacy_dpss_qid(qid: Qid | str) -> bool:
    """Return whether *qid* belongs to the transcript-only legacy namespace."""

    if isinstance(qid, str):
        qid = Qid.decode(qid)
    return qid.kind == LEGACY_DPSS_QID_KIND


def is_dpss_qid(qid: Qid | str) -> bool:
    """Backward-compatible alias for :func:`is_legacy_dpss_qid`."""

    return is_legacy_dpss_qid(qid)


def dpss_context_key(qid: Qid | str) -> DPSSContextKey:
    """Extract the context lookup key from a DPSS qid."""

    if isinstance(qid, str):
        qid = Qid.decode(qid)
    if qid.kind != DPSS_QID_KIND:
        raise ValueError(f"expected qid kind {DPSS_QID_KIND!r}, got {qid.kind!r}")

    fields = dict(qid.fields)
    missing = [
        key
        for key in (
            "epoch",
            "sid",
            "source_committee",
            "successor_committee",
        )
        if key not in fields
    ]
    if missing:
        raise ValueError(f"missing DPSS qid fields: {', '.join(missing)}")

    return (
        int(fields["epoch"]),
        fields["sid"],
        fields["source_committee"],
        fields["successor_committee"],
        int(fields.get("transfer_seq", 0)),
    )


def dpss_qid_matches_context(qid: Qid | str, ctx: DPSSContext) -> bool:
    try:
        return dpss_context_key(qid) == ctx.key
    except ValueError:
        return False


def make_genesis_shares(
    *,
    g_hex: str,
    h_hex: str,
    secret: int,
    secret_hat: int,
    holder_ids: Sequence[int],
    service_t: int,
    holder_eval_points: Mapping[int, int] | None = None,
    phi_coeffs: Sequence[int] | None = None,
    phi_hat_coeffs: Sequence[int] | None = None,
) -> DPSSGenesisShares:
    """Create a local old-committee sharing for Step-3/4 tests.

    This is not a production key-generation protocol.  It models the state
    that ACSS/DKG will eventually provide: each old holder already has a share
    pair and the DPSS context knows the corresponding Pedersen commitments.
    """

    if service_t < 0:
        raise ValueError("service_t must be non-negative")
    holder_ids = [int(holder_id) for holder_id in holder_ids]
    if len(holder_ids) < service_t + 1:
        raise ValueError("need at least service_t+1 holders")
    if len(set(holder_ids)) != len(holder_ids):
        raise ValueError("holder_ids must be distinct")

    if holder_eval_points is None:
        eval_points = {
            holder_id: idx + 1
            for idx, holder_id in enumerate(holder_ids)
        }
    else:
        eval_points = {
            int(holder_id): int(eval_point)
            for holder_id, eval_point in holder_eval_points.items()
        }
        missing = [holder_id for holder_id in holder_ids if holder_id not in eval_points]
        if missing:
            raise ValueError(f"missing eval points for holders: {missing}")

    if any(eval_point <= 0 for eval_point in eval_points.values()):
        raise ValueError("holder eval points must be positive")
    if len(set(eval_points.values())) != len(eval_points):
        raise ValueError("holder eval points must be distinct")

    if phi_coeffs is None:
        phi = sample_poly(secret, service_t)
    else:
        phi = [int(coeff) for coeff in phi_coeffs]
        if len(phi) != service_t + 1:
            raise ValueError("phi_coeffs must have length service_t+1")
        if int(phi[0]) != int(secret):
            raise ValueError("phi_coeffs constant term must equal secret")
    if phi_hat_coeffs is None:
        phi_hat = sample_poly(secret_hat, service_t)
    else:
        phi_hat = [int(coeff) for coeff in phi_hat_coeffs]
        if len(phi_hat) != service_t + 1:
            raise ValueError("phi_hat_coeffs must have length service_t+1")
        if int(phi_hat[0]) != int(secret_hat):
            raise ValueError("phi_hat_coeffs constant term must equal secret_hat")
    shares = {
        holder_id: (
            eval_poly(phi, eval_points[holder_id]),
            eval_poly(phi_hat, eval_points[holder_id]),
        )
        for holder_id in holder_ids
    }

    max_eval_point = max(eval_points.values())
    commitment_values = [
        eval_poly(phi, eval_point)
        for eval_point in range(max_eval_point + 1)
    ]
    commitment_blinds = [
        eval_poly(phi_hat, eval_point)
        for eval_point in range(max_eval_point + 1)
    ]
    old_commitment_vector = batch_pedersen_commit(
        g_hex,
        h_hex,
        commitment_values,
        commitment_blinds,
    )

    return DPSSGenesisShares(
        secret=int(secret),
        secret_hat=int(secret_hat),
        service_t=int(service_t),
        holder_eval_points=eval_points,
        holder_shares=shares,
        old_commitment_vector=old_commitment_vector,
    )


def _as_int_tuple(values: Sequence) -> Tuple[int, ...]:
    return tuple(int(value) for value in values)


def _role_entries_from_extra(extra: Mapping) -> list[dict]:
    role_data = extra.get("dpss_role_keys", {})
    if isinstance(role_data, Mapping) and "role_keys" in role_data:
        return list(role_data["role_keys"])
    if "role_keys" in extra and isinstance(extra["role_keys"], Sequence):
        return list(extra["role_keys"])
    return []


def _default_dpss_role_keys(M: int, g_hex: str) -> list[dict]:
    return [
        {
            "rsk": str(17 + 12 * node_id),
            "rpk": derive_public_key(17 + 12 * node_id, g_hex),
        }
        for node_id in range(int(M))
    ]


def build_dpss_local_config(
    extra: Mapping,
    *,
    g_hex: str,
    h_hex: str,
    my_id: int,
    M: int,
    service_t: int,
    epoch: int | str = 0,
) -> DPSSLocalConfig:
    """Build Step-5 DPSS runtime state from config extras.

    The preferred config shape is ``extra["dpss_smoke"]`` plus
    ``extra["dpss_role_keys"]``.  If absent, a deterministic local smoke setup
    is derived so older configs can still exercise the DPSS glue path.
    """

    extra = dict(extra or {})
    cfg = dict(extra.get("dpss") or extra.get("dpss_smoke") or {})
    formal_node_local = (
        int(cfg.get("version", 0)) >= 2
        and cfg.get("private_material_scope") == "node-local"
    )
    if formal_node_local:
        forbidden = {
            "holder_shares",
            "secret",
            "secret_hat",
            "phi_coeffs",
            "phi_hat_coeffs",
            "role_keys",
        }
        leaked = sorted(forbidden.intersection(cfg))
        if leaked:
            raise ValueError(
                "formal node-local DPSS config contains shared private fields: "
                + ", ".join(leaked)
            )
        if "dpss_role_keys" in extra:
            raise ValueError(
                "formal node-local DPSS config must not contain dpss_role_keys"
            )
        role_entries = []
    else:
        role_entries = _role_entries_from_extra(extra) or list(
            cfg.get("role_keys", [])
        )
        if not role_entries:
            role_entries = _default_dpss_role_keys(M, g_hex)

    cfg_service_t = int(cfg.get("service_t", service_t))
    committee_size_override = cfg.get("committee_size")
    if committee_size_override is None:
        committee_size_override = os.environ.get(
            "DAG_TOB_DPSS_SERVICE_COMMITTEE_SIZE"
        )
    if committee_size_override is None:
        committee_size = min(int(M), max(cfg_service_t + 2, cfg_service_t + 1, 3))
    else:
        committee_size = int(committee_size_override)
        if committee_size < cfg_service_t + 1:
            raise ValueError("DPSS committee_size must be at least service_t+1")
        if committee_size > int(M):
            raise ValueError("DPSS committee_size cannot exceed M")
    source_committee = _as_int_tuple(
        cfg.get("source_committee", list(range(committee_size)))
    )
    successor_committee = _as_int_tuple(
        cfg.get("successor_committee", list(range(committee_size)))
    )
    if formal_node_local:
        configured_n_r = int(cfg.get("n_r", len(source_committee)))
        configured_t_r = int(cfg.get("t_r", cfg_service_t))
        if configured_t_r != cfg_service_t:
            raise ValueError("formal DPSS t_r/service_t mismatch")
        if (
            len(source_committee) != configured_n_r
            or len(successor_committee) != configured_n_r
        ):
            raise ValueError("formal DPSS committee size does not match n_r")
        if len(set(source_committee)) != configured_n_r or len(
            set(successor_committee)
        ) != configured_n_r:
            raise ValueError("formal DPSS committees must contain distinct nodes")
        if 2 * configured_t_r >= configured_n_r:
            raise ValueError("formal DPSS requires t_r < n_r/2")
    holder_eval_points = {
        int(node_id): int(eval_point)
        for node_id, eval_point in dict(
            cfg.get(
                "holder_eval_points",
                {
                    str(node_id): idx + 1
                    for idx, node_id in enumerate(source_committee)
                },
            )
        ).items()
    }
    successor_role_indices = {
        int(node_id): int(role_idx)
        for node_id, role_idx in dict(
            cfg.get(
                "successor_role_indices",
                {
                    str(node_id): idx
                    for idx, node_id in enumerate(successor_committee)
                },
            )
        ).items()
    }

    if formal_node_local:
        successor_rpk_list = [
            str(value) for value in cfg.get("successor_rpk_list", ())
        ]
        if len(successor_rpk_list) != len(successor_committee):
            raise ValueError(
                "formal DPSS config successor_rpk_list length mismatch"
            )
        local_successor = cfg.get("my_successor_role")
        role_rsks = {}
        if local_successor is not None:
            if not isinstance(local_successor, Mapping):
                raise ValueError("my_successor_role must be an object")
            if int(local_successor.get("node_id", -1)) != int(my_id):
                raise ValueError("my_successor_role node binding mismatch")
            if int(local_successor.get("role_idx", -1)) != int(
                successor_role_indices.get(int(my_id), -2)
            ):
                raise ValueError("my_successor_role index binding mismatch")
            if int(local_successor.get("epoch", -1)) != int(
                cfg.get("epoch", 0)
            ) or int(local_successor.get("transfer_seq", -1)) != int(
                cfg.get("transfer_seq", 0)
            ):
                raise ValueError("my_successor_role epoch/transfer binding mismatch")
            role_rsks[int(my_id)] = int(local_successor["rsk"])
            expected_rpk = derive_public_key(
                role_rsks[int(my_id)], g_hex
            )
            role_idx = successor_role_indices[int(my_id)]
            def _point_coordinates(value):
                x_hex, y_hex = str(value).strip("()").split(",")
                return int(x_hex, 16), int(y_hex, 16)

            if _point_coordinates(expected_rpk) != _point_coordinates(
                successor_rpk_list[role_idx]
            ):
                raise ValueError("my_successor_role private/public key mismatch")
        elif int(my_id) in successor_role_indices:
            raise ValueError("successor node is missing my_successor_role")
    else:
        successor_rpk_list = [
            str(role_entries[node_id]["rpk"])
            for node_id in successor_committee
        ]
        role_rsks = {
            int(node_id): int(role_entries[int(node_id)]["rsk"])
            for node_id in range(min(len(role_entries), int(M)))
        }

    if formal_node_local:
        if "old_commitment_vector" not in cfg:
            raise ValueError("formal DPSS config requires old_commitment_vector")
        holder_shares = {}
        local_share = cfg.get("my_holder_share")
        if local_share is not None:
            if not isinstance(local_share, Mapping):
                raise ValueError("my_holder_share must be an object")
            if int(local_share.get("node_id", -1)) != int(my_id):
                raise ValueError("my_holder_share node binding mismatch")
            if int(my_id) not in source_committee:
                raise ValueError("non-holder config contains my_holder_share")
            if int(local_share.get("eval_point", -1)) != int(
                holder_eval_points[int(my_id)]
            ):
                raise ValueError("my_holder_share evaluation-point mismatch")
            holder_shares[int(my_id)] = (
                int(local_share["s"]),
                int(local_share["s_hat"]),
            )
        elif int(my_id) in source_committee:
            raise ValueError("source holder is missing my_holder_share")
        old_commitment_vector = list(cfg["old_commitment_vector"])
    elif "holder_shares" in cfg and "old_commitment_vector" in cfg:
        holder_shares = {
            int(node_id): (
                int(share["s"]),
                int(share["s_hat"]),
            )
            for node_id, share in dict(cfg["holder_shares"]).items()
        }
        old_commitment_vector = list(cfg["old_commitment_vector"])
    else:
        secret = int(cfg.get("secret", 42))
        secret_hat = int(cfg.get("secret_hat", 7))
        phi_coeffs = cfg.get(
            "phi_coeffs",
            [secret] + [1000 + idx for idx in range(cfg_service_t)],
        )
        phi_hat_coeffs = cfg.get(
            "phi_hat_coeffs",
            [secret_hat] + [2000 + idx for idx in range(cfg_service_t)],
        )
        genesis = make_genesis_shares(
            g_hex=g_hex,
            h_hex=h_hex,
            secret=secret,
            secret_hat=secret_hat,
            holder_ids=source_committee,
            service_t=cfg_service_t,
            holder_eval_points=holder_eval_points,
            phi_coeffs=phi_coeffs,
            phi_hat_coeffs=phi_hat_coeffs,
        )
        holder_shares = dict(genesis.holder_shares)
        old_commitment_vector = list(genesis.old_commitment_vector)

    epoch_value = cfg.get("epoch", epoch)
    try:
        epoch_value = int(epoch_value)
    except (TypeError, ValueError):
        epoch_value = 0

    ctx = DPSSContext(
        epoch=epoch_value,
        sid=str(cfg.get("sid", "dpss-smoke")),
        source_committee=str(cfg.get("source_committee_id", "source")),
        successor_committee=str(cfg.get("successor_committee_id", "successor")),
        transfer_seq=int(cfg.get("transfer_seq", 0)),
        g_hex=g_hex,
        h_hex=h_hex,
        successor_rpk_list=successor_rpk_list,
        service_t=cfg_service_t,
        old_commitment_vector=old_commitment_vector,
        holder_eval_points=holder_eval_points,
        successor_role_indices=successor_role_indices,
    )

    my_id = int(my_id)
    my_holder_share = holder_shares.get(my_id)
    return DPSSLocalConfig(
        ctx=ctx,
        source_committee=source_committee,
        successor_committee=successor_committee,
        holder_shares=dict(holder_shares),
        my_holder_share=my_holder_share,
        my_holder_eval_point=holder_eval_points.get(my_id)
        if my_holder_share is not None
        else None,
        my_successor_rsk=role_rsks.get(my_id)
        if my_id in successor_role_indices
        else None,
        my_successor_role_idx=successor_role_indices.get(my_id),
    )


def _wipe_erasable_state(erasable_state: Sequence | None) -> None:
    for buf in erasable_state or ():
        try:
            for idx in range(len(buf)):
                buf[idx] = 0
        except TypeError:
            continue


def make_resharing_transcript(
    *,
    g_hex: str,
    h_hex: str,
    my_share_s: int,
    my_share_s_hat: int,
    successor_rpk_list: Sequence[str],
    dealer_eval_point: int,
    service_t: int,
    my_id: int,
    key_base_hex: str | None = None,
) -> PVSSTranscript:
    """Create one old-holder PVSS transcript without enqueueing it."""

    return pvss_deal(
        g_hex,
        h_hex,
        successor_rpk_list,
        my_share_s,
        my_share_s_hat,
        service_t,
        dealer_id=my_id,
        dealer_eval_point=dealer_eval_point,
        key_base_hex=key_base_hex,
    )


async def run_resharing(
    *,
    dag_tob,
    g_hex: str,
    h_hex: str,
    my_share_s: int,
    my_share_s_hat: int,
    successor_rpk_list: Sequence[str],
    dealer_eval_point: int,
    service_t: int,
    epoch: int,
    sid: str,
    source_committee: str,
    successor_committee: str,
    transfer_seq: int,
    my_id: int,
    key_base_hex: str | None = None,
    erasable_state: Sequence | None = None,
) -> DPSSResharingResult:
    """Run one old-holder resharing action and enqueue its PVSS transcript."""

    try:
        transcript = make_resharing_transcript(
            g_hex=g_hex,
            h_hex=h_hex,
            my_share_s=my_share_s,
            my_share_s_hat=my_share_s_hat,
            successor_rpk_list=successor_rpk_list,
            dealer_eval_point=dealer_eval_point,
            service_t=service_t,
            my_id=my_id,
            key_base_hex=key_base_hex,
        )
        qid = encode_dpss_qid(
            epoch=epoch,
            sid=sid,
            source_committee=source_committee,
            successor_committee=successor_committee,
            transfer_seq=transfer_seq,
        )
        mid = f"node_{int(my_id)}"
        dag_tob.t_broadcast(qid, mid, transcript.to_bytes())
        return DPSSResharingResult(qid=qid, mid=mid, transcript=transcript)
    finally:
        _wipe_erasable_state(erasable_state)


def _stable_entry(entry: PoolEntry) -> Tuple[str, str]:
    mid, payload = entry
    return str(mid), bytes(payload).hex()


def _payload_to_transcript(payload: bytes) -> PVSSTranscript:
    """Decode a PVSS transcript from a TOB payload.

    Step 2.3 uses raw ``PVSSTranscript.to_bytes()`` payloads.  The small
    envelope fallback keeps the manager compatible with a later ``TobMessage``
    wrapper without changing the justifier surface.
    """

    payload = bytes(payload)
    try:
        return PVSSTranscript.from_bytes(payload)
    except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError, ValueError):
        pass

    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("unsupported DPSS payload")

    payload_obj = data.get("payload", data)
    if not isinstance(payload_obj, Mapping):
        raise ValueError("unsupported DPSS payload envelope")

    transcript = payload_obj.get("transcript")
    if transcript is None:
        transcript = payload_obj.get("pvss_transcript")
    if transcript is None:
        transcript_b64 = payload_obj.get("transcript_b64")
        if transcript_b64 is not None:
            try:
                transcript = base64.b64decode(str(transcript_b64))
            except binascii.Error as exc:
                raise ValueError("invalid transcript_b64") from exc
    if transcript is None:
        transcript_hex = payload_obj.get("transcript_hex")
        if transcript_hex is not None:
            transcript = bytes.fromhex(str(transcript_hex))
    if transcript is None:
        raise ValueError("missing PVSS transcript in DPSS payload")

    if isinstance(transcript, Mapping):
        transcript = json.dumps(
            transcript,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    elif isinstance(transcript, str):
        transcript = transcript.encode("utf-8")
    return PVSSTranscript.from_bytes(transcript)


class DPSSJustifierManager(Justifier):
    """Legacy context-aware justifier for bare PVSS transcript sets.

    DAG-TOB dispatches by ``qid.kind == "dpss"``.  This manager then uses the
    structured qid fields to find the transfer-specific ``DPSSContext``.  It is
    deliberately not registered for the current ``ocdpss`` namespace.
    """

    def __init__(self, contexts: Sequence[DPSSContext] | None = None):
        self.contexts: Dict[DPSSContextKey, DPSSContext] = {}
        for ctx in contexts or ():
            self.register_context(ctx)

    def register_context(self, ctx: DPSSContext) -> None:
        self.contexts[ctx.key] = ctx

    def unregister_context(self, ctx_or_key: DPSSContext | DPSSContextKey) -> None:
        key = ctx_or_key.key if isinstance(ctx_or_key, DPSSContext) else ctx_or_key
        self.contexts.pop(key, None)

    def get_context(self, qid: Qid | str) -> DPSSContext:
        key = dpss_context_key(qid)
        try:
            return self.contexts[key]
        except KeyError as exc:
            raise KeyError(f"unknown DPSS context: {key}") from exc

    def _transcript_is_valid(
        self,
        ctx: DPSSContext,
        transcript: PVSSTranscript,
        seen_eval_points: Optional[Set[int]] = None,
    ) -> bool:
        dealer_id = int(transcript.dealer_id)
        dealer_eval_point = int(transcript.dealer_eval_point)
        if dealer_id not in ctx.holder_eval_points:
            return False
        if dealer_eval_point != ctx.holder_eval_points[dealer_id]:
            return False
        if seen_eval_points is not None and dealer_eval_point in seen_eval_points:
            return False
        if not pvss_verify(
            ctx.g_hex,
            ctx.h_hex,
            ctx.successor_rpk_list,
            transcript,
            ctx.service_t,
            key_base_hex=ctx.key_base,
        ):
            return False
        if not pvss_verify_anchor(transcript, ctx.old_commitment_vector):
            return False
        return True

    def select_justified_subset(
        self,
        qid: Qid,
        entries: Sequence[PoolEntry],
    ) -> Optional[List[PoolEntry]]:
        try:
            ctx = self.get_context(qid)
        except (KeyError, ValueError):
            return None

        selected: List[PoolEntry] = []
        seen_eval_points: Set[int] = set()
        for mid, payload in sorted(entries, key=_stable_entry):
            try:
                transcript = _payload_to_transcript(payload)
                if not self._transcript_is_valid(ctx, transcript, seen_eval_points):
                    continue
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                UnicodeDecodeError,
                ValueError,
            ):
                continue

            selected.append((str(mid), bytes(payload)))
            seen_eval_points.add(int(transcript.dealer_eval_point))
            if len(selected) >= ctx.service_t + 1:
                return selected
        return None

    def verify_qid_set(
        self,
        qid: Qid,
        entries: Sequence[PoolEntry],
    ) -> bool:
        try:
            ctx = self.get_context(qid)
        except (KeyError, ValueError):
            return False
        if len(entries) < ctx.service_t + 1:
            return False

        seen_eval_points: Set[int] = set()
        for _mid, payload in entries:
            try:
                transcript = _payload_to_transcript(payload)
                if not self._transcript_is_valid(ctx, transcript, seen_eval_points):
                    return False
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                UnicodeDecodeError,
                ValueError,
            ):
                return False
            seen_eval_points.add(int(transcript.dealer_eval_point))
        return True


class DPSSReconstructionManager:
    """Recover local successor shares from delivered DPSS blocks."""

    def __init__(self):
        self.contexts: Dict[DPSSContextKey, DPSSContext] = {}
        self.local_successor_state: Dict[DPSSContextKey, dict] = {}
        self.completed: Set[DPSSContextKey] = set()
        self.results: Dict[DPSSContextKey, DPSSReconstructionResult] = {}

    def register_context(
        self,
        ctx: DPSSContext,
        *,
        my_node_id: int,
        my_rsk: int,
        on_new_share=None,
    ) -> None:
        my_node_id = int(my_node_id)
        if my_node_id not in ctx.successor_role_indices:
            raise ValueError(f"node {my_node_id} is not in successor_role_indices")
        my_role_idx = int(ctx.successor_role_indices[my_node_id])
        self.contexts[ctx.key] = ctx
        self.local_successor_state[ctx.key] = {
            "my_node_id": my_node_id,
            "my_rsk": int(my_rsk),
            "my_role_idx": my_role_idx,
            "on_new_share": on_new_share,
        }

    def unregister_context(self, ctx_or_key: DPSSContext | DPSSContextKey) -> None:
        key = ctx_or_key.key if isinstance(ctx_or_key, DPSSContext) else ctx_or_key
        self.contexts.pop(key, None)
        self.local_successor_state.pop(key, None)
        self.completed.discard(key)
        self.results.pop(key, None)

    def on_deliver(
        self,
        qid_raw: str,
        entries: Sequence[PoolEntry],
    ) -> List[DPSSReconstructionResult]:
        """Process one per-query TOB delivery and reconstruct a local share."""

        reconstructed: List[DPSSReconstructionResult] = []
        qid = Qid.decode(qid_raw)
        if qid.kind != DPSS_QID_KIND:
            return reconstructed
        try:
            key = dpss_context_key(qid)
        except ValueError:
            return reconstructed
        if key in self.completed:
            return reconstructed

        ctx = self.contexts.get(key)
        local = self.local_successor_state.get(key)
        if ctx is None or local is None:
            return reconstructed

        share_pairs: List[Tuple[int, int]] = []
        eval_points: List[int] = []
        seen_eval_points: Set[int] = set()
        for _mid, payload in sorted(entries, key=_stable_entry):
            try:
                transcript = _payload_to_transcript(payload)
                if not DPSSJustifierManager()._transcript_is_valid(
                    ctx,
                    transcript,
                    seen_eval_points,
                ):
                    continue
                share_pair = pvss_deliver(
                    ctx.g_hex,
                    ctx.h_hex,
                    transcript,
                    local["my_rsk"],
                    local["my_role_idx"],
                )
            except (
                IndexError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                UnicodeDecodeError,
                ValueError,
            ):
                continue

            share_pairs.append(share_pair)
            eval_points.append(int(transcript.dealer_eval_point))
            seen_eval_points.add(int(transcript.dealer_eval_point))
            if len(share_pairs) >= ctx.service_t + 1:
                break

        if len(share_pairs) < ctx.service_t + 1:
            return reconstructed

        share_s, share_s_hat = pvss_interpolate(
            eval_points,
            share_pairs,
            ctx.service_t,
        )
        result = DPSSReconstructionResult(
            key=key,
            qid=str(qid_raw),
            my_node_id=local["my_node_id"],
            my_role_idx=local["my_role_idx"],
            share_s=share_s,
            share_s_hat=share_s_hat,
            dealer_eval_points=tuple(eval_points[: ctx.service_t + 1]),
        )
        self.completed.add(key)
        self.results[key] = result
        callback = local.get("on_new_share")
        if callback is not None:
            callback(share_s, share_s_hat)
        reconstructed.append(result)
        return reconstructed
