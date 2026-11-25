"""Immutable A²DPSS fixtures for paired DAG/ACS TOB experiments."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from beaver.a2dpss import (
    A2DPSSContext,
    A2DPSSContribution,
    A2DPSSHolderMetrics,
    a2dpss_contribution_digest,
    make_a2dpss_contribution,
)
from beaver.handover_fixtures import build_handover_local_config


FIXTURE_KIND = "atcaas-a2dpss-paired-fixture"
FIXTURE_VERSION = 1
FIXTURE_PROTOCOL = "a2dpss"
FIXTURE_ALGORITHM = 4
_FIXTURE_DIGEST_DOMAIN = b"ATCaaS/A2DPSS/paired-fixture/v1\x00"
_INITIAL_SHARING_DIGEST_DOMAIN = b"ATCaaS/A2DPSS/initial-sharing/v1\x00"
_ROLE_KEY_DIGEST_DOMAIN = b"ATCaaS/A2DPSS/successor-role-keys/v1\x00"


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _raw_g1(point: Mapping) -> str:
    return (
        b"\x04"
        + int(point["X"]).to_bytes(48, byteorder="big")
        + int(point["Y"]).to_bytes(48, byteorder="big")
    ).hex()


def g_h_from_config_extra(extra: Mapping) -> tuple[str, str]:
    srs = json.loads(base64.b64decode(extra["SRS"]).decode("utf-8"))
    return _raw_g1(srs["Pk"]["G1_g"][0]), _raw_g1(srs["Pk"]["G1_h"][0])


@dataclass(frozen=True)
class A2DPSSFixtureEntry:
    sender: int
    mid: str
    payload: bytes
    holder_metrics: Mapping[str, float | int]

    def to_obj(self) -> dict:
        return {
            "sender": self.sender,
            "mid": self.mid,
            "payload_hex": self.payload.hex(),
            "holder_metrics": dict(self.holder_metrics),
        }


class A2DPSSPairedFixture:
    """One public contribution set replayed by both ordering protocols."""

    def __init__(
        self,
        *,
        trial_id: str,
        trial_epoch: str,
        context: A2DPSSContext,
        initial_sharing_digest: str,
        successor_role_key_digest: str,
        entries: tuple[A2DPSSFixtureEntry, ...],
    ) -> None:
        if not isinstance(trial_id, str) or not trial_id:
            raise ValueError("trial_id must be a non-empty string")
        if not isinstance(trial_epoch, str) or not trial_epoch:
            raise ValueError("trial_epoch must be a non-empty string")
        if not isinstance(context, A2DPSSContext):
            raise TypeError("context must be an A2DPSSContext")
        for name, digest in (
            ("initial_sharing_digest", initial_sharing_digest),
            ("successor_role_key_digest", successor_role_key_digest),
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or digest.lower() != digest
            ):
                raise ValueError(f"{name} must be lowercase sha256 hex")
            bytes.fromhex(digest)

        expected_senders = set(context.holder_eval_points)
        actual_senders = {entry.sender for entry in entries}
        if actual_senders != expected_senders or len(entries) != len(actual_senders):
            raise ValueError("fixture sender set does not match source committee")
        payload_digests = set()
        metric_fields = {
            "pvss_seconds",
            "pvss_cpu_seconds",
            "serialization_seconds",
            "contribution_bytes",
            "pvss_bytes",
        }
        for entry in entries:
            if isinstance(entry.sender, bool) or not isinstance(entry.sender, int):
                raise ValueError("fixture sender must be an integer")
            if not isinstance(entry.mid, str):
                raise ValueError("fixture mid must be a string")
            if not isinstance(entry.payload, bytes):
                raise ValueError("fixture payload must be bytes")
            if entry.mid != f"node_{entry.sender}":
                raise ValueError("fixture mid does not bind holder identity")
            contribution = A2DPSSContribution.from_bytes(entry.payload)
            if contribution.context_digest != context.context_digest:
                raise ValueError("fixture contribution/context digest mismatch")
            if contribution.holder_id != entry.sender:
                raise ValueError("fixture contribution/sender mismatch")
            if (
                context.holder_eval_points.get(entry.sender)
                != contribution.holder_eval_point
            ):
                raise ValueError("fixture contribution evaluation point mismatch")
            digest = a2dpss_contribution_digest(entry.payload)
            if digest in payload_digests:
                raise ValueError("fixture contains duplicate contribution bytes")
            payload_digests.add(digest)
            if set(entry.holder_metrics) != metric_fields:
                raise ValueError("fixture holder metric fields are not canonical")
            for field, raw in entry.holder_metrics.items():
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    raise ValueError(f"fixture holder metric {field} is invalid")
                if not math.isfinite(float(raw)) or float(raw) < 0.0:
                    raise ValueError(f"fixture holder metric {field} is invalid")
            for field in ("contribution_bytes", "pvss_bytes"):
                if not isinstance(entry.holder_metrics[field], int):
                    raise ValueError(f"fixture holder metric {field} must be int")
            if entry.holder_metrics["contribution_bytes"] != len(entry.payload):
                raise ValueError("fixture contribution byte metric mismatch")

        self.trial_id = trial_id
        self.trial_epoch = trial_epoch
        self.context = context
        self.initial_sharing_digest = initial_sharing_digest
        self.successor_role_key_digest = successor_role_key_digest
        self.entries = tuple(sorted(entries, key=lambda entry: entry.sender))
        self._entries_by_sender = {
            entry.sender: entry for entry in self.entries
        }

    @classmethod
    def from_obj(cls, value: Mapping) -> "A2DPSSPairedFixture":
        if not isinstance(value, Mapping):
            raise ValueError("A2DPSS paired fixture must be an object")
        expected = {
            "kind",
            "version",
            "protocol",
            "algorithm",
            "trial_id",
            "trial_epoch",
            "context",
            "initial_sharing_digest",
            "successor_role_key_digest",
            "entries",
        }
        if set(value) != expected:
            raise ValueError("A2DPSS paired fixture fields are not canonical")
        if value["kind"] != FIXTURE_KIND or value["version"] != FIXTURE_VERSION:
            raise ValueError("unsupported A2DPSS paired fixture kind/version")
        if (
            value["protocol"] != FIXTURE_PROTOCOL
            or value["algorithm"] != FIXTURE_ALGORITHM
        ):
            raise ValueError("A2DPSS paired fixture protocol metadata mismatch")
        entries = []
        for raw in value["entries"]:
            if set(raw) != {"sender", "mid", "payload_hex", "holder_metrics"}:
                raise ValueError("A2DPSS fixture entry fields are not canonical")
            if isinstance(raw["sender"], bool) or not isinstance(
                raw["sender"], int
            ):
                raise ValueError("fixture sender must be an integer")
            if not isinstance(raw["payload_hex"], str):
                raise ValueError("fixture payload_hex must be a string")
            payload = bytes.fromhex(raw["payload_hex"])
            if payload.hex() != raw["payload_hex"]:
                raise ValueError("fixture payload must use canonical lowercase hex")
            entries.append(
                A2DPSSFixtureEntry(
                    sender=raw["sender"],
                    mid=str(raw["mid"]),
                    payload=payload,
                    holder_metrics=dict(raw["holder_metrics"]),
                )
            )
        return cls(
            trial_id=str(value["trial_id"]),
            trial_epoch=str(value["trial_epoch"]),
            context=A2DPSSContext.from_obj(value["context"]),
            initial_sharing_digest=str(value["initial_sharing_digest"]),
            successor_role_key_digest=str(value["successor_role_key_digest"]),
            entries=tuple(entries),
        )

    @classmethod
    def from_file(cls, path: str) -> "A2DPSSPairedFixture":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_obj(json.load(handle))

    def to_fixture_obj(self) -> dict:
        return {
            "kind": FIXTURE_KIND,
            "version": FIXTURE_VERSION,
            "protocol": FIXTURE_PROTOCOL,
            "algorithm": FIXTURE_ALGORITHM,
            "trial_id": self.trial_id,
            "trial_epoch": self.trial_epoch,
            "context": self.context.to_obj(),
            "initial_sharing_digest": self.initial_sharing_digest,
            "successor_role_key_digest": self.successor_role_key_digest,
            "entries": [entry.to_obj() for entry in self.entries],
        }

    def digest(self) -> str:
        return hashlib.sha256(
            _FIXTURE_DIGEST_DOMAIN + _canonical_json(self.to_fixture_obj())
        ).hexdigest()

    def qid(self) -> str:
        return self.context.encode_qid()

    def source_holder_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.context.holder_eval_points))

    def successor_node_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.context.successor_role_indices))

    def entry_for_sender(self, sender: int) -> A2DPSSFixtureEntry:
        try:
            return self._entries_by_sender[int(sender)]
        except KeyError as exc:
            raise KeyError(f"node {sender} is not a fixture holder") from exc

    def public_entries(self) -> list[tuple[str, bytes]]:
        return [(entry.mid, entry.payload) for entry in self.entries]

    def contribution_digests(self) -> tuple[str, ...]:
        return tuple(
            a2dpss_contribution_digest(entry.payload)
            for entry in self.entries
        )

    def holder_metrics_summary(self) -> dict:
        entries = [
            {"sender": entry.sender, **dict(entry.holder_metrics)}
            for entry in self.entries
        ]
        fields = (
            "pvss_seconds",
            "pvss_cpu_seconds",
            "serialization_seconds",
            "contribution_bytes",
            "pvss_bytes",
        )
        return {
            "entry_count": len(entries),
            "totals": {
                field: sum(entry[field] for entry in entries)
                for field in fields
            },
            "entries": entries,
        }


def build_paired_fixture(
    *,
    config_path: str,
    trial_id: str,
    trial_epoch: str,
) -> A2DPSSPairedFixture:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    extra = config["extra"]
    g_hex, h_hex = g_h_from_config_extra(extra)
    M = int(config.get("M", config["N"]))
    service_t = int(config.get("t_r", config["t"]))
    local = build_handover_local_config(
        extra,
        g_hex=g_hex,
        h_hex=h_hex,
        my_id=0,
        M=M,
        service_t=service_t,
        epoch=0,
    )
    legacy = local.ctx
    context = A2DPSSContext(
        epoch=legacy.epoch,
        sid=f"a2dpss-paired-{trial_id}",
        source_committee=legacy.source_committee,
        successor_committee=legacy.successor_committee,
        transfer_seq=0,
        g_hex=legacy.g_hex,
        h_hex=legacy.h_hex,
        key_base=legacy.key_base,
        successor_rpk_list=legacy.successor_rpk_list,
        service_t=legacy.service_t,
        old_commitment_vector=legacy.old_commitment_vector,
        holder_eval_points=legacy.holder_eval_points,
        successor_role_indices=legacy.successor_role_indices,
    )
    holder_shares = dict(local.holder_shares)
    missing_holders = set(context.holder_eval_points).difference(holder_shares)
    if missing_holders:
        config_dir = Path(config_path).resolve().parent
        for holder_id in sorted(missing_holders):
            holder_path = config_dir / f"local.{holder_id}.json"
            if not holder_path.is_file():
                raise ValueError(
                    f"missing node-local holder config {holder_path}"
                )
            with holder_path.open("r", encoding="utf-8") as handle:
                holder_config = json.load(handle)
            holder_local = build_handover_local_config(
                holder_config["extra"],
                g_hex=g_hex,
                h_hex=h_hex,
                my_id=holder_id,
                M=M,
                service_t=service_t,
                epoch=0,
            )
            if holder_local.my_holder_share is None:
                raise ValueError(
                    f"node-local holder {holder_id} has no private share"
                )
            holder_shares[holder_id] = holder_local.my_holder_share

    initial_sharing_digest = hashlib.sha256(
        _INITIAL_SHARING_DIGEST_DOMAIN
        + _canonical_json(
            {
                "holder_shares": {
                    str(holder): [str(value) for value in shares]
                    for holder, shares in sorted(holder_shares.items())
                },
                "old_commitment_vector": list(context.old_commitment_vector),
            }
        )
    ).hexdigest()
    successor_role_key_digest = hashlib.sha256(
        _ROLE_KEY_DIGEST_DOMAIN
        + _canonical_json(
            {
                "successor_rpk_list": list(context.successor_rpk_list),
                "successor_role_indices": {
                    str(node): role
                    for node, role in sorted(
                        context.successor_role_indices.items()
                    )
                },
            }
        )
    ).hexdigest()
    entries = []
    for holder_id in sorted(context.holder_eval_points):
        share_s, share_s_hat = holder_shares[holder_id]
        metrics = A2DPSSHolderMetrics()
        contribution = make_a2dpss_contribution(
            context=context,
            holder_id=holder_id,
            share_s=share_s,
            share_s_hat=share_s_hat,
            metrics=metrics,
        )
        started = time.perf_counter()
        payload = contribution.to_bytes()
        metrics.serialization_seconds = time.perf_counter() - started
        metrics.contribution_bytes = len(payload)
        entries.append(
            A2DPSSFixtureEntry(
                sender=holder_id,
                mid=f"node_{holder_id}",
                payload=payload,
                holder_metrics={
                    "pvss_seconds": metrics.pvss_seconds,
                    "pvss_cpu_seconds": metrics.pvss_cpu_seconds,
                    "serialization_seconds": metrics.serialization_seconds,
                    "contribution_bytes": metrics.contribution_bytes,
                    "pvss_bytes": len(contribution.transcript.to_bytes()),
                },
            )
        )
    return A2DPSSPairedFixture(
        trial_id=trial_id,
        trial_epoch=trial_epoch,
        context=context,
        initial_sharing_digest=initial_sharing_digest,
        successor_role_key_digest=successor_role_key_digest,
        entries=tuple(entries),
    )


__all__ = [
    "FIXTURE_KIND",
    "FIXTURE_VERSION",
    "FIXTURE_PROTOCOL",
    "FIXTURE_ALGORITHM",
    "A2DPSSFixtureEntry",
    "A2DPSSPairedFixture",
    "build_paired_fixture",
    "g_h_from_config_extra",
]
