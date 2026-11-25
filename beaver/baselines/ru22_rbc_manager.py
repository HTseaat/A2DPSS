"""YOSO-RBC transport for one-shot RU22 contribution batches."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from beaver.baselines.ru22_dpss import (
    RU22Context,
    RU22ContributionBatch,
    validate_ru22_contribution_batch_shape,
)
from beaver.dag_tob import DagTOB
from beaver.yosorbc import YosoRBC


RU22_BATCH_RBC_VERSION = 1
_RU22_BATCH_RBC_KIND = "atcaas-ru22-contribution-batch-rbc"


def _context_tag(context: RU22Context) -> str:
    return context.context_digest[:16]


def ru22_batch_rbc_prefix(
    epoch: str,
    context: RU22Context,
    holder_role_id: int,
) -> str:
    return (
        f"RU22:{epoch}:CONTRIBUTION_RBC:"
        f"{_context_tag(context)}:{int(holder_role_id)}"
    )


def ru22_batch_echo_label(
    epoch: str,
    context: RU22Context,
    holder_role_id: int,
) -> bytes:
    return (
        f"ATCAAS:RBC:{epoch}:RU22_CONTRIBUTION:"
        f"{_context_tag(context)}:{int(holder_role_id)}:ECHO"
    ).encode("utf-8")


def ru22_batch_ready_label(
    epoch: str,
    context: RU22Context,
    holder_role_id: int,
) -> bytes:
    return (
        f"ATCAAS:RBC:{epoch}:RU22_CONTRIBUTION:"
        f"{_context_tag(context)}:{int(holder_role_id)}:READY"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate RU22 RBC payload field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class RU22BatchSize:
    batch_bytes: int
    pvr_count: int
    pvr_bytes: int
    pvss_bytes: int
    pvss_proof_bytes: int
    dleq_bytes: int
    proof_bytes: int
    ciphertext_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            field: int(getattr(self, field))
            for field in self.__dataclass_fields__
        }


def classify_ru22_batch(batch_bytes: bytes) -> RU22BatchSize:
    """Classify a batch without executing PVSS or DLEQ verification."""

    raw = bytes(batch_bytes)
    batch = RU22ContributionBatch.from_bytes(raw)
    pvrs = batch.pvr_entries
    ciphertext_bytes = 0
    pvss_proof_bytes = 0
    for pvr in pvrs:
        transcript = pvr.pvss_transcript
        ciphertext_bytes += sum(
            len(_canonical_json(result[field]))
            for result in transcript.enc_results
            for field in ("enc_s", "enc_hat_s")
        )
        if transcript.derived_c_v_vector is not None:
            ciphertext_bytes += sum(
                len(_canonical_json(ciphertext))
                for ciphertext in transcript.derived_c_v_vector
            )
        pvss_proof_bytes += len(_canonical_json(transcript.pi_ve))
    dleq_bytes = sum(
        len(pvr.old_to_new_dleq_proof.to_bytes()) for pvr in pvrs
    )
    return RU22BatchSize(
        batch_bytes=len(raw),
        pvr_count=len(pvrs),
        pvr_bytes=sum(len(pvr.to_bytes()) for pvr in pvrs),
        pvss_bytes=sum(
            len(pvr.pvss_transcript.to_bytes()) for pvr in pvrs
        ),
        pvss_proof_bytes=pvss_proof_bytes,
        dleq_bytes=dleq_bytes,
        proof_bytes=pvss_proof_bytes + dleq_bytes,
        ciphertext_bytes=ciphertext_bytes,
    )


@dataclass
class RU22RBCMetrics:
    outbound_batches: int = 0
    delivered_batches: int = 0
    rejected_payloads: int = 0
    outbound_envelope_bytes: int = 0
    outbound_batch_bytes: int = 0
    outbound_pvr_bytes: int = 0
    outbound_pvss_bytes: int = 0
    outbound_pvss_proof_bytes: int = 0
    outbound_dleq_bytes: int = 0
    outbound_proof_bytes: int = 0
    outbound_ciphertext_bytes: int = 0
    delivered_batch_bytes: int = 0
    delivered_pvr_bytes: int = 0
    delivered_pvss_bytes: int = 0
    delivered_pvss_proof_bytes: int = 0
    delivered_dleq_bytes: int = 0
    delivered_proof_bytes: int = 0
    delivered_ciphertext_bytes: int = 0

    def record_outbound(
        self, envelope_size: int, sizes: RU22BatchSize
    ) -> None:
        self.outbound_batches += 1
        self.outbound_envelope_bytes += int(envelope_size)
        self.outbound_batch_bytes += sizes.batch_bytes
        self.outbound_pvr_bytes += sizes.pvr_bytes
        self.outbound_pvss_bytes += sizes.pvss_bytes
        self.outbound_pvss_proof_bytes += sizes.pvss_proof_bytes
        self.outbound_dleq_bytes += sizes.dleq_bytes
        self.outbound_proof_bytes += sizes.proof_bytes
        self.outbound_ciphertext_bytes += sizes.ciphertext_bytes

    def record_delivered(self, sizes: RU22BatchSize) -> None:
        self.delivered_batches += 1
        self.delivered_batch_bytes += sizes.batch_bytes
        self.delivered_pvr_bytes += sizes.pvr_bytes
        self.delivered_pvss_bytes += sizes.pvss_bytes
        self.delivered_pvss_proof_bytes += sizes.pvss_proof_bytes
        self.delivered_dleq_bytes += sizes.dleq_bytes
        self.delivered_proof_bytes += sizes.proof_bytes
        self.delivered_ciphertext_bytes += sizes.ciphertext_bytes

    def snapshot(self) -> dict[str, int]:
        return {
            field: int(getattr(self, field))
            for field in self.__dataclass_fields__
        }


@dataclass
class _RU22RBCInstance:
    context: RU22Context
    holder_role_id: int
    rbc: YosoRBC
    reactor_task: asyncio.Task
    output_task: asyncio.Task
    sender_task: Optional[asyncio.Task] = None
    outbound_digest: Optional[str] = None
    delivered: bool = False
    stopped: bool = False


class RU22ContributionRBCManager:
    """Run one YOSO-RBC instance per RU22 source role."""

    def __init__(
        self,
        *,
        dag: DagTOB,
        epoch: str,
        my_id: int,
        source_role_nodes: Mapping[int, int],
        M: int,
        t: int,
        threshold_fraction: float,
        expected_committee_size: int,
        committee_m: int,
        quorum_h: int,
        send,
        shared_subscribe,
        crypto,
        vrf_sk,
        vrf_pks,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not isinstance(source_role_nodes, Mapping):
            raise TypeError("source_role_nodes must be a mapping")
        normalized_roles: dict[int, int] = {}
        for role, node in source_role_nodes.items():
            if (
                isinstance(role, bool)
                or not isinstance(role, int)
                or role < 0
                or isinstance(node, bool)
                or not isinstance(node, int)
                or node < 0
            ):
                raise ValueError(
                    "source_role_nodes must map non-negative integers"
                )
            normalized_roles[role] = node
        if not normalized_roles:
            raise ValueError("source_role_nodes must be non-empty")
        if len(set(normalized_roles.values())) != len(normalized_roles):
            raise ValueError("source_role_nodes must be one-to-one")

        self.dag = dag
        self.epoch = str(epoch)
        self.my_id = int(my_id)
        self.source_role_nodes = dict(sorted(normalized_roles.items()))
        self.M = int(M)
        self.t = int(t)
        self.threshold_fraction = float(threshold_fraction)
        self.expected_committee_size = int(expected_committee_size)
        self.committee_m = int(committee_m)
        self.quorum_h = int(quorum_h)
        self.send = send
        self.shared_subscribe = shared_subscribe
        self.crypto = crypto
        self.vrf_sk = vrf_sk
        self.vrf_pks = vrf_pks
        self.log = logger or logging.getLogger(
            f"ru22_rbc_manager.{self.my_id}"
        )
        self.instances: Dict[Tuple[str, int], _RU22RBCInstance] = {}
        self.delivered_batches: set[Tuple[str, int]] = set()
        self.metrics = RU22RBCMetrics()
        self._stopping = False

    @staticmethod
    def _key(
        context: RU22Context, holder_role_id: int
    ) -> Tuple[str, int]:
        return context.context_digest, int(holder_role_id)

    def _validate_role(
        self, context: RU22Context, holder_role_id: int
    ) -> int:
        holder_role_id = int(holder_role_id)
        if holder_role_id not in context.holder_eval_points:
            raise ValueError("holder role is not in the RU22 source committee")
        if set(self.source_role_nodes) != set(context.source_role_indices):
            raise ValueError("source role/node map does not match context")
        return holder_role_id

    def ensure_instance(
        self, context: RU22Context, holder_role_id: int
    ) -> None:
        if self._stopping:
            raise RuntimeError("RU22 contribution RBC manager is stopping")
        if not isinstance(context, RU22Context):
            raise TypeError("context must be an RU22Context")
        holder_role_id = self._validate_role(context, holder_role_id)
        key = self._key(context, holder_role_id)
        if key in self.instances:
            return

        pbk, pvk, pks, sk, srs = self.crypto
        rbc = YosoRBC(
            pks,
            sk,
            pbk,
            pvk,
            self.expected_committee_size,
            self.t,
            srs,
            self.my_id,
            self.send,
            None,
            prefix=ru22_batch_rbc_prefix(
                self.epoch, context, holder_role_id
            ),
            shared_subscribe=self.shared_subscribe,
            threshold_fraction=self.threshold_fraction,
            network_size=self.M,
            vrf_sk=self.vrf_sk,
            vrf_pks=self.vrf_pks,
            expected_committee_size=self.expected_committee_size,
            committee_m=self.committee_m,
            quorum_h=self.quorum_h,
            ready_amplify_threshold=(self.quorum_h + 1) // 2 + 1,
            vrf_echo_label=ru22_batch_echo_label(
                self.epoch, context, holder_role_id
            ),
            vrf_ready_label=ru22_batch_ready_label(
                self.epoch, context, holder_role_id
            ),
        )
        reactor_task = asyncio.create_task(rbc.reactor())
        output_task = asyncio.create_task(
            self._watch_output(context, holder_role_id, rbc)
        )
        self.instances[key] = _RU22RBCInstance(
            context=context,
            holder_role_id=holder_role_id,
            rbc=rbc,
            reactor_task=reactor_task,
            output_task=output_task,
        )

    def ensure_context(
        self, context: RU22Context, holder_role_ids: Iterable[int]
    ) -> None:
        for holder_role_id in holder_role_ids:
            self.ensure_instance(context, int(holder_role_id))

    async def broadcast_batch(
        self,
        context: RU22Context,
        *,
        holder_role_id: int,
        batch_bytes: bytes,
    ) -> None:
        holder_role_id = self._validate_role(context, holder_role_id)
        if self.source_role_nodes[holder_role_id] != self.my_id:
            raise ValueError(
                "only the node assigned to this source role may start sender"
            )
        raw = bytes(batch_bytes)
        batch = RU22ContributionBatch.from_bytes(raw)
        validate_ru22_contribution_batch_shape(context, batch)
        if batch.holder_role_id != holder_role_id:
            raise ValueError("batch holder does not match RBC holder role")

        self.ensure_instance(context, holder_role_id)
        inst = self.instances[self._key(context, holder_role_id)]
        digest = hashlib.sha256(raw).hexdigest()
        if inst.sender_task is not None:
            if inst.outbound_digest != digest:
                raise ValueError(
                    "source role attempted conflicting RU22 batch broadcasts"
                )
            return

        payload = self._encode_payload(context, holder_role_id, raw)
        sizes = classify_ru22_batch(raw)
        inst.outbound_digest = digest
        self.metrics.record_outbound(len(payload), sizes)

        async def _send() -> None:
            await asyncio.sleep(0.1)
            await inst.rbc.sender(payload)

        inst.sender_task = asyncio.create_task(_send())

    @staticmethod
    def _encode_payload(
        context: RU22Context,
        holder_role_id: int,
        batch_bytes: bytes,
    ) -> bytes:
        raw = bytes(batch_bytes)
        batch = RU22ContributionBatch.from_bytes(raw)
        return _canonical_json(
            {
                "kind": _RU22_BATCH_RBC_KIND,
                "version": RU22_BATCH_RBC_VERSION,
                "qid": context.encode_qid(),
                "context_digest": context.context_digest,
                "holder_role_id": int(holder_role_id),
                "mid": f"node_{int(holder_role_id)}",
                "batch_digest": batch.batch_digest,
                "batch_hex": raw.hex(),
            }
        )

    @staticmethod
    def _decode_payload(
        payload: bytes,
        *,
        context: RU22Context,
        holder_role_id: int,
    ) -> tuple[str, bytes, RU22ContributionBatch]:
        try:
            data = json.loads(
                bytes(payload).decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("malformed RU22 RBC payload") from exc
        if not isinstance(data, dict):
            raise ValueError("RU22 RBC payload must be a JSON object")
        expected_fields = {
            "kind",
            "version",
            "qid",
            "context_digest",
            "holder_role_id",
            "mid",
            "batch_digest",
            "batch_hex",
        }
        if set(data) != expected_fields:
            raise ValueError("RU22 RBC payload fields do not match version 1")
        if data["kind"] != _RU22_BATCH_RBC_KIND:
            raise ValueError("unexpected RU22 RBC payload kind")
        if (
            isinstance(data["version"], bool)
            or data["version"] != RU22_BATCH_RBC_VERSION
        ):
            raise ValueError("unsupported RU22 RBC payload version")
        holder_role_id = int(holder_role_id)
        if (
            isinstance(data["holder_role_id"], bool)
            or data["holder_role_id"] != holder_role_id
        ):
            raise ValueError("RU22 RBC holder does not match instance")
        if data["qid"] != context.encode_qid():
            raise ValueError("RU22 RBC qid does not match instance")
        if data["context_digest"] != context.context_digest:
            raise ValueError("RU22 RBC context digest does not match instance")
        mid = f"node_{holder_role_id}"
        if data["mid"] != mid:
            raise ValueError("RU22 RBC mid does not match holder role")
        encoded = data["batch_hex"]
        if not isinstance(encoded, str):
            raise ValueError("RU22 RBC batch_hex must be a string")
        try:
            raw = bytes.fromhex(encoded)
        except ValueError as exc:
            raise ValueError("RU22 RBC batch_hex is malformed") from exc
        if encoded != raw.hex():
            raise ValueError("RU22 RBC batch_hex must be canonical lowercase hex")
        batch = RU22ContributionBatch.from_bytes(raw)
        if data["batch_digest"] != batch.batch_digest:
            raise ValueError("RU22 RBC batch_digest mismatch")
        validate_ru22_contribution_batch_shape(context, batch)
        if batch.holder_role_id != holder_role_id:
            raise ValueError("RU22 RBC batch holder mismatch")
        return mid, raw, batch

    async def _watch_output(
        self,
        context: RU22Context,
        holder_role_id: int,
        rbc: YosoRBC,
    ) -> None:
        key = self._key(context, holder_role_id)
        try:
            payload, _ = await rbc.output_queue.get()
            mid, raw, _batch = self._decode_payload(
                payload,
                context=context,
                holder_role_id=holder_role_id,
            )
            if key not in self.delivered_batches:
                sizes = classify_ru22_batch(raw)
                self.dag.deliver_message(context.encode_qid(), mid, raw)
                self.delivered_batches.add(key)
                self.metrics.record_delivered(sizes)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.metrics.rejected_payloads += 1
            self.log.warning(
                "[RU22:RBC] rejected holder_role=%d payload: %s",
                holder_role_id,
                exc,
            )
        finally:
            inst = self.instances.get(key)
            if inst is not None:
                inst.delivered = key in self.delivered_batches

    async def _stop_instance(self, key: Tuple[str, int]) -> None:
        inst = self.instances.pop(key, None)
        if inst is None or inst.stopped:
            return
        inst.stopped = True
        tasks = [
            task
            for task in (
                inst.sender_task,
                inst.output_task,
                inst.reactor_task,
            )
            if task is not None
        ]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self, timeout: Optional[float] = None) -> bool:
        if self._stopping:
            return self.active_count() == 0
        self._stopping = True
        stop_all = asyncio.gather(
            *(
                self._stop_instance(key)
                for key in list(self.instances)
            ),
            return_exceptions=True,
        )
        if timeout is None:
            await stop_all
            return True
        try:
            await asyncio.wait_for(stop_all, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            self.log.warning(
                "[RU22:RBC] stop timed out after %.3fs; active=%d",
                timeout,
                self.active_count(),
            )
            return False

    def active_count(self) -> int:
        return len(self.instances)


__all__ = [
    "RU22_BATCH_RBC_VERSION",
    "RU22BatchSize",
    "RU22ContributionRBCManager",
    "RU22RBCMetrics",
    "classify_ru22_batch",
    "ru22_batch_echo_label",
    "ru22_batch_rbc_prefix",
    "ru22_batch_ready_label",
]
