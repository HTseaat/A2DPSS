"""YOSO-RBC transport for complete Algorithm-5 OCDPSS holder bundles.

This path is deliberately separate from ``DPSSTranscriptRBCManager``: the
legacy manager transports a bare PVSS transcript, while Algorithm 5 must bind
``(transcript, operation share, operation proof)`` in one RBC instance.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from beaver.dag_tob import DagTOB
from beaver.ocdpss import OCDPSSBundle, OCDPSSContext
from beaver.yosorbc import YosoRBC


OCDPSS_BUNDLE_RBC_VERSION = 1
_OCDPSS_BUNDLE_RBC_KIND = "atcaas-ocdpss-bundle-rbc"


def _context_tag(context: OCDPSSContext) -> str:
    return context.context_digest[:16]


def ocdpss_bundle_rbc_prefix(
    epoch: str, context: OCDPSSContext, holder_id: int
) -> str:
    return f"OCDPSS:{epoch}:BUNDLE_RBC:{_context_tag(context)}:{int(holder_id)}"


def ocdpss_bundle_echo_label(
    epoch: str, context: OCDPSSContext, holder_id: int
) -> bytes:
    return (
        f"ATCAAS:RBC:{epoch}:OCDPSS_BUNDLE:{_context_tag(context)}:"
        f"{int(holder_id)}:ECHO"
    ).encode("utf-8")


def ocdpss_bundle_ready_label(
    epoch: str, context: OCDPSSContext, holder_id: int
) -> bytes:
    return (
        f"ATCAAS:RBC:{epoch}:OCDPSS_BUNDLE:{_context_tag(context)}:"
        f"{int(holder_id)}:READY"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate OCDPSS RBC payload field: {key}")
        result[key] = value
    return result


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class OCDPSSBundleSize:
    bundle_bytes: int
    pvss_bytes: int
    operation_share_bytes: int
    operation_proof_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "bundle_bytes": self.bundle_bytes,
            "pvss_bytes": self.pvss_bytes,
            "operation_share_bytes": self.operation_share_bytes,
            "operation_proof_bytes": self.operation_proof_bytes,
        }


def classify_ocdpss_bundle(bundle_bytes: bytes) -> OCDPSSBundleSize:
    raw = bytes(bundle_bytes)
    bundle = OCDPSSBundle.from_bytes(raw)
    wire = bundle.to_obj()
    return OCDPSSBundleSize(
        bundle_bytes=len(raw),
        pvss_bytes=len(bundle.transcript.to_bytes()),
        operation_share_bytes=len(_canonical_json(wire["operation_share"])),
        operation_proof_bytes=len(_canonical_json(wire["operation_proof"])),
    )


@dataclass
class OCDPSSRBCMetrics:
    outbound_bundles: int = 0
    delivered_bundles: int = 0
    rejected_payloads: int = 0
    outbound_envelope_bytes: int = 0
    outbound_bundle_bytes: int = 0
    outbound_pvss_bytes: int = 0
    outbound_operation_share_bytes: int = 0
    outbound_operation_proof_bytes: int = 0
    delivered_bundle_bytes: int = 0
    delivered_pvss_bytes: int = 0
    delivered_operation_share_bytes: int = 0
    delivered_operation_proof_bytes: int = 0

    def record_outbound(self, envelope_size: int, sizes: OCDPSSBundleSize) -> None:
        self.outbound_bundles += 1
        self.outbound_envelope_bytes += int(envelope_size)
        self.outbound_bundle_bytes += sizes.bundle_bytes
        self.outbound_pvss_bytes += sizes.pvss_bytes
        self.outbound_operation_share_bytes += sizes.operation_share_bytes
        self.outbound_operation_proof_bytes += sizes.operation_proof_bytes

    def record_delivered(self, sizes: OCDPSSBundleSize) -> None:
        self.delivered_bundles += 1
        self.delivered_bundle_bytes += sizes.bundle_bytes
        self.delivered_pvss_bytes += sizes.pvss_bytes
        self.delivered_operation_share_bytes += sizes.operation_share_bytes
        self.delivered_operation_proof_bytes += sizes.operation_proof_bytes

    def snapshot(self) -> dict[str, int]:
        return {
            field: int(getattr(self, field))
            for field in self.__dataclass_fields__
        }


@dataclass
class _BundleRBCInstance:
    context: OCDPSSContext
    holder_id: int
    rbc: YosoRBC
    reactor_task: asyncio.Task
    output_task: asyncio.Task
    sender_task: Optional[asyncio.Task] = None
    outbound_digest: Optional[str] = None
    delivered: bool = False
    stopped: bool = False


class OCDPSSBundleRBCManager:
    """Run one YOSO-RBC instance for each source holder's full bundle."""

    def __init__(
        self,
        *,
        dag: DagTOB,
        epoch: str,
        my_id: int,
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
        self.dag = dag
        self.epoch = str(epoch)
        self.my_id = int(my_id)
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
            f"ocdpss_rbc_manager.{self.my_id}"
        )
        self.instances: Dict[Tuple[str, int], _BundleRBCInstance] = {}
        self.delivered_bundles = set()
        self.metrics = OCDPSSRBCMetrics()
        self._stopping = False

    @staticmethod
    def _key(context: OCDPSSContext, holder_id: int) -> Tuple[str, int]:
        return context.context_digest, int(holder_id)

    def ensure_instance(self, context: OCDPSSContext, holder_id: int) -> None:
        if self._stopping:
            raise RuntimeError("OCDPSS bundle RBC manager is stopping")
        if not isinstance(context, OCDPSSContext):
            raise TypeError("context must be an OCDPSSContext")
        holder_id = int(holder_id)
        if holder_id not in context.holder_eval_points:
            raise ValueError("holder_id is not in the OCDPSS source committee")
        key = self._key(context, holder_id)
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
            prefix=ocdpss_bundle_rbc_prefix(self.epoch, context, holder_id),
            shared_subscribe=self.shared_subscribe,
            threshold_fraction=self.threshold_fraction,
            network_size=self.M,
            vrf_sk=self.vrf_sk,
            vrf_pks=self.vrf_pks,
            expected_committee_size=self.expected_committee_size,
            committee_m=self.committee_m,
            quorum_h=self.quorum_h,
            ready_amplify_threshold=(self.quorum_h + 1) // 2 + 1,
            vrf_echo_label=ocdpss_bundle_echo_label(
                self.epoch, context, holder_id
            ),
            vrf_ready_label=ocdpss_bundle_ready_label(
                self.epoch, context, holder_id
            ),
        )
        reactor_task = asyncio.create_task(rbc.reactor())
        output_task = asyncio.create_task(
            self._watch_output(context, holder_id, rbc)
        )
        self.instances[key] = _BundleRBCInstance(
            context=context,
            holder_id=holder_id,
            rbc=rbc,
            reactor_task=reactor_task,
            output_task=output_task,
        )

    def ensure_context(
        self, context: OCDPSSContext, holder_ids: Iterable[int]
    ) -> None:
        before = len(self.instances)
        for holder_id in holder_ids:
            self.ensure_instance(context, int(holder_id))
        if len(self.instances) != before:
            self.log.info(
                "[ALGORITHM-5:OCDPSS:RBC] ensure_context qid=%s active=%d",
                context.encode_qid(),
                self.active_count(),
            )

    async def broadcast_bundle(
        self,
        context: OCDPSSContext,
        *,
        holder_id: int,
        bundle_bytes: bytes,
    ) -> None:
        holder_id = int(holder_id)
        if holder_id != self.my_id:
            raise ValueError("only the local source holder may start its RBC sender")
        raw = bytes(bundle_bytes)
        bundle = OCDPSSBundle.from_bytes(raw)
        if bundle.context_digest != context.context_digest:
            raise ValueError("bundle context does not match RBC context")
        if bundle.holder_id != holder_id:
            raise ValueError("bundle holder does not match RBC holder")
        expected_eval_point = context.holder_eval_points.get(holder_id)
        if bundle.holder_eval_point != expected_eval_point:
            raise ValueError("bundle eval point does not match RBC context")

        self.ensure_instance(context, holder_id)
        inst = self.instances[self._key(context, holder_id)]
        digest = hashlib.sha256(raw).hexdigest()
        if inst.sender_task is not None:
            if inst.outbound_digest != digest:
                raise ValueError("source holder attempted conflicting OCDPSS broadcasts")
            return

        payload = self._encode_payload(context, holder_id, raw)
        sizes = classify_ocdpss_bundle(raw)
        inst.outbound_digest = digest
        self.metrics.record_outbound(len(payload), sizes)

        async def _send() -> None:
            await asyncio.sleep(0.1)
            await inst.rbc.sender(payload)

        inst.sender_task = asyncio.create_task(_send())
        self.log.info(
            "[ALGORITHM-5:OCDPSS:RBC] sender started qid=%s "
            "holder=%d envelope_bytes=%d "
            "bundle_bytes=%d pvss_bytes=%d operation_proof_bytes=%d",
            context.encode_qid(),
            holder_id,
            len(payload),
            sizes.bundle_bytes,
            sizes.pvss_bytes,
            sizes.operation_proof_bytes,
        )

    @staticmethod
    def _encode_payload(
        context: OCDPSSContext,
        holder_id: int,
        bundle_bytes: bytes,
    ) -> bytes:
        holder_id = int(holder_id)
        return _canonical_json(
            {
                "kind": _OCDPSS_BUNDLE_RBC_KIND,
                "version": OCDPSS_BUNDLE_RBC_VERSION,
                "qid": context.encode_qid(),
                "context_digest": context.context_digest,
                "holder_id": holder_id,
                "mid": f"node_{holder_id}",
                "bundle_hex": bytes(bundle_bytes).hex(),
            }
        )

    @staticmethod
    def _decode_payload(
        payload: bytes,
        *,
        context: OCDPSSContext,
        holder_id: int,
    ) -> tuple[str, bytes, OCDPSSBundle]:
        try:
            data = json.loads(
                bytes(payload).decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("malformed OCDPSS RBC payload") from exc
        if not isinstance(data, dict):
            raise ValueError("OCDPSS RBC payload must be a JSON object")
        expected_fields = {
            "kind",
            "version",
            "qid",
            "context_digest",
            "holder_id",
            "mid",
            "bundle_hex",
        }
        if set(data) != expected_fields:
            raise ValueError("OCDPSS RBC payload fields do not match version 1")
        if data["kind"] != _OCDPSS_BUNDLE_RBC_KIND:
            raise ValueError("unexpected OCDPSS RBC payload kind")
        if isinstance(data["version"], bool) or data["version"] != 1:
            raise ValueError("unsupported OCDPSS RBC payload version")

        holder_id = int(holder_id)
        if isinstance(data["holder_id"], bool) or data["holder_id"] != holder_id:
            raise ValueError("OCDPSS RBC holder does not match instance")
        if data["qid"] != context.encode_qid():
            raise ValueError("OCDPSS RBC qid does not match instance")
        if data["context_digest"] != context.context_digest:
            raise ValueError("OCDPSS RBC context digest does not match instance")
        mid = f"node_{holder_id}"
        if data["mid"] != mid:
            raise ValueError("OCDPSS RBC mid does not match holder")
        if not isinstance(data["bundle_hex"], str):
            raise ValueError("OCDPSS RBC bundle_hex must be a string")
        try:
            raw = bytes.fromhex(data["bundle_hex"])
        except ValueError as exc:
            raise ValueError("OCDPSS RBC bundle_hex is malformed") from exc
        if data["bundle_hex"] != raw.hex():
            raise ValueError("OCDPSS RBC bundle_hex must be canonical lowercase hex")
        bundle = OCDPSSBundle.from_bytes(raw)
        if bundle.context_digest != context.context_digest:
            raise ValueError("OCDPSS RBC bundle context digest mismatch")
        if bundle.holder_id != holder_id:
            raise ValueError("OCDPSS RBC bundle holder mismatch")
        if bundle.holder_eval_point != context.holder_eval_points[holder_id]:
            raise ValueError("OCDPSS RBC bundle eval point mismatch")
        return mid, raw, bundle

    async def _watch_output(
        self,
        context: OCDPSSContext,
        holder_id: int,
        rbc: YosoRBC,
    ) -> None:
        key = self._key(context, holder_id)
        try:
            payload, _ = await rbc.output_queue.get()
            mid, raw, _bundle = self._decode_payload(
                payload,
                context=context,
                holder_id=holder_id,
            )
            sizes = classify_ocdpss_bundle(raw)
            self.dag.deliver_message(context.encode_qid(), mid, raw)
            self.delivered_bundles.add(key)
            self.metrics.record_delivered(sizes)
            self.log.info(
                "[ALGORITHM-5:OCDPSS:RBC] delivered qid=%s "
                "holder=%d mid=%s bundle_bytes=%d",
                context.encode_qid(),
                holder_id,
                mid,
                sizes.bundle_bytes,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.metrics.rejected_payloads += 1
            self.log.warning(
                "[ALGORITHM-5:OCDPSS:RBC] rejected holder=%d payload: %s",
                holder_id,
                exc,
            )
        finally:
            inst = self.instances.get(key)
            if inst is not None:
                inst.delivered = key in self.delivered_bundles

    async def _stop_instance(self, key: Tuple[str, int]) -> None:
        inst = self.instances.pop(key, None)
        if inst is None or inst.stopped:
            return
        inst.stopped = True
        tasks = [
            task
            for task in (inst.sender_task, inst.output_task, inst.reactor_task)
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
            *(self._stop_instance(key) for key in list(self.instances)),
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
                "[ALGORITHM-5:OCDPSS:RBC] stop timed out "
                "after %.3fs; active=%d",
                timeout,
                self.active_count(),
            )
            return False

    def active_count(self) -> int:
        return len(self.instances)


__all__ = [
    "OCDPSS_BUNDLE_RBC_VERSION",
    "OCDPSSBundleRBCManager",
    "OCDPSSBundleSize",
    "OCDPSSRBCMetrics",
    "classify_ocdpss_bundle",
    "ocdpss_bundle_echo_label",
    "ocdpss_bundle_rbc_prefix",
    "ocdpss_bundle_ready_label",
]
