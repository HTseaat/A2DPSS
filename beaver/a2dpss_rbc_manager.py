"""YOSO-RBC transport for complete Algorithm-4 A²DPSS contributions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from beaver.a2dpss import A2DPSSContext, A2DPSSContribution
from beaver.dag_tob import DagTOB
from beaver.yosorbc import YosoRBC


A2DPSS_CONTRIBUTION_RBC_VERSION = 1
_A2DPSS_CONTRIBUTION_RBC_KIND = (
    "atcaas-a2dpss-contribution-rbc"
)


def _context_tag(context: A2DPSSContext) -> str:
    return context.context_digest[:16]


def a2dpss_contribution_rbc_prefix(
    epoch: str,
    context: A2DPSSContext,
    holder_id: int,
) -> str:
    return (
        f"A2DPSS:{epoch}:CONTRIBUTION_RBC:"
        f"{_context_tag(context)}:{int(holder_id)}"
    )


def a2dpss_contribution_echo_label(
    epoch: str,
    context: A2DPSSContext,
    holder_id: int,
) -> bytes:
    return (
        f"ATCAAS:RBC:{epoch}:A2DPSS_CONTRIBUTION:"
        f"{_context_tag(context)}:{int(holder_id)}:ECHO"
    ).encode("utf-8")


def a2dpss_contribution_ready_label(
    epoch: str,
    context: A2DPSSContext,
    holder_id: int,
) -> bytes:
    return (
        f"ATCAAS:RBC:{epoch}:A2DPSS_CONTRIBUTION:"
        f"{_context_tag(context)}:{int(holder_id)}:READY"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"duplicate A2DPSS RBC payload field: {key}"
            )
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
class A2DPSSContributionSize:
    contribution_bytes: int
    pvss_bytes: int
    proof_bytes: int
    ciphertext_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "contribution_bytes": self.contribution_bytes,
            "pvss_bytes": self.pvss_bytes,
        }


def classify_a2dpss_contribution(
    contribution_bytes: bytes,
) -> A2DPSSContributionSize:
    raw = bytes(contribution_bytes)
    contribution = A2DPSSContribution.from_bytes(raw)
    transcript = contribution.transcript
    # Count serialized cryptographic component values, excluding recipient ids,
    # public keys, field names, and surrounding transcript JSON framing.
    ciphertext_bytes = sum(
        len(_canonical_json(result[field]))
        for result in transcript.enc_results
        for field in ("enc_s", "enc_hat_s")
    )
    if transcript.derived_c_v_vector is not None:
        ciphertext_bytes += sum(
            len(_canonical_json(ciphertext))
            for ciphertext in transcript.derived_c_v_vector
        )
    return A2DPSSContributionSize(
        contribution_bytes=len(raw),
        pvss_bytes=len(transcript.to_bytes()),
        proof_bytes=len(_canonical_json(transcript.pi_ve)),
        ciphertext_bytes=ciphertext_bytes,
    )


@dataclass
class A2DPSSRBCMetrics:
    outbound_contributions: int = 0
    delivered_contributions: int = 0
    rejected_payloads: int = 0
    outbound_envelope_bytes: int = 0
    outbound_contribution_bytes: int = 0
    outbound_pvss_bytes: int = 0
    outbound_proof_bytes: int = 0
    outbound_ciphertext_bytes: int = 0
    delivered_contribution_bytes: int = 0
    delivered_pvss_bytes: int = 0
    delivered_proof_bytes: int = 0
    delivered_ciphertext_bytes: int = 0

    def record_outbound(
        self,
        envelope_size: int,
        sizes: A2DPSSContributionSize,
    ) -> None:
        self.outbound_contributions += 1
        self.outbound_envelope_bytes += int(envelope_size)
        self.outbound_contribution_bytes += sizes.contribution_bytes
        self.outbound_pvss_bytes += sizes.pvss_bytes
        self.outbound_proof_bytes += sizes.proof_bytes
        self.outbound_ciphertext_bytes += sizes.ciphertext_bytes

    def record_delivered(
        self,
        sizes: A2DPSSContributionSize,
    ) -> None:
        self.delivered_contributions += 1
        self.delivered_contribution_bytes += sizes.contribution_bytes
        self.delivered_pvss_bytes += sizes.pvss_bytes
        self.delivered_proof_bytes += sizes.proof_bytes
        self.delivered_ciphertext_bytes += sizes.ciphertext_bytes

    def snapshot(self) -> dict[str, int]:
        return {
            field: int(getattr(self, field))
            for field in self.__dataclass_fields__
        }


@dataclass
class _ContributionRBCInstance:
    context: A2DPSSContext
    holder_id: int
    rbc: YosoRBC
    reactor_task: asyncio.Task
    output_task: asyncio.Task
    sender_task: Optional[asyncio.Task] = None
    outbound_digest: Optional[str] = None
    delivered: bool = False
    stopped: bool = False


class A2DPSSContributionRBCManager:
    """Run one YOSO-RBC instance per A²DPSS source-holder contribution."""

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
        self.expected_committee_size = int(
            expected_committee_size
        )
        self.committee_m = int(committee_m)
        self.quorum_h = int(quorum_h)
        self.send = send
        self.shared_subscribe = shared_subscribe
        self.crypto = crypto
        self.vrf_sk = vrf_sk
        self.vrf_pks = vrf_pks
        self.log = logger or logging.getLogger(
            f"a2dpss_rbc_manager.{self.my_id}"
        )
        self.instances: Dict[
            Tuple[str, int],
            _ContributionRBCInstance,
        ] = {}
        self.delivered_contributions = set()
        self.metrics = A2DPSSRBCMetrics()
        self._stopping = False

    @staticmethod
    def _key(
        context: A2DPSSContext,
        holder_id: int,
    ) -> Tuple[str, int]:
        return context.context_digest, int(holder_id)

    def ensure_instance(
        self,
        context: A2DPSSContext,
        holder_id: int,
    ) -> None:
        if self._stopping:
            raise RuntimeError(
                "A2DPSS contribution RBC manager is stopping"
            )
        if not isinstance(context, A2DPSSContext):
            raise TypeError("context must be an A2DPSSContext")
        holder_id = int(holder_id)
        if holder_id not in context.holder_eval_points:
            raise ValueError(
                "holder_id is not in the A2DPSS source committee"
            )
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
            prefix=a2dpss_contribution_rbc_prefix(
                self.epoch,
                context,
                holder_id,
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
            vrf_echo_label=a2dpss_contribution_echo_label(
                self.epoch,
                context,
                holder_id,
            ),
            vrf_ready_label=a2dpss_contribution_ready_label(
                self.epoch,
                context,
                holder_id,
            ),
        )
        reactor_task = asyncio.create_task(rbc.reactor())
        output_task = asyncio.create_task(
            self._watch_output(context, holder_id, rbc)
        )
        self.instances[key] = _ContributionRBCInstance(
            context=context,
            holder_id=holder_id,
            rbc=rbc,
            reactor_task=reactor_task,
            output_task=output_task,
        )

    def ensure_context(
        self,
        context: A2DPSSContext,
        holder_ids: Iterable[int],
    ) -> None:
        before = len(self.instances)
        for holder_id in holder_ids:
            self.ensure_instance(context, int(holder_id))
        if len(self.instances) != before:
            self.log.info(
                "[ALGORITHM-4:A2DPSS:RBC] ensure_context "
                "qid=%s active=%d",
                context.encode_qid(),
                self.active_count(),
            )

    async def broadcast_contribution(
        self,
        context: A2DPSSContext,
        *,
        holder_id: int,
        contribution_bytes: bytes,
    ) -> None:
        holder_id = int(holder_id)
        if holder_id != self.my_id:
            raise ValueError(
                "only the local source holder may start its RBC sender"
            )
        raw = bytes(contribution_bytes)
        contribution = A2DPSSContribution.from_bytes(raw)
        if contribution.context_digest != context.context_digest:
            raise ValueError(
                "contribution context does not match RBC context"
            )
        if contribution.holder_id != holder_id:
            raise ValueError(
                "contribution holder does not match RBC holder"
            )
        expected_eval_point = context.holder_eval_points.get(
            holder_id
        )
        if contribution.holder_eval_point != expected_eval_point:
            raise ValueError(
                "contribution eval point does not match RBC context"
            )

        self.ensure_instance(context, holder_id)
        inst = self.instances[self._key(context, holder_id)]
        digest = hashlib.sha256(raw).hexdigest()
        if inst.sender_task is not None:
            if inst.outbound_digest != digest:
                raise ValueError(
                    "source holder attempted conflicting A2DPSS broadcasts"
                )
            return

        payload = self._encode_payload(context, holder_id, raw)
        sizes = classify_a2dpss_contribution(raw)
        inst.outbound_digest = digest
        self.metrics.record_outbound(len(payload), sizes)

        async def _send() -> None:
            await asyncio.sleep(0.1)
            await inst.rbc.sender(payload)

        inst.sender_task = asyncio.create_task(_send())
        self.log.info(
            "[ALGORITHM-4:A2DPSS:RBC] sender started "
            "qid=%s holder=%d envelope_bytes=%d "
            "contribution_bytes=%d pvss_bytes=%d",
            context.encode_qid(),
            holder_id,
            len(payload),
            sizes.contribution_bytes,
            sizes.pvss_bytes,
        )

    @staticmethod
    def _encode_payload(
        context: A2DPSSContext,
        holder_id: int,
        contribution_bytes: bytes,
    ) -> bytes:
        holder_id = int(holder_id)
        return _canonical_json(
            {
                "kind": _A2DPSS_CONTRIBUTION_RBC_KIND,
                "version": A2DPSS_CONTRIBUTION_RBC_VERSION,
                "qid": context.encode_qid(),
                "context_digest": context.context_digest,
                "holder_id": holder_id,
                "mid": f"node_{holder_id}",
                "contribution_hex": bytes(
                    contribution_bytes
                ).hex(),
            }
        )

    @staticmethod
    def _decode_payload(
        payload: bytes,
        *,
        context: A2DPSSContext,
        holder_id: int,
    ) -> tuple[str, bytes, A2DPSSContribution]:
        try:
            data = json.loads(
                bytes(payload).decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (
            TypeError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError(
                "malformed A2DPSS RBC payload"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(
                "A2DPSS RBC payload must be a JSON object"
            )
        expected_fields = {
            "kind",
            "version",
            "qid",
            "context_digest",
            "holder_id",
            "mid",
            "contribution_hex",
        }
        if set(data) != expected_fields:
            raise ValueError(
                "A2DPSS RBC payload fields do not match version 1"
            )
        if data["kind"] != _A2DPSS_CONTRIBUTION_RBC_KIND:
            raise ValueError(
                "unexpected A2DPSS RBC payload kind"
            )
        if (
            isinstance(data["version"], bool)
            or data["version"]
            != A2DPSS_CONTRIBUTION_RBC_VERSION
        ):
            raise ValueError(
                "unsupported A2DPSS RBC payload version"
            )

        holder_id = int(holder_id)
        if (
            isinstance(data["holder_id"], bool)
            or data["holder_id"] != holder_id
        ):
            raise ValueError(
                "A2DPSS RBC holder does not match instance"
            )
        if data["qid"] != context.encode_qid():
            raise ValueError(
                "A2DPSS RBC qid does not match instance"
            )
        if data["context_digest"] != context.context_digest:
            raise ValueError(
                "A2DPSS RBC context digest does not match instance"
            )
        mid = f"node_{holder_id}"
        if data["mid"] != mid:
            raise ValueError(
                "A2DPSS RBC mid does not match holder"
            )
        encoded = data["contribution_hex"]
        if not isinstance(encoded, str):
            raise ValueError(
                "A2DPSS RBC contribution_hex must be a string"
            )
        try:
            raw = bytes.fromhex(encoded)
        except ValueError as exc:
            raise ValueError(
                "A2DPSS RBC contribution_hex is malformed"
            ) from exc
        if encoded != raw.hex():
            raise ValueError(
                "A2DPSS RBC contribution_hex must be canonical lowercase hex"
            )
        contribution = A2DPSSContribution.from_bytes(raw)
        if contribution.context_digest != context.context_digest:
            raise ValueError(
                "A2DPSS RBC contribution context digest mismatch"
            )
        if contribution.holder_id != holder_id:
            raise ValueError(
                "A2DPSS RBC contribution holder mismatch"
            )
        if (
            contribution.holder_eval_point
            != context.holder_eval_points[holder_id]
        ):
            raise ValueError(
                "A2DPSS RBC contribution eval point mismatch"
            )
        return mid, raw, contribution

    async def _watch_output(
        self,
        context: A2DPSSContext,
        holder_id: int,
        rbc: YosoRBC,
    ) -> None:
        key = self._key(context, holder_id)
        try:
            payload, _ = await rbc.output_queue.get()
            mid, raw, _contribution = self._decode_payload(
                payload,
                context=context,
                holder_id=holder_id,
            )
            sizes = classify_a2dpss_contribution(raw)
            self.dag.deliver_message(
                context.encode_qid(),
                mid,
                raw,
            )
            self.delivered_contributions.add(key)
            self.metrics.record_delivered(sizes)
            self.log.info(
                "[ALGORITHM-4:A2DPSS:RBC] delivered "
                "qid=%s holder=%d mid=%s contribution_bytes=%d",
                context.encode_qid(),
                holder_id,
                mid,
                sizes.contribution_bytes,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.metrics.rejected_payloads += 1
            self.log.warning(
                "[ALGORITHM-4:A2DPSS:RBC] rejected "
                "holder=%d payload: %s",
                holder_id,
                exc,
            )
        finally:
            inst = self.instances.get(key)
            if inst is not None:
                inst.delivered = (
                    key in self.delivered_contributions
                )

    async def _stop_instance(
        self,
        key: Tuple[str, int],
    ) -> None:
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
            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

    async def stop(
        self,
        timeout: Optional[float] = None,
    ) -> bool:
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
                "[ALGORITHM-4:A2DPSS:RBC] stop timed out "
                "after %.3fs; active=%d",
                timeout,
                self.active_count(),
            )
            return False

    def active_count(self) -> int:
        return len(self.instances)


__all__ = [
    "A2DPSS_CONTRIBUTION_RBC_VERSION",
    "A2DPSSContributionRBCManager",
    "A2DPSSContributionSize",
    "A2DPSSRBCMetrics",
    "a2dpss_contribution_echo_label",
    "a2dpss_contribution_rbc_prefix",
    "a2dpss_contribution_ready_label",
    "classify_a2dpss_contribution",
]
