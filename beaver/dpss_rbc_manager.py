"""Application-level YosoRBC for legacy transcript-only DPSS smokes.

Formal Algorithm 4 broadcasts an ``A2DPSSContribution`` and Algorithm 5
broadcasts a complete OCDPSS bundle through their dedicated managers.  This
compatibility manager accepts only the older bare-PVSS payload and must not be
used by formal or paired experiment paths.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from beaver.dag_tob import DagTOB
from beaver.pvss import PVSSTranscript
from beaver.yosorbc import YosoRBC
from beaver.legacy_yoso_dpss import DPSSContext


def _ctx_tag(ctx: DPSSContext) -> str:
    return hashlib.sha256(ctx.encode_qid().encode("utf-8")).hexdigest()[:16]


def dpss_transcript_rbc_prefix(epoch: str, ctx: DPSSContext, holder_id: int) -> str:
    return f"DPSS:{epoch}:TRANSCRIPT_RBC:{_ctx_tag(ctx)}:{int(holder_id)}"


def dpss_transcript_echo_label(epoch: str, ctx: DPSSContext, holder_id: int) -> bytes:
    return (
        f"ATCAAS:RBC:{epoch}:DPSS_TRANSCRIPT:{_ctx_tag(ctx)}:{int(holder_id)}:ECHO"
    ).encode("utf-8")


def dpss_transcript_ready_label(epoch: str, ctx: DPSSContext, holder_id: int) -> bytes:
    return (
        f"ATCAAS:RBC:{epoch}:DPSS_TRANSCRIPT:{_ctx_tag(ctx)}:{int(holder_id)}:READY"
    ).encode("utf-8")


@dataclass
class _TranscriptRBCInstance:
    ctx: DPSSContext
    holder_id: int
    rbc: YosoRBC
    reactor_task: asyncio.Task
    output_task: asyncio.Task
    sender_task: Optional[asyncio.Task] = None
    delivered: bool = False
    stopped: bool = False


class DPSSTranscriptRBCManager:
    """Run one legacy YosoRBC instance per source-holder transcript."""

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
    ):
        self.dag = dag
        self.epoch = epoch
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
        self.log = logger or logging.getLogger(f"dpss_rbc_manager.{my_id}")

        self.instances: Dict[Tuple[Tuple[int, str, str, str, int], int], _TranscriptRBCInstance] = {}
        self.delivered_transcripts = set()
        self._stopping = False

    def ensure_instance(self, ctx: DPSSContext, holder_id: int) -> None:
        holder_id = int(holder_id)
        key = (ctx.key, holder_id)
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
            prefix=dpss_transcript_rbc_prefix(self.epoch, ctx, holder_id),
            shared_subscribe=self.shared_subscribe,
            threshold_fraction=self.threshold_fraction,
            network_size=self.M,
            vrf_sk=self.vrf_sk,
            vrf_pks=self.vrf_pks,
            expected_committee_size=self.expected_committee_size,
            committee_m=self.committee_m,
            quorum_h=self.quorum_h,
            ready_amplify_threshold=(self.quorum_h + 1) // 2 + 1,
            vrf_echo_label=dpss_transcript_echo_label(self.epoch, ctx, holder_id),
            vrf_ready_label=dpss_transcript_ready_label(self.epoch, ctx, holder_id),
        )
        reactor_task = asyncio.create_task(rbc.reactor())
        output_task = asyncio.create_task(self._watch_output(ctx, holder_id, rbc))
        self.instances[key] = _TranscriptRBCInstance(
            ctx=ctx,
            holder_id=holder_id,
            rbc=rbc,
            reactor_task=reactor_task,
            output_task=output_task,
        )

    def ensure_context(self, ctx: DPSSContext, holder_ids: Iterable[int]) -> None:
        created_before = len(self.instances)
        for holder_id in holder_ids:
            self.ensure_instance(ctx, int(holder_id))
        if len(self.instances) != created_before:
            self.log.info(
                "[DPSS-RBC] ensure_context qid=%s active=%d",
                ctx.encode_qid(),
                self.active_count(),
            )

    async def broadcast_transcript(
        self,
        ctx: DPSSContext,
        *,
        holder_id: int,
        transcript: PVSSTranscript,
    ) -> None:
        holder_id = int(holder_id)
        self.ensure_instance(ctx, holder_id)
        inst = self.instances[(ctx.key, holder_id)]
        if inst.sender_task is not None:
            return

        payload = self._encode_payload(ctx, holder_id, transcript)

        async def _send():
            await asyncio.sleep(0.1)
            await inst.rbc.sender(payload)

        inst.sender_task = asyncio.create_task(_send())
        self.log.info(
            "[DPSS-RBC] sender started qid=%s holder=%d bytes=%d",
            ctx.encode_qid(),
            holder_id,
            len(payload),
        )

    async def _watch_output(self, ctx: DPSSContext, holder_id: int, rbc: YosoRBC) -> None:
        key = (ctx.key, int(holder_id))
        try:
            payload, _ = await rbc.output_queue.get()
            data = self._decode_payload(payload)
            if data["qid"] != ctx.encode_qid() or int(data["holder_id"]) != int(holder_id):
                self.log.warning(
                    "[DPSS-RBC] prefix holder=%d delivered mismatched payload, ignored",
                    holder_id,
                )
                return
            mid = data["mid"]
            transcript_bytes = bytes.fromhex(data["transcript_hex"])
            # Receive side: retain the transcript in the active or delivered
            # pool and re-run buffer promotion (a newly available transcript
            # may satisfy a vertex's Available gate).  Use deliver_message,
            # not t_broadcast, so this never re-disseminates -- this manager
            # *is* the transcript's RBC.
            qid = ctx.encode_qid()
            self.dag.deliver_message(qid, mid, transcript_bytes)
            self.delivered_transcripts.add(key)
            self.log.info(
                "[DPSS-RBC] delivered qid=%s holder=%d mid=%s active=%d delivered=%d",
                qid,
                holder_id,
                mid,
                len(self.dag.Pool.get(qid, set())),
                len(self.dag.DeliveredPool.get(qid, set())),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.info(
                "[DPSS-RBC] output watcher holder=%d ended: %s",
                holder_id,
                exc,
            )
        finally:
            inst = self.instances.get(key)
            if inst is not None:
                inst.delivered = key in self.delivered_transcripts

    @staticmethod
    def _encode_payload(
        ctx: DPSSContext,
        holder_id: int,
        transcript: PVSSTranscript,
    ) -> bytes:
        mid = f"node_{int(holder_id)}"
        return json.dumps(
            {
                "version": 1,
                "qid": ctx.encode_qid(),
                "holder_id": int(holder_id),
                "mid": mid,
                "transcript_hex": transcript.to_bytes().hex(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _decode_payload(payload: bytes) -> dict:
        data = json.loads(bytes(payload).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("DPSS RBC payload must be a JSON object")
        if int(data.get("version", -1)) != 1:
            raise ValueError("unsupported DPSS RBC payload version")
        for key in ("qid", "holder_id", "mid", "transcript_hex"):
            if key not in data:
                raise ValueError(f"missing DPSS RBC payload field: {key}")
        return data

    async def _stop_instance(
        self,
        key: Tuple[Tuple[int, str, str, str, int], int],
    ) -> None:
        inst = self.instances.pop(key, None)
        if inst is None or inst.stopped:
            return
        inst.stopped = True
        for task in (inst.sender_task, inst.output_task, inst.reactor_task):
            if task is not None and not task.done():
                task.cancel()
        tasks = [
            task for task in (inst.sender_task, inst.output_task, inst.reactor_task)
            if task is not None
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self, timeout: Optional[float] = None) -> bool:
        if self._stopping:
            return True
        self._stopping = True
        keys = list(self.instances)
        stop_all = asyncio.gather(
            *(self._stop_instance(key) for key in keys),
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
                "[DPSS-RBC] stop timed out after %.3fs; active=%d",
                timeout,
                self.active_count(),
            )
            return False

    def active_count(self) -> int:
        return len(self.instances)
