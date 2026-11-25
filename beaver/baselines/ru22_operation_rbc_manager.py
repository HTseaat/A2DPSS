"""YOSO-RBC transport for RU22 operation-coupled contribution batches."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from beaver.baselines.ru22_operation import (
    RU22OperationContributionBatch,
    build_ru22_operation_qid,
)
from beaver.baselines.ru22_rbc_manager import RU22ContributionRBCManager
from beaver.threshold_operation_contract import ThresholdOperationContext


_KIND = "atcaas-ru22-operation-contribution-batch-rbc"
_VERSION = 1


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate RU22 operation RBC field: {key}")
        result[key] = value
    return result


class RU22OperationContributionRBCManager(RU22ContributionRBCManager):
    """Reuse RU22's one-RBC-per-source topology with application wire bytes."""

    def __init__(
        self, *, operation_context: ThresholdOperationContext, trial_id: str,
        **kwargs,
    ) -> None:
        if not isinstance(operation_context, ThresholdOperationContext):
            raise TypeError("operation_context has the wrong type")
        if not isinstance(trial_id, str) or not trial_id:
            raise ValueError("trial_id must be non-empty")
        self.operation_context = operation_context
        self.trial_id = trial_id
        super().__init__(**kwargs)

    def operation_qid(self, context) -> str:
        return build_ru22_operation_qid(
            context, self.operation_context, trial_id=self.trial_id
        ).encode()

    async def broadcast_batch(
        self, context, *, holder_role_id: int, batch_bytes: bytes,
    ) -> None:
        holder_role_id = self._validate_role(context, holder_role_id)
        if self.source_role_nodes[holder_role_id] != self.my_id:
            raise ValueError("only the assigned RU22 source role may send")
        raw = bytes(batch_bytes)
        batch = RU22OperationContributionBatch.from_bytes(raw)
        if (
            batch.qid != self.operation_qid(context)
            or batch.base_context_digest != context.context_digest
            or batch.operation_context_digest != self.operation_context.context_digest
            or batch.trial_id != self.trial_id
            or batch.holder_role_id != holder_role_id
            or context.holder_eval_points.get(holder_role_id)
            != batch.holder_eval_point
        ):
            raise ValueError("RU22 operation batch binding mismatch")
        self.ensure_instance(context, holder_role_id)
        instance = self.instances[self._key(context, holder_role_id)]
        digest = hashlib.sha256(raw).hexdigest()
        if instance.sender_task is not None:
            if instance.outbound_digest != digest:
                raise ValueError("RU22 operation source attempted equivocation")
            return
        envelope = self._encode_operation_payload(
            context, holder_role_id, raw
        )
        instance.outbound_digest = digest
        self.metrics.outbound_batches += 1
        self.metrics.outbound_envelope_bytes += len(envelope)
        self.metrics.outbound_batch_bytes += len(raw)

        async def _send() -> None:
            await asyncio.sleep(0.1)
            await instance.rbc.sender(envelope)

        instance.sender_task = asyncio.create_task(_send())

    def _encode_operation_payload(
        self, context, holder_role_id: int, raw: bytes,
    ) -> bytes:
        batch = RU22OperationContributionBatch.from_bytes(raw)
        return _canonical_json({
            "kind": _KIND,
            "version": _VERSION,
            "qid": self.operation_qid(context),
            "context_digest": context.context_digest,
            "operation_context_digest": self.operation_context.context_digest,
            "trial_id": self.trial_id,
            "holder_role_id": holder_role_id,
            "mid": f"node_{holder_role_id}",
            "batch_digest": batch.batch_digest,
            "batch_hex": raw.hex(),
        })

    def _decode_operation_payload(
        self, payload: bytes, *, context, holder_role_id: int,
    ) -> tuple[str, bytes, RU22OperationContributionBatch]:
        try:
            data = json.loads(
                bytes(payload).decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number: {value}")
                ),
            )
        except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("malformed RU22 operation RBC payload") from exc
        expected = {
            "kind", "version", "qid", "context_digest",
            "operation_context_digest", "trial_id", "holder_role_id", "mid",
            "batch_digest", "batch_hex",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("RU22 operation RBC fields mismatch")
        if bytes(payload) != _canonical_json(data):
            raise ValueError("RU22 operation RBC envelope is not canonical")
        mid = f"node_{holder_role_id}"
        if (
            data["kind"] != _KIND
            or type(data["version"]) is not int
            or data["version"] != _VERSION
            or data["qid"] != self.operation_qid(context)
            or data["context_digest"] != context.context_digest
            or data["operation_context_digest"]
            != self.operation_context.context_digest
            or data["trial_id"] != self.trial_id
            or type(data["holder_role_id"]) is not int
            or data["holder_role_id"] != holder_role_id
            or data["mid"] != mid
        ):
            raise ValueError("RU22 operation RBC identity mismatch")
        encoded = data["batch_hex"]
        if not isinstance(encoded, str):
            raise ValueError("RU22 operation RBC batch_hex must be text")
        try:
            raw = bytes.fromhex(encoded)
        except ValueError as exc:
            raise ValueError("malformed RU22 operation RBC batch hex") from exc
        if raw.hex() != encoded:
            raise ValueError("RU22 operation RBC batch hex is non-canonical")
        batch = RU22OperationContributionBatch.from_bytes(raw)
        if (
            data["batch_digest"] != batch.batch_digest
            or batch.qid != data["qid"]
            or batch.holder_role_id != holder_role_id
        ):
            raise ValueError("RU22 operation RBC batch mismatch")
        return mid, raw, batch

    async def _watch_output(self, context, holder_role_id: int, rbc) -> None:
        key = self._key(context, holder_role_id)
        try:
            payload, _ = await rbc.output_queue.get()
            mid, raw, _batch = self._decode_operation_payload(
                payload, context=context, holder_role_id=holder_role_id
            )
            if key not in self.delivered_batches:
                self.dag.deliver_message(self.operation_qid(context), mid, raw)
                self.delivered_batches.add(key)
                self.metrics.delivered_batches += 1
                self.metrics.delivered_batch_bytes += len(raw)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.metrics.rejected_payloads += 1
            self.log.warning(
                "[RU22:OP:RBC] rejected holder_role=%d payload: %s",
                holder_role_id, exc,
            )
        finally:
            instance = self.instances.get(key)
            if instance is not None:
                instance.delivered = key in self.delivered_batches


__all__ = ["RU22OperationContributionRBCManager"]
