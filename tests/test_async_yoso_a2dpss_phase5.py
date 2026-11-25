"""Phase 5 contracts for predicate-gated network ACS proposals."""

from __future__ import annotations

import asyncio

import pytest

import beaver.async_yoso_acs as async_yoso_acs
from beaver.a2dpss import A2DPSSContribution
from beaver.a2dpss_acs_adapter import A2DPSSACSAdapter
from beaver.async_yoso_acs import NetworkAsyncYosoACS
from beaver.async_yoso_tob import (
    AsyncYosoTOB,
    BroadcastEnvelope,
    MessageRef,
    PendingEntry,
)
from scripts.run_async_yoso_tob import _validate_a2dpss_proposal_once


class _LoopbackYosoRBC:
    """Minimal RBC boundary used to exercise ACSPROP admission."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.output_queue = asyncio.Queue()
        self._stopped = asyncio.Event()

    async def reactor(self) -> None:
        await self._stopped.wait()

    async def sender(self, payload: bytes) -> None:
        await self.output_queue.put((payload, None))


def _network_acs(monkeypatch, *, epoch: str) -> NetworkAsyncYosoACS:
    monkeypatch.setattr(async_yoso_acs, "YosoRBC", _LoopbackYosoRBC)
    monkeypatch.setattr(
        async_yoso_acs,
        "generator_hex_from_srs",
        lambda _srs: "00",
    )
    return NetworkAsyncYosoACS(
        my_id=0,
        M=1,
        t=0,
        epoch=epoch,
        send=None,
        shared_subscribe=None,
        crypto=(None, None, None, None, {}),
        vrf_sk=None,
        vrf_pks=None,
        collect_all=True,
    )


def _detailed_validator(adapter, pending, *, session_id="phase5"):
    records = []
    result_cache = {}

    def validate(proposer, refs):
        return _validate_a2dpss_proposal_once(
            node_id=0,
            session_id=session_id,
            proposer=proposer,
            proposal_refs=refs,
            adapter=adapter,
            pending=pending,
            records=records,
            result_cache=result_cache,
        )

    validate.predicate_records = records
    validate.result_cache = result_cache
    return validate


def test_receiver_charges_each_proposer_block_exactly_once(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = A2DPSSACSAdapter(setup["justifier"])
    refs = [setup["refs"][10], setup["refs"][11]]
    records = []
    result_cache = {}

    def validate(proposer):
        return _validate_a2dpss_proposal_once(
            node_id=7,
            session_id="phase2-count",
            proposer=proposer,
            proposal_refs=refs,
            adapter=adapter,
            pending=setup["pending"],
            records=records,
            result_cache=result_cache,
        )

    assert validate(3).accepted
    after_first = setup["justifier"].validation_metrics_snapshot()
    assert validate(3).accepted
    after_duplicate = setup["justifier"].validation_metrics_snapshot()
    assert after_duplicate == after_first
    assert len(records) == 1

    # Identical refs from another proposer are a distinct candidate block and
    # must execute the formal predicate again, as required by Phase 1.
    assert validate(4).accepted
    assert len(records) == 2
    assert {record["proposer"] for record in records} == {3, 4}
    assert len({record["block_id"] for record in records}) == 2
    assert all(record["node"] == 7 for record in records)


def test_network_acs_buffers_proposal_until_transcripts_are_pending(
    monkeypatch,
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = A2DPSSACSAdapter(setup["justifier"])
    tob = AsyncYosoTOB(my_id=0, M=13, epoch="phase5-buffer")
    refs = [setup["refs"][10], setup["refs"][11]]
    acs = _network_acs(monkeypatch, epoch="phase5-buffer")
    validator = _detailed_validator(
        adapter,
        tob.pending,
        session_id="qid-buffer",
    )

    async def scenario():
        task = asyncio.create_task(
            acs.collect_proposals(
                refs,
                session_id="qid-buffer",
                proposal_validator=validator,
            )
        )
        await asyncio.sleep(0.18)
        assert not task.done()
        buffered = acs.proposal_metrics_snapshot()
        assert buffered["received"] == 1
        assert buffered["buffered"] == 1
        assert buffered["buffered_current"] == 1
        assert buffered["accepted"] == 0

        for holder_id in (10, 11):
            tob.schedule_message(
                BroadcastEnvelope(
                    sender=holder_id,
                    counter=0,
                    mid=f"node_{holder_id}",
                    payload=setup["entries"][holder_id][1],
                )
            )
        return await asyncio.wait_for(task, timeout=15.0)

    core = asyncio.run(scenario())

    assert [proposal.proposer for proposal in core.proposals] == [0]
    assert core.flatten_refs() == refs
    metrics = acs.proposal_metrics_snapshot()
    assert metrics["checked"] == 1
    assert metrics["accepted"] == 1
    assert metrics["rejected"] == 0
    assert metrics["buffered_current"] == 0
    assert metrics["validation_invocations"] >= 2
    assert metrics["validator_attempts"] >= 2
    assert metrics["accounting_scope"] == "admission-callback-attempts"
    assert metrics["buffered_wait_seconds"] > 0
    assert len(metrics["records"]) == 1
    record = metrics["records"][0]
    assert record["session_id"] == "qid-buffer"
    assert record["proposer"] == 0
    assert record["accepted"] is True
    assert record["reason"] == "accepted"
    assert record["qid_count"] == 1
    assert record["transcript_count"] == 2
    assert record["buffered_wait_seconds"] > 0
    assert len(validator.predicate_records) == 1
    assert validator.predicate_records[0]["proposer"] == 0
    assert validator.predicate_records[0]["accepted"] is True


def test_network_acs_never_counts_unresolved_proposal(
    monkeypatch,
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = A2DPSSACSAdapter(setup["justifier"])
    tob = AsyncYosoTOB(my_id=0, M=13, epoch="phase5-missing")
    acs = _network_acs(monkeypatch, epoch="phase5-missing")

    async def scenario():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                acs.collect_proposals(
                    [setup["refs"][10], setup["refs"][11]],
                    session_id="qid-missing",
                    proposal_validator=_detailed_validator(
                        adapter, tob.pending
                    ),
                ),
                timeout=0.32,
            )

    asyncio.run(scenario())

    metrics = acs.proposal_metrics_snapshot()
    assert metrics["received"] == 1
    assert metrics["buffered"] == 1
    assert metrics["buffered_current"] == 0
    assert metrics["checked"] == 0
    assert metrics["accepted"] == 0
    assert metrics["rejected"] == 0
    assert metrics["records"][0]["reason"] == "unresolved-at-stage-close"


def test_network_acs_rejects_predicate_false_proposal_before_core(
    monkeypatch,
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = A2DPSSACSAdapter(setup["justifier"])
    refs = [setup["refs"][10], setup["refs"][12]]
    pending = {
        setup["refs"][10]: setup["pending"][setup["refs"][10]],
        setup["refs"][12]: PendingEntry(
            ref=setup["refs"][12],
            mid="node_99",
            payload=setup["entries"][12][1],
            received_at=0.0,
        ),
    }
    acs = _network_acs(monkeypatch, epoch="phase5-invalid")

    with pytest.raises(
        RuntimeError, match="without enough justified proposals"
    ):
        asyncio.run(
            acs.collect_proposals(
                refs,
                session_id="qid-invalid",
                proposal_validator=_detailed_validator(adapter, pending),
            )
        )

    metrics = acs.proposal_metrics_snapshot()
    assert metrics["received"] == 1
    assert metrics["checked"] == 1
    assert metrics["accepted"] == 0
    assert metrics["rejected"] == 1
    assert metrics["records"][0]["reason"] == "predicate-false"
    assert metrics["records"][0]["qid_count"] == 1
    assert metrics["records"][0]["transcript_count"] == 2


def test_network_acs_rejects_block_if_any_qid_segment_is_unknown(
    monkeypatch,
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    adapter = A2DPSSACSAdapter(setup["justifier"])
    valid_refs = [setup["refs"][10], setup["refs"][11]]
    unknown_ref = MessageRef(99, 0)
    source = setup["contributions"][12]
    unknown_context = A2DPSSContribution(
        holder_id=source.holder_id,
        holder_eval_point=source.holder_eval_point,
        context_digest="00" * 32,
        transcript=source.transcript,
    )
    pending = dict(setup["pending"])
    pending[unknown_ref] = PendingEntry(
        ref=unknown_ref,
        mid="node_12",
        payload=unknown_context.to_bytes(),
        received_at=0.0,
    )
    acs = _network_acs(monkeypatch, epoch="phase5-mixed")

    with pytest.raises(
        RuntimeError, match="without enough justified proposals"
    ):
        asyncio.run(
            acs.collect_proposals(
                valid_refs + [unknown_ref],
                session_id="qid-mixed",
                proposal_validator=_detailed_validator(adapter, pending),
            )
        )

    metrics = acs.proposal_metrics_snapshot()
    assert metrics["accepted"] == 0
    assert metrics["rejected"] == 1
    assert metrics["records"][0]["reason"].startswith("validator-error:")
