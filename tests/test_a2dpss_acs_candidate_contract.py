"""Contracts for the A²DPSS predicate used by ACS proposals.

The candidate predicate intentionally differs from the existing DAG delivery
predicate: a candidate is an all-valid superset of size at least ``t_r + 1``;
the reconstruction set ``T`` remains an exact, canonical threshold subset.

The candidate API is independent from the existing exact-threshold DAG
delivery predicate.  Phase 1 also requires every formal predicate invocation
to repeat the full contribution checks, even for byte-identical blocks.
"""

from __future__ import annotations

import itertools

import pytest

from beaver.a2dpss import A2DPSSJustifierManager
from beaver.a2dpss_acs_adapter import A2DPSSACSAdapter
from beaver.dag_tob import Qid


def _candidate_entries(setup, *holder_ids):
    return [setup["entries"][holder_id] for holder_id in holder_ids]


def test_exact_threshold_valid_entries_pass_candidate_predicate(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup

    assert setup["justifier"].verify_candidate_set(
        setup["context"].build_qid(),
        _candidate_entries(setup, 10, 11),
    )


def test_valid_superset_passes_candidate_predicate(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup

    assert setup["justifier"].verify_candidate_set(
        setup["context"].build_qid(),
        _candidate_entries(setup, 12, 10, 11),
    )


def test_candidate_below_threshold_fails(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup

    assert not setup["justifier"].verify_candidate_set(
        setup["context"].build_qid(),
        _candidate_entries(setup, 10),
    )


@pytest.mark.parametrize(
    "invalid_entry_key",
    ("wrong_anchor_entry", "tampered_pvss_entry"),
)
def test_one_invalid_pvss_or_anchor_rejects_entire_candidate(
    a2dpss_acs_contract_setup,
    invalid_entry_key,
) -> None:
    setup = a2dpss_acs_contract_setup
    candidate = [
        setup["entries"][10],
        setup["entries"][11],
        setup[invalid_entry_key],
    ]

    assert not setup["justifier"].verify_candidate_set(
        setup["context"].build_qid(),
        candidate,
    )


def test_duplicate_holder_identity_rejects_entire_candidate(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    candidate = [
        setup["entries"][10],
        setup["entries"][10],
        setup["entries"][11],
    ]

    assert not setup["justifier"].verify_candidate_set(
        setup["context"].build_qid(),
        candidate,
    )


def test_unregistered_qid_fails_candidate_predicate(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    unknown = Qid.build("a2dpss", context_digest="00" * 32)

    assert not setup["justifier"].verify_candidate_set(
        unknown,
        _candidate_entries(setup, 10, 11),
    )


def test_deduplicated_union_of_valid_candidates_remains_valid(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    left = _candidate_entries(setup, 10, 11)
    right = _candidate_entries(setup, 11, 12)
    union = list(dict.fromkeys(left + right))

    assert setup["justifier"].verify_candidate_set(
        setup["context"].build_qid(),
        left,
    )
    assert setup["justifier"].verify_candidate_set(
        setup["context"].build_qid(),
        right,
    )
    assert setup["justifier"].verify_candidate_set(
        setup["context"].build_qid(),
        union,
    )


def test_superset_selects_an_exact_deterministic_T(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    qid = setup["context"].build_qid()
    superset = _candidate_entries(setup, 12, 10, 11)

    selected = setup["justifier"].select_justified_subset(qid, superset)

    assert selected == _candidate_entries(setup, 10, 11)
    assert len(selected) == setup["context"].service_t + 1


def test_every_superset_permutation_selects_the_same_T(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    qid = setup["context"].build_qid()
    superset = _candidate_entries(setup, 10, 11, 12)
    expected = _candidate_entries(setup, 10, 11)

    for permutation in itertools.permutations(superset):
        assert setup["justifier"].select_justified_subset(
            qid, list(permutation)
        ) == expected


def test_preverified_selection_repeats_structure_but_not_crypto(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    manager = setup["justifier"]
    qid = setup["context"].build_qid()
    entries = _candidate_entries(setup, 12, 10, 11)
    before = manager.validation_metrics_snapshot()

    assert manager.select_preverified_subset(qid, entries) == (
        _candidate_entries(setup, 10, 11)
    )
    assert manager.select_preverified_subset(
        qid,
        [entries[0], entries[0], entries[1]],
    ) is None
    assert manager.select_preverified_subset(
        qid,
        [("node_99", entries[0][1]), entries[1]],
    ) is None

    after = manager.validation_metrics_snapshot()
    assert after == before


def test_existing_dag_exact_set_contract_is_unchanged(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    qid = setup["context"].build_qid()

    assert setup["justifier"].verify_qid_set(
        qid,
        _candidate_entries(setup, 10, 11),
    )
    assert not setup["justifier"].verify_qid_set(
        qid,
        _candidate_entries(setup, 10, 11, 12),
    )


@pytest.mark.parametrize(
    "predicate_name",
    (
        "verify_candidate_set",
        "verify_qid_set",
        "select_justified_subset",
    ),
)
def test_formal_predicates_revalidate_identical_block_bytes(
    a2dpss_acs_contract_setup,
    predicate_name,
) -> None:
    setup = a2dpss_acs_contract_setup
    manager = A2DPSSJustifierManager([setup["context"]])
    qid = setup["context"].build_qid()
    entries = _candidate_entries(setup, 10, 11)

    # Warm the auxiliary cache first.  Formal block predicates must ignore it.
    for entry in entries:
        assert manager.validate_contribution(
            setup["context"], *entry
        ) is not None
        assert manager.validate_contribution(
            setup["context"], *entry
        ) is not None
    before = manager.validation_metrics_snapshot()

    predicate = getattr(manager, predicate_name)
    first = predicate(qid, entries)
    second = predicate(qid, entries)
    after = manager.validation_metrics_snapshot()

    assert bool(first) and bool(second)
    assert after["cache_hits"] == before["cache_hits"]
    # Two entries in each of two identical blocks: every stage runs four times.
    for stage in ("schema_identity", "pvss", "anchor"):
        assert (
            after[stage]["invocations"]
            - before[stage]["invocations"]
        ) == 4


def test_two_candidate_ref_blocks_with_identical_refs_revalidate(
    a2dpss_acs_contract_setup,
) -> None:
    setup = a2dpss_acs_contract_setup
    manager = A2DPSSJustifierManager([setup["context"]])
    adapter = A2DPSSACSAdapter(manager)
    refs = [setup["refs"][10], setup["refs"][11]]
    before = manager.validation_metrics_snapshot()

    assert adapter.validate_candidate_refs(refs, setup["pending"])
    assert adapter.validate_candidate_refs(refs, setup["pending"])
    after = manager.validation_metrics_snapshot()

    for stage in ("schema_identity", "pvss", "anchor"):
        assert (
            after[stage]["invocations"]
            - before[stage]["invocations"]
        ) == 4
