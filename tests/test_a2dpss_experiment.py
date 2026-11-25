import json
from pathlib import Path

import pytest

from beaver.a2dpss import A2DPSSJustifierManager, A2DPSSPublicFinalizer
from beaver.a2dpss_experiment import (
    A2DPSSPairedFixture,
    build_paired_fixture,
)
from scripts.analyze_a2dpss_tob_pair import analyze_pair


CONFIG = Path("conf/admpc_100_4_4/local.0.json")


@pytest.fixture(scope="module")
def paired_fixture():
    return build_paired_fixture(
        config_path=str(CONFIG),
        trial_id="phase5-test",
        trial_epoch="phase5-test-epoch",
    )


def test_paired_fixture_round_trip_and_public_finalization(
    paired_fixture, tmp_path
):
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps(paired_fixture.to_fixture_obj(), sort_keys=True),
        encoding="utf-8",
    )
    loaded = A2DPSSPairedFixture.from_file(str(fixture_path))

    assert loaded.digest() == paired_fixture.digest()
    assert loaded.qid() == paired_fixture.qid()
    assert loaded.source_holder_ids() == (0, 1, 2)
    assert len(set(loaded.contribution_digests())) == 3

    serialized = loaded.to_fixture_obj()
    assert "holder_shares" not in serialized
    assert "successor_private_keys" not in serialized

    justifier = A2DPSSJustifierManager([loaded.context])
    selected = justifier.select_justified_subset(
        loaded.context.build_qid(), loaded.public_entries()
    )
    assert selected is not None
    assert len(selected) == loaded.context.service_t + 1
    result = A2DPSSPublicFinalizer(justifier).on_deliver(
        loaded.qid(), selected
    )
    assert result is not None
    assert result.context_digest == loaded.context.context_digest


def test_paired_fixture_rejects_noncanonical_or_tampered_data(paired_fixture):
    extra = paired_fixture.to_fixture_obj()
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="canonical"):
        A2DPSSPairedFixture.from_obj(extra)

    tampered = paired_fixture.to_fixture_obj()
    tampered["entries"][0]["sender"] = 99
    with pytest.raises(ValueError):
        A2DPSSPairedFixture.from_obj(tampered)


def test_pair_analyzer_fails_closed_when_a_protocol_has_no_summaries(
    paired_fixture,
):
    result = analyze_pair(
        dag={},
        acs={},
        fixture=paired_fixture,
        expected_nodes=4,
        dag_exit_code=1,
        acs_exit_code=0,
    )
    assert result["status"] == "failed"
    assert any("DAG runner exited" in item for item in result["failures"])
    assert any("summary nodes" in item for item in result["failures"])
