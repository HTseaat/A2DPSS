import json
import os
from pathlib import Path

import pytest

from beaver.baselines.bhk22_experiment import BHK22DAGExperimentManifest, BHK22NodePrivateShard
from beaver.baselines.ru22_experiment import RU22DAGExperimentManifest, RU22DAGLocalShard
from beaver.pedersen_shamir import eval_poly
from beaver.yoso_adpss_workload import (
    ARM_ORDER, YOSOADPSSNodePrivateShard, build_yoso_adpss_fixture,
    interpolate_polynomial, load_yoso_adpss_public_fixture,
    verify_same_initial_workload, write_yoso_adpss_fixture,
)


CONFIG = Path(os.environ.get("YOSO_ADPSS_TEST_CONFIG", "/tmp/yoso-adpss-step0-m5-config"))


def _build(seed="unit-seed", trace="12" * 32):
    if not CONFIG.is_dir():
        pytest.skip("local Step-0 five-node config is unavailable")
    return build_yoso_adpss_fixture(
        config_directory=CONFIG, trial_id="unit", trial_epoch="1",
        seed=seed, role_trace_digest=trace, network_profile={"kind": "test"},
    )


def test_interpolate_polynomial_round_trip():
    coefficients = (17, 9, 4)
    assert interpolate_polynomial([(x, eval_poly(coefficients, x)) for x in (1, 2, 3)]) == coefficients


def test_common_fixture_is_reproducible_private_and_cross_arm_bound(tmp_path):
    first, second = _build(), _build()
    assert first.manifest.digest() == second.manifest.digest()
    assert verify_same_initial_workload(first.manifest, first.arms)
    assert {arm.common_workload_digest for arm in first.arms.values()} == {first.manifest.digest()}
    write_yoso_adpss_fixture(first, tmp_path)
    manifest, arms = load_yoso_adpss_public_fixture(tmp_path)
    assert verify_same_initial_workload(manifest, arms)
    public_text = (tmp_path / "manifest.json").read_text() + "".join(
        (tmp_path / "arms" / f"{arm}.json").read_text() for arm in ARM_ORDER
    )
    for forbidden in ("share_s", "share_s_hat", "recipient_secret_key", "coin_private_key"):
        assert forbidden not in public_text
    shard = YOSOADPSSNodePrivateShard.from_obj(json.loads((tmp_path / "node.0.private.json").read_text()))
    assert shard.node_id == 0


def test_protocol_shards_validate_and_ru_candidates_share_same_opening():
    bundle = _build()
    ru_manifest = RU22DAGExperimentManifest.from_obj(bundle.arms["ru22"].protocol_manifest)
    bhk_manifest = BHK22DAGExperimentManifest.from_obj(bundle.arms["bhk22"].protocol_manifest)
    for node, common_shard in bundle.shards.items():
        ru = RU22DAGLocalShard.from_obj(common_shard.ru22, ru_manifest)
        BHK22NodePrivateShard.from_obj(common_shard.bhk22, bhk_manifest)
        if ru.source:
            openings = {(item.share_s, item.share_s_hat) for item in ru.source.local_vps_shares}
            assert len(openings) == 1


def test_seed_and_trace_change_public_binding():
    assert _build("seed-a").manifest.digest() != _build("seed-b").manifest.digest()
    assert _build(trace="21" * 32).manifest.digest() != _build(trace="22" * 32).manifest.digest()
