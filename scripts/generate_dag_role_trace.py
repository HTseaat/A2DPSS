from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from beaver.atcaas_params import ATCaaSProtocolParameters
from beaver.protocol_trace import CommitteeRoleTraceArtifact
from beaver.vrf import vrf_prove_sortition, vrf_verify_sortition
from beaver.vrf import (
    BERNOULLI_COMMITTEE_MODE,
    FIXED_SIZE_COMMITTEE_MODE,
    fixed_size_committee,
    normalize_committee_selection_mode,
)
from scripts.run_dag_tob_phase3 import _vertex_committee_label
from scripts.run_key_gen_atcaas import _srs_generators


def _load_vrf_material(config_dir: Path, M: int):
    configs = []
    for node_id in range(M):
        path = config_dir / f"local.{node_id}.json"
        if not path.is_file():
            raise ValueError(f"missing node config: {path}")
        configs.append(json.loads(path.read_text(encoding="utf-8")))

    params = ATCaaSProtocolParameters.from_mapping(configs[0])
    if params.M != M:
        raise ValueError(f"config M={params.M} does not match requested M={M}")
    for node_id, config in enumerate(configs):
        if int(config.get("my_id", -1)) != node_id:
            raise ValueError(f"local.{node_id}.json has the wrong my_id")
        if ATCaaSProtocolParameters.from_mapping(config) != params:
            raise ValueError("node configs do not share identical parameters")

    extras = [config.get("extra") or {} for config in configs]
    vrf_pks = extras[0].get("vrf_pks")
    if not isinstance(vrf_pks, list) or len(vrf_pks) != M:
        raise ValueError("config does not contain M shared VRF public keys")
    if any(extra.get("vrf_pks") != vrf_pks for extra in extras[1:]):
        raise ValueError("node configs disagree on VRF public keys")
    vrf_sks = [extra.get("vrf_sk") for extra in extras]
    if any(secret is None for secret in vrf_sks):
        raise ValueError("every node config must contain its local VRF secret key")

    encoded_srs = extras[0].get("SRS")
    if not isinstance(encoded_srs, str):
        raise ValueError("config does not contain a base64-encoded SRS")
    if any(extra.get("SRS") != encoded_srs for extra in extras[1:]):
        raise ValueError("node configs disagree on the SRS")
    g_hex, _ = _srs_generators(base64.b64decode(encoded_srs))
    return params, vrf_sks, vrf_pks, g_hex


def selected_roles_for_round(
    *,
    epoch: str,
    parent_round: int,
    M: int,
    threshold_fraction: float,
    vrf_sks: Sequence,
    vrf_pks: Sequence,
    g_hex: str,
    selection_mode: str = BERNOULLI_COMMITTEE_MODE,
    committee_size: Optional[int] = None,
    fixed_committee_seed: str = "",
) -> Tuple[int, ...]:
    label = _vertex_committee_label(epoch, parent_round)
    selection_mode = normalize_committee_selection_mode(selection_mode)
    if selection_mode == FIXED_SIZE_COMMITTEE_MODE:
        if committee_size is None:
            raise ValueError("fixed-size selection requires committee_size")
        return fixed_size_committee(
            label=label,
            population_size=M,
            committee_size=committee_size,
            seed=fixed_committee_seed,
        )
    selected: List[int] = []
    for node_id in range(M):
        is_selected, proof = vrf_prove_sortition(
            label=label,
            vrf_sk=vrf_sks[node_id],
            g_hex=g_hex,
            threshold_fraction=threshold_fraction,
        )
        if is_selected:
            if not vrf_verify_sortition(
                label=label,
                proof=proof,
                sender_id=node_id,
                vrf_pks=vrf_pks,
                g_hex=g_hex,
                threshold_fraction=threshold_fraction,
            ):
                raise ValueError(
                    f"node {node_id} produced an invalid VRF proof for round "
                    f"{parent_round + 1}"
                )
            selected.append(node_id)
    return tuple(selected)


def build_trace_for_epoch(
    *,
    epoch: str,
    rounds: int,
    M: int,
    threshold_fraction: float,
    vrf_sks: Sequence,
    vrf_pks: Sequence,
    g_hex: str,
    selection_mode: str = BERNOULLI_COMMITTEE_MODE,
    committee_size: Optional[int] = None,
    fixed_committee_seed: str = "",
) -> Tuple[CommitteeRoleTraceArtifact, Dict[int, Tuple[int, ...]]]:
    roles_by_round = {
        round_id: selected_roles_for_round(
            epoch=epoch,
            parent_round=round_id - 1,
            M=M,
            threshold_fraction=threshold_fraction,
            vrf_sks=vrf_sks,
            vrf_pks=vrf_pks,
            g_hex=g_hex,
            selection_mode=selection_mode,
            committee_size=committee_size,
            fixed_committee_seed=fixed_committee_seed,
        )
        for round_id in range(1, rounds + 1)
    }
    roles_by_wave = {
        wave: roles_by_round[4 * (wave - 1) + 1]
        for wave in range(1, (rounds + 3) // 4 + 1)
        if 4 * (wave - 1) + 1 <= rounds
    }
    return CommitteeRoleTraceArtifact(roles_by_wave), roles_by_round


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a DAG role-only trace from the node-local VRF keys and "
            "find an epoch whose per-round committees satisfy configured bounds."
        )
    )
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--M", required=True, type=int)
    parser.add_argument("--epoch-prefix", required=True)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--min-size", type=int)
    parser.add_argument("--max-size", type=int)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument(
        "--selection-mode",
        default=BERNOULLI_COMMITTEE_MODE,
        choices=sorted(
            {BERNOULLI_COMMITTEE_MODE, FIXED_SIZE_COMMITTEE_MODE}
        ),
    )
    parser.add_argument(
        "--fixed-committee-seed",
        help="public fixture seed; defaults to the selected epoch",
    )
    parser.add_argument(
        "--indices",
        type=int,
        default=10,
        help="number of leader role namespaces to include in the artifact",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.rounds < 1 or args.max_attempts < 1 or args.indices < 1:
        raise ValueError("rounds, max-attempts, and indices must be positive")
    params, vrf_sks, vrf_pks, g_hex = _load_vrf_material(
        args.config_dir, args.M
    )
    min_size = params.quorum_h if args.min_size is None else args.min_size
    max_size = params.m_a if args.max_size is None else args.max_size
    if not 0 < min_size <= max_size <= args.M:
        raise ValueError("committee bounds must satisfy 0 < min <= max <= M")

    threshold_fraction = params.n_a / params.M
    rejected = []
    for attempt in range(args.max_attempts):
        epoch = (
            args.epoch_prefix
            if args.max_attempts == 1
            else f"{args.epoch_prefix}-{attempt:04d}"
        )
        artifact, roles_by_round = build_trace_for_epoch(
            epoch=epoch,
            rounds=args.rounds,
            M=params.M,
            threshold_fraction=threshold_fraction,
            vrf_sks=vrf_sks,
            vrf_pks=vrf_pks,
            g_hex=g_hex,
            selection_mode=args.selection_mode,
            committee_size=params.n_a,
            fixed_committee_seed=(
                args.fixed_committee_seed
                if args.fixed_committee_seed is not None
                else epoch
            ),
        )
        if args.selection_mode == FIXED_SIZE_COMMITTEE_MODE:
            fixture_seed = (
                args.fixed_committee_seed
                if args.fixed_committee_seed is not None
                else epoch
            )
            artifact = CommitteeRoleTraceArtifact(
                {
                    index: selected_roles_for_round(
                        epoch=epoch,
                        parent_round=4 * (index - 1),
                        M=params.M,
                        threshold_fraction=threshold_fraction,
                        vrf_sks=vrf_sks,
                        vrf_pks=vrf_pks,
                        g_hex=g_hex,
                        selection_mode=args.selection_mode,
                        committee_size=params.n_a,
                        fixed_committee_seed=fixture_seed,
                    )
                    for index in range(1, args.indices + 1)
                }
            )
        invalid = {
            round_id: len(roles)
            for round_id, roles in roles_by_round.items()
            if not min_size <= len(roles) <= max_size
        }
        if invalid:
            rejected.append({"epoch": epoch, "invalid_round_sizes": invalid})
            continue

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "epoch": epoch,
                    "output": str(args.output),
                    "trace_digest": artifact.digest(),
                    "threshold_fraction": threshold_fraction,
                    "selection_mode": args.selection_mode,
                    "fixed_committee_seed": (
                        args.fixed_committee_seed
                        if args.fixed_committee_seed is not None
                        else epoch
                    )
                    if args.selection_mode == FIXED_SIZE_COMMITTEE_MODE
                    else None,
                    "committee_bounds": [min_size, max_size],
                    "committee_roles_by_round": {
                        str(round_id): list(roles)
                        for round_id, roles in roles_by_round.items()
                    },
                },
                sort_keys=True,
            )
        )
        return

    raise SystemExit(
        "no epoch satisfied committee bounds: "
        + json.dumps(rejected[-10:], sort_keys=True)
    )


if __name__ == "__main__":
    main()
