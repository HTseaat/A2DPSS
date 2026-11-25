"""Build one immutable A²DPSS fixture for paired DAG/ACS replay."""

from __future__ import annotations

import argparse
import json

from beaver.a2dpss_experiment import build_paired_fixture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--trial-epoch", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fixture = build_paired_fixture(
        config_path=args.config,
        trial_id=args.trial_id,
        trial_epoch=args.trial_epoch,
    )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(
            fixture.to_fixture_obj(),
            handle,
            sort_keys=True,
            separators=(",", ":"),
        )
        handle.write("\n")
    print(
        json.dumps(
            {
                "fixture_digest": fixture.digest(),
                "trial_id": fixture.trial_id,
                "trial_epoch": fixture.trial_epoch,
                "qid": fixture.qid(),
                "contribution_digests": list(
                    fixture.contribution_digests()
                ),
                "holder_metrics": fixture.holder_metrics_summary(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
