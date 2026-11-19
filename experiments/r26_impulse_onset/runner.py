"""Run C2 through the frozen combined-system study without editing its registry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.r26_delivery_pair import study
from experiments.r26_delivery_pair.controller import Controller
from experiments.r26_impulse_onset.candidate import Detector


def install_factory():
    original = study.factory

    def factory(key):
        if key == "c2":
            return Detector, Controller
        return original(key)

    study.factory = factory


def evaluate(panel, arms):
    install_factory()
    rows = []
    for case, impacts in study.panel(panel):
        rows.append(study.evaluate(case, impacts, arms))
    return {
        "schema": "r26-detached-onset-study.v1",
        "panel": panel,
        "classification": "disclosed finite laboratory simulation; not ORIGINAL, independent challenge, hardware, deployment, or safety proof",
        "arms": list(arms),
        "rows": rows,
        "episodes": sum(row["episodes"] for row in rows),
        "promotion_eligible": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("normal", "target"), required=True)
    parser.add_argument("--arms", nargs="+", choices=("r10", "r19", "r24", "pair", "c2"),
                        required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
        parser.error("fresh regular output required")
    if len(args.arms) != len(set(args.arms)):
        parser.error("unique arms required")
    data = evaluate(args.panel, tuple(args.arms))
    args.out.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
