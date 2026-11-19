"""Verify the frozen C2 slew floors against held-separate normal signals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.r26_impulse_onset.calibrate import digest, maxima


KEYS = ("horizontal", "positive_vertical", "negative_vertical")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
        parser.error("fresh regular output required")
    model = json.loads(args.model.read_bytes())
    fit_rows, fit_high = maxima(args.fit)
    audit_rows, audit_high = maxima(args.audit)
    fit_expected = [model["fit_max_change_n"][key] for key in KEYS]
    floors = [model["change_floor_n"][key] for key in KEYS]
    gates = {
        "fit_archive_exact": digest(args.fit) == model["fit_archive_sha256"],
        "fit_ids_exact": sorted(fit_rows) == model["fit_episode_ids"],
        "fit_max_exact": fit_high.tolist() == fit_expected,
        "audit_six_held_separate": len(audit_rows) == 6,
        "audit_all_below_frozen_floors": all(value < floor
                                               for value, floor in zip(audit_high, floors)),
    }
    result = {
        "schema": "r26-detached-onset-normal-verification.v1",
        "fit_archive_sha256": digest(args.fit),
        "audit_archive_sha256": digest(args.audit),
        "fit_rows": fit_rows,
        "audit_rows": audit_rows,
        "audit_max_change_n": dict(zip(KEYS, audit_high.tolist())),
        "frozen_change_floor_n": dict(zip(KEYS, floors)),
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
