"""Fit the preregistered eight-sample detached-force slew floors from normals."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


LAG = 8


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def maxima(path):
    archive = np.load(path)
    rows = {}
    high = np.zeros(3)
    for name in archive.files:
        force = np.asarray(archive[name][:, 1:4], float)
        valid = np.isfinite(force).all(axis=1)
        delta = np.full_like(force, np.nan)
        delta[LAG:] = force[LAG:] - force[:-LAG]
        paired = valid & np.r_[np.zeros(LAG, dtype=bool), valid[:-LAG]]
        values = [float(np.linalg.norm(delta[paired, :2], axis=1).max()),
                  float(delta[paired, 2].max()),
                  float((-delta[paired, 2]).max())]
        rows[name] = values
        high = np.maximum(high, values)
    return rows, high


def model(path):
    rows, high = maxima(path)
    return {
        "schema": "r26-detached-onset-model.v1",
        "fit_archive_sha256": digest(path),
        "fit_episode_ids": sorted(rows),
        "fit_episode_count": len(rows),
        "lag_samples": LAG,
        "sample_period_s": 0.001,
        "formula": "1.2 * normal signed eight-sample change maximum + 0.5 N",
        "fit_max_change_n": {
            "horizontal": float(high[0]),
            "positive_vertical": float(high[1]),
            "negative_vertical": float(high[2]),
        },
        "change_floor_n": {
            "horizontal": float(1.2 * high[0] + 0.5),
            "positive_vertical": float(1.2 * high[1] + 0.5),
            "negative_vertical": float(1.2 * high[2] + 0.5),
        },
        "fit_rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
        parser.error("fresh regular output required")
    data = model(args.fit)
    if data["fit_episode_count"] != 7:
        parser.error("exactly seven preregistered fit normals required")
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
