"""Fit the scheduled approach-torque envelope from normal-only R23 traces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


START, STOP = 1200, 1400


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("fresh output required")
    capture_raw, raw_bytes = args.capture.read_bytes(), args.raw.read_bytes()
    capture = json.loads(capture_raw)
    traces = np.load(args.raw)
    calibration = [row for row in capture["rows"] if row["split"] == "calibration_normal"]
    if len(calibration) != 12 or any(row["summary"]["contact_present"] for row in calibration):
        raise ValueError("exactly 12 force-free normal calibration traces required")
    keys = [row["key"] + "__tau_s" for row in calibration]
    values = np.stack([traces[key][START:STOP] for key in keys])
    center = np.median(values, axis=0)
    spread = np.max(np.abs(values - center), axis=0)
    innovation = np.diff(values, axis=1)
    floor = .25 * np.quantile(np.abs(innovation), .999, axis=(0, 1)) + 1e-6
    bound = spread + floor
    payload = {
        "schema": "r23-normal-approach-torque-envelope.v1",
        "classification": "normal-only finite simulator calibration; not hardware or safety proof",
        "plan_sha256": capture["plan_sha256"],
        "capture_sha256": hashlib.sha256(capture_raw).hexdigest(),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "fit_row_keys": [row["key"] for row in calibration],
        "start_index": START, "stop_index": STOP, "dt_s": .001,
        "score_threshold": 3.0, "persistence_samples": 8, "latch_samples": 160,
        "innovation_floor_joint_Nm": floor.tolist(),
        "center_joint_Nm": center.tolist(), "bound_joint_Nm": bound.tolist(),
    }
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"fit_rows": len(calibration), "samples": STOP - START}))


if __name__ == "__main__":
    main()
