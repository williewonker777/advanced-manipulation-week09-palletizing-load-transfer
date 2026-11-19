"""Save and directly certify actual stock BOTH force arrays for two cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r16_payload_contract.controller import Controller
from experiments.r18_paired_impulses.runner import run_episode
from experiments.r19_impulse_end.detector import Detector
from experiments.r20_force_certificate.certificate import exact_trace


def read(path):
    return json.loads(Path(path).read_text())


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def mismatches(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        raise ValueError("array shape mismatch")
    if left.ndim == 1:
        return np.flatnonzero(left != right).tolist()
    return np.flatnonzero(np.any(left != right, axis=tuple(range(1, left.ndim)))).tolist()


def run_case(case, height, early, model):
    detector = Detector()
    detector.model_path = Path(model)
    metrics, log = run_episode(
        Controller(height, early), case["mass"], tuple(case["com"]), detector,
        pick_offset=case.get("offset"), impacts=case["impacts"],
    )
    if metrics["diverged"]:
        raise ValueError("divergent direct cross-check")
    expected = exact_trace(case, "both")
    actual = {
        "t": np.asarray(log["t"]),
        "native": np.asarray(log["f_ext_xyz_native"]),
        "extra": np.asarray(log["f_ext_xyz_second"]),
        "combined": np.asarray(log["f_ext_xyz_committed"]),
        "scalar": np.asarray(log["f_ext"]),
    }
    mismatch = {key: mismatches(actual[key], expected[key]) for key in actual}
    batched = np.linalg.norm(actual["native"], axis=1)
    norm_ticks = np.flatnonzero(batched != actual["scalar"])
    norm_rows = [{
        "tick": int(k), "time_s": float(actual["t"][k]),
        "signed_xyz_N": actual["native"][k].tolist(),
        "actual_stock_scalar_N": float(actual["scalar"][k]),
        "batched_axis_norm_N": float(batched[k]),
        "absolute_difference_N": float(abs(actual["scalar"][k] - batched[k])),
    } for k in norm_ticks]
    if any(mismatch.values()):
        raise ValueError("actual stock force differs from exact certificate")
    return {
        "id": case["id"], "diverged": False,
        "exact_mismatch_ticks": mismatch,
        "all_actual_arrays_exact": True,
        "actual_array_sha256": {key: array_sha(value) for key, value in actual.items()},
        "batched_axis_norm_difference_rows": norm_rows,
        "actual": {key: value.tolist() for key, value in actual.items()},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=Path, required=True)
    parser.add_argument("--early", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--certificate-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("new output path required")
    try:
        plan, cert = read(args.plan), read(args.certificate_plan)
        cases = {row["id"]: row for row in plan["fresh"]}
        ids = cert["case_ids"]
        if len(ids) != 2 or any(case not in cases for case in ids):
            raise ValueError("expected the two frozen certificate cases")
        height, early = read(args.height), read(args.early)
        rows = [run_case(cases[case], height, early, args.model) for case in ids]
        result = {
            "schema": "r20-force-direct-stock-crosscheck.v1",
            "provenance_only_not_detection_attempt": True,
            "episodes": len(rows), "detector": "frozen R19", "rows": rows,
        }
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({row["id"]: {
        "signed_mismatch_ticks": {key: value for key, value in row["exact_mismatch_ticks"].items()
                                  if key in ("native", "extra", "combined")},
        "scalar_mismatch_ticks": row["exact_mismatch_ticks"]["scalar"],
        "axis_norm_difference_ticks": [item["tick"]
                                       for item in row["batched_axis_norm_difference_rows"]],
    } for row in rows}))


if __name__ == "__main__":
    main()
