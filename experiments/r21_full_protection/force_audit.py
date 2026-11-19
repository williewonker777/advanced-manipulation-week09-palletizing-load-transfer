"""Independent exact-input audit for the 180 fixed D4 artificial forces."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np

import my_torque
from sim import runner
from tools.check_integrity import check_integrity
from tools.d4_robustness import build_cases


ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TracePlant(runner.Plant):
    latest: "TracePlant | None" = None

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.inputs: list[np.ndarray] = []
        type(self).latest = self

    def push(self, force: Any) -> Any:
        self.inputs.append(np.asarray(force, dtype=float).copy())
        return super().push(force)


def independent_expected(t: np.ndarray, collision: Any) -> np.ndarray:
    onset, force = collision
    force = np.asarray(force, dtype=float)
    rows = []
    for instant in t:
        age = instant - onset
        scale = min(age / 0.02, 1.0, (0.12 - age) / 0.02)
        rows.append(max(scale, 0.0) * force)
    return np.asarray(rows)


def audit(case: dict[str, Any]) -> dict[str, Any]:
    original = runner.Plant
    TracePlant.latest = None
    runner.Plant = TracePlant
    try:
        _, log = runner.run_episode(
            my_torque.Controller(), box_mass=case["mass"], box_com=tuple(case["com"]),
            detector=None, collision=case["collision"], pick_offset=case.get("pick_offset"))
    finally:
        runner.Plant = original
    plant = TracePlant.latest
    if plant is None:
        raise RuntimeError("instrumented Plant was not constructed")
    actual = np.asarray(plant.inputs, dtype=float)
    t = np.asarray(log["t"])[:len(actual)]
    expected = independent_expected(t, case["collision"])
    scalar_stock_order = np.asarray([np.linalg.norm(row) for row in actual])
    nonzero = np.flatnonzero(scalar_stock_order > 0)
    onset = int(np.searchsorted(t, case["collision"][0]))
    return {
        "case_id": case["case_id"], "onset_index": onset,
        "sample_count": len(actual),
        "signed_input_exact": bool(np.array_equal(actual, expected)),
        "stock_scalar_exact": bool(np.array_equal(np.asarray(log["f_ext"])[:len(actual)], scalar_stock_order)),
        "onset_vector_exactly_zero": bool(np.array_equal(actual[onset], np.zeros(3))),
        "first_nonzero_offset": int(nonzero[0] - onset),
        "last_nonzero_offset": int(nonzero[-1] - onset),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        parser.error("new output path required")
    raw = args.plan.read_bytes()
    plan = json.loads(raw)
    for relative, digest in plan["source_pins"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError("source pin differs: " + relative)
    cases = [case for case in build_cases("my_torque") if case["kind"] == "collision"]
    matrix = (json.dumps(build_cases("my_torque"), sort_keys=True,
                         separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(cases) != 180 or hashlib.sha256(matrix).hexdigest() != plan["matrix_sha256"]:
        raise ValueError("fixed D4 collision matrix differs")
    before = {relative: sha(ROOT / relative) for relative in plan["source_pins"]}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = list(pool.imap(audit, cases))
    after = {relative: sha(ROOT / relative) for relative in plan["source_pins"]}
    if before != after:
        raise RuntimeError("source changed during audit")
    result = {
        "schema": "r21-full-d4-force-audit.v1",
        "classification": "exact stock-input audit; no detector outcomes",
        "plan_sha256": hashlib.sha256(raw).hexdigest(),
        "kit_integrity": check_integrity(ROOT),
        "cases": len(rows),
        "all_signed_input_exact": all(row["signed_input_exact"] for row in rows),
        "all_stock_scalar_exact": all(row["stock_scalar_exact"] for row in rows),
        "onset_exact_zero_cases": sum(row["onset_vector_exactly_zero"] for row in rows),
        "first_nonzero_offset_counts": {
            str(value): sum(row["first_nonzero_offset"] == value for row in rows)
            for value in sorted({row["first_nonzero_offset"] for row in rows})},
        "last_nonzero_offset_counts": {
            str(value): sum(row["last_nonzero_offset"] == value for row in rows)
            for value in sorted({row["last_nonzero_offset"] for row in rows})},
        "source_pins_before": before, "source_pins_after": after, "rows": rows,
    }
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "cases", "all_signed_input_exact", "all_stock_scalar_exact",
        "onset_exact_zero_cases", "first_nonzero_offset_counts", "last_nonzero_offset_counts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
