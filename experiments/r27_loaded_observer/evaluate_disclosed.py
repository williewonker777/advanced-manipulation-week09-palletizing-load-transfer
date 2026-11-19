"""Evaluate the frozen R27 observer on only the three disclosed R26 singles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from experiments.r26_holdout import evaluator as lab
from experiments.r26_yaw_acquisition.candidate import Detector as R26
from experiments.r27_loaded_observer.candidate import Detector as R27


PROTOCOL_SHA256 = "d489e5f9985d84e681310132692eec2a51f9ccc0c91a2652247f5c8f9025c10a"
CANDIDATE_SHA256 = "6c24a8d1944e3c5c23fd2c6a24f1b2afce1054c4817ec443cf5513bb21f87d4b"
FIRST_SHA256 = "854459ef3292f3c20eae7bc8ba829a83eae70e6ce2b5e66f5b89abeb6836489a"
CASES = ("s26_long_carry_signed", "s26_long_place_lateral",
         "s26_long_transfer_rotated")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Probe(R27):
    def reset(self):
        super().reset()
        self.r27_diagnostic = []

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
        state = self.loaded_force_state
        self.r27_diagnostic.append({
            "tick": int(round(1000.0 * float(t))),
            "report": bool(report),
            "extra": bool(self.loaded_force_extra),
            "eligible": bool(self.loaded_force_eligible),
            "owner": bool(state.owner),
            "pending": int(state.pending),
            "onset_score_N": float(state.last_onset_score),
            "sustain_score_N": float(state.last_sustain_score),
            "projection_N": float(state.last_projection),
            "onset_count": int(state.onset_count),
        })
        return residual, report


def run(case, variant, detector_type, impacts):
    holders = []

    def factory():
        item = detector_type()
        holders.append(item)
        return item

    _, log = lab.run_episode(case, variant, factory, impacts)
    if len(holders) != 1:
        raise ValueError("unexpected detector construction count")
    return log, holders[0]


def rises(bits):
    values = np.asarray(bits, bool)
    return np.flatnonzero(values & ~np.r_[False, values[:-1]]).astype(int).tolist()


def evaluate(case, variant_name, variant, archive_row, plan):
    assignment = {"none": (None, None), "single": (case["impact"], None)}
    traces, detectors = {}, {}
    for arm, detector_type in (("r26", R26), ("r27", Probe)):
        traces[arm], detectors[arm] = {}, {}
        for twin, impacts in assignment.items():
            traces[arm][twin], detectors[arm][twin] = run(
                case, variant, detector_type, impacts)
    start, boundary = lab.actual_start(traces["r26"]["single"]["t"], case["impact"])
    end = min(6000, start + plan["event_window_ms"])
    records = {}
    for arm in ("r26", "r27"):
        none, single = traces[arm]["none"], traces[arm]["single"]
        records[arm] = {
            "event": lab.event(single, [none], start, end, plan),
            "twins": {
                twin: lab.common(log, assignment[twin], [start] if twin == "single" else [], plan)
                for twin, log in traces[arm].items()
            },
            "prefix": lab.certified.prefix(none, single, start, start),
            "boundary": boundary,
            "physical_signature": lab.certified.trace_signature(traces[arm]),
        }
    expected = archive_row["records"]["r26"]
    reproduction = {
        "r26_event_exact": records["r26"]["event"] == expected["event"],
        "r26_twins_exact": records["r26"]["twins"] == expected["twins"],
        "r26_prefix_exact": records["r26"]["prefix"] == expected["prefix"],
        "r26_boundary_exact": boundary == expected["boundary"],
        "r26_physics_exact": records["r26"]["physical_signature"]
        == archive_row["r25_physics_sha256"],
        "r27_physics_exact": records["r27"]["physical_signature"]
        == archive_row["r25_physics_sha256"],
    }
    if not all(reproduction.values()):
        raise ValueError("disclosed R26 reproduction differs")
    diagnostics = {}
    for twin in ("none", "single"):
        rows = detectors["r27"][twin].r27_diagnostic
        diagnostics[twin] = {
            "extra_rise_indices": [rows[index]["tick"] for index in rises(
                [row["extra"] for row in rows])],
            "extra_active_samples": int(sum(row["extra"] for row in rows)),
            "owner_samples": int(sum(row["owner"] for row in rows)),
            "eligible_samples": int(sum(row["eligible"] for row in rows)),
            "onset_count": int(max(row["onset_count"] for row in rows)),
            "onset_score_peak_N": float(max(row["onset_score_N"] for row in rows)),
            "sustain_score_peak_N": float(max(row["sustain_score_N"] for row in rows)),
        }
    return {
        "id": case["id"],
        "variant": variant_name,
        "role": "target_miss" if case["id"] != CASES[-1] else "detected_comparator",
        "episodes": 4,
        "actual_boundary": boundary,
        "reproduction": reproduction,
        "records": records,
        "r27_diagnostics": diagnostics,
        "r27_vs_r26": {
            "new_attributable_rises": sorted(set(records["r27"]["event"]["attributable_rise_indices"])
                                             - set(records["r26"]["event"]["attributable_rise_indices"])),
            "lost_attributable_rises": sorted(set(records["r26"]["event"]["attributable_rise_indices"])
                                              - set(records["r27"]["event"]["attributable_rise_indices"])),
            "none_detect_trace_exact": bool(np.array_equal(
                traces["r26"]["none"]["detect"], traces["r27"]["none"]["detect"])),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh output required")
    if mujoco.__version__ != "3.17.0":
        raise RuntimeError("MuJoCo 3.17.0 required")
    candidate = Path(__file__).with_name("candidate.py")
    if (sha(args.protocol) != PROTOCOL_SHA256 or sha(candidate) != CANDIDATE_SHA256
            or sha(args.first) != FIRST_SHA256):
        raise ValueError("frozen forced inputs differ")
    protocol = json.loads(args.protocol.read_bytes())
    archive = json.loads(args.first.read_bytes())
    plan = json.loads(lab.PLAN.read_bytes())
    rows = []
    for case_id in CASES:
        case = next(item for item in plan["single"] if item["id"] == case_id)
        for variant_name in protocol["known_cases"]["mechanics"]:
            archived = next(item for item in archive["rows"]
                            if item["id"] == case_id and item["variant"] == variant_name)
            rows.append(evaluate(case, variant_name, plan["variants"][variant_name],
                                 archived, plan))
            print(f"known {len(rows)}/9", flush=True)
    targets = [row for row in rows if row["role"] == "target_miss"]
    comparators = [row for row in rows if row["role"] == "detected_comparator"]
    gates = {
        "both_stock_targets_strict": all(
            row["records"]["r27"]["event"]["strict"] for row in targets
            if row["variant"] == "stock"),
        "target_zero_duplicate": all(
            row["records"]["r27"]["event"]["duplicate_rises"] == 0 for row in targets),
        "target_no_none_rises": all(
            not row["records"]["r27"]["twins"]["none"]["false_times_s"] for row in targets),
        "comparator_event_exact": all(
            row["records"]["r27"]["event"] == row["records"]["r26"]["event"]
            for row in comparators),
        "comparator_no_extra": all(
            not row["r27_vs_r26"]["new_attributable_rises"] for row in comparators),
        "all_none_detect_exact": all(row["r27_vs_r26"]["none_detect_trace_exact"] for row in rows),
        "all_force_prefix_physics_exact": all(
            all(row["reproduction"].values())
            and row["records"]["r27"]["prefix"]["passed"]
            and all(twin["force_exact"]["passed"]
                    for twin in row["records"]["r27"]["twins"].values())
            for row in rows),
    }
    result = {
        "schema": "r27-loaded-observer-disclosed-result.v1",
        "classification": protocol["classification"],
        "runtime": {"mujoco": mujoco.__version__, "cpu_workers": 1},
        "episodes": sum(row["episodes"] for row in rows),
        "protocol_sha256": sha(args.protocol),
        "candidate_sha256": sha(candidate),
        "evaluator_script_sha256": sha(__file__),
        "gates": gates,
        "passed": all(gates.values()),
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "passed": result["passed"],
                      "target_events": {row["id"] + "__" + row["variant"]:
                          row["records"]["r27"]["event"] for row in targets}}, indent=2))


if __name__ == "__main__":
    main()
