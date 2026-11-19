"""Run the preregistered disclosed pickup-duration and four-twin development rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r23_detached.free import Detector as R23
from experiments.r23_holdout import evaluator as r23_eval
from experiments.r24_composed.detector import Detector as R24
from experiments.r24_holdout import evaluator as r24_eval
from experiments.r25_pickup.candidate import Detector as R25


ROOT = Path(__file__).resolve().parents[2]
R23_PLAN = ROOT / "experiments/r23_holdout/plan.json"
R24_PLAN = ROOT / "experiments/r24_holdout/plan.json"
ARMS = {"r23": R23, "r24": R24, "r25": R25}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervals(bits):
    values = np.asarray(bits, bool)
    rises = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    falls = np.flatnonzero(~values & np.r_[False, values[:-1]])
    if len(values) and values[-1]:
        falls = np.r_[falls, len(values)]
    return [[int(a), int(b)] for a, b in zip(rises, falls, strict=True)]


def probe(base):
    class Probe(base):
        def reset(self):
            super().reset()
            self.r25_state_edges = []
            self._r25_previous = None

        def update(self, t, th, thd, tau_m, tau_s, grip):
            residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
            state = (
                bool(report), bool(self.yaw_active),
                bool(getattr(self, "pick_event_active", False)),
                bool(getattr(self, "pick_completion_active", False)),
                bool(getattr(self, "pick_completion_selected", False)),
            )
            if state != self._r25_previous:
                self.r25_state_edges.append({
                    "index": int(round(1000 * float(t))),
                    "output": state[0], "yaw": state[1],
                    "r24_latch": state[2], "completion_active": state[3],
                    "completion_selected": state[4],
                    "yaw_value_Nm": (None if self.yaw_value is None
                                     else float(self.yaw_value)),
                    "completion_reason": getattr(self, "pick_completion_reason", None),
                })
                self._r25_previous = state
            return residual, report

    return Probe


def run(evaluator, case, variant, base, impacts):
    detector = probe(base)()
    _, log = evaluator.run_episode(case, variant, lambda: detector, impacts)
    return detector, log


def common_record(detector, log, impacts, evaluator):
    return {
        "output_intervals": intervals(log["detect"]),
        "active_samples": int(np.asarray(log["detect"], bool).sum()),
        "force_intervals": intervals(np.any(np.asarray(log["f_ext_xyz_committed"]) != 0.0,
                                             axis=1)),
        "force_exact": evaluator.certified.exact_force(log, impacts),
        "physical_sha256": evaluator.certified.physical_hash(log),
        "state_edges": detector.r25_state_edges,
    }


def duration_row(plan):
    case = next(row for row in plan["single"]
                if row["id"] == "s24_transfer_offaxis_early")
    variant = plan["variants"]["stock"]
    traces = {}
    for name, base in ARMS.items():
        traces[name] = {
            "none": run(r24_eval, case, variant, base, (None, None)),
            "force": run(r24_eval, case, variant, base, (case["impact"], None)),
        }
    start, boundary = r24_eval.certified.actual_start(
        traces["r23"]["force"][1]["t"], case["impact"])
    records = {}
    for name, twins in traces.items():
        none, force = twins["none"], twins["force"]
        records[name] = {
            "event": r24_eval.event_record(
                force[1], [none[1]], start,
                min(6000, start + plan["event_window_ms"]),
                plan["quiet_ms"], plan["reference_twin_ms"]),
            "none": common_record(none[0], none[1], (None, None), r24_eval),
            "force": common_record(force[0], force[1], (case["impact"], None), r24_eval),
        }
    for twin in ("none", "force"):
        hashes = {name: row[twin]["physical_sha256"] for name, row in records.items()}
        if any(value != hashes["r23"] for value in hashes.values()):
            raise ValueError("duration-row physical trace differs by detector")
    event = records["r25"]["event"]
    gates = {
        "r23_plus_five_active_samples": (
            event["active_samples_in_slot"] <=
            records["r23"]["event"]["active_samples_in_slot"] + 5),
        "same_or_earlier_strict_rise": (
            event["strict"] and
            event["delays_ms"][0] <= records["r23"]["event"]["delays_ms"][0] + 5),
        "zero_duplicate": event["duplicate_rises"] == 0,
        "none_quiet": records["r25"]["none"]["active_samples"] == 0,
        "force_and_physics_exact": all(
            row[twin]["force_exact"]["passed"]
            for row in records.values() for twin in ("none", "force")),
    }
    return {"id": case["id"], "variant": variant["kind"], "boundary": boundary,
            "records": records, "gates": gates, "passed": all(gates.values())}


def pair_rows(plan):
    case = next(row for row in plan["paired"]
                if row["id"] == "p23_transfer_close_diagonals")
    assignment = {"none": (None, None), "first": (case["impacts"][0], None),
                  "second": (None, case["impacts"][1]), "both": tuple(case["impacts"])}
    rows = []
    for variant in plan["variants"].values():
        traces = {name: {twin: run(r23_eval, case, variant, base, impacts)
                         for twin, impacts in assignment.items()}
                  for name, base in ARMS.items()}
        starts = [r23_eval.certified.actual_start(
            traces["r23"]["both"][1]["t"], impact)[0] for impact in case["impacts"]]
        first, second = starts
        window = plan["event_window_ms"]
        windows = {
            "isolated_first": ("first", first, min(second, first + window), ["none"]),
            "isolated_second": ("second", second, min(6000, second + window), ["none"]),
            "both_first": ("both", first, min(second, first + window), ["none", "second"]),
            "both_second": ("both", second, min(6000, second + window), ["first"]),
        }
        records = {}
        for name, twins in traces.items():
            records[name] = {
                "events": {slot: r23_eval.event_record(
                    twins[twin][1], [twins[ref][1] for ref in refs], begin, end,
                    plan["quiet_ms"], plan["reference_twin_ms"])
                           for slot, (twin, begin, end, refs) in windows.items()},
                "twins": {twin: common_record(pair[0], pair[1], assignment[twin], r23_eval)
                           for twin, pair in twins.items()},
            }
        for twin in assignment:
            hashes = {name: row["twins"][twin]["physical_sha256"]
                      for name, row in records.items()}
            if any(value != hashes["r23"] for value in hashes.values()):
                raise ValueError("pair physical trace differs by detector")
        joint, candidate = records["r24"]["events"], records["r25"]["events"]
        gates = {
            "retain_joint_strict_slots": all(
                not event["strict"] or candidate[slot]["strict"]
                for slot, event in joint.items()),
            "zero_candidate_duplicate": all(
                event["duplicate_rises"] == 0 for event in candidate.values()),
            "none_quiet": records["r25"]["twins"]["none"]["active_samples"] == 0,
            "no_later_joint_strict_slot": all(
                not event["strict"] or
                candidate[slot]["delays_ms"][0] <= event["delays_ms"][0] + 5
                for slot, event in joint.items()),
            "force_and_physics_exact": all(
                row["twins"][twin]["force_exact"]["passed"]
                for row in records.values() for twin in assignment),
        }
        rows.append({"id": case["id"], "variant": variant["kind"],
                     "starts": starts, "records": records, "gates": gates,
                     "passed": all(gates.values())})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    r23_plan, r24_plan = json.loads(R23_PLAN.read_bytes()), json.loads(R24_PLAN.read_bytes())
    duration = duration_row(r24_plan)
    pairs = pair_rows(r23_plan)
    doc = {
        "schema": "r25-pickup-targeted-development.v1",
        "classification": "Preregistered disclosed rows only; not independent R25 evidence",
        "episodes": 6 + 3 * 3 * 4,
        "sources": {
            "candidate": sha(ROOT / "experiments/r25_pickup/candidate.py"),
            "r24_joint": sha(ROOT / "experiments/r24_composed/detector.py"),
            "r23_plan": sha(R23_PLAN), "r24_plan": sha(R24_PLAN),
        },
        "duration": duration,
        "pairs": pairs,
        "passed": duration["passed"] and all(row["passed"] for row in pairs),
    }
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "episodes": doc["episodes"], "passed": doc["passed"],
        "duration": duration["records"]["r25"]["event"],
        "pairs": [{"variant": row["variant"], "passed": row["passed"],
                   "r25": row["records"]["r25"]["events"]} for row in pairs],
    }, indent=2))


if __name__ == "__main__":
    main()
