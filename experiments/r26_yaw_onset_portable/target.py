"""Replay the disclosed signed-yaw acquisition target using the sealed v2 lab."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from experiments.r25_composed.detector import Detector as R25
from experiments.r26_yaw_acquisition.candidate import Detector as Acquisition
from experiments.r26_yaw_portable import evaluator as sealed


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SHA256 = "212f4e79367cb636c0143a2d1f8aa7ba5584aaaca5ecde6b412d01a4985b6784"
CLASSIFICATION = (
    "Previously disclosed R25 signed-yaw acquisition diagnostic; opt-in only, "
    "with no default promotion or unopened R26 independent evidence."
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ObservedAcquisition(Acquisition):
    """Record telemetry without changing the frozen candidate's output."""

    def reset(self):
        super().reset()
        self.acquisition_trace = []

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
        self.acquisition_trace.append({
            "tick": int(round(1000 * float(t))),
            "inferred_event_identity": bool(self.yaw_acquisition_episode_active),
            "reacquisition_consumed": bool(self.yaw_reacquisition_consumed),
            "reacquisition_count": int(self.yaw_reacquisition_count),
            "provenance_complete": bool(self.yaw_reacquisition_provenance_complete),
        })
        return residual, report


def tick_intervals(trace, key):
    intervals = sealed.bool_intervals([row[key] for row in trace])
    return [[trace[first]["tick"], trace[last]["tick"]]
            for first, last in intervals]


def telemetry(detector):
    trace = detector.acquisition_trace
    counts = [row["reacquisition_count"] for row in trace]
    increments = [trace[index]["tick"] for index in range(1, len(trace))
                  if counts[index] > counts[index - 1]]
    inferred = [row["inferred_event_identity"] for row in trace]
    consumed = [row["reacquisition_consumed"] for row in trace]
    return {
        "samples": len(trace),
        "inferred_continuous_event": {
            "meaning": "telemetry only; never ORed into the Boolean alarm",
            "active_samples": int(sum(inferred)),
            "intervals": tick_intervals(trace, "inferred_event_identity"),
        },
        "consumed_same_source_reacquisition": {
            "meaning": "telemetry identifying yaw-only reacquisitions removed from the alarm",
            "active_samples": int(sum(consumed)),
            "intervals": tick_intervals(trace, "reacquisition_consumed"),
            "count": int(counts[-1]) if counts else 0,
            "count_increment_ticks": increments,
        },
        "provenance_incomplete_samples": int(sum(
            not row["provenance_complete"] for row in trace)),
    }


def alarm(log):
    values = np.asarray(log["detect"], dtype=bool)
    rises = sealed.rise_indices(values)
    return {
        "meaning": "actual Boolean detector output only",
        "rise_ticks": rises,
        "first_rise_tick": rises[0] if rises else None,
        "active_samples": int(values.sum()),
        "intervals": sealed.bool_intervals(values),
    }


def run_arm(case, variant, detector, impact):
    observed, log = sealed.run_episode(case, variant, detector, impact)
    return observed, log, alarm(log)


def evaluate():
    if mujoco.__version__ != "3.17.0":
        raise ValueError("authenticated course MuJoCo 3.17.0 required")
    candidate_path = ROOT / "experiments/r26_yaw_acquisition/candidate.py"
    if digest(candidate_path) != CANDIDATE_SHA256:
        raise ValueError("frozen signed-yaw acquisition candidate differs")
    registry = sealed.registration()
    case, scoring = registry["case"], registry["scoring"]
    rows = []
    for variant_name, variant in registry["variants"].items():
        expected = registry["expected"][variant_name]
        arms = {}
        traces = {}
        detectors = {}
        for arm_name, detector in (("r25", R25), ("acquisition", ObservedAcquisition)):
            detectors[arm_name] = {}
            traces[arm_name] = {}
            arms[arm_name] = {}
            for twin, impact in (("none", None), ("force", case["impact"])):
                observed, log, actual_alarm = run_arm(case, variant, detector, impact)
                detectors[arm_name][twin] = observed
                traces[arm_name][twin] = log
                arms[arm_name][twin] = {"alarm": actual_alarm}
                if arm_name == "acquisition":
                    arms[arm_name][twin]["telemetry"] = telemetry(observed)

        interval = sealed.force_interval(traces["acquisition"]["force"])
        if interval is None:
            raise ValueError("registered signed force absent")
        start = interval["first"]
        end = min(6000, start + scoring["event_window_ms"])
        r25_rises = arms["r25"]["force"]["alarm"]["rise_ticks"]
        new_rises = arms["acquisition"]["force"]["alarm"]["rise_ticks"]
        r25_event = sealed.event(traces["r25"]["force"], traces["r25"]["none"],
                                 start, end, scoring)
        new_event = sealed.event(traces["acquisition"]["force"],
                                 traces["acquisition"]["none"], start, end, scoring)
        expected_new = expected["r25_rises"][:1]
        new_telemetry = arms["acquisition"]["force"]["telemetry"]
        gates = {
            "continuous_force_exact": (
                [interval["first"], interval["last"]] == expected["force_interval"]
                and interval["samples"] == 180 and interval["continuous"]),
            "all_forces_exact": all(
                sealed.exact_force(traces[arm][twin],
                                   ((case["impact"], None) if twin == "force"
                                    else (None, None)))["passed"]
                for arm in traces for twin in traces[arm]),
            "physical_signatures_exact": all(
                sealed.lab.certified.trace_signature({
                    "none": traces[arm]["none"], "single": traces[arm]["force"]
                }) == expected["physical_signature"] for arm in traces),
            "prefixes_exact": all(
                sealed.lab.certified.prefix(traces[arm]["none"],
                                            traces[arm]["force"], start, start)["passed"]
                for arm in traces),
            "r25_registered_edges_exact": r25_rises == expected["r25_rises"],
            "acquisition_first_edge_only": new_rises == expected_new,
            "first_edge_unchanged": new_rises[:1] == r25_rises[:1],
            "strict_twin_attribution": (
                r25_event["raw_rise_indices"] == r25_rises
                and r25_event["attributable_rise_indices"] == r25_rises
                and new_event["raw_rise_indices"] == new_rises
                and new_event["attributable_rise_indices"] == new_rises),
            "acquisition_one_strict_rise": bool(
                new_event["strict"] and new_event["duplicate_rises"] == 0),
            "none_twins_have_no_alarm_edges": all(
                not arms[arm]["none"]["alarm"]["rise_ticks"] for arm in arms),
            "telemetry_consumption_count_exact": (
                new_telemetry["consumed_same_source_reacquisition"]["count"]
                == len(expected["r25_rises"]) - 1),
            "telemetry_provenance_complete": (
                new_telemetry["provenance_incomplete_samples"] == 0),
            "telemetry_traces_complete": all(
                arms["acquisition"][twin]["telemetry"]["samples"] == 6000
                for twin in ("none", "force")),
            "no_stuck_tail": all(
                not bool(traces[arm][twin]["detect"][-1])
                for arm in traces for twin in traces[arm]),
        }
        rows.append({
            "variant": variant_name,
            "force_interval": interval,
            "arms": arms,
            "events": {"r25": r25_event, "acquisition": new_event},
            "gates": gates,
            "passed": all(gates.values()),
        })
    return {
        "schema": "r26-known-signed-yaw-acquisition-diagnostic.v1",
        "classification": CLASSIFICATION,
        "independent": False,
        "unopened_r26_conditions_used": False,
        "default_promoted": False,
        "original_course_physics": True,
        "candidate_sha256": CANDIDATE_SHA256,
        "registry_sha256": digest(sealed.REGISTRATION),
        "registered_input": {"case": case, "variants": registry["variants"]},
        "case": case,
        "episodes": 12,
        "semantics": {
            "alarm": "Boolean Detector.update report; only these samples are alarm-active.",
            "inferred_continuous_event": (
                "Diagnostic identity telemetry spanning certified gaps; not an alarm."),
            "consumed_reacquisition": (
                "Diagnostic telemetry for same-source yaw-only reacquisitions suppressed "
                "from the alarm; other or uncertain routes fail open."),
        },
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
        "limitations": [
            "previously disclosed finite simulator target only",
            "not unopened R26 independent evidence, hardware, deployment, or safety proof",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh output required")
    result = evaluate()
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    for row in result["rows"]:
        old = row["arms"]["r25"]["force"]["alarm"]
        new = row["arms"]["acquisition"]["force"]
        inferred = new["telemetry"]["inferred_continuous_event"]
        consumed = new["telemetry"]["consumed_same_source_reacquisition"]
        print(f"{row['variant']}: force={row['force_interval']['first']}.."
              f"{row['force_interval']['last']} R25={old['rise_ticks']} "
              f"new={new['alarm']['rise_ticks']} alarm_active={new['alarm']['active_samples']} "
              f"inferred_active={inferred['active_samples']} consumed={consumed['count']}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
