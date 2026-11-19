"""Replay R18 and physical V2 on the exact frozen impulse-end tuple set."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiments.r16_payload_contract.controller import Controller as D
from experiments.r18_paired_impulses import runner, scoring
from experiments.r18_single_pass import paired_replay as replay
from experiments.r19_close_pairs import assay
from experiments.r19_event_state.physical_detector import Detector as V2
from tools.r14_paired_events import match_false_events, preserve_tool
from tools.r5_metrics import event_result


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("single_pass", "event")
TWINS = scoring.TWINS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def assignments(case):
    first, second = case["impacts"]
    return {"none": (None, None), "first": (first, None),
            "second": (None, second), "both": (first, second)}


def episode(arm, case, height, early, model, impacts):
    if arm == "single_pass":
        return replay.episode(arm, case, height, early, model, impacts)
    detector = V2()
    detector.model_path = model
    metrics, log = runner.run_episode(
        D(height, early), case["mass"], tuple(case["com"]), detector,
        pick_offset=case.get("offset"), impacts=impacts)
    log["r18_diverged"] = metrics["diverged"]
    if not scoring.finite_grid(log):
        raise ValueError("incomplete " + case["id"] + "/" + arm)
    return log


def save_trace(path, trace):
    arrays = {key: value for key, value in trace.items() if isinstance(value, np.ndarray)}
    diverged = trace.get("r18_diverged")
    if type(diverged) is not bool:
        raise ValueError("measured boolean r18_diverged required")
    arrays["r18_diverged"] = np.asarray(diverged, dtype=np.bool_)
    np.savez_compressed(path, **arrays)


def load_trace(path):
    with np.load(path) as source:
        output = {key: source[key] for key in source.files}
    divergence = output.pop("r18_diverged", None)
    if not isinstance(divergence, np.ndarray) or divergence.shape != () or divergence.dtype != np.bool_:
        raise ValueError("invalid persisted r18_diverged")
    output["r18_diverged"] = bool(divergence.item())
    return output


def event(trace):
    return event_result(trace["t"], trace["detect"], {}, trace["f_tool"])


def normal_case(index, case, height, early, model, raw_dir, impulse_normals):
    impulse_normal = impulse_normals[case["id"]]
    logs = {arm: episode(arm, case, height, early, model, (None, None)) for arm in ARMS}
    for arm, trace in logs.items():
        save_trace(Path(raw_dir) / f"{case['id']}_{arm}_none.npz", trace)
    events = {arm: event(trace) for arm, trace in logs.items()}
    r18, v2 = logs["single_pass"], logs["event"]
    physical = {key: bool(np.array_equal(v2[key], r18[key]))
                for key in replay.PHYSICAL + ("r",)}
    forward = match_false_events(events["event"]["false_alarm_times"],
                                 events["single_pass"]["false_alarm_times"])
    reverse = match_false_events(events["single_pass"]["false_alarm_times"],
                                 events["event"]["false_alarm_times"])
    gates = {
        "complete_finite": all(scoring.finite_grid(trace) for trace in logs.values()),
        "no_force": all(not np.any(trace["f_ext_xyz_committed"]) for trace in logs.values()),
        "V2_R18_physics_and_residual_exact": all(physical.values()),
        "V2_R18_bits_exact": bool(np.array_equal(v2["detect"], r18["detect"])),
        "no_added_false": forward["passed"],
        "no_lost_false": reverse["passed"],
        "no_added_or_lost_tool": (preserve_tool(events["event"], events["single_pass"])
                                   and preserve_tool(events["single_pass"], events["event"])),
        "verified_impulse_normal_R18_D_physics": bool(
            impulse_normal["passed"]
            and impulse_normal["gates"]["D_R18_candidate_physics"]
            and impulse_normal["gates"]["R18_residual_parity"]
            and impulse_normal["gates"]["R18_bits_equal"]),
    }
    return {"id": case["id"], "episodes": 2, "events": events,
            "physics": physical, "false_match": {"forward": forward, "reverse": reverse},
            "impulse_normal_gates": impulse_normal["gates"],
            "gates": gates, "passed": all(gates.values())}


def first_fall(arm, case):
    onset = int(round(1000 * case["impacts"][0][0]))
    second = int(round(1000 * case["impacts"][1][0]))
    intervals = arm["levels"]["both"]["on_intervals_ms"]
    match = next((pair for pair in intervals if pair[0] >= onset and pair[0] < second), None)
    return {
        "interval_ms": match,
        "duration_from_first_onset_ms": None if match is None else match[1] - onset,
        "clear_before_second_ms": None if match is None else second - match[1],
    }


def arm_delta(v2, impulse, case):
    slots = {}
    for name in assay.SLOTS:
        left, right = v2["slots"][name], impulse["slots"][name]
        slots[name] = {
            "V2_raw_ms": left["raw_delays_ms"],
            "V2_credited_ms": left["credited_delays_ms"],
            "impulse_raw_ms": right["raw_delays_ms"],
            "impulse_credited_ms": right["credited_delays_ms"],
            "V2_high_at_onset": left["high_at_onset"],
            "impulse_high_at_onset": right["high_at_onset"],
            "V2_quiet": left["quiet_before_onset"],
            "impulse_quiet": right["quiet_before_onset"],
            "raw_rise_delta": len(right["raw_delays_ms"]) - len(left["raw_delays_ms"]),
            "credited_delta": len(right["credited_delays_ms"]) - len(left["credited_delays_ms"]),
        }
    strict_v2 = all(len(v2["slots"][name]["raw_delays_ms"]) ==
                    len(v2["slots"][name]["credited_delays_ms"]) == 1 for name in assay.SLOTS)
    strict_impulse = all(len(impulse["slots"][name]["raw_delays_ms"]) ==
                         len(impulse["slots"][name]["credited_delays_ms"]) == 1 for name in assay.SLOTS)
    return {
        "slots": slots,
        "strict_all_four_V2": strict_v2,
        "strict_all_four_impulse": strict_impulse,
        "strict_both_second_V2": len(v2["slots"]["both_second"]["credited_delays_ms"]) == 1,
        "strict_both_second_impulse": len(impulse["slots"]["both_second"]["credited_delays_ms"]) == 1,
        "first_alarm_fall": {"V2": first_fall(v2, case), "impulse": first_fall(impulse, case)},
        "active_sample_delta_impulse_minus_V2": {
            twin: impulse["levels"][twin]["active_samples"] - v2["levels"][twin]["active_samples"]
            for twin in TWINS},
        "impulse_vs_V2_safety": assay.compare(impulse, v2),
    }


def force_case(index, case, height, early, model, quiet, window, raw_dir, impulse_rows):
    impulse_row = impulse_rows[case["id"]]
    traces, arms, raws = {}, {}, {}
    variants = assignments(case)
    for arm in ARMS:
        twins = {"none": load_trace(Path(raw_dir) / f"{case['id']}_{arm}_none.npz")}
        for twin in ("first", "second", "both"):
            twins[twin] = episode(arm, case, height, early, model, variants[twin])
        traces[arm] = twins
        arms[arm], raws[arm] = assay.characterize(case, twins, quiet, window)
    physics = {twin: {key: bool(np.array_equal(traces["event"][twin][key],
                                               traces["single_pass"][twin][key]))
                      for key in replay.PHYSICAL + ("r",)} for twin in TWINS}
    impulse = impulse_row["arms"]["candidate"]
    r18_signature = replay.signature(raws["single_pass"])
    gates = {
        "actual_R18_raw_signature_matches_gate": (
            r18_signature == impulse_row["assessment_signatures"]["single_pass"]),
        "actual_R18_character_matches_gate": arms["single_pass"] == impulse_row["arms"]["single_pass"],
        "V2_R18_physics_residual_exact_all_twins": all(
            all(values.values()) for values in physics.values()),
        "actual_signed_force_and_prefix_exact": (
            arms["single_pass"]["signed_and_scalar_exact"]
            and all(arms["single_pass"]["r18_gates"][name] for name in (
                "complete_finite_strict_grid", "force_free_prefixes",
                "all_required_prefixes_exact", "distinct_non_noop_impacts", "no_sign_inversion"))),
        "impulse_gate_D_and_R18_physics_verified": (
            all(impulse_row["physics"]["D_coupled"].values())
            and all(impulse_row["physics"]["single_pass"].values())),
    }
    return {
        "id": case["id"], "episodes": 6,
        "actual_R18_assessment_sha256": r18_signature,
        "V2_assessment_sha256": replay.signature(raws["event"]),
        "arms": arms,
        "physics": physics,
        "V2_to_impulse": arm_delta(arms["event"], impulse, case),
        "gates": gates, "passed": all(gates.values()),
    }


def parallel(cases, function, args, folder, jobs):
    folder.mkdir()
    rows = {}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(function, index, case, *args): case
                   for index, case in enumerate(cases)}
        for future in as_completed(futures):
            row = future.result()
            write(folder / (row["id"] + ".json"), row)
            rows[row["id"]] = row
            print(folder.name, row["id"], row["passed"], flush=True)
    return [rows[case["id"]] for case in cases]


def source_pins(args, impulse_files):
    impulse_paths = {}
    for path in impulse_files:
        key = ("impulse_report" if path.name == "report.json"
               else f"impulse_{path.parent.name}_{path.stem}")
        if key in impulse_paths:
            raise ValueError("duplicate impulse evidence pin key")
        impulse_paths[key] = path
    paths = {
        "plan": args.plan,
        "height": args.height,
        "early": args.early,
        "model": args.model,
        "module": Path(__file__),
        "V2_detector": ROOT / "experiments/r19_event_state/physical_detector.py",
        "V2_state": ROOT / "experiments/r19_event_state/state.py",
        "R18_recovery": ROOT / "experiments/r18_single_pass/recovery.py",
        "R18_detector": ROOT / "experiments/r18_single_pass/detector.py",
        "paired_runner": ROOT / "experiments/r18_paired_impulses/runner.py",
        "paired_scorer": ROOT / "experiments/r18_paired_impulses/scoring.py",
        "assay": ROOT / "experiments/r19_close_pairs/assay.py",
        **impulse_paths,
    }
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("height", "early", "model", "plan", "out"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--impulse-dir", type=Path, required=True,
                     help="private completed impulse-gate run containing report/normal/novel JSON")
    cli.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    args = cli.parse_args(argv)
    if args.out.exists():
        cli.error("fresh output required")
    plan = read(args.plan)
    if plan.get("schema") != "r19-impulse-gate-plan.v1" or len(plan.get("new", [])) != 6:
        cli.error("fixed six-case impulse plan required")
    cases = plan["new"]
    report_path = args.impulse_dir / "report.json"
    impulse_files = [args.impulse_dir / "novel" / (case["id"] + ".json") for case in cases]
    missing = [str(path) for path in [report_path, *impulse_files] if not path.is_file()]
    if missing:
        cli.error("missing complete impulse evidence: " + ", ".join(missing))
    impulse_report = read(report_path)
    impulse_rows = {path.stem: read(path) for path in impulse_files}
    normal_files = [args.impulse_dir / "normal" / (case["id"] + ".json") for case in cases]
    if any(not path.is_file() for path in normal_files):
        cli.error("missing impulse no-force gate")
    impulse_normals = {path.stem: read(path) for path in normal_files}
    if (impulse_report.get("schema") != "r19-impulse-gate-result.v1"
            or set(impulse_rows) != {case["id"] for case in cases}
            or set(impulse_normals) != set(impulse_rows)):
        cli.error("incomplete impulse evidence contract")
    expected_gate_pins = {
        "plan": sha(args.plan), "height": sha(args.height), "early": sha(args.early),
        "r17_model": sha(args.model),
        "candidate_detector": sha(ROOT / "experiments/r19_impulse_end/detector.py"),
        "candidate_state": sha(ROOT / "experiments/r19_impulse_end/state.py"),
        "close_assay": sha(ROOT / "experiments/r19_close_pairs/assay.py"),
        "paired_runner": sha(ROOT / "experiments/r18_paired_impulses/runner.py"),
        "paired_scorer": sha(ROOT / "experiments/r18_paired_impulses/scoring.py"),
    }
    if any(impulse_report.get("pins", {}).get(name) != digest
           for name, digest in expected_gate_pins.items()):
        cli.error("impulse evidence source authentication failed")
    reported_novel = {row["id"]: row for row in impulse_report["stages"]["novel"]["rows"]}
    reported_normal = {row["id"]: row for row in impulse_report["stages"]["normal"]["rows"]}
    if reported_novel != impulse_rows or reported_normal != impulse_normals:
        cli.error("impulse row files differ from authenticated report")
    args.out.mkdir()
    raw_dir = args.out / "raw"
    raw_dir.mkdir()
    pins = source_pins(args, [report_path, *impulse_files, *normal_files])
    write(args.out / "INPUT_PINS.json", pins)
    height, early = read(args.height), read(args.early)
    normal = parallel(cases, normal_case,
                      (height, early, args.model, raw_dir, impulse_normals),
                      args.out / "normal", args.jobs)
    stages = {"normal": {"rows": normal, "passed": all(row["passed"] for row in normal)}}
    if stages["normal"]["passed"]:
        forced = parallel(cases, force_case,
                          (height, early, args.model, plan["quiet_ms"], plan["window_ms"],
                           raw_dir, impulse_rows), args.out / "forced", args.jobs)
        stages["forced"] = {"rows": forced, "passed": all(row["passed"] for row in forced)}
    if source_pins(args, [report_path, *impulse_files, *normal_files]) != pins:
        raise ValueError("source or evidence drift")
    forced = stages.get("forced", {}).get("rows", [])
    summary = {
        "schema": "r19-same-tuple-result.v1",
        "research_only": True,
        "plan_sha256": sha(args.plan),
        "stages": stages,
        "same_six_tuple_ids": [case["id"] for case in cases],
        "episodes": sum(row["episodes"] for stage in stages.values() for row in stage["rows"]),
        "strict_all_four": {
            "V2": sum(row["V2_to_impulse"]["strict_all_four_V2"] for row in forced),
            "impulse": sum(row["V2_to_impulse"]["strict_all_four_impulse"] for row in forced),
        },
        "strict_both_second": {
            "V2": sum(row["V2_to_impulse"]["strict_both_second_V2"] for row in forced),
            "impulse": sum(row["V2_to_impulse"]["strict_both_second_impulse"] for row in forced),
        },
        "limitations": ["same fixed six tuples; no tuning or new force", "no host timing claim"],
    }
    write(args.out / "report.json", summary)
    print(json.dumps({"normal": stages["normal"]["passed"],
                      "forced": stages.get("forced", {}).get("passed"),
                      "episodes": summary["episodes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
