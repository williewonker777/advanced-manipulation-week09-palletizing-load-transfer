"""Compare a named R27 detector against the frozen R26 C3 history baseline.

Baseline mode authenticates prior C3 evidence and actively reproduces:
- all 29 direct official-grid episode records,
- the authentic stock evaluate.py table (66 displayed cells), and
- a deterministic seven-case D4 anchor spanning every phase/kind plus case 299.

Candidate mode is intentionally inert until an explicit R27 entrypoint, source
hash, and authored overlay closure are provided. It then evaluates the full 385
D4 cases against C3 itself, not R25, plus both forms of the official grid.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from experiments.r22_census import check
from experiments.r24_holdout import evaluator as certified_lab
from experiments.r24_loaded.d4 import load_cases
from experiments.r26_history import d4 as d4_base
from experiments.r26_history import official as official_base
from tools import r21_replay as overlay
from tools import r26_signed_yaw_replay as signed_release
from tools.r20_official_compare import summarize, table


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
C3 = "experiments.r26_yaw_acquisition.candidate:Detector"
C3_SHA256 = "212f4e79367cb636c0143a2d1f8aa7ba5584aaaca5ecde6b412d01a4985b6784"
MATRIX_SHA256 = "bc4b3dbb925cfb8609f535e1420f8b3f20785769b7f339708ca914b6992db791"
D4_FILES = ("d4-shard-0-of-4.json", "d4-shard-1.json", "d4-shard-2-of-4.json")
OFFICIAL_FILE = "official.json"
D4_COMPARISON_FILE = "d4-comparison.json"
CLOSURE_SCHEMA = "r27-history-candidate-closure.v1"


def sha(value):
    if isinstance(value, (bytes, bytearray)):
        return hashlib.sha256(value).hexdigest()
    return hashlib.sha256(Path(value).read_bytes()).hexdigest()


def save(path, document):
    with Path(path).open("x") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def json_value(value):
    """Normalize tuples and NumPy-compatible scalars through the JSON contract."""
    return json.loads(json.dumps(value, allow_nan=False))


def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode())


def source_path(entrypoint):
    module, symbol = entrypoint.rsplit(":", 1)
    if (symbol != "Detector" or not module.startswith("experiments.r27_")
            or module.startswith("experiments.r27_loaded_holdout")):
        raise ValueError("named non-holdout R27 Detector entrypoint required")
    path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
    if (not path.is_file() or path.is_symlink()
            or not path.resolve().is_relative_to(ROOT.resolve())):
        raise ValueError("regular repository-local R27 candidate source required")
    return path


def resolve(entrypoint, expected_sha):
    path = (ROOT / "experiments/r26_yaw_acquisition/candidate.py"
            if entrypoint == C3 else source_path(entrypoint))
    if sha(path) != expected_sha:
        raise ValueError("R27 candidate source hash differs")
    module, symbol = entrypoint.rsplit(":", 1)
    return getattr(importlib.import_module(module), symbol)


def official_tasks():
    tasks = []
    for mass in (0.0, 4.0, 8.0):
        tasks.append(("control", mass, None))
    for mass in (0.0, 8.0):
        tasks.append(("reference", mass, None))
        for time_value in (1.42, 3.0, 3.86):
            for force in (10.0, 15.0, 40.0, 100.0):
                tasks.append(("collision", mass, [time_value, [0.0, force, 0.0]]))
    if len(tasks) != 29:
        raise AssertionError("official grid accounting changed")
    return tasks


def displayed_cell_count(cells):
    return (sum(len(row) for row in cells["control"].values())
            + sum(len(row["cells"]) for row in cells["collisions"]))


def d4_anchor_ids(cases):
    groups = {}
    for case in cases:
        key = (case["kind"], case.get("controller"), case.get("detector"), case.get("phase"))
        groups.setdefault(key, []).append(case["case_id"])
    return sorted({0, 299, len(cases) - 1, *(min(ids) for ids in groups.values())})


def capture_d4(job):
    entrypoint, expected_sha, case = job
    detector = resolve(entrypoint, expected_sha)
    return check.old.episode_record(detector, case)


def capture_official(job):
    entrypoint, expected_sha, task = job
    resolve(entrypoint, expected_sha)
    return official_base.capture((entrypoint, *task))


def load_c3_d4(history):
    history = Path(history)
    records, pins = {}, {}
    for name in D4_FILES:
        path = history / name
        document = json.loads(path.read_bytes())
        if (document.get("schema") != "r26-acquisition-d4-shard.v1"
                or document.get("candidate") != C3
                or document.get("candidate_sha256") != C3_SHA256
                or document.get("matrix_sha256") != MATRIX_SHA256):
            raise ValueError("unexpected C3 D4 shard: " + name)
        pins[name] = sha(path)
        for key, record in document["records"].items():
            index = int(key)
            if index in records:
                raise ValueError("duplicate C3 D4 case")
            records[index] = record
    if set(records) != set(range(385)):
        raise ValueError("C3 D4 archive is not exactly 385 complete cases")
    return records, pins


def validate_d4_published(history, records):
    path = Path(history) / D4_COMPARISON_FILE
    document = json.loads(path.read_bytes())
    if (document.get("schema") != "r26-acquisition-d4-comparison.v1"
            or document.get("candidate") != C3
            or document.get("candidate_sha256") != C3_SHA256
            or document.get("matrix_sha256") != MATRIX_SHA256
            or document.get("cases") != 385
            or document.get("fresh_candidate_episodes") != 385
            or document.get("regressions") or not document.get("passed")):
        raise ValueError("published C3 D4 comparison contract differs")
    rows = document["rows"]
    if [row["id"] for row in rows] != list(range(385)):
        raise ValueError("published C3 D4 row order/coverage differs")
    cases = load_cases()
    if len(cases) != 385 or check.MATRIX != MATRIX_SHA256:
        raise ValueError("live D4 matrix differs")
    references = {check.old.reference_key(case): case["case_id"]
                  for case in cases if case["kind"] == "reference"}
    for case, row in zip(cases, rows, strict=True):
        index = case["case_id"]
        ref = references[check.old.reference_key(case)] if case["kind"] == "collision" else index
        score = check.score(case, records[index], records[ref])
        if (row["candidate"] != score
                or row["candidate_active_samples"] != d4_base.occupied(records[index])
                or row["identity"] != json_value(case)):
            raise ValueError("published C3 D4 row differs: " + str(index))
    return document, sha(path)


def load_c3_official(history):
    path = Path(history) / OFFICIAL_FILE
    document = json.loads(path.read_bytes())
    if (document.get("schema") != "r26-acquisition-original-29.v1"
            or document.get("candidate") != C3
            or document.get("candidate_sha256") != C3_SHA256
            or document.get("per_arm_episodes") != 29
            or document.get("episodes") != 58
            or document.get("regressions") or not document.get("passed")
            or document.get("evaluate_sha256") != sha(ROOT / "evaluate.py")):
        raise ValueError("published C3 direct official archive differs")
    rows = document["arms"].get(C3)
    if not isinstance(rows, list) or len(rows) != 29:
        raise ValueError("C3 direct official archive is incomplete")
    for task, row in zip(official_tasks(), rows, strict=True):
        if (row["role"], row["mass"], row["collision"]) != task:
            raise ValueError("C3 direct official row order differs")
    return rows, sha(path)


def base_overlay_blobs():
    closure = signed_release.closure()
    blobs, pins = signed_release.source_blobs(closure)
    for name in ("__init__.py", "candidate.py"):
        relative = "experiments/r26_yaw_acquisition/" + name
        raw = (ROOT / relative).read_bytes()
        blobs[relative] = raw
        pins[relative] = sha(raw)
    if pins["experiments/r26_yaw_acquisition/candidate.py"] != C3_SHA256:
        raise ValueError("frozen C3 source differs")
    return blobs, pins


def load_candidate_closure(path, entrypoint, candidate_sha):
    document = json.loads(Path(path).read_bytes())
    files, overlay_files = document.get("files"), document.get("overlay_files")
    if (document.get("schema") != CLOSURE_SCHEMA
            or document.get("entrypoint") != entrypoint
            or document.get("candidate_sha256") != candidate_sha
            or document.get("kit_included") is not False
            or document.get("loaded_holdout_included") is not False
            or not isinstance(files, dict) or not files
            or not isinstance(overlay_files, list) or set(files) != set(overlay_files)):
        raise ValueError("R27 candidate closure contract differs")
    candidate_relative = source_path(entrypoint).relative_to(ROOT).as_posix()
    if files.get(candidate_relative) != candidate_sha:
        raise ValueError("candidate is not pinned in closure")
    for name, expected in files.items():
        relative = Path(name)
        source = ROOT / relative
        if (relative.is_absolute() or ".." in relative.parts
                or any(part in ("sim", "model", "r27_loaded_holdout") for part in relative.parts)
                or relative.name in ("evaluate.py", "run.py")
                or source.is_symlink() or not source.is_file()
                or len(expected) != 64 or sha(source) != expected):
            raise ValueError("invalid/stale candidate closure member: " + name)
        if b"r27_loaded_holdout" in source.read_bytes():
            raise ValueError("candidate closure references the excluded loaded holdout: " + name)
    return document


def actual_official(kit, entrypoint, candidate_sha, closure, parent):
    runtime = signed_release.runtime_gate(kit)
    if closure is None:
        blobs, pins = base_overlay_blobs()
    else:
        release = load_candidate_closure(closure, entrypoint, candidate_sha)
        blobs = {name: (ROOT / name).read_bytes() for name in release["overlay_files"]}
        pins = {name: sha(raw) for name, raw in blobs.items()}
        if pins != release["files"]:
            raise ValueError("candidate closure snapshot differs")
    manifest_raw = (ROOT / overlay.KIT_MANIFEST).read_bytes()
    kit_pins = overlay.authenticate_kit(kit, manifest_raw)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MUJOCO_GL="disable")
    environment.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="r27-official-", dir=parent) as folder:
        work = Path(folder)
        overlay.install_overlay(kit, work, manifest_raw, blobs, pins)
        module = entrypoint.rsplit(":", 1)[0]
        process = subprocess.run(
            [sys.executable, "-B", "evaluate.py", "--ctrl", "my_torque",
             "--detector", module, "--out", "result.txt"],
            cwd=work, env=environment, text=True, capture_output=True)
        if process.returncode:
            raise RuntimeError("stock evaluate.py failed: " + process.stderr.strip())
        episodes = sum(line.strip().startswith("실행 중: 박스")
                       for line in process.stdout.splitlines())
        if episodes != 29:
            raise ValueError("incomplete stock evaluate.py run")
        cells = table((work / "result.txt").read_text())
        if displayed_cell_count(cells) != 66:
            raise ValueError("stock evaluate.py table is not 66 displayed cells")
        if (kit_pins != overlay.authenticate_kit(kit, manifest_raw)
                or kit_pins != overlay.authenticate_kit(work, manifest_raw)
                or any(sha(ROOT / name) != expected or sha(work / name) != expected
                       for name, expected in pins.items())):
            raise ValueError("kit or authored source changed during stock evaluation")
    return {"runtime": runtime, "entrypoint": entrypoint, "episodes": 29,
            "displayed_cells": 66, "cells": cells, "counts": summarize(cells),
            "evaluate_sha256": overlay.EVALUATOR_SHA256,
            "kit_manifest_sha256": sha(manifest_raw), "source_sha256": pins}


def official_gates(reference, candidate, references, candidate_refs):
    gates = {"physical_exact": reference["physical_sha256"] == candidate["physical_sha256"],
             "trajectory_cost_exact": all(reference["metrics"][name] ==
                                            candidate["metrics"][name]
                                            for name in ("rms_mm", "max_mm", "saturation_pct")),
             "no_new_false": candidate["metrics"]["false_alarms"] <=
                             reference["metrics"]["false_alarms"],
             "no_stuck_tail": not candidate["tail"]}
    old_strict = new_strict = False
    if reference["role"] == "control":
        gates["tool_floor"] = certified_lab.prior.tool_floor(
            reference["tool"], candidate["tool"], 5)
    elif reference["role"] == "collision":
        old_strict = official_base.strict(reference, references[reference["mass"]])
        new_strict = official_base.strict(candidate, candidate_refs[candidate["mass"]])
        gates.update({
            "strict_floor": not old_strict or new_strict,
            "latency_plus_5_ms": (not old_strict or candidate["metrics"]["detect_delay_ms"]
                                  <= reference["metrics"]["detect_delay_ms"] + 5),
            "duration_plus_5_samples": (not old_strict or
                candidate["event"]["active_samples"] <= reference["event"]["active_samples"] + 5),
            "no_new_duplicate": (candidate["event"]["duplicate_rises"]
                                  <= reference["event"]["duplicate_rises"]),
            "quiet_preserved": (not reference["event"]["quiet_before"]
                                or candidate["event"]["quiet_before"]),
        })
    return gates, old_strict, new_strict


def d4_comparison(cases, baseline, candidate):
    references = {check.old.reference_key(case): case["case_id"]
                  for case in cases if case["kind"] == "reference"}
    rows, regressions, gains = [], [], []
    for case in cases:
        index = case["case_id"]
        ref = references[check.old.reference_key(case)] if case["kind"] == "collision" else index
        old_score = check.score(case, baseline[index], baseline[ref])
        new_score = check.score(case, candidate[index], candidate[ref])
        gates = check.compare(new_score, old_score)
        gates.update({
            "latency_plus_5_ms": (not old_score["strict"] or
                (new_score["strict"] and new_score["delay_ms"] <= old_score["delay_ms"] + 5)),
            "duration_plus_5_samples": (not old_score["strict"] or
                d4_base.occupied(candidate[index]) <= d4_base.occupied(baseline[index]) + 5),
            "tool_time_plus_5_ms": (not old_score["tool"]["detected"] or
                (new_score["tool"]["detected"]
                 and min(candidate[index]["tool_rise_indices"])
                 <= min(baseline[index]["tool_rise_indices"]) + 5)),
            "physical_force_prefix_exact": (
                candidate[index]["physical_sha256"] == baseline[index]["physical_sha256"]
                and all(new_score["valid"].values())),
        })
        failed = [name for name, value in gates.items() if not value]
        rows.append({"id": index, "kind": case["kind"], "c3": old_score,
                     "candidate": new_score,
                     "c3_active_samples": d4_base.occupied(baseline[index]),
                     "candidate_active_samples": d4_base.occupied(candidate[index]),
                     "gates": gates, "failures": failed})
        if failed:
            regressions.append({"id": index, "failures": failed})
        if new_score["strict"] and not old_score["strict"]:
            gains.append(index)
    return rows, regressions, gains


def baseline(args):
    out = args.out_dir
    if out.exists() or out.is_symlink():
        raise ValueError("fresh baseline output directory required")
    out.mkdir(parents=True)
    if sha(ROOT / "experiments/r26_yaw_acquisition/candidate.py") != C3_SHA256:
        raise ValueError("frozen C3 source differs")
    records, shard_pins = load_c3_d4(args.history)
    published, comparison_pin = validate_d4_published(args.history, records)
    cases = load_cases()
    anchors = d4_anchor_ids(cases)
    with ProcessPoolExecutor(args.workers) as pool:
        fresh_anchors = list(pool.map(capture_d4,
            [(C3, C3_SHA256, cases[index]) for index in anchors]))
    anchor_gates = {str(index): fresh == records[index]
                    for index, fresh in zip(anchors, fresh_anchors, strict=True)}
    anchor_doc = {"schema": "r27-c3-d4-anchor.v1", "ids": anchors,
                  "records": {str(i): r for i, r in zip(anchors, fresh_anchors, strict=True)},
                  "exact_archive": anchor_gates, "passed": all(anchor_gates.values())}
    save(out / "d4-anchor.json", anchor_doc)

    archived_official, official_pin = load_c3_official(args.history)
    tasks = official_tasks()
    with ProcessPoolExecutor(args.workers) as pool:
        fresh_official = list(pool.map(capture_official,
            [(C3, C3_SHA256, task) for task in tasks]))
    exact_official = [fresh == old for fresh, old in
                      zip(fresh_official, archived_official, strict=True)]
    direct_doc = {"schema": "r27-c3-direct-official-anchor.v1", "episodes": 29,
                  "records": fresh_official, "exact_archive": exact_official,
                  "passed": all(exact_official)}
    save(out / "direct-official.json", direct_doc)

    actual = actual_official(args.kit, C3, C3_SHA256, None, out.parent)
    prior_actual = json.loads(args.authentic_official.read_bytes())
    expected_cells = prior_actual["results"]["signed_acquisition"]["cells"]
    actual["exact_prior_66_cells"] = actual["cells"] == expected_cells
    actual["prior_authentic_official_sha256"] = sha(args.authentic_official)
    actual["passed"] = actual["exact_prior_66_cells"]
    save(out / "stock-evaluate.json", actual)

    strict = sum(row["candidate"]["strict"] for row in published["rows"])
    source_files = (HERE / "guard.py", HERE / "make_closure.py", HERE / "test_guard.py")
    receipt = {
        "schema": "r27-c3-history-baseline.v1",
        "classification": "R26 C3 baseline only; no R27 candidate was loaded or executed",
        "candidate_executed": False, "c3_entrypoint": C3, "c3_sha256": C3_SHA256,
        "matrix_sha256": MATRIX_SHA256, "d4_cases": 385,
        "d4_archive_complete_exact_once": set(records) == set(range(385)),
        "d4_record_set_sha256": canonical_sha(records),
        "d4_strict_cases": strict, "d4_published_regressions": published["regressions"],
        "d4_anchor_ids": anchors, "d4_anchor_exact": all(anchor_gates.values()),
        "direct_official_episodes": 29, "direct_official_exact": all(exact_official),
        "direct_official_record_set_sha256": canonical_sha(archived_official),
        "stock_evaluate_episodes": 29, "stock_evaluate_displayed_cells": 66,
        "stock_evaluate_exact": actual["exact_prior_66_cells"],
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files},
        "input_sha256": {**{name: pin for name, pin in shard_pins.items()},
            D4_COMPARISON_FILE: comparison_pin, OFFICIAL_FILE: official_pin,
            "authentic-official.json": sha(args.authentic_official)},
        "output_sha256": {name: sha(out / name) for name in
                          ("d4-anchor.json", "direct-official.json", "stock-evaluate.json")},
    }
    receipt["passed"] = all((receipt["d4_archive_complete_exact_once"],
                             receipt["d4_anchor_exact"], receipt["direct_official_exact"],
                             receipt["stock_evaluate_exact"],
                             not receipt["d4_published_regressions"]))
    save(out / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in (
        "d4_cases", "d4_strict_cases", "d4_anchor_ids", "d4_anchor_exact",
        "direct_official_exact", "stock_evaluate_exact", "passed")}, indent=2))
    if not receipt["passed"]:
        raise SystemExit(1)


def candidate(args):
    out = args.out_dir
    if out.exists() or out.is_symlink():
        raise ValueError("fresh candidate output directory required")
    receipt = json.loads(args.baseline_receipt.read_bytes())
    baseline_dir = args.baseline_receipt.parent
    if receipt.get("schema") != "r27-c3-history-baseline.v1" or not receipt.get("passed"):
        raise ValueError("passing C3 baseline receipt required")
    for name, expected in receipt["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("R27 history gate source changed: " + name)
    for name, expected in receipt["output_sha256"].items():
        if sha(baseline_dir / name) != expected:
            raise ValueError("C3 baseline output changed: " + name)
    closure_document = load_candidate_closure(
        args.closure, args.candidate, args.candidate_sha256)
    resolve(args.candidate, args.candidate_sha256)
    for name in (*D4_FILES, D4_COMPARISON_FILE, OFFICIAL_FILE):
        if sha(args.history / name) != receipt["input_sha256"][name]:
            raise ValueError("C3 history input changed after baseline: " + name)
    records, shard_pins = load_c3_d4(args.history)
    if any(receipt["input_sha256"].get(name) != pin for name, pin in shard_pins.items()):
        raise ValueError("C3 D4 archive changed after baseline")
    if canonical_sha(records) != receipt["d4_record_set_sha256"]:
        raise ValueError("C3 D4 record signature changed after baseline")
    cases = load_cases()
    out.mkdir(parents=True)
    before = sha(source_path(args.candidate))
    with ProcessPoolExecutor(args.workers) as pool:
        candidate_records = list(pool.map(capture_d4,
            [(args.candidate, args.candidate_sha256, case) for case in cases]))
    if before != sha(source_path(args.candidate)):
        raise ValueError("R27 candidate changed during D4")
    by_id = {case["case_id"]: row for case, row in zip(cases, candidate_records, strict=True)}
    rows, regressions, gains = d4_comparison(cases, records, by_id)
    d4_doc = {"schema": "r27-candidate-vs-c3-d4.v1",
              "candidate": args.candidate, "candidate_sha256": args.candidate_sha256,
              "matrix_sha256": MATRIX_SHA256, "cases": 385, "episodes": 385,
              "rows": rows, "regressions": regressions, "strict_gains": gains,
              "passed": not regressions}
    save(out / "d4.json", d4_doc)

    old_official, official_pin = load_c3_official(args.history)
    if canonical_sha(old_official) != receipt["direct_official_record_set_sha256"]:
        raise ValueError("C3 direct official signature changed after baseline")
    tasks = official_tasks()
    with ProcessPoolExecutor(args.workers) as pool:
        new_official = list(pool.map(capture_official,
            [(args.candidate, args.candidate_sha256, task) for task in tasks]))
    old_refs = {row["mass"]: row for row in old_official if row["role"] == "reference"}
    new_refs = {row["mass"]: row for row in new_official if row["role"] == "reference"}
    official_rows, official_regressions, official_gains = [], [], []
    for old, new in zip(old_official, new_official, strict=True):
        if (old["role"], old["mass"], old["collision"]) != (
                new["role"], new["mass"], new["collision"]):
            raise ValueError("future official row order differs")
        gates, old_strict, new_strict = official_gates(old, new, old_refs, new_refs)
        failures = [name for name, value in gates.items() if not value]
        item = {"role": old["role"], "mass": old["mass"], "collision": old["collision"],
                "c3": old, "candidate": new, "c3_strict": old_strict,
                "candidate_strict": new_strict, "gates": gates, "failures": failures}
        official_rows.append(item)
        if failures:
            official_regressions.append({"role": old["role"], "mass": old["mass"],
                                         "collision": old["collision"], "failures": failures})
        if new_strict and not old_strict:
            official_gains.append({"mass": old["mass"], "collision": old["collision"]})
    direct = {"schema": "r27-candidate-vs-c3-direct-official.v1",
              "candidate": args.candidate, "candidate_sha256": args.candidate_sha256,
              "episodes": 29, "baseline_archive_sha256": official_pin,
              "rows": official_rows, "regressions": official_regressions,
              "strict_gains": official_gains, "passed": not official_regressions}
    save(out / "direct-official.json", direct)

    actual = actual_official(args.kit, args.candidate, args.candidate_sha256,
                             args.closure, out.parent)
    baseline_actual = json.loads((baseline_dir / "stock-evaluate.json").read_bytes())
    actual["exact_c3_66_cells"] = actual["cells"] == baseline_actual["cells"]
    actual["passed"] = actual["exact_c3_66_cells"]
    save(out / "stock-evaluate.json", actual)
    closure_changed = any(sha(ROOT / name) != expected
                          for name, expected in closure_document["files"].items())
    result = {"schema": "r27-candidate-vs-c3-history-verdict.v1",
              "candidate": args.candidate, "candidate_sha256": args.candidate_sha256,
              "baseline_receipt_sha256": sha(args.baseline_receipt),
              "d4": {"cases": 385, "regressions": regressions,
                     "strict_gains": gains, "passed": not regressions},
              "direct_official": {"episodes": 29, "regressions": official_regressions,
                                  "strict_gains": official_gains,
                                  "passed": not official_regressions},
              "stock_evaluate": {"episodes": 29, "displayed_cells": 66,
                                 "exact_c3": actual["exact_c3_66_cells"],
                                 "passed": actual["passed"]},
              "candidate_changed_during_execution": before != sha(source_path(args.candidate))}
    result["closure_changed_during_execution"] = closure_changed
    result["passed"] = (result["d4"]["passed"] and result["direct_official"]["passed"]
                        and result["stock_evaluate"]["passed"]
                        and not result["candidate_changed_during_execution"]
                        and not result["closure_changed_during_execution"])
    result["output_sha256"] = {name: sha(out / name) for name in
                               ("d4.json", "direct-official.json", "stock-evaluate.json")}
    save(out / "verdict.json", result)
    print(json.dumps({k: result[k] for k in ("d4", "direct_official",
                                             "stock_evaluate", "passed")}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    base = command.add_parser("baseline")
    base.add_argument("--history", required=True, type=Path)
    base.add_argument("--authentic-official", required=True, type=Path)
    base.add_argument("--kit", required=True, type=Path)
    base.add_argument("--out-dir", required=True, type=Path)
    base.add_argument("--workers", choices=(1, 2), default=2, type=int)
    future = command.add_parser("candidate")
    future.add_argument("--candidate", required=True)
    future.add_argument("--candidate-sha256", required=True)
    future.add_argument("--closure", required=True, type=Path)
    future.add_argument("--baseline-receipt", required=True, type=Path)
    future.add_argument("--history", required=True, type=Path)
    future.add_argument("--kit", required=True, type=Path)
    future.add_argument("--out-dir", required=True, type=Path)
    future.add_argument("--workers", choices=(1, 2), default=2, type=int)
    args = parser.parse_args(argv)
    try:
        (baseline if args.command == "baseline" else candidate)(args)
    except (ValueError, OSError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
