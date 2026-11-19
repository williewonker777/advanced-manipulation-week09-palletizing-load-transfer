"""Execute R31 and the frozen R33 composition independently on disclosed D4 inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
from pathlib import Path

from experiments.r21_full_protection import runner_v2 as physical
from experiments.r22_census import check
from experiments.r31_held_identity.candidate import Detector as Identity
from tools.d4_robustness import build_cases

ROOT = Path(__file__).resolve().parents[2]
MATRIX = "bc4b3dbb925cfb8609f535e1420f8b3f20785769b7f339708ca914b6992db791"
SMOKE = (193, 253, 313)
AXIAL19 = (193, 194, 203, 204, 213, 214,
           253, 254, 263, 264, 273, 274, 283,
           313, 314, 323, 324, 333, 334)
SCOPES = {"smoke": SMOKE, "axial19": AXIAL19, "full-d4": None}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registration() -> dict:
    path = Path(__file__).with_name("registry.json")
    obj = json.loads(path.read_bytes())
    if obj.get("schema") != "r33-vertical-replay-registry.v1":
        raise ValueError("wrong R33 replay registry")
    candidate = obj.get("candidate", {})
    entry = candidate.get("entry", "")
    parts = entry.partition(":")
    if not parts[1] or parts[2] != "Detector":
        raise ValueError("invalid registered R33 entry")
    source = ROOT / (parts[0].replace(".", "/") + ".py")
    if sha(source) != candidate.get("sha256"):
        raise ValueError("registered R33 candidate differs")
    dependencies = obj.get("dependencies", {})
    if not dependencies or source.relative_to(ROOT).as_posix() not in dependencies:
        raise ValueError("incomplete R33 runtime registration")
    for name, pin in dependencies.items():
        path = ROOT / name
        if (not (name.startswith("experiments/r33_")
                 or name == "experiments/r31_held_identity/candidate.py") or path.is_symlink()
                or not path.is_file() or sha(path) != pin):
            raise ValueError("registered R33 dependency differs: " + name)
    return obj


def candidate_type() -> type:
    entry = registration()["candidate"]["entry"]
    module, _, kind = entry.partition(":")
    return getattr(importlib.import_module(module), kind)


def selected_cases(scope: str) -> list[dict]:
    cases = build_cases("my_torque")
    if len(cases) != 385 or hashlib.sha256(physical.canonical_cases(cases)).hexdigest() != MATRIX:
        raise ValueError("D4 matrix differs")
    requested = SCOPES[scope]
    if requested is None:
        return cases
    by_id = {case["case_id"]: case for case in cases}
    references = {physical.reference_key(case): case for case in cases if case["kind"] == "reference"}
    selected: set[int] = set()
    for index in requested:
        case = by_id[index]
        if case["kind"] != "collision":
            raise ValueError("disclosed vertical id is not a collision")
        selected.add(index)
        selected.add(references[physical.reference_key(case)]["case_id"])
    return [by_id[index] for index in sorted(selected)]


def record(job: tuple[dict, str]) -> tuple[int, str, dict]:
    case, arm = job
    detector = Identity if arm == "r31" else candidate_type()
    return case["case_id"], arm, physical.episode_record(detector, case)


def summarize(scope: str, jobs: int) -> dict:
    cases = selected_cases(scope)
    arms = ("r31", "r33")
    work = [(case, arm) for case in cases for arm in arms]
    if jobs == 1:
        captured = list(map(record, work))
    else:
        with ProcessPoolExecutor(jobs) as pool:
            captured = list(pool.map(record, work))
    records: dict[int, dict[str, dict]] = {}
    for index, arm, value in captured:
        if arm in records.setdefault(index, {}):
            raise ValueError("duplicate arm record")
        records[index][arm] = value
    if any(set(value) != set(arms) for value in records.values()):
        raise ValueError("missing independently executed arm")

    references = {
        physical.reference_key(case): case["case_id"]
        for case in cases if case["kind"] == "reference"
    }
    rows = []
    for case in cases:
        index = case["case_id"]
        pair = records[index]
        ref_index = references[physical.reference_key(case)] if case["kind"] == "collision" else index
        scores = {arm: check.score(case, pair[arm], records[ref_index][arm]) for arm in arms}
        valid = all(all(score["valid"].values()) for score in scores.values())
        same_physics = pair["r31"]["physical_sha256"] == pair["r33"]["physical_sha256"]
        bounds = (pair["r33"]["force_first_nonzero_index"], pair["r33"]["force_last_nonzero_index"])
        first = None if not scores["r33"]["strict"] else pair["r33"]["onset_index"] + scores["r33"]["delay_ms"]
        in_force = first is not None and bounds[0] is not None and bounds[0] <= first <= bounds[1]
        protection = check.compare(scores["r33"], scores["r31"])
        rows.append({
            "id": index,
            "input": case,
            "matched_reference_id": ref_index,
            "independent_executions": 2,
            "physical_sha256_equal": same_physics,
            "input_certified": bool(valid and same_physics),
            "r33_first_rise_in_force": in_force,
            "protection_vs_r31": protection,
            "scores": scores,
            "records": pair,
        })
    collision_rows = [row for row in rows if row["input"]["kind"] == "collision"]
    summary = {}
    for arm in arms:
        summary[arm] = {
            "strict": sum(row["scores"][arm]["strict"] for row in collision_rows),
            "duplicates": sum(row["scores"][arm]["duplicate"] for row in collision_rows),
            "false_rises": sum(len(row["scores"][arm]["false"]) for row in rows),
            "masked": sum(row["scores"][arm]["masked"] for row in collision_rows),
            "tool_detected": sum(row["scores"][arm]["tool"]["detected"] for row in rows),
        }
    gains = [row["id"] for row in collision_rows if row["scores"]["r33"]["strict"] and not row["scores"]["r31"]["strict"]]
    losses = [row["id"] for row in collision_rows if not row["scores"]["r33"]["strict"] and row["scores"]["r31"]["strict"]]
    return {
        "schema": "r33-vertical-physical-replay.v1",
        "scope": scope,
        "course_original": False,
        "default_changed": False,
        "candidate": registration()["candidate"],
        "matrix_sha256": MATRIX,
        "arms": list(arms),
        "inputs": len(cases),
        "collisions": len(collision_rows),
        "episodes": len(work),
        "actual_independent_executions_per_input": 2,
        "summary": summary,
        "gains_vs_r31": gains,
        "losses_vs_r31": losses,
        "input_failures": [row["id"] for row in rows if not row["input_certified"]],
        "not_in_force_gains": [row["id"] for row in collision_rows if row["id"] in gains and not row["r33_first_rise_in_force"]],
        "protection_failures": [row["id"] for row in rows if not all(row["protection_vs_r31"].values())],
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=SCOPES)
    parser.add_argument("--jobs", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh output required")
    obj = summarize(args.scope, args.jobs)
    with args.out.open("x") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: obj[key] for key in (
        "scope", "episodes", "summary", "gains_vs_r31", "losses_vs_r31",
        "input_failures", "not_in_force_gains", "protection_failures")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
