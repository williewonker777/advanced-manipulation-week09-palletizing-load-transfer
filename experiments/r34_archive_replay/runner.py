"""Run frozen R31/R33 and an optional registered R34 on disclosed placement inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
import re
from pathlib import Path
from typing import Any

from experiments.r21_full_protection import runner_v2 as physical
from experiments.r22_census import check
from experiments.r31_held_identity.candidate import Detector as R31
from experiments.r33_vertical_replay import runner as r33
from tools.d4_robustness import build_cases

ROOT = Path(__file__).resolve().parents[2]
MATRIX = "bc4b3dbb925cfb8609f535e1420f8b3f20785769b7f339708ca914b6992db791"
SMOKE = (313, 314)
PLACEMENT9 = (309, 310, 313, 314, 318, 323, 324, 333, 334)
ENTRY = re.compile(r"^experiments\.r34_[A-Za-z0-9_]+(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
SCOPES: dict[str, tuple[int, ...] | None] = {
    "smoke": SMOKE,
    "placement9": PLACEMENT9,
    "full-place": None,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _has_symlink_component(path: Path, root: Path) -> bool:
    root = root.absolute()
    current = path.absolute()
    if not current.is_relative_to(root):
        return True
    while True:
        if current.is_symlink():
            return True
        if current == root:
            return False
        current = current.parent


def _safe_r34_name(name: str) -> bool:
    path = Path(name)
    return bool(
        name.startswith("experiments/r34_")
        and path.suffix in {".py", ".json"}
        and not path.is_absolute()
        and ".." not in path.parts
        and not any(part.startswith(".") for part in path.parts)
        and not any(token in name.lower() for token in ("private", "sealed", "validation", "result"))
    )


def optional_registration(path: Path | None, root: Path = ROOT) -> dict[str, Any] | None:
    """Authenticate an explicit public R34 source manifest; absence is a valid two-arm run."""
    if path is None:
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular R34 registry required")
    obj = json.loads(path.read_bytes())
    candidate = obj.get("candidate", {})
    entry = candidate.get("entry", "")
    module, separator, kind = entry.partition(":")
    files = obj.get("files", {})
    source_name = module.replace(".", "/") + ".py"
    if (obj.get("schema") != "r34-placement-opt-in-registry.v1"
            or obj.get("public_sources_only") is not True
            or obj.get("kit_included") is not False
            or not separator or kind != "Detector"
            or ENTRY.fullmatch(module) is None
            or not isinstance(files, dict) or not files or len(files) > 64
            or source_name not in files
            or candidate.get("sha256") != files.get(source_name)):
        raise ValueError("invalid R34 opt-in registry")
    for name, pin in files.items():
        source = root / name
        if (not isinstance(name, str) or not isinstance(pin, str) or len(pin) != 64
                or not _safe_r34_name(name) or _has_symlink_component(source, root)
                or not source.is_file()
                or sha(source) != pin):
            raise ValueError("R34 opt-in source differs: " + str(name))
    return obj


def detector_type(arm: str, registry_path: str | None) -> type:
    if arm == "r31":
        return R31
    if arm == "r33":
        return r33.candidate_type()
    if arm != "r34" or registry_path is None:
        raise ValueError("unregistered replay arm")
    entry = optional_registration(Path(registry_path))["candidate"]["entry"]
    module, _, kind = entry.partition(":")
    return getattr(importlib.import_module(module), kind)


def selected_cases(scope: str) -> list[dict[str, Any]]:
    cases = build_cases("my_torque")
    if len(cases) != 385 or hashlib.sha256(physical.canonical_cases(cases)).hexdigest() != MATRIX:
        raise ValueError("D4 matrix differs")
    focus = SCOPES[scope]
    if focus is None:
        focus = tuple(case["case_id"] for case in cases
                      if case["kind"] == "collision" and case["phase"] == "place")
    by_id = {case["case_id"]: case for case in cases}
    references = {physical.reference_key(case): case for case in cases if case["kind"] == "reference"}
    selected: set[int] = set()
    for index in focus:
        case = by_id[index]
        if case["kind"] != "collision" or case["phase"] != "place":
            raise ValueError("disclosed placement id is not a placement collision")
        selected.add(index)
        selected.add(references[physical.reference_key(case)]["case_id"])
    return [by_id[index] for index in sorted(selected)]


def record(job: tuple[dict[str, Any], str, str | None]) -> tuple[int, str, dict[str, Any]]:
    case, arm, registry_path = job
    return case["case_id"], arm, physical.episode_record(detector_type(arm, registry_path), case)


def _arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, int]:
    collisions = [row for row in rows if row["input"]["kind"] == "collision"]
    return {
        "strict": sum(row["scores"][arm]["strict"] for row in collisions),
        "duplicates": sum(row["scores"][arm]["duplicate"] for row in collisions),
        "raw_duplicates": sum(row["scores"][arm]["raw_duplicate"] for row in collisions),
        "false_rises": sum(len(row["scores"][arm]["false"]) for row in rows),
        "masked": sum(row["scores"][arm]["masked"] for row in collisions),
        "stuck": sum(row["scores"][arm]["stuck"] for row in rows),
        "tool_detected": sum(row["scores"][arm]["tool"]["detected"] for row in rows),
    }


def summarize(scope: str, jobs: int, registry_path: Path | None = None) -> dict[str, Any]:
    optional = optional_registration(registry_path)
    cases = selected_cases(scope)
    arms = ("r31", "r33") + (("r34",) if optional else ())
    registry_arg = None if registry_path is None else str(registry_path.resolve())
    work = [(case, arm, registry_arg) for case in cases for arm in arms]
    if jobs == 1:
        captured = list(map(record, work))
    else:
        with ProcessPoolExecutor(jobs) as pool:
            captured = list(pool.map(record, work))
    records: dict[int, dict[str, dict[str, Any]]] = {}
    for index, arm, value in captured:
        if arm in records.setdefault(index, {}):
            raise ValueError("duplicate arm record")
        records[index][arm] = value
    if any(set(value) != set(arms) for value in records.values()):
        raise ValueError("missing independently executed arm")

    references = {physical.reference_key(case): case["case_id"]
                  for case in cases if case["kind"] == "reference"}
    rows = []
    for case in cases:
        index = case["case_id"]
        pair = records[index]
        ref_index = references[physical.reference_key(case)] if case["kind"] == "collision" else index
        scores = {arm: check.score(case, pair[arm], records[ref_index][arm]) for arm in arms}
        physical_equal = all(pair[arm]["physical_sha256"] == pair["r31"]["physical_sha256"]
                             for arm in arms)
        valid = all(all(score["valid"].values()) for score in scores.values())
        bounds = {arm: (pair[arm]["force_first_nonzero_index"],
                        pair[arm]["force_last_nonzero_index"]) for arm in arms}
        first = {arm: (None if not scores[arm]["strict"] else
                       pair[arm]["onset_index"] + scores[arm]["delay_ms"]) for arm in arms}
        in_force = {arm: (value is not None and bounds[arm][0] is not None
                          and bounds[arm][0] <= value <= bounds[arm][1])
                    for arm, value in first.items()}
        comparisons = {f"{arm}_vs_r31": check.compare(scores[arm], scores["r31"])
                       for arm in arms if arm != "r31"}
        if "r34" in arms:
            comparisons["r34_vs_r33"] = check.compare(scores["r34"], scores["r33"])
        rows.append({
            "id": index,
            "input": case,
            "matched_reference_id": ref_index,
            "independent_executions": len(arms),
            "physical_sha256_equal": physical_equal,
            "input_certified": bool(valid and physical_equal),
            "first_rise_in_force": in_force,
            "protection": comparisons,
            "scores": scores,
            "records": pair,
        })

    collision_rows = [row for row in rows if row["input"]["kind"] == "collision"]
    gains = {arm: [row["id"] for row in collision_rows
                   if row["scores"][arm]["strict"] and not row["scores"]["r31"]["strict"]]
             for arm in arms if arm != "r31"}
    losses = {arm: [row["id"] for row in collision_rows
                    if not row["scores"][arm]["strict"] and row["scores"]["r31"]["strict"]]
              for arm in arms if arm != "r31"}
    protection_failures = {
        label: [row["id"] for row in rows if not all(row["protection"][label].values())]
        for label in next(iter(rows))["protection"]
    }
    registrations: dict[str, Any] = {"r33": r33.registration()["candidate"]}
    if optional:
        registrations["r34"] = optional["candidate"]
    return {
        "schema": "r34-placement-physical-replay.v1",
        "scope": scope,
        "course_original": False,
        "default_changed": False,
        "kit_changed": False,
        "matrix_sha256": MATRIX,
        "arms": list(arms),
        "registrations": registrations,
        "inputs": len(cases),
        "collisions": len(collision_rows),
        "episodes": len(work),
        "actual_independent_executions_per_input": len(arms),
        "summary": {arm: _arm_summary(rows, arm) for arm in arms},
        "gains_vs_r31": gains,
        "losses_vs_r31": losses,
        "input_failures": [row["id"] for row in rows if not row["input_certified"]],
        "not_in_force_gains": {
            arm: [row["id"] for row in collision_rows
                  if row["id"] in gains[arm] and not row["first_rise_in_force"][arm]]
            for arm in gains
        },
        "protection_failures": protection_failures,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=SCOPES)
    parser.add_argument("--jobs", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--r34-registry", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh output required")
    obj = summarize(args.scope, args.jobs, args.r34_registry)
    with args.out.open("x") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: obj[key] for key in (
        "scope", "arms", "episodes", "summary", "gains_vs_r31", "losses_vs_r31",
        "input_failures", "not_in_force_gains", "protection_failures")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
