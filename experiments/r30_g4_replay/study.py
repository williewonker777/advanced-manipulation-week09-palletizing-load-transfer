"""Run four disclosed G4 tool contacts through four frozen detector arms."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.r29_g4_matrix_replay import matrix as shared


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE / "plan.json"
EXPECTED_EPISODES = 16


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(entry):
    module, name = entry.rsplit(":", 1)
    return getattr(importlib.import_module(module), name)


def contact_intervals(values, threshold=1.0):
    """Return inclusive contiguous sample bounds above the registered tool threshold."""
    active = np.asarray(values, float) > float(threshold)
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    return [[int(a), int(b)] for a, b in zip(starts, ends, strict=True)]


def edge_attribution(rises, intervals):
    if not intervals:
        return [{"tick": int(tick), "relative_to_tool": "no_registered_tool_contact",
                 "delay_from_first_contact_ms": None, "delay_from_last_contact_ms": None}
                for tick in rises]
    first, last = intervals[0][0], intervals[-1][1]
    rows = []
    for tick in rises:
        during = any(begin <= tick <= end for begin, end in intervals)
        relation = ("before_first_contact" if tick < first else
                    "during_contact" if during else "between_contact_intervals"
                    if tick <= last else "after_last_contact")
        rows.append({"tick": int(tick), "relative_to_tool": relation,
                     "delay_from_first_contact_ms": int(tick - first),
                     "delay_from_last_contact_ms": int(tick - last)})
    return rows


def verify_plan(plan):
    if (plan.get("schema") != "r30-g4-disclosed-replay-plan.v1"
            or plan.get("episode_accounting", {}).get("total_actual") != EXPECTED_EPISODES
            or len(plan.get("cases", [])) != 4 or len(plan.get("arms", {})) != 4):
        raise ValueError("wrong disclosed R30 replay plan")
    for arm, entry in plan["arms"].items():
        module = entry.rsplit(":", 1)[0]
        source = ROOT / (module.replace(".", "/") + ".py")
        if digest(source) != plan["candidate_sha256"][arm]:
            raise ValueError("frozen detector source changed: " + arm)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        parser.error("--out must be fresh")
    shared.verify()
    plan = verify_plan(json.loads(PLAN.read_bytes()))
    controller = resolve(plan["controller"])
    arms = {name: resolve(entry) for name, entry in plan["arms"].items()}
    rows = []
    episodes = 0
    for case in plan["cases"]:
        records = {}
        physical = set()
        for name, detector in arms.items():
            metrics, log, ctrl = shared.run(case, detector, controller, (None, None))
            episodes += 1
            base = shared.record(metrics, log, ctrl, case["mass"], (None, None), [], 320)
            intervals = contact_intervals(log["f_tool"], plan["tool_threshold_N"])
            base["true_tool_intervals_inclusive"] = intervals
            base["first_tool_tick"] = None if not intervals else intervals[0][0]
            base["last_tool_tick"] = None if not intervals else intervals[-1][1]
            base["rise_attribution"] = edge_attribution(base["rises"], intervals)
            base["all_rises_after_contact_began"] = bool(
                intervals and base["rises"] and all(x >= intervals[0][0] for x in base["rises"]))
            records[name] = base
            physical.add(json.dumps(base["physical_sha256"], sort_keys=True))
        memory = records["memory"]
        rows.append({
            "id": case["id"], "source_id": case["source_id"], "case": case,
            "records": records,
            "same_physics_all_arms": len(physical) == 1,
            "all_forces_exact": all(row["force_exact"]["passed"] for row in records.values()),
            "edge_delta_vs_memory": {
                name: {"memory_rises": memory["rises"], "arm_rises": row["rises"],
                       "first_edge_delta_ms": (None if not memory["rises"] or not row["rises"]
                                               else int(row["rises"][0] - memory["rises"][0])),
                       "active_delta": int(row["active_samples"] - memory["active_samples"])}
                for name, row in records.items() if name != "memory"
            },
        })
    if episodes != EXPECTED_EPISODES:
        raise ValueError("wrong episode count")
    document = {
        "schema": "r30-g4-disclosed-replay-result.v1",
        "classification": plan["classification"],
        "plan_sha256": digest(PLAN),
        "base_replay_registration_sha256": digest(shared.PROTOCOL),
        "runtime": "MuJoCo 3.17.0",
        "arms": plan["arms"],
        "candidate_sha256": plan["candidate_sha256"],
        "episodes": episodes,
        "rows": rows,
        "physics_exact_cases": sum(row["same_physics_all_arms"] for row in rows),
        "force_exact_episodes": sum(record["force_exact"]["passed"]
                                    for row in rows for record in row["records"].values()),
        "claim_status": "disclosed_replay_only_no_promotion",
        "limits": plan["claim_limits"],
    }
    if document["physics_exact_cases"] != 4 or document["force_exact_episodes"] != 16:
        raise ValueError("passive detector physics or force identity changed")
    args.out.write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": episodes, "physics_exact_cases": 4,
                      "force_exact_episodes": 16,
                      "events": {row["id"]: {name: record["rises"]
                                  for name, record in row["records"].items()}
                                 for row in rows}}, sort_keys=True))


if __name__ == "__main__":
    main()
