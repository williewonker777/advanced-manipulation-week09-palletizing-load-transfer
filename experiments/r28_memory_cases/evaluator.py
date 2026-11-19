"""Replay the disclosed R28 landing cases with the frozen observation-memory arm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from experiments.r28_landing_replay import evaluator as authored


HERE = Path(__file__).resolve().parent
AUTHORED_PLAN = Path(authored.__file__).with_name("plan.json")
AUTHORED_PLAN_SHA256 = "ced65c332af9e02480ffbfa50462ab71340cb19530ff54a09c43cba5d45af0ee"
MEMORY_ENTRYPOINT = "experiments.r28_contact_memory.candidate:Detector"
MEMORY_SHA256 = "85c4107c3939c7624dd77040ad322a99595c8e6941c7b75751c516a33be7a07b"
FINITE_ENTRYPOINT = "experiments.r28_finite_contact.candidate:Detector"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def summarize(result):
    """Extract the handoff question without replacing authored physical records."""
    tool = next(row for row in result["tool_rows"]
                if row["id"] == "known_5p25kg_29mm_tool")
    quiet = next(row for row in result["tool_rows"] if row["id"].endswith("zero_offset"))
    positives = {row["id"]: row for row in result["positive_control_rows"]}
    v3_edges = tool["records"]["r27_v3"]["none"]["rises"]
    memory_edges = tool["records"]["r28"]["none"]["rises"]
    removed = [edge for edge in v3_edges if edge not in memory_edges]
    memory_positive = {
        name: {
            "none_rises": row["records"]["r28"]["none"]["rises"],
            "force_rises": row["records"]["r28"]["force"]["rises"],
            "event": row["records"]["r28"]["event"],
            "force_physical_sha256": row["records"]["r28"]["force"]["physical_sha256"],
            "force_exact": row["records"]["r28"]["force"]["force_exact"],
        }
        for name, row in positives.items()
    }
    integrity = all(all(row["integrity_gates"].values())
                    for row in result["tool_rows"] + result["positive_control_rows"])
    both_y = all(row["records"]["r28"]["event"]["strict"] for row in positives.values())
    removed_four = len(removed) == 4
    tool_preserved = tool["records"]["r28"]["none"]["tool"]["detected"]
    quiet_not_broadened = (
        quiet["records"]["r28"]["none"]["late_false_rises_s"]
        == quiet["records"]["c1"]["none"]["late_false_rises_s"]
    )
    return {
        "arm_labels": {"c1": "c1", "r27_v3": "v3", "r28": "observation_memory",
                       "pulse_diagnostic": "finite_rejected"},
        "tool_pair": {
            row["id"]: {name: record["none"]["rises"]
                         for name, record in row["records"].items()}
            for row in (tool, quiet)
        },
        "positive_controls": memory_positive,
        "removed_v3_tool_edges": removed,
        "gates": {
            "memory_recovers_both_y_controls": both_y,
            "memory_removes_four_known_v3_tool_edges": removed_four,
            "memory_preserves_real_tool": tool_preserved,
            "memory_does_not_broaden_quiet_twin_vs_c1": quiet_not_broadened,
            "all_authored_force_and_physical_integrity_gates": integrity,
        },
        "core_question_passed": all((both_y, removed_four, tool_preserved,
                                     quiet_not_broadened, integrity)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        parser.error("fresh --out required")
    plan_raw = AUTHORED_PLAN.read_bytes()
    if digest(plan_raw) != AUTHORED_PLAN_SHA256:
        parser.error("authored disclosed plan changed")
    plan = json.loads(plan_raw)
    plan["arms"]["r28"] = MEMORY_ENTRYPOINT
    plan["arms"]["pulse_diagnostic"] = FINITE_ENTRYPOINT
    plan["classification"] = (
        "Known disclosed observation-memory replay; not blind, enrolled, promoted, or hardware-qualified."
    )
    with tempfile.TemporaryDirectory(prefix="r28-memory-cases-", dir=args.out.parent) as folder:
        temporary = Path(folder)
        modified_plan = temporary / "plan.json"
        base_result = temporary / "authored-result.json"
        modified_plan.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
        previous = authored.PLAN
        try:
            authored.PLAN = modified_plan
            authored.main(["--out", str(base_result)])
        finally:
            authored.PLAN = previous
        authored_result = json.loads(base_result.read_bytes())
    if (authored_result.get("episodes") != 24
            or authored_result.get("arms", {}).get("r28") != MEMORY_ENTRYPOINT
            or authored_result.get("arms", {}).get("pulse_diagnostic") != FINITE_ENTRYPOINT):
        raise ValueError("incomplete or wrong authored replay")
    result = {
        "schema": "r28-memory-known-case-delivery.v1",
        "classification": plan["classification"],
        "episodes": authored_result["episodes"],
        "authored_evaluator_sha256": digest(Path(authored.__file__).read_bytes()),
        "authored_plan_sha256": AUTHORED_PLAN_SHA256,
        "modified_plan_sha256": authored_result["plan_sha256"],
        "memory_candidate_sha256": MEMORY_SHA256,
        "summary": summarize(authored_result),
        "authored_result": authored_result,
        "claim_status": "known_replay_only_no_fresh_enrollment",
    }
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": result["episodes"], **result["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
