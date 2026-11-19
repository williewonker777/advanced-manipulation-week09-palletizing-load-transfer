"""Paired C1 evaluation of every D4 event row affected by the demand guard."""
from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

from experiments.r9_saturation.headroom_guard import Controller as Guard
from sim.nominal import Nominal
from tools import d4_robustness
from tools import sweep

BASELINE = "r1_servo"
CANDIDATE = "experiments.r9_saturation.headroom_guard"
DETECTOR = "experiments.r8_final.C1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_guarded(mass, com):
    controller = Guard()
    controller.nom = Nominal()
    controller.nom_box = Nominal()
    controller.nom_box.set_payload(mass, com)
    controller.box_mass, controller.box_com = mass, tuple(com)
    controller.reset()
    return controller.endpoint_advance_guarded


def affected_cases(module):
    original = d4_robustness.build_cases("my_servo", detector=DETECTOR)
    events = [row for row in original
              if row["kind"] in ("collision", "offset")
              and is_guarded(row["mass"], row["com"])]
    reference_coms = sorted({tuple(row["com"]) for row in events if row["collision"] is not None})
    references = [dict(controller=module, detector=DETECTOR,
                       case_id=f"matched-reference-{index}", mass=8.0, com=com,
                       collision=None, pick_offset=None, kind="matched_reference",
                       phase="none", direction="none", force_n=0.0)
                  for index, com in enumerate(reference_coms)]
    paired_events = [dict(row, controller=module, detector=DETECTOR) for row in events]
    return references + paired_events


def design():
    rows = affected_cases(BASELINE)
    return {
        "scope": "all D4 collision/offset rows whose payload activates the guard, plus matched references",
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "detector": DETECTOR,
        "source_sha256": {"r1_servo.py": sha("r1_servo.py"),
                           "headroom_guard.py": sha("experiments/r9_saturation/headroom_guard.py"),
                           "C1.py": sha("experiments/r8_final/C1.py")},
        "inventory_per_controller": {
            "total": len(rows),
            "matched_reference": sum(row["kind"] == "matched_reference" for row in rows),
            "collision": sum(row["kind"] == "collision" for row in rows),
            "offset": sum(row["kind"] == "offset" for row in rows),
        },
        "gates": {"baseline_detected_event_losses": 0, "baseline_tool_event_losses": 0,
                  "conditions_with_increased_false_alarms": 0,
                  "candidate_saturated_episodes": 0, "candidate_diverged": 0,
                  "normal_each_cost_ratio_max": 1.05},
        "classification": "existing tools.sweep.classify_rows with matched C1 references",
        "development_only": True,
        "no_retuning_after_run": True,
    }


def run_rows(payloads, workers):
    if workers == 1:
        rows = [sweep.run_case(row) for row in payloads]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            rows = list(pool.imap(sweep.run_case, payloads))
    sweep.classify_rows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    current = design()
    if args.register:
        args.register.write_text(json.dumps(current, indent=2) + "\n")
        print(json.dumps(current, indent=2))
        return
    if not args.design or not args.output:
        parser.error("--design and --output are required for execution")
    if json.loads(args.design.read_text()) != current:
        raise RuntimeError("registered design/source pins changed")
    all_rows = {label: run_rows(affected_cases(module), args.workers)
                for label, module in (("R1", BASELINE), ("guard", CANDIDATE))}
    base = {str(row["case_id"]): row for row in all_rows["R1"]}
    candidate = {str(row["case_id"]): row for row in all_rows["guard"]}
    losses, tool_losses, fa_increase = [], [], []
    for case_id, row in base.items():
        other = candidate[case_id]
        if row.get("collision_outcome") == "detected" and other.get("collision_outcome") != "detected":
            losses.append(case_id)
        if row["tool_contact_peak_n"] > 1 and row["tool_contact_delay_ms"] is not None \
                and other["tool_contact_delay_ms"] is None:
            tool_losses.append(case_id)
        if other["false_alarms"] > row["false_alarms"]:
            fa_increase.append(case_id)
    normals = [str(row["case_id"]) for row in all_rows["R1"]
               if row["kind"] == "matched_reference"]
    fields = ("rms_mm", "max_mm", "torque_step_place_nm", "table_force_place_max_n")
    cost = {field: max(candidate[i][field] / max(base[i][field], 1e-12) for i in normals)
            for field in fields}
    summary = {
        label: {"rows": len(rows),
                "detected": sum(row.get("collision_outcome") == "detected" for row in rows),
                "tool_contacts": sum(row["tool_contact_peak_n"] > 1 for row in rows),
                "false_alarm_rises": sum(row["false_alarms"] for row in rows),
                "saturated_episodes": sum(row["saturation_pct"] > 0 for row in rows),
                "diverged": sum(row["diverged"] for row in rows)}
        for label, rows in all_rows.items()
    }
    summary.update(baseline_detected_losses=losses, baseline_tool_event_losses=tool_losses,
                   conditions_with_increased_false_alarms=fa_increase,
                   normal_candidate_over_R1_max=cost)
    args.output.write_text(json.dumps({"design": current, "summary": summary,
                                       "rows": all_rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
