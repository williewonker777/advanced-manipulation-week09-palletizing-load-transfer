"""Build the compact R24 repeat result from completed disclosed-suite evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FILES = {
    "reproduction": "reproduction.json",
    "channels": "channel_attribution.json",
    "force_boundaries": "force_boundaries.json",
    "target": "trial3_target.json",
    "r23_full": "r23_full_protection.json",
    "d4": "d4_full.json",
    "weak": "weak.json",
    "pairs": "pairs.json",
    "official": "official/summary.json",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    docs = {name: json.loads((args.evidence / relative).read_bytes())
            for name, relative in FILES.items()}
    target_rows = docs["target"]["rows"]
    target = {row["variant"]: row["records"]["r23"]["events"]
              for row in target_rows}
    for events in target.values():
        assert len(events["isolated_first"]["raw_rise_indices"]) == 1
        assert len(events["both_first"]["raw_rise_indices"]) == 1
    d4 = docs["d4"]
    full = docs["r23_full"]["summary"]
    official = docs["official"]
    boundaries = docs["force_boundaries"]
    assert docs["reproduction"]["exact_vs_disclosed_rows"]
    assert d4["summary"]["base"] == d4["summary"]["trial"]
    assert not d4["changed"] and not d4["regressions"] and not d4["invalid"]
    assert docs["weak"]["changed"] == []
    assert docs["pairs"]["changed"] == [] and docs["pairs"]["strict_losses"] == []
    assert official["raw_table_equal"]
    document = {
        "schema": "r24-disclosed-repeat-result.v1",
        "classification": "Publicly disclosed development/protection suites only; no R24 private suite, deployment, hardware, hard-real-time, or safety claim",
        "published_candidate": False,
        "frozen_research_component": "experiments.r24_repeat.trial:Detector",
        "root_cause": "The first 120 ms physical impulse crosses the actual grip/model transition. The inherited yaw state resets at pickup and re-accumulates the still-present same force, producing two rises. The true second impulse is a later R10-baseline route in stock/contact and is already missed under actuator skew.",
        "correction": "When yaw was active on the immediately preceding sensor report, compensate the modeled incoming payload momentum and preserve that one pickup event. End it only after the existing R10 suspicion evidence and existing eight-sample release persistence are both quiet. No nominal time, case, mass, COM, mechanics, programmed force, threshold change, or global cooldown is used.",
        "target": {
            "episodes": docs["target"]["episodes"],
            "duplicate_slots_before": 6,
            "duplicate_slots_after": 0,
            "first_rises_after": {mechanics: {
                "isolated_first": events["isolated_first"]["raw_rise_indices"],
                "both_first": events["both_first"]["raw_rise_indices"],
            } for mechanics, events in target.items()},
            "true_second_after": {mechanics: {
                "isolated_second": events["isolated_second"]["raw_rise_indices"],
                "both_second": events["both_second"]["raw_rise_indices"],
            } for mechanics, events in target.items()},
            "new_false_or_safety_floor_failure": False,
        },
        "protection": {
            "r23_disclosed_rows": full["rows"],
            "r23_disclosed_episodes": full["episodes"],
            "r23_changed_rows": full["changed_rows"],
            "strict_slots_before_after": [full["base_strict_slots"],
                                           full["trial_strict_slots"]],
            "d4_episodes": d4["episodes"],
            "d4_summary": d4["summary"],
            "weak_episodes": docs["weak"]["episodes"],
            "weak_strict": docs["weak"]["strict"],
            "repeat_episodes": docs["pairs"]["episodes"],
            "repeat_strict": docs["pairs"]["strict"],
            "official_episodes": official["episodes"],
            "official_equal": official["raw_table_equal"],
        },
        "qualification_limit": {
            "reason_no_candidate": "The predeclared two-real-impact gate remains unmet: actuator-skew misses the second impact in both isolated and paired twins, as the selected R23 base already did. The paired stock/contact second rise also cannot have a 24 ms quiet prefix because the fixed physical pulses provide only 21 force-free samples.",
            "physical_force_spans": boundaries["spans"],
            "force_free_between_samples": boundaries["force_free_between_samples"],
            "quiet_contract_samples": boundaries["quiet_contract_samples"],
            "gap_meets_quiet_contract": boundaries["gap_meets_quiet_contract"],
            "actuator_skew_second_rises": target["actuator_skew"]["isolated_second"]["raw_rise_indices"],
            "stock_contact_both_second_quiet": {
                mechanics: target[mechanics]["both_second"]["quiet_before"]
                for mechanics in ("stock", "contact_surface")},
            "preexisting_r23_absolute_safety_failures": full["safety_failures"],
        },
        "evidence_sha256": {name: sha(args.evidence / relative)
                            for name, relative in FILES.items()},
    }
    args.out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"published_candidate": False,
                      "target_duplicates": [6, 0],
                      "strict_slots": document["protection"]["strict_slots_before_after"],
                      "d4": document["protection"]["d4_summary"]}, indent=2))


if __name__ == "__main__":
    main()
