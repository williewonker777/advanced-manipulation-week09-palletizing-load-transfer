"""Run the frozen R27 source on the separated normal-audit split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r26_delivery_pair import probe
from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r14_stock_handoff.qualification import compact
from experiments.r27_landing_integration.candidate import Detector as Candidate


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rises(log):
    bits = np.asarray(log["detect"], bool)
    return np.flatnonzero(bits & ~np.r_[False, bits[:-1]]).astype(int).tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    plan = json.loads((HERE / "plan.json").read_bytes())
    cases, split = probe.cases()
    if split["audit"] != plan["normal_split"]["audit"]:
        raise ValueError("audit split differs from preregistration")
    rows = []
    for key in split["audit"]:
        systems = {}
        for name, detector in (("c1", C1), ("candidate", Candidate)):
            _, ctrl, metrics, log = probe.run(cases[key], detector)
            systems[name] = {
                "rises": rises(log),
                "active_samples": int(np.asarray(log["detect"], bool).sum()),
                "false_alarm_times": metrics["false_alarm_times"],
                "tool_contact_delay_ms": metrics["tool_contact_delay_ms"],
                "physical_sha256": probe.replay.physical_signature(log),
                "cost": compact(metrics, log, ctrl, cases[key]["mass"])["cost"],
            }
        old, new = systems["c1"], systems["candidate"]
        gates = {
            "same_physics": old["physical_sha256"] == new["physical_sha256"],
            "same_rises": old["rises"] == new["rises"],
            "same_active_samples": old["active_samples"] == new["active_samples"],
            "same_false_times": old["false_alarm_times"] == new["false_alarm_times"],
            "same_tool_delay": old["tool_contact_delay_ms"] == new["tool_contact_delay_ms"],
            "same_cost": old["cost"] == new["cost"],
        }
        rows.append({"id": key, "systems": systems, "gates": gates,
                     "passed": all(gates.values())})
    doc = {
        "schema": "r27-landing-separated-normal-audit.v1",
        "classification": "Separated normal audit after source/model freeze; no positive inputs.",
        "sources": {name: sha(HERE / name) for name in
                    ("plan.json", "model.json", "candidate.py", "audit.py")},
        "episodes": 2 * len(rows), "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": doc["episodes"], "passed": doc["passed"],
                      "failures": [row["id"] for row in rows if not row["passed"]]},
                     indent=2))


if __name__ == "__main__":
    main()
