"""Separated normal audit for the frozen support-aware candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r14_stock_handoff.qualification import compact
from experiments.r26_delivery_pair import probe
from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r27_landing_integration.support_aware import Detector as Candidate


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(case, detector):
    _, ctrl, metrics, log = probe.run(case, detector)
    bits = np.asarray(log["detect"], bool)
    return {
        "rises": np.flatnonzero(bits & ~np.r_[False, bits[:-1]]).astype(int).tolist(),
        "active_samples": int(bits.sum()),
        "false_alarm_times": metrics["false_alarm_times"],
        "tool_contact_delay_ms": metrics["tool_contact_delay_ms"],
        "physical_sha256": probe.replay.physical_signature(log),
        "cost": compact(metrics, log, ctrl, case["mass"])["cost"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    plan = json.loads((HERE / "plan.json").read_bytes())
    cases, split = probe.cases()
    if split["audit"] != plan["normal_split"]["audit"]:
        raise ValueError("audit split differs")
    rows = []
    for key in split["audit"]:
        systems = {"c1": record(cases[key], C1),
                   "support_aware": record(cases[key], Candidate)}
        old, new = systems.values()
        gates = {field: old[field] == new[field] for field in old}
        rows.append({"id": key, "systems": systems, "gates": gates,
                     "passed": all(gates.values())})
    doc = {
        "schema": "r27-landing-support-aware-normal-audit.v1",
        "classification": "Separated normal audit before challenged outcomes.",
        "sources": {name: sha(HERE / name) for name in
                    ("support_plan.json", "model.json", "support_aware.py")},
        "episodes": 2 * len(rows), "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": doc["episodes"], "passed": doc["passed"],
                      "failures": [row["id"] for row in rows if not row["passed"]]}, indent=2))


if __name__ == "__main__":
    main()
