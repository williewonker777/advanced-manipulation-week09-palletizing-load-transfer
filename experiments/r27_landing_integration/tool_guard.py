"""Check disclosed genuine tool-contact offsets with frozen C1 and R27."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r27_landing_integration.candidate import Detector as Candidate
from experiments.r27_landing_integration.mechanism import FROZEN, HERE, run, sha


OFFSETS = {
    "tool_high": (0.0, 0.0, 0.03),
    "tool_lateral": (0.015, 0.0, 0.0),
    "tool_low": (0.0, 0.0, -0.03),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    import mujoco
    if mujoco.__version__ != "3.17.0" or any(
            sha(HERE / name) != expected for name, expected in FROZEN.items()):
        raise ValueError("runtime or frozen source/model pin differs")
    rows = []
    for key, offset in OFFSETS.items():
        records = {name: run(base, (None, None), offset=offset)[2]
                   for name, base in (("c1", C1), ("r27", Candidate))}
        old, new = records["c1"]["tool"], records["r27"]["tool"]
        gates = {
            "same_physics": (records["c1"]["physical_sha256"] ==
                             records["r27"]["physical_sha256"]),
            "no_new_prealarm": not new["prealarm"] or old["prealarm"],
            "preserve_true_tool": (not old["detected"] or
                                   (new["detected"] and
                                    new["delay_ms"] <= old["delay_ms"] + 5)),
            "no_extra_rises": len(records["r27"]["rises"]) <=
                              len(records["c1"]["rises"]),
        }
        rows.append({"id": key, "offset": offset, "records": records,
                     "gates": gates, "passed": all(gates.values())})
    doc = {
        "schema": "r27-landing-genuine-tool-guard.v1",
        "classification": "Disclosed tool offsets; not blind.",
        "runtime": {"mujoco": mujoco.__version__}, "sources": dict(FROZEN),
        "episodes": 2 * len(rows), "rows": rows,
        "genuine_contact_rows": [row["id"] for row in rows
                                 if row["records"]["c1"]["tool"]["present"]],
        "passed": all(row["passed"] for row in rows),
    }
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": doc["episodes"],
                      "genuine_contact_rows": doc["genuine_contact_rows"],
                      "passed": doc["passed"],
                      "tool": {row["id"]: {name: record["tool"]
                               for name, record in row["records"].items()}
                               for row in rows}}, indent=2))


if __name__ == "__main__":
    main()
