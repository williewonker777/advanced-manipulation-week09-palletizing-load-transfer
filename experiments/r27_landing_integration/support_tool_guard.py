"""Known and preregistered unseen tool offsets for support-aware source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r27_landing_integration.mechanism import run
from experiments.r27_landing_integration.support_aware import Detector


OFFSETS = {
    "known_high": (0.0, 0.0, 0.03),
    "known_lateral": (0.015, 0.0, 0.0),
    "known_low": (0.0, 0.0, -0.03),
    "unseen_high_25mm": (0.0, 0.0, 0.025),
    "unseen_high_35mm": (0.0, 0.0, 0.035),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    rows = []
    for key, offset in OFFSETS.items():
        records = {name: run(base, (None, None), offset=offset)[2]
                   for name, base in (("c1", C1), ("support_aware", Detector))}
        old, new = records.values()
        gates = {
            "same_physics": old["physical_sha256"] == new["physical_sha256"],
            "same_rises": old["rises"] == new["rises"],
            "same_active_samples": old["active_samples"] == new["active_samples"],
            "same_tool": old["tool"] == new["tool"],
        }
        rows.append({"id": key, "offset": offset, "records": records,
                     "gates": gates, "passed": all(gates.values())})
    doc = {"schema": "r27-landing-support-aware-tool-guard.v1",
           "classification": "Known plus preregistered unseen tool offsets after freeze.",
           "episodes": 2 * len(rows), "rows": rows,
           "genuine_contact_rows": [row["id"] for row in rows
             if row["records"]["c1"]["tool"]["present"]],
           "passed": all(row["passed"] for row in rows)}
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": doc["episodes"], "passed": doc["passed"],
                      "genuine_contact_rows": doc["genuine_contact_rows"],
                      "failures": [row["id"] for row in rows if not row["passed"]]}, indent=2))


if __name__ == "__main__":
    main()
