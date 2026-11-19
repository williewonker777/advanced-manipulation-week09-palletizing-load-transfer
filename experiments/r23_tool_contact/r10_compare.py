"""Compare the original R10 detector on frozen R23 same-normal twins."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import my_detector
from experiments.r23_tool_contact.twins import episode, rises


HERE = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--reference-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("fresh output required")
    plan_raw = args.plan.read_bytes()
    plan = json.loads(plan_raw)
    variants = json.loads((HERE.parents[0] / "r22_challenge/plan.json").read_bytes())["variants"]
    reference = np.load(args.reference_raw)
    rows = []
    for case in plan["cases"]:
        for mechanic in plan["mechanics"]:
            twins = {}
            for twin, offset in (("normal", None), ("offset", case["offset"])):
                _, arrays = episode(case, variants[mechanic], my_detector.Detector, offset)
                twins[twin] = arrays
                prefix = case["id"] + "__" + mechanic + "__base__" + twin + "__"
                physical_exact = all(np.array_equal(arrays[name], reference[prefix + name])
                                     for name in ("t", "q", "tau", "f_tool", "f_table", "f_ext"))
                if not physical_exact:
                    raise ValueError("R10 and R22 physical traces differ")
            contact = twins["offset"]["f_tool"] > 1.
            active = np.flatnonzero(contact)
            bits = twins["offset"]["detect"]
            if len(active):
                first, last = active[[0, -1]]
                after = np.flatnonzero(bits[first:min(6000, last + 301)])
                delay = None if not len(after) else int(after[0])
                prealarm = bool(bits[:first].any())
            else:
                delay, prealarm = None, False
            rows.append({"case": case["id"], "mechanic": mechanic,
                         "contact_present": bool(len(active)),
                         "contact_peak_N": float(twins["offset"]["f_tool"].max()),
                         "normal_rises": rises(twins["normal"]["detect"]).astype(int).tolist(),
                         "offset_rises": rises(bits).astype(int).tolist(),
                         "contact_delay_ms": delay, "contact_prealarm": prealarm})
    payload = {"schema": "r23-r10-tool-contact-comparison.v1",
               "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
               "r10_sha256": hashlib.sha256(Path(my_detector.__file__).read_bytes()).hexdigest(),
               "rows": rows}
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": len(rows), "contacts": sum(row["contact_present"] for row in rows),
                      "detected": sum(row["contact_delay_ms"] is not None for row in rows)}))


if __name__ == "__main__":
    main()
