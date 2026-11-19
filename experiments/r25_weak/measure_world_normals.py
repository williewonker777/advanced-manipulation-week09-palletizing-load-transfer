"""Measure the locked force-free fit or audit panel for world-horizontal force."""
from pathlib import Path
import argparse
import json

import numpy as np

from experiments.r24_holdout import evaluator as lab
from experiments.r25_weak.measure_world_horizontal import Probe, summarize


def main(args):
    plan = json.loads(args.plan.read_text())
    frozen = json.loads(args.frozen.read_text())
    mechanics = (["stock", "damping_coupled"] if args.split == "fit" else
                 ["actuator_delay_gain", "contact_armature"])
    cases = [("negative", row) for row in plan["negative"]]
    cases += [("single", row) for row in plan["single"]]
    rows, arrays = [], {}
    for kind, case in cases:
        for name in mechanics:
            probe = Probe()
            _, log = lab.run_episode(case, plan["variants"][name], lambda: probe,
                                     (None, None))
            common = lab.record_common(log, (None, None), [], plan)
            expected = next(row for row in frozen["rows"]
                            if row["id"] == case["id"] and row["variant"] == name)
            reference = (expected["records"]["r24"] if kind == "negative" else
                         expected["records"]["r24"]["twins"]["none"])
            if any(reference.get(key) != value for key, value in common.items()):
                raise ValueError(f"frozen force-free mismatch: {case['id']} {name}")
            data = np.asarray(probe.samples25, float)
            key = f"{case['id']}__{name}"
            arrays[key] = data
            rows.append({"id": case["id"], "kind": kind, "mechanics": name,
                         "frozen_common_exact": True, **summarize(data)})
            print(f"{args.split} {len(rows)}/{len(cases) * len(mechanics)}", flush=True)
    with args.raw.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    maximum = max(row["corrected_norm_max_n"] for row in rows)
    doc = {"split": args.split, "episodes": len(rows),
           "maximum_corrected_norm_n": maximum, "rows": rows}
    if args.split == "fit":
        doc["locked_threshold_n"] = 1.10 * maximum + .25
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({key: value for key, value in doc.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--split", choices=("fit", "audit"), required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
