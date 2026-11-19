"""Verify the R23 extension is inert for servo and unsupported controller modes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import types

import numpy as np

from experiments.r22_challenge.evaluator import local_plant
from experiments.r22_light.candidate import Detector as Light
from experiments.r23_tool_contact.candidate import Detector as Candidate
import my_servo
import my_torque
import sim.runner as stock


HERE = Path(__file__).resolve().parent


class Unsupported(my_torque.Controller):
    MODE = None


def episode(controller_type, detector_type, case):
    plant = local_plant(None, {"kind": "stock"})
    function = types.FunctionType(stock.run_episode.__code__,
                                  dict(stock.run_episode.__globals__, Plant=plant),
                                  stock.run_episode.__name__, stock.run_episode.__defaults__,
                                  stock.run_episode.__closure__)
    function.__kwdefaults__ = stock.run_episode.__kwdefaults__
    metrics, log = function(controller_type(), box_mass=case["mass"], box_com=tuple(case["com"]),
                            pick_offset=case["offset"], detector=detector_type(), collision=None)
    return metrics, log


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("fresh output required")
    cases = ({"id": "normal", "mass": 5.42, "com": [-.009, .022, -.014], "offset": None},
             {"id": "offset", "mass": 5.42, "com": [-.009, .022, -.014],
              "offset": [-.0042, .0033, .0012]})
    rows = []
    for mode, controller in (("servo", my_servo.Controller), ("unsupported", Unsupported)):
        for case in cases:
            base_metrics, base = episode(controller, Light, case)
            new_metrics, new = episode(controller, Candidate, case)
            exact = {name: bool(np.array_equal(base[name], new[name])) for name in
                     ("t", "q", "tau", "r", "detect", "f_tool", "f_table", "f_ext")}
            rows.append({"mode": mode, "case": case["id"], "exact": exact,
                         "metrics_exact": all(base_metrics[key] == new_metrics[key]
                                              for key in base_metrics
                                              if key not in ("ctrl_time_us", "r_peak"))})
    payload = {"schema": "r23-tool-contact-fallback-check.v1",
               "candidate_sha256": hashlib.sha256((HERE / "candidate.py").read_bytes()).hexdigest(),
               "rows": rows, "passed": all(all(row["exact"].values()) and row["metrics_exact"] for row in rows)}
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"passed": payload["passed"], "rows": len(rows)}))


if __name__ == "__main__":
    main()
