"""Compare public support state at the v1 pickup gain and tool-high counterexample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.r26_delivery_pair.controller import Controller
from experiments.r27_landing_integration.candidate import Detector as V1
from sim.runner import run_episode


class Probe(V1):
    latest = None

    def reset(self):
        super().reset()
        self.trace = []
        Probe.latest = self

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
        packet = self._shared
        estimator = getattr(self.ctrl, "shared_estimator", None)
        vector = self.grip_force_value
        self.trace.append({
            "index": int(round(1000 * float(t))), "grip": bool(grip),
            "output": bool(report), "extra": bool(self.grip_force_extra),
            "force_xyz_N": None if vector is None else np.asarray(vector).tolist(),
            "horizontal_N": None if vector is None else float(np.hypot(*vector[:2])),
            "table_allowed": bool(packet.last_table_allowed),
            "observed_table_normal_N": float(max(0.0, packet.last_table_wrench[2]))
                                      if packet.last_table_allowed else 0.0,
            "handoff_alpha": float(getattr(self.ctrl, "hybrid_alpha", 0.0)),
            "support_valid": bool(getattr(estimator, "lambda_support_valid", False)),
            "estimated_payload_support_fraction": (None if estimator is None else
                float(getattr(estimator, "lambda_support", float("nan")))),
        })
        return residual, report


def episode(collision=None, offset=None):
    metrics, log = run_episode(
        Controller(), box_mass=4.0, box_com=(0.02, -0.01, 0.03),
        pick_offset=offset, detector=Probe(), collision=collision)
    rows = Probe.latest.trace
    bits = np.asarray(log["detect"], bool)
    rises = np.flatnonzero(bits & ~np.r_[False, bits[:-1]]).astype(int).tolist()
    selected = [row for row in rows if row["extra"] or
                (row["horizontal_N"] is not None and row["horizontal_N"] > 14.78106064144941)]
    return {"rises": rises, "false_alarm_times": metrics["false_alarm_times"],
            "tool_contact_delay_ms": metrics["tool_contact_delay_ms"],
            "selected": selected}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    rows = {
        "pickup_y15": episode((1.42, [0.0, 15.0, 0.0])),
        "tool_high": episode(None, (0.0, 0.0, 0.03)),
        "normal": episode(),
    }
    doc = {"schema": "r27-landing-support-diagnosis.v1",
           "classification": "Post-v1 diagnostic only; no candidate selection claim.",
           "episodes": 3, "rows": rows}
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    for name, row in rows.items():
        compact = [{key: value for key, value in sample.items() if key not in ("force_xyz_N",)}
                   for sample in row["selected"] if sample["extra"]]
        print(name, "rises", row["rises"], "extra", compact)


if __name__ == "__main__":
    main()
