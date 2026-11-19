"""Identify public detector-route provenance on disclosed G4 landing normals."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r26_delivery_pair.controller import Controller
from experiments.r28_contact_memory.candidate import Detector as Memory
from experiments.r28_landing_replay import evaluator as replay
from experiments.r15_delayed_wrench.observer import NormalForce
from sim import config as C


class RouteTrace:
    route_trace = None

    def reset(self):
        super().reset()
        released = self.released_vector

        class CaptureReleased:
            def __init__(self, inner):
                self.inner = inner
                self.last = None

            def feed(self, *args):
                value = self.inner.feed(*args)
                self.last = None if value is None else np.asarray(value).copy()
                return value

        self.released_vector = CaptureReleased(released)
        self.diagnostic_empty_inverse = NormalForce(
            self.nom, self.nom_box, dt=C.DT, viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)
        self.diagnostic_empty_inverse_value = None
        type(self).route_trace = []

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, detected = super().update(t, th, thd, tau_m, tau_s, grip)
        inverse = self.diagnostic_empty_inverse.update(t, th, tau_s, False)
        self.diagnostic_empty_inverse_value = None if inverse is None else inverse.force_n.copy()
        shared = self._shared
        support = getattr(self.ctrl, "shared_estimator", None)
        normal = getattr(self, "normal_model", None)
        event = getattr(self, "detached_event", None)
        type(self).route_trace.append({
            "detected": bool(detected),
            "grip": bool(grip),
            "cf": bool(getattr(self, "cf_detected", False)),
            "cf_alarm": int(getattr(self, "cf_alarm", 0)),
            "cf_fast": int(getattr(self, "cf_fast", 0)),
            "normal": bool(getattr(normal, "detected", False)),
            "r5": bool(getattr(self, "r5_extra", False)),
            "r8": bool(getattr(self, "r8_extra", False)),
            "detached_pulse_remaining": int(getattr(event, "pulse", 0)),
            "detached_active": bool(getattr(event, "direction", None) is not None),
            "detached_suppress_fast": bool(getattr(event, "suppress_fast", False)),
            "memory_extra": bool(getattr(self, "grip_force_extra", False)),
            "memory_owned": bool(getattr(self, "load_ownership", False)),
            "r29_suppressed": bool(getattr(self, "r29_suppressed", False)),
            "r29_unloading_owned": bool(getattr(self, "unloading_owned", False)),
            "r29_rebound_pending": bool(getattr(self, "rebound_pending", False)),
            "released_force": (None if self.released_vector.last is None
                               else self.released_vector.last.tolist()),
            "released_status": {
                "bias_ready": bool(self.released_vector.inner.bias is not None),
                "observer_exists": bool(self.released_vector.inner.observer is not None),
                "observer_residual_samples": (0 if self.released_vector.inner.observer is None
                                              else len(self.released_vector.inner.observer.residual)),
                "observer_position_samples": (0 if self.released_vector.inner.observer is None
                                              else len(self.released_vector.inner.observer.pos)),
            },
            "continuous_empty_force": (None if self.diagnostic_empty_inverse_value is None
                                       else self.diagnostic_empty_inverse_value.tolist()),
            "table_allowed": bool(getattr(shared, "last_table_allowed", False)),
            "table_wrench": np.asarray(getattr(shared, "last_table_wrench", np.zeros(6)))[:3].tolist(),
            "residual_wrench": np.asarray(shared.last_residual_wrench)[:3].tolist(),
            "fast_innovation": np.asarray(shared.last_fast_innovation)[:3].tolist(),
            "support_valid": bool(getattr(support, "lambda_support_valid", False)),
            "support_fraction": (None if support is None or not np.isfinite(
                getattr(support, "lambda_support", np.nan)) else float(support.lambda_support)),
        })
        return residual, detected


class C1Trace(RouteTrace, C1):
    pass


class MemoryTrace(RouteTrace, Memory):
    pass


def source_names(row):
    names = []
    if row["cf"]:
        names.append("counterfactual")
    if row["normal"]:
        names.append("normal_model")
    if row["r5"]:
        names.append("r5")
    if row["r8"]:
        names.append("r8")
    if row["detached_pulse_remaining"]:
        names.append("detached_pulse")
    if row["memory_extra"]:
        names.append("memory_extra")
    return names


def record(case, detector):
    log = replay.run_episode(case, detector, (None, None), Controller)
    trace = detector.route_trace
    edges = replay.edge_indices(log).astype(int).tolist()
    rows = []
    for edge in edges:
        lo, hi = max(0, edge - 2), min(6000, edge + 3)
        rows.append({
            "edge": edge,
            "actual_tool_N": float(log["f_tool"][edge]),
            "actual_table_N": float(log["f_table"][edge]),
            "sources": source_names(trace[edge]),
            "online": trace[edge],
            "neighborhood": [{"sample": i, **trace[i]} for i in range(lo, hi)],
        })
    return {
        "rises": edges,
        "active_samples": int(np.count_nonzero(log["detect"])),
        "force_exact": replay.force_exact(log, (None, None)),
        "physical_sha256": replay.physical_hash(log),
        "tool": replay.tool_record(log, 1.0, 300),
        "edges": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        parser.error("fresh --out required")
    plan = json.loads(replay.PLAN.read_bytes())
    tool, quiet = plan["known_tool_pair"]
    zero = dict(plan["positive_controls"][0])
    zero.pop("impact")
    cases = (tool, quiet, zero)
    result = {"schema": "r29-g4-route-diagnostic.v1", "classification":
              "Disclosed known normals; actual contact values are offline attribution only.",
              "episodes": len(cases) * 2, "rows": []}
    for case in cases:
        result["rows"].append({
            "id": case["id"],
            "c1": record(case, C1Trace),
            "memory": record(case, MemoryTrace),
        })
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({row["id"]: {arm: [(edge["edge"], edge["sources"], edge["actual_tool_N"])
                                                   for edge in row[arm]["edges"]]
                                         for arm in ("c1", "memory")}
                      for row in result["rows"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
