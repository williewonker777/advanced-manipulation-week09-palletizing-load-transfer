"""Reproduce and attribute the disclosed R25 long-force duplicate edges."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.r25_composed.detector import Detector as R25
from experiments.r25_holdout import evaluator


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_SHA = "83987dba8ce14fb6cc0283021d6bd062ac2fedb317ceb8da67c3d4b2b8c4f4fe"
CASE_ID = "s25_long_transfer_signed"
ROUTES = ("report", "r10", "normal", "horizontal", "yaw", "light", "detached",
          "pickup_hold", "pickup_completion", "continuation_observed", "continuation_bridge")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def vector(value):
    if value is None:
        return None
    result = np.asarray(value, float)
    return result.tolist() if np.isfinite(result).all() else None


class Observed(R25):
    def reset(self):
        super().reset()
        self.r26_samples = []

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
        tick = int(round(1000 * float(t)))
        if 1540 <= tick <= 1940:
            packet = self._shared
            normal = getattr(self, "normal_model", None)
            light = getattr(getattr(self, "light_witness", None), "state", None)
            detached = getattr(getattr(self, "detached", None), "evidence", None)
            continuation = getattr(self, "continuation", None)
            bound = None
            if self.yaw_value is not None:
                bound = float(self.bounds[int(bool(grip))])
            self.r26_samples.append({
                "tick": tick,
                "grip": bool(grip),
                "report": bool(report),
                "r10": bool(packet.detected),
                "normal": bool(normal is not None and normal.detected),
                "normal_last_active": bool(normal is not None and normal.last_active),
                "normal_error_xy": vector(None if normal is None else normal.last_error),
                "normal_error_ratio": (None if normal is None else
                                       finite(np.linalg.norm(normal.last_error) / normal.threshold)),
                "horizontal": bool(self.horizontal_detected),
                "horizontal_score": finite(self.horizontal_score),
                "yaw": bool(self.yaw_active),
                "yaw_value": finite(self.yaw_value),
                "yaw_bound": bound,
                "yaw_ratio": (None if self.yaw_value is None or bound is None else
                              abs(float(self.yaw_value)) / bound),
                "light": bool(light is not None and light.active),
                "light_score": finite(None if light is None else light.score),
                "detached": bool(detached is not None and detached.active),
                "detached_score": finite(None if detached is None else detached.score),
                "detached_vector_xy": vector(getattr(getattr(self, "detached", None), "vector", None)),
                "pickup_hold": bool(getattr(self, "pick_event_active", False)),
                "pickup_completion": bool(getattr(self, "pick_completion_active", False)),
                "continuation_observed": bool(continuation is not None and continuation.observed),
                "continuation_bridge": bool(continuation is not None and continuation.bridge),
                "continuation_n": int(0 if continuation is None else continuation.n),
                "continuation_mean": vector(None if continuation is None else continuation.mean),
                "r10_slow_score": finite(packet.last_score),
                "r10_fast_score": finite(packet.last_fast_score),
                "residual_wrench": vector(residual),
            })
        return residual, report


def edges(samples, key):
    rises, falls = [], []
    before = False
    for sample in samples:
        now = bool(sample[key])
        if now and not before:
            rises.append(sample["tick"])
        elif before and not now:
            falls.append(sample["tick"])
        before = now
    return {"rises": rises, "falls": falls}


def intervals(samples, key):
    result, start = [], None
    for sample in samples:
        active = bool(sample[key])
        if active and start is None:
            start = sample["tick"]
        elif not active and start is not None:
            result.append([start, sample["tick"] - 1])
            start = None
    if start is not None:
        result.append([start, samples[-1]["tick"]])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    if sha(args.archive) != ARCHIVE_SHA:
        raise ValueError("unexpected disclosed R25 archive")
    archive = json.loads(args.archive.read_bytes())
    plan = json.loads(evaluator.PLAN.read_bytes())
    case = next(row for row in plan["single"] if row["id"] == CASE_ID)
    old = {row["variant"]: row for row in archive["rows"] if row["id"] == CASE_ID}
    output = {"schema": "r26-long-source-diagnosis.v1", "case": case,
              "archive_sha256": sha(args.archive), "variants": {}}
    for name, variant in plan["variants"].items():
        none_detector, force_detector = Observed(), Observed()
        _, none = evaluator.run_episode(case, variant, lambda: none_detector, (None, None))
        _, forced = evaluator.run_episode(case, variant, lambda: force_detector, (case["impact"], None))
        start, boundary = evaluator.actual_start(forced["t"], case["impact"])
        event = evaluator.event(forced, [none], start,
                                min(6000, start + plan["event_window_ms"]), plan)
        expected = old[name]["records"]["r25"]
        if event != expected["event"] or boundary != expected["boundary"]:
            raise ValueError("frozen event mismatch: " + name)
        if evaluator.certified.trace_signature({"none": none, "single": forced}) != old[name]["r24_physics_sha256"]:
            raise ValueError("frozen physical signature mismatch: " + name)
        if not evaluator.exact_force(forced, (case["impact"], None))["passed"]:
            raise ValueError("force mismatch: " + name)
        for sample in force_detector.r26_samples:
            tick = sample["tick"]
            applied = np.asarray(forced["f_ext_xyz_committed"][tick], float)
            sample["actual_force_xyz_N"] = applied.tolist()
            sample["actual_force_norm_N"] = float(np.linalg.norm(applied))
        edge_samples = [sample for sample in force_detector.r26_samples
                        if sample["tick"] in event["raw_rise_indices"]]
        output["variants"][name] = {
            "event": event,
            "boundary": boundary,
            "none_report_edges": edges(none_detector.r26_samples, "report"),
            "forced_edges": {route: edges(force_detector.r26_samples, route) for route in ROUTES},
            "forced_intervals": {route: intervals(force_detector.r26_samples, route) for route in ROUTES},
            "samples_at_output_rises": edge_samples,
            "samples": force_detector.r26_samples,
        }
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: {"event": row["event"],
                                  "route_intervals": row["forced_intervals"]}
                      for name, row in output["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
