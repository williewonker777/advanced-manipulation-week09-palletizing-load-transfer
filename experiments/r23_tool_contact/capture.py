"""Capture detector-visible signals for the predeclared R23 tool-contact study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import types

import numpy as np

from experiments.r22_challenge.evaluator import local_plant
from experiments.r22_light.candidate import Detector as Light
from my_torque import Controller
import sim.runner as stock


HERE = Path(__file__).resolve().parent
PLAN = HERE / "plan.json"


class Probe(Light):
    def reset(self):
        self.sensor = {name: [] for name in ("t", "th", "thd", "tau_m", "tau_s", "grip")}
        super().reset()

    def update(self, t, th, thd, tau_m, tau_s, grip):
        for name, value in (("t", t), ("th", th), ("thd", thd), ("tau_m", tau_m),
                            ("tau_s", tau_s), ("grip", grip)):
            self.sensor[name].append(np.asarray(value).copy())
        return super().update(t, th, thd, tau_m, tau_s, grip)


def episode(case, variant):
    plant = local_plant(None, variant)
    function = types.FunctionType(stock.run_episode.__code__,
                                  dict(stock.run_episode.__globals__, Plant=plant),
                                  stock.run_episode.__name__, stock.run_episode.__defaults__,
                                  stock.run_episode.__closure__)
    function.__kwdefaults__ = stock.run_episode.__kwdefaults__
    detector = Probe()
    metrics, log = function(Controller(), box_mass=case["mass"], box_com=tuple(case["com"]),
                            pick_offset=case.get("offset"), detector=detector,
                            collision=case.get("impact"))
    if metrics["diverged"] or len(log["t"]) != 6000:
        raise ValueError("complete finite trace required")
    arrays = {name: np.asarray(value) for name, value in detector.sensor.items()}
    arrays.update({name: np.asarray(log[name]) for name in ("r", "detect", "f_tool", "f_table",
                                                            "f_ext", "q", "tau")})
    return metrics, arrays


def rises(bits):
    bits = np.asarray(bits, bool)
    return np.flatnonzero(bits & ~np.r_[False, bits[:-1]])


def summarize(arrays):
    t = arrays["t"]
    contact = arrays["f_tool"] > 1.
    active = np.flatnonzero(contact)
    first, last = (int(active[0]), int(active[-1])) if len(active) else (None, None)
    tau_step = np.linalg.norm(np.diff(arrays["tau_s"], axis=0, prepend=arrays["tau_s"][:1]), axis=1)
    mismatch = np.linalg.norm(arrays["tau_s"] - arrays["tau_m"], axis=1)
    residual = np.linalg.norm(arrays["r"], axis=1)
    velocity = np.linalg.norm(arrays["thd"], axis=1)
    pick = (t >= 1.25) & (t < 1.75)
    row = {
        "contact_present": bool(len(active)), "contact_first": first, "contact_last": last,
        "contact_first_s": None if first is None else float(t[first]),
        "contact_duration_ms": int(len(active)), "contact_peak_N": float(arrays["f_tool"].max()),
        "contact_impulse_Ns": float(arrays["f_tool"].sum() * .001),
        "base_rises": rises(arrays["detect"]).astype(int).tolist(),
        "pick_max_residual_norm": float(residual[pick].max()),
        "pick_max_abs_residual_joint": np.max(np.abs(arrays["r"][pick]), axis=0).tolist(),
        "pick_max_tau_sensor_step": float(tau_step[pick].max()),
        "pick_max_tau_mismatch": float(mismatch[pick].max()),
        "pick_max_velocity": float(velocity[pick].max()),
    }
    if first is not None:
        window = slice(max(0, first - 20), min(len(t), last + 121))
        row.update({
            "contact_window_max_residual_norm": float(residual[window].max()),
            "contact_window_max_abs_residual_joint": np.max(np.abs(arrays["r"][window]), axis=0).tolist(),
            "contact_window_max_tau_sensor_step": float(tau_step[window].max()),
            "contact_window_max_tau_mismatch": float(mismatch[window].max()),
            "contact_window_max_velocity": float(velocity[window].max()),
        })
    return row


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.raw.exists():
        parser.error("fresh outputs required")
    plan_raw = PLAN.read_bytes()
    plan = json.loads(plan_raw)
    rows, blobs = [], {}
    for split in ("calibration_normal", "development", "fresh", "external_force_checks"):
        for case in plan[split]:
            for mechanic in plan["mechanics"]:
                variant = json.loads((HERE.parents[0] / "r22_challenge/plan.json").read_bytes())["variants"][mechanic]
                metrics, arrays = episode(case, variant)
                key = split + "__" + case["id"] + "__" + mechanic
                rows.append({"key": key, "split": split, "case": case, "mechanic": mechanic,
                             "metrics": jsonable(metrics), "summary": summarize(arrays)})
                for name, value in arrays.items():
                    blobs[key + "__" + name] = value
    np.savez_compressed(args.raw, **blobs)
    payload = {"schema": "r23-tool-contact-capture.v1",
               "plan_sha256": hashlib.sha256(plan_raw).hexdigest(), "rows": rows,
               "raw_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": len(rows), "contacts": sum(r["summary"]["contact_present"] for r in rows)}))


if __name__ == "__main__":
    main()
