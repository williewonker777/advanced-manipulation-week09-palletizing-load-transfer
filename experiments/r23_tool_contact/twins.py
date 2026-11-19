"""Run same-normal twins for R23 tool-contact characterization and protection."""
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
from my_torque import Controller
import sim.runner as stock


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recorder(base):
    class Recording(base):
        def reset(self):
            self.r23_tau_s = []
            super().reset()

        def update(self, t, th, thd, tau_m, tau_s, grip):
            self.r23_tau_s.append(np.asarray(tau_s).copy())
            return super().update(t, th, thd, tau_m, tau_s, grip)
    return Recording


def episode(case, variant, detector_type, offset):
    plant = local_plant(None, variant)
    function = types.FunctionType(stock.run_episode.__code__,
                                  dict(stock.run_episode.__globals__, Plant=plant),
                                  stock.run_episode.__name__, stock.run_episode.__defaults__,
                                  stock.run_episode.__closure__)
    function.__kwdefaults__ = stock.run_episode.__kwdefaults__
    detector = recorder(detector_type)()
    metrics, log = function(Controller(), box_mass=case["mass"], box_com=tuple(case["com"]),
                            pick_offset=offset, detector=detector, collision=None)
    if metrics["diverged"] or len(log["t"]) != 6000:
        raise ValueError("complete finite trace required")
    return metrics, {name: np.asarray(log[name]) for name in
                     ("t", "q", "tau", "r", "detect", "f_tool", "f_table", "f_ext")} | {
                         "tau_s": np.asarray(detector.r23_tau_s)}


def rises(bits):
    bits = np.asarray(bits, bool)
    return np.flatnonzero(bits & ~np.r_[False, bits[:-1]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.raw.exists():
        parser.error("fresh outputs required")
    plan_raw = args.plan.read_bytes()
    plan = json.loads(plan_raw)
    if plan.get("candidate_sha256") != sha(HERE / "candidate.py"):
        raise ValueError("candidate differs from predeclared plan")
    if "model_sha256" in plan and plan["model_sha256"] != sha(HERE / "model.json"):
        raise ValueError("model differs from predeclared plan")
    variants = json.loads((HERE.parents[0] / "r22_challenge/plan.json").read_bytes())["variants"]
    model = json.loads((HERE / "model.json").read_bytes())
    center, bound = np.asarray(model["center_joint_Nm"]), np.asarray(model["bound_joint_Nm"])
    start, stop = model["start_index"], model["stop_index"]
    rows, blobs = [], {}
    for case in plan["cases"]:
        for mechanic in plan["mechanics"]:
            traces = {}
            for arm, detector in (("base", Light), ("candidate", Candidate)):
                for twin, offset in (("normal", None), ("offset", case["offset"])):
                    metrics, arrays = episode(case, variants[mechanic], detector, offset)
                    traces[arm, twin] = (metrics, arrays)
                    key = case["id"] + "__" + mechanic + "__" + arm + "__" + twin
                    for name, value in arrays.items():
                        blobs[key + "__" + name] = value
            base_normal, base_offset = traces["base", "normal"][1], traces["base", "offset"][1]
            cand_normal, cand_offset = traces["candidate", "normal"][1], traces["candidate", "offset"][1]
            physical_exact = all(np.array_equal(traces["base", twin][1][name],
                                                traces["candidate", twin][1][name])
                                 for twin in ("normal", "offset")
                                 for name in ("t", "q", "tau", "f_tool", "f_table", "f_ext"))
            contact = base_offset["f_tool"] > 1.
            active = np.flatnonzero(contact)
            first, last = (int(active[0]), int(active[-1])) if len(active) else (None, None)
            bits = cand_offset["detect"]
            edge = rises(bits)
            if first is None:
                delay, prealarm, contact_rises = None, False, []
            else:
                after = np.flatnonzero(bits[first:min(6000, last + 301)])
                delay = None if not len(after) else int(after[0])
                prealarm = bool(bits[:first].any())
                contact_rises = edge[(edge >= first) & (edge < min(6000, last + 301))].astype(int).tolist()
            delta_tau = np.linalg.norm(base_offset["tau_s"] - base_normal["tau_s"], axis=1)
            delta_r = np.linalg.norm(base_offset["r"] - base_normal["r"], axis=1)
            score = np.max(np.abs(base_offset["tau_s"][start:stop] - center) / bound, axis=1)
            rows.append({"case": case, "mechanic": mechanic,
                         "contact": {"present": bool(len(active)), "first_index": first, "last_index": last,
                                     "duration_ms": int(len(active)), "peak_N": float(base_offset["f_tool"].max()),
                                     "impulse_Ns": float(base_offset["f_tool"].sum() * .001)},
                         "sensor_visible": {"max_tau_s_delta_vs_same_normal_Nm": float(delta_tau.max()),
                                            "max_r_delta_vs_same_normal_Nm": float(delta_r.max()),
                                            "max_normal_envelope_score": float(score.max())},
                         "base": {"normal_rises": rises(base_normal["detect"]).astype(int).tolist(),
                                  "offset_rises": rises(base_offset["detect"]).astype(int).tolist()},
                         "candidate": {"normal_rises": rises(cand_normal["detect"]).astype(int).tolist(),
                                       "offset_rises": edge.astype(int).tolist(), "contact_delay_ms": delay,
                                       "contact_prealarm": prealarm, "contact_window_rises": contact_rises},
                         "physical_exact": physical_exact,
                         "normal_exact": bool(np.array_equal(base_normal["detect"], cand_normal["detect"])),
                         "offset_exact": bool(np.array_equal(base_offset["detect"], cand_offset["detect"]))})
    np.savez_compressed(args.raw, **blobs)
    contacts = [row for row in rows if row["contact"]["present"]]
    nuisances = [row for row in rows if not row["contact"]["present"]]
    gates = {"physical_exact": all(row["physical_exact"] for row in rows),
             "normal_exact": all(row["normal_exact"] for row in rows),
             "nuisance_exact": all(row["offset_exact"] for row in nuisances),
             "contacts_detected": bool(contacts) and all(
                 row["candidate"]["contact_delay_ms"] is not None
                 and row["candidate"]["contact_delay_ms"] <= 120
                 and not row["candidate"]["contact_prealarm"]
                 and len(row["candidate"]["contact_window_rises"]) <= 1 for row in contacts)}
    protected = plan["schema"] == "r23-tool-contact-protected-plan.v1"
    payload = {"schema": "r23-tool-contact-twins.v1", "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
               "candidate_sha256": sha(HERE / "candidate.py"), "model_sha256": sha(HERE / "model.json"),
               "protected": protected, "counts": {"rows": len(rows), "contacts": len(contacts),
                                                    "nuisances": len(nuisances)},
               "gates": gates, "passed": all(gates.values()) if protected else None, "rows": rows,
               "raw_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"protected": protected, "counts": payload["counts"], "gates": gates,
                      "passed": payload["passed"]}))


if __name__ == "__main__":
    main()
