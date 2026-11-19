"""Post-result diagnostic only: compare one long pulse with its short first-pulse twin."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r25_composed.detector import Detector
from experiments.r25_holdout import evaluator as lab
from experiments.r25_weak.world_horizontal import WorldHorizontalWitness


class Observed(Detector):
    def reset(self):
        super().reset()
        # Never OR this port with the frozen candidate; it is a measurement only.
        self.probe = WorldHorizontalWitness(self.nom, self.nom_box, threshold=8.394415779390961)
        self.measurements = []

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
        self.probe(t, th, tau_s, grip, self._shared)
        n = self.normal_model
        self.measurements.append((float(n.ema/n.threshold) if n and n.last_active else 0.0,
                                  self.probe.last_score, float(self._shared.last_score)))
        return residual, report


def run(raw):
    j = json.loads(raw)
    if j.get("schema") != "r25-independent-variable-pulse-result.v1" or j["episodes"] != 198:
        raise ValueError("complete independent acquisition required")
    rows = [r for r in j["rows"] if r["kind"] == "paired"]
    def rank(row):
        ev = row["records"]["r25"]["events"]
        long, short = ev["long_single"], ev["isolated_first"]
        return (not (long["strict"] and not short["attributable_rise_indices"]),
                not (long["strict"] and not short["strict"]), row["variant"] != "stock",
                row["id"], row["variant"])
    row = min(rows, key=rank)
    plan = json.loads(lab.PLAN.read_bytes())
    c = next(c for c in plan["paired"] if c["id"] == row["id"])
    v = next(v for v in plan["variants"].values() if v["kind"] == row["variant"])
    lo, f, s = c["long"], *c["impacts"]
    if lo[0] != f[0] or lo[2] != f[2]:
        raise ValueError("duration-only comparison required")
    twins = {"none": (None,None), "long": (lo,None), "first": (f,None),
             "second": (None,s), "both": (f,s)}
    logs, meters = {}, {}
    for tw, inp in twins.items():
        det = Observed()
        _, log = lab.run_episode(c, v, lambda:det, inp)
        if not lab.exact_force(log, inp)["passed"]:
            raise ValueError("force trace changed")
        logs[tw], meters[tw] = log, np.array(det.measurements)
    if lab.certified.trace_signature(logs) != row["r24_physics_sha256"]:
        raise ValueError("physical trace changed")
    t = logs["both"]["t"]
    i = lab.actual_start(t, f)[0]
    k = lab.actual_start(t, s)[0]
    hi = min(6000, i + plan["event_window_ms"])
    kh = min(6000, k + plan["event_window_ms"])
    slots = {"long_single": ("long",i,hi,["none"]), "isolated_first": ("first",i,min(k,hi),["none"]),
             "isolated_second": ("second",k,kh,["none"]), "both_first": ("both",i,min(k,hi),["none","second"]),
             "both_second": ("both",k,kh,["first"])}
    ledger = {name:lab.event(logs[tw],[logs[r] for r in refs],a,b,plan)
              for name,(tw,a,b,refs) in slots.items()}
    if ledger != row["records"]["r25"]["events"]:
        raise ValueError("observational instrumentation changed a scientific result")
    evidence = {}
    for name in ("none","first","long"):
        log = logs[name]
        applied = log["f_ext_xyz_committed"]
        active = np.flatnonzero(np.any(applied != 0,axis=1))
        peak = np.max(meters[name][i:hi], axis=0)
        evidence[name] = {"nonzero_force_samples":int(len(active)),
                          "actual_applied_peak_n":float(np.max(np.linalg.norm(applied,axis=1))),
                          "first_280ms_peaks":{"original_horizontal_ema_ratio":float(peak[0]),
                                               "diagnostic_support_compensated_force_n":float(peak[1]),
                                               "original_directional_score":float(peak[2])}}
    return {"schema":"r25-post-independent-pulse-length-probe.v1",
            "classification":"Post-disclosure diagnostic only; no detector, criterion or independent score changed.",
            "selection_rule":"Prefer a strict long pulse with zero attributable short-first rise, then a strict long pulse and non-strict short, then stock and lexical identity.",
            "source_result_sha256":hashlib.sha256(raw).hexdigest(),"episodes":len(twins),
            "case":c["id"],"mechanics":row["variant"],"nominal_force_n":lo[2],
            "declared_durations_ms":[f[1],lo[1]],"same_actual_five_twin_physics":True,
            "all_five_event_records_reproduced":True,"signals":evidence,"scientific_events":ledger}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    doc=run(args.first.read_bytes())
    with args.out.open("x") as out:
        json.dump(doc,out,indent=2,allow_nan=False)
        out.write("\n")
    print(doc["case"],doc["mechanics"],doc["declared_durations_ms"],doc["signals"])
