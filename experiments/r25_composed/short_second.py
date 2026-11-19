"""Post-disclosure measurement of the missed 55 ms second pulse; no output fusion or tuning."""
import argparse
from pathlib import Path
import hashlib
import json

import numpy as np

from experiments.r25_composed.pulse_diagnosis import Observed
from experiments.r25_holdout import evaluator as lab


class Meter(Observed):
    def reset(self):
        super().reset()
        self.vector, self.level = [], []

    def update(self, *inputs):
        residual, report = super().update(*inputs)
        self.vector.append(self.probe.last_vector.copy())
        self.level.append(bool(self._shared.detected))
        return residual, report


def measure(raw):
    result = json.loads(raw)
    case_id, mechanics = "p25_close_transfer_cross", "stock"
    row = next(r for r in result["rows"] if r["id"] == case_id and r["variant"] == mechanics)
    plan = json.loads(lab.PLAN.read_bytes())
    c = next(c for c in plan["paired"] if c["id"] == case_id)
    first, second = c["impacts"]
    twin = {"none":(None,None), "long":(c["long"],None), "first":(first,None),
            "second":(None,second), "both":(first,second)}
    logs, meters = {}, {}
    for name, inp in twin.items():
        det = Meter()
        _, log = lab.run_episode(c, plan["variants"][mechanics], lambda:det, inp)
        if not lab.exact_force(log, inp)["passed"]:
            raise ValueError("physical force mismatch")
        logs[name], meters[name] = log, det
    if lab.certified.trace_signature(logs) != row["r24_physics_sha256"]:
        raise ValueError("trajectory does not reproduce the recorded runtime")
    t = logs["both"]["t"]
    begin, bounds = lab.actual_start(t, second)
    end = min(len(t), begin + plan["event_window_ms"])
    sl = slice(begin, end)
    evidence = {}
    for slot, name, ref in (("isolated_second","second","none"), ("both_second","both","first")):
        event = lab.event(logs[name], [logs[ref]], begin, end, plan)
        if event != row["records"]["r25"]["events"][slot]:
            raise ValueError("independent reported event no longer reproduced")
        direction = np.asarray(second[2][:2], float)
        direction /= np.linalg.norm(direction)
        dep = np.array(meters[name].vector)[sl] - np.array(meters[ref].vector)[sl]
        proj = dep @ direction
        i = int(np.argmax(proj))
        observations = {}
        for n in (name, ref):
            vals = np.array(meters[n].measurements)[sl]
            force = logs[n]["f_ext_xyz_second"][sl]
            observations[n] = {"peak_committed_second_force_N":float(np.max(np.linalg.norm(force,axis=1))),
                               "peak_base_horizontal_ema_over_threshold":float(vals[:,0].max()),
                               "peak_optional_support_compensated_force_N":float(vals[:,1].max()),
                               "original_directional_detector_active_samples":
                                   int(np.count_nonzero(np.array(meters[n].level)[sl]))}
        evidence[slot] = {"event":event, "twins":observations,
                          "peak_matched_horizontal_difference_along_applied_direction_N":float(proj[i]),
                          "peak_ms_after_first_nonzero":i}
    return {"schema":"r25-post-disclosure-short-second-witness.v1",
            "qualification":False, "decision":"Diagnosis only; no change to frozen R25 or any threshold.",
            "selection":"After the primary result, inspect the disclosed stock transfer-cross named in the missed-second family.",
            "case":case_id,"variant":mechanics,"registered_second":second,"actual_second":bounds,
            "earlier_frozen_optional_force_threshold_N":8.394415779390961,
            "units":"Force in N; all maxima inside the registered 280 ms second slot. Matched-reference subtraction is offline only and unavailable to an operating detector.",
            "episodes":5,"original_physics_signature":row["r24_physics_sha256"],
            "two_recorded_events_and_all_five_dynamics_reproduced":True,
            "independent_result_sha256":hashlib.sha256(raw).hexdigest(),"comparisons":evidence}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    q=p.parse_args()
    j=measure(q.first.read_bytes())
    with q.out.open("x") as out:
        json.dump(j,out,allow_nan=False,indent=2)
        out.write("\n")
    print(j["comparisons"])
