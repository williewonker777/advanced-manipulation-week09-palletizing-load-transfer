"""Two declined, unchanged hypotheses on a now-disclosed second pulse and signed-long control."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import json

from experiments.r25_composed.detector import Detector as Selected
from experiments.r25_composed.with_weak import Detector as Immediate
from experiments.r25_composed.deferred_weak import Detector as Deferred
from experiments.r25_holdout import evaluator as trial


def one(job):
    name, kind, c, v, plan = job
    other = {"immediate":Immediate,"deferred":Deferred}[name]
    row = trial.one((kind,c,v,{"r24_joint":Selected,"r25":other},plan))
    return name, row


def run(raw):
    plan = json.loads(trial.PLAN.read_bytes())
    original = json.loads(raw)
    known = {r["id"]: r for r in original["rows"] if r["variant"] == "stock"}
    cases = [("paired","p25_close_transfer_cross"),("single","s25_long_transfer_signed")]
    jobs = [(d, k, next(c for c in plan[k] if c["id"] == case), plan["variants"]["stock"], plan)
            for d in ("immediate","deferred") for k,case in cases]
    with ProcessPoolExecutor(2) as pool:
        measured = list(pool.map(one,jobs))
    rows = []
    for name, result in measured:
        ref = known[result["id"]]
        if (result["records"]["r24_joint"] != ref["records"]["r25"] or
                result["r24_physics_sha256"] != ref["r24_physics_sha256"]):
            raise ValueError("unchanged selected comparator or dynamics differ")
        def summary(record):
            ev=record.get("events",{"single":record.get("event")})
            return {slot: {k:e[k] for k in ("strict","delays_ms","duplicate_rises",
                                            "active_samples_in_slot","quiet_before")}
                    for slot,e in ev.items()}
        rows.append({"arm":name, "case":result["id"],"variant":result["variant"],
                     "episodes":result["episodes"],"selected":summary(result["records"]["r24_joint"]),
                     "declined":summary(result["records"]["r25"]),
                     "false_rises_by_twin":{tw:v["false_times_s"] for tw,v in result["records"]["r25"]["twins"].items()},
                     "scientific_positive_and_protections_pass":result["passed"],
                     "full_selected_record_and_dynamics_reproduced":True})
    return {"schema":"r25-post-disclosure-rejected-arm-contrast.v1",
            "independent_qualification":False,"candidate_or_threshold_changed":False,
            "selection_changed":False,"qualification_scope":"Only post-disclosure stock two-case analysis. The unchanged optional arms that failed development were not submitted to the independent evaluation.",
            "choice_of_cases":"Inspect the missed short second whose measured signal crossed the optional boundary, and a known long single with duplicate reports as a control.",
            "source_result_sha256":hashlib.sha256(raw).hexdigest(),
            "episodes":sum(r["episodes"] for r in rows),"rows":rows}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    q=p.parse_args()
    result=run(q.first.read_bytes())
    with q.out.open("x") as out:
        json.dump(result,out,indent=2,allow_nan=False)
        out.write("\n")
    print([(r["arm"],r["case"],r["scientific_positive_and_protections_pass"])for r in result["rows"]])
