"""R26 vs the frozen R25, with actual physical twins and inherited condition-by-condition guard floors."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
from pathlib import Path
from types import FunctionType

from experiments.r25_composed.detector import Detector as Baseline
from experiments.r25_fragment import study as older
from experiments.r25_holdout import evaluator as previous
from experiments.r25_weak.nearby_plan import heldback_plan

ANCHOR="r25_frozen"
ROOT=Path(__file__).resolve().parents[2]


def construct(key):
    if key==ANCHOR:
        return Baseline
    mod,member=key.split(":")
    if not mod.startswith("experiments.r26_") or member!="Detector":
        raise ValueError("registered R26 detector required")
    return getattr(importlib.import_module(mod),member)


def plan_jobs(panel,arm):
    if panel.startswith("r25"):
        plan=json.loads(previous.PLAN.read_bytes())
        jobs=[(panel,k,c,v,(arm,),plan) for k in ("negative","single","paired")
              for c in plan[k] for v in plan["variants"].values()]
        return [x for x in jobs if panel=="r25" or x[3]["kind"]=="stock"]
    if panel=="weak32":
        reg=json.loads((ROOT/"experiments/r25_weak/PLAN.json").read_bytes())
        from experiments.r24_holdout import evaluator as e
        p=heldback_plan(json.loads(e.PLAN.read_bytes()),reg)
        return [(panel,"single",c,v,(arm,),p) for c in p["single"] for v in p["variants"].values()]
    if panel=="grip":
        from experiments.r24_holdout import evaluator as e
        s=json.loads((ROOT/"experiments/r25_composed/plan.json").read_bytes())["supplemental_grip_pairs"]
        p=json.loads(e.PLAN.read_bytes());jobs=[]
        for i,t in enumerate(s["first_times"]):
            for j,gap in enumerate(s["separations_s"]):
                for k,f in enumerate(s["second_N"]):
                    case={"id":f"grip-{i}{j}{k}","phase":"pickup","mass":s["mass_kg"],"com":s["com_m"],
                          "offset":None,"impacts":[[t,s["first_N"]],[t+gap,f]]}
                    jobs.extend((panel,"paired",case,p["variants"][m],(arm,),p) for m in s["mechanics"])
        return jobs
    return older.jobs(panel,(arm,))


def capture(spec):
    panel,k,c,v,keys,plan=spec
    arm=keys[0]
    if panel.startswith("r25"):
        j=previous.one((k,c,v,{"r24_joint":Baseline,"r25":construct(arm)},plan))
        return {"kind":k,"id":c["id"],"variant":v["kind"],"episodes":j["episodes"],
                "physical_signature":j["r24_physics_sha256"],
                "records":{ANCHOR:j["records"]["r24_joint"],arm:j["records"]["r25"]}}
    fn=FunctionType(older.capture.__code__,dict(older.capture.__globals__,Joint=Baseline,constructor=construct))
    result=fn(spec)
    result["records"][ANCHOR]=result["records"].pop(older.ANCHOR)
    return result


def run(panel,arm,workers,archive=None):
    import mujoco
    if mujoco.__version__!="3.17.0":
        raise ValueError("course MuJoCo 3.17.0 required")
    with ProcessPoolExecutor(workers) as pool:rows=list(pool.map(capture,plan_jobs(panel,arm)))
    # These are the original relative guards; keep the registered R24/R25 logic intact.
    guard=FunctionType(older.guards.__code__,dict(older.guards.__globals__,ANCHOR=ANCHOR))
    summary=FunctionType(older.summarize.__code__,dict(older.summarize.__globals__,ANCHOR=ANCHOR,guards=guard))
    counts,reg=summary(rows,[arm])
    for total in counts.values():
        total["all_registered_pair_slots"]=total.pop("all_four")
    archive_sha=None
    if archive:
        raw=archive.read_bytes();archive_sha=hashlib.sha256(raw).hexdigest()
        if not panel.startswith("r25") or archive_sha!="83987dba8ce14fb6cc0283021d6bd062ac2fedb317ceb8da67c3d4b2b8c4f4fe":
            raise ValueError("authentic earlier R25 course run required")
        earlier={(r["id"],r["variant"]):r for r in json.loads(raw)["rows"]}
        for row in rows:
            old=earlier[row["id"],row["variant"]]
            if row["physical_signature"]!=old["r24_physics_sha256"] or row["records"][ANCHOR]!=old["records"]["r25"]:
                raise ValueError("old physics/event record did not reproduce")
    return {"schema":"r26-disclosed-comparison.v1","independent":False,"panel":panel,"arm":arm,
            "real_episodes":sum(r["episodes"] for r in rows),"counts":counts,"regressions":reg[arm],
            "verified_r25_archive_sha256":archive_sha,"rows":rows}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--panel",required=True,choices=("r25","r25-stock","weak32","r24","r23","nearby","stress","grip","sample","target"))
    p.add_argument("--arm",required=True);p.add_argument("--archive",type=Path)
    p.add_argument("--workers",type=int,choices=(1,2),default=2);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error("fresh output required")
    j=run(a.panel,a.arm,a.workers,a.archive)
    with a.out.open("x")as f:json.dump(j,f,allow_nan=False);f.write("\n")
    print(j["real_episodes"],j["counts"],j["regressions"])
