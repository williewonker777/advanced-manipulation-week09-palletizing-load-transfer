"""Measure a 3-D unloaded observer and compare authentic matched landing controllers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.r26_delivery_pair.observer import ReleasedVector
from experiments.r14_stock_handoff.detector import Detector as Coupled
from experiments.r24_transfer_replay import evaluator as replay
from experiments.r26_delivery_pair.controller import Controller

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "r24_transfer_replay"


class Probe(Coupled):
    def reset(self):
        super().reset()
        self.sensor = ReleasedVector(self.nom,self.nom_box)
        self.meter = []

    def update(self,t,th,thd,motor,spring,closed):
        residual,alarm = super().update(t,th,thd,motor,spring,closed)
        f = self.sensor.feed(t,th,spring,closed)
        self.meter.append([np.nan]*3 if f is None else f)
        return residual,alarm


def run(case, cls, inp=None, control=Controller):
    import mujoco
    if mujoco.__version__ != "3.17.0":
        raise ValueError("course MuJoCo 3.17.0 required")
    det,ctrl=cls(),control()
    runner=replay.run_function(case["mechanics"]["table_solref_time_scale"])
    m,log=runner(ctrl,case["mass"],tuple(case["com"]),det,collision=inp,pick_offset=case["offset"])
    log["signed_force"]=np.asarray(log["plant"].r24_committed_force)
    exact=replay.expected_force(log["t"],inp)
    if (m["diverged"] or len(log["t"])!=6000 or not np.array_equal(log["signed_force"],exact)
            or not np.array_equal(np.asarray(log["plant"].r24_requested_force),exact)):
        raise ValueError("invalid physics, sample count, or applied input")
    return det,ctrl,m,log


def cases():
    p=json.loads((HERE/"plan.json").read_bytes())
    known=json.loads((ORIGINAL/"plan.json").read_bytes())
    out={c["id"]:c for c in known["known_replay"]+known["new_normal"]}
    panels={"fit":list(p["calibration_fit_existing_normals"]),
            "audit":list(p["normal_audit_existing"])}
    for tag in ("fit","audit"):
        for i,(mass,com,offset,scale) in enumerate(p["additional_force_free_"+tag]):
            key=f"{tag}-{i}";panels[tag].append(key)
            out[key]={"id":key,"mass":mass,"com":com,"offset":offset,"mechanics":{"table_solref_time_scale":scale}}
    return out,panels


def capture(job):
    key,case,impact=job
    detector,_,metrics,log=run(case,Probe,impact)
    vector=np.asarray(detector.meter,float);active=np.where(np.isfinite(vector).all(axis=1))[0]
    if not len(active):raise ValueError("no valid empty-arm estimates")
    data=np.column_stack((log["t"],vector,log["detect"],log["signed_force"]))
    summary={"id":key,"physical":replay.physical_signature(log),"reported_rises":replay.rise_times(log),
             "official_false_times":metrics["false_alarm_times"],
             "valid_samples":len(active),"first_tick":int(active[0]),
             "max_xyz":vector[active].max(axis=0).tolist(),"min_xyz":vector[active].min(axis=0).tolist(),
             "max_xy":float(np.linalg.norm(vector[active,:2],axis=1).max())}
    if "expected" in case and impact is None:
        exp=case["expected"]
        if summary["physical"]!=exp["r24_physical_sha256"] or summary["official_false_times"]!=exp["r24_false"]:
            raise ValueError("known R24 control or report changed")
    return key,summary,data


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--panel",choices=("fit","audit","target"),required=True)
    p.add_argument("--out",type=Path,required=True);args=p.parse_args()
    args.out.mkdir();reg=json.loads((ORIGINAL/"plan.json").read_bytes())
    allcases,panels=cases()
    if args.panel=="target":
        base=reg["impulse_base"];jobs=[("none",base,None)]+[(x["id"],base,(x["time_s"],x["force_n"])) for x in reg["late_impulses"]]
    else:jobs=[(k,allcases[k],None) for k in panels[args.panel]]
    with ProcessPoolExecutor(2) as pool: result=list(pool.map(capture,jobs))
    np.savez_compressed(args.out/"signals.npz",**{k:a for k,_,a in result})
    doc={"schema":"r26-physical-landing-vector-probe.v1","panel":args.panel,"rows":[s for _,s,_ in result],
         "episodes":len(jobs),"signal_order":"t, fx,fy,fz, existing_alarm, committed applied force x,y,z",
         "signals_sha256":hashlib.sha256((args.out/"signals.npz").read_bytes()).hexdigest()}
    (args.out/"rows.json").write_text(json.dumps(doc,allow_nan=False,indent=2)+"\n")
    print(args.panel,[(s["id"],s["max_xyz"],s["min_xyz"])for _,s,_ in result])
