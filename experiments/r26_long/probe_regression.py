"""Inspect the disclosed +5 duration-floor counterexample for the first ownership trial."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from experiments.r25_holdout import evaluator
from experiments.r26_long.candidate import Detector


class Probe(Detector):
    def reset(self):
        super().reset(); self.samples=[]
    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,report=super().update(t,th,thd,tau_m,tau_s,grip)
        tick=int(round(1000*float(t)))
        if 1560<=tick<=1710:
            self.samples.append({"tick":tick,"report":bool(report),"yaw":bool(self.yaw_active),
                                 "yaw_value":None if self.yaw_value is None else float(self.yaw_value),
                                 "identity":bool(self.yaw_identity.active),
                                 "bridge":bool(self.yaw_identity_bridge),
                                 "mean":float(self.yaw_identity.mean),
                                 "normal_last_active":bool(self.normal_model.last_active),
                                 "normal_error_xy":np.asarray(self.normal_model.last_error,float).tolist(),
                                 "normal_error_ratio":float(np.linalg.norm(self.normal_model.last_error)/self.normal_model.threshold),
                                 "residual_wrench":np.asarray(residual,float).tolist(),
                                 "r10_score":float(self._shared.last_score)})
        return residual,report


def edges(samples,key):
    out=[];before=False
    for row in samples:
        now=bool(row[key])
        if now!=before:out.append({"tick":row["tick"],"active":now,"sample":row})
        before=now
    return out


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args(argv)
    if a.out.exists() or a.out.is_symlink():p.error('fresh --out required')
    plan=json.loads(evaluator.PLAN.read_bytes());case=next(x for x in plan['paired'] if x['id']=='p25_close_transfer_cross');variant=plan['variants']['stock']
    d=Probe();_,log=evaluator.run_episode(case,variant,lambda:d,(case['impacts'][0],None))
    for row in d.samples:row['actual_force_N']=float(np.linalg.norm(log['f_ext_xyz_committed'][row['tick']]))
    doc={'schema':'r26-long-v1-regression-probe.v1','case':case['id'],'variant':'stock','twin':'first',
         'report_edges':edges(d.samples,'report'),'yaw_edges':edges(d.samples,'yaw'),
         'bridge_edges':edges(d.samples,'bridge'),'samples':d.samples}
    a.out.write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:doc[k] for k in ('report_edges','yaw_edges','bridge_edges')},indent=2))
if __name__=='__main__':main()
