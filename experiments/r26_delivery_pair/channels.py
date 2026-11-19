"""Compare the existing counterfactual detector source against the external empty-arm force."""
import json
from pathlib import Path
import sys

import numpy as np
from experiments.r26_delivery_pair.probe import ORIGINAL, Probe, run


class Channels(Probe):
    def reset(self):
        super().reset()
        self.chan=[]

    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,alarm=super().update(t,th,thd,tau_m,tau_s,grip)
        k=min(round(float(t)*1000),len(self.wrench_coef)-1)
        basis=np.array([1.,float(self.box_mass)])
        dw=np.einsum("fc,c->f",self.wrench_coef[k],basis)
        df=np.einsum("fc,c->f",self.fast_coef[k],basis)
        s=self._shared
        self.chan.append([self.cf_detected,self.cf_alarm,self.cf_fast,self.cf_clear,*s.last_residual_wrench[:3]-dw[:3],
                          *s.last_fast_innovation[:3]-df[:3],*self.meter[-1]])
        return residual,alarm


if __name__=="__main__":
    out=Path(sys.argv[1]);out.mkdir()
    p=json.loads((ORIGINAL/"plan.json").read_bytes());res={}
    for f in p["late_impulses"]:
        det,_,_,log=run(p["impulse_base"],Channels,(f["time_s"],f["force_n"]))
        res[f["id"]]=np.column_stack([log["t"],log["detect"],np.asarray(det.chan)])
    np.savez_compressed(out/"channels.npz",**res)
