"""Independent decision replay after deterministic public-clock subtraction."""
from pathlib import Path
import numpy as np
import my_detector


class Detector(my_detector.Detector):
    model_path=None
    def reset(self):
        super().reset()
        with np.load(Path(self.model_path),allow_pickle=False) as z:
            self.wrench_coef=z['wrench_coef'];self.fast_coef=z['fast_coef']
        self.cf_alarm=0;self.cf_fast=0;self.cf_clear=0;self.cf_detected=False
        self.parity_mismatches=0;self.corrected_samples=[]
    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,original=super().update(t,th,thd,tau_m,tau_s,grip);p=self._shared;k=min(round(float(t)*1000),len(self.wrench_coef)-1)
        basis=np.array([1.,float(self.box_mass)]);dw=np.einsum('fc,c->f',self.wrench_coef[k],basis);df=np.einsum('fc,c->f',self.fast_coef[k],basis)
        wrench=p.last_wrench-dw;res=p.last_residual_wrench-dw;innovation=p.last_fast_innovation-df
        over,score=p._directional_alarm(res,wrench,p.last_table_allowed);fast_over,fast_score=p._fast_directional_alarm(innovation)
        if grip:fast_over=False;fast_score=0.
        self.cf_alarm=self.cf_alarm+1 if over else 0;self.cf_fast=self.cf_fast+1 if fast_over else 0
        if over or fast_over:self.cf_clear=0
        else:self.cf_clear+=1
        if self.cf_alarm>=p.persistence or self.cf_fast>=p.fast_persistence:self.cf_detected=True
        elif self.cf_clear>=p.release_persistence:self.cf_detected=False
        # Preserve every learned OR branch explicitly, including coincidence
        # with the original main alarm that the counterfactual replay removes.
        learned=bool(getattr(self.normal_model,'detected',False)
                     or getattr(self,'r5_extra',False)
                     or getattr(self,'r8_extra',False))
        replay=bool(self.cf_detected or learned)
        if not np.any(dw) and not np.any(df):self.parity_mismatches+=int(replay!=bool(original))
        if 3.88<=t<=3.93:self.corrected_samples.append({'t':float(t),'main_score':float(score),'fast_score':float(fast_score),'dw':dw.tolist(),'df':df.tolist(),'original':bool(original),'replay':replay})
        return residual,replay


def bound(model):
    class Bound(Detector):model_path=model
    return Bound
