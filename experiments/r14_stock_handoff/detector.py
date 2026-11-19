"""R12 coupled detector loaded from transparent pinned coefficient JSON."""
from pathlib import Path
import hashlib
import json
import numpy as np
import my_detector

DEFAULT_COEFFICIENTS = Path(__file__).resolve().parents[2] / 'models/r12/counterfactual_coefficients.json'


def _load(path):
    raw = Path(path).read_bytes(); document = json.loads(raw)
    if document.get('schema') != 'r12-counterfactual-coefficients.v1':
        raise ValueError('coefficient schema mismatch')
    arrays = []
    for name in ('wrench_coef','fast_coef'):
        record = document['arrays'][name]
        value = np.asarray(record['values'], dtype=np.float64)
        if (record['dtype'] != 'float64' or list(value.shape) != record['shape']
                or not np.isfinite(value).all()
                or hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() != record['raw_c_order_sha256']):
            raise ValueError('invalid coefficient array: '+name)
        arrays.append(value)
    return (*arrays, hashlib.sha256(raw).hexdigest())


class Detector(my_detector.Detector):
    coefficient_path = DEFAULT_COEFFICIENTS
    def reset(self):
        super().reset()
        self.wrench_coef,self.fast_coef,self.coefficient_sha256 = _load(self.coefficient_path)
        self.cf_alarm=0; self.cf_fast=0; self.cf_clear=0; self.cf_detected=False
    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,_ = super().update(t,th,thd,tau_m,tau_s,grip)
        p=self._shared; k=min(round(float(t)*1000),len(self.wrench_coef)-1)
        basis=np.array([1.,float(self.box_mass)])
        dw=np.einsum('fc,c->f',self.wrench_coef[k],basis)
        df=np.einsum('fc,c->f',self.fast_coef[k],basis)
        wrench=p.last_wrench-dw; corrected=p.last_residual_wrench-dw
        innovation=p.last_fast_innovation-df
        over,_=p._directional_alarm(corrected,wrench,p.last_table_allowed)
        fast_over,_=p._fast_directional_alarm(innovation)
        if grip: fast_over=False
        self.cf_alarm=self.cf_alarm+1 if over else 0
        self.cf_fast=self.cf_fast+1 if fast_over else 0
        if over or fast_over:self.cf_clear=0
        else:self.cf_clear+=1
        if self.cf_alarm>=p.persistence or self.cf_fast>=p.fast_persistence:self.cf_detected=True
        elif self.cf_clear>=p.release_persistence:self.cf_detected=False
        learned=bool(getattr(self.normal_model,'detected',False)
                     or getattr(self,'r5_extra',False) or getattr(self,'r8_extra',False))
        return residual,bool(self.cf_detected or learned)
