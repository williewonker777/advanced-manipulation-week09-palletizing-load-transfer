"""Attribute observed alarms in R23 FREE. Ground truth is joined offline only after simulation."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.r23_detached.free import Detector as Free
from experiments.r23_holdout import evaluator as held


class Probe(Free):
    def reset(self):
        super().reset()
        self.rows = []

    def update(self, t, th, thd, tau_m, tau_s, grip):
        result = super().update(t, th, thd, tau_m, tau_s, grip)
        if 3.58 <= t < 4.15:
            s = self._shared
            force = np.array(s.last_residual_wrench[:3])
            force[2] = min(force[2], 0.)
            scores = np.r_[np.abs(force) / s.force_threshold,
                           np.abs(s.last_residual_wrench[3:]) / s.moment_threshold,
                           0. if s.last_table_allowed else max(0., s.last_wrench[2]) / s.out_of_height_fz_threshold]
            normal = self.normal_model
            self.rows.append({'t':round(t,3), 'grip':bool(grip),'det':bool(result[1]),'r10':bool(self.r8_base_detect),
                     'directional':bool(s.detected),'normal_model':bool(normal.detected) if normal else False,
                     'horizontal':bool(self.horizontal_detected),'yaw':bool(self.yaw_active),
                     'light':bool(self.light_witness.state.active),'detached':bool(self.detached.evidence.active),
                     'table_allowed':bool(s.last_table_allowed),'scores':scores.tolist(),
                     'yaw_score':None if self.yaw_value is None else float(abs(self.yaw_value)/self.bounds[int(bool(grip))]),
                     'wrench':s.last_wrench.tolist(),'residual_wrench':s.last_residual_wrench.tolist(),
                     'normal_error':normal.last_error.tolist() if normal else None,'normal_ema':float(normal.ema/normal.threshold) if normal else 0.,
                     'th':th.tolist(),'tau_s':tau_s.tolist(),'tau_m':tau_m.tolist(),'estimated_load':float(s.load_fraction)})
        return result


def measurements(case, variant, impact):
    detector = Probe()
    _, log = held.run_episode(case, variant, lambda:detector, (impact,None))
    cert = held.certified
    if not cert.exact_force(log,(impact,None))['passed']:
        raise ValueError('applied force differs')
    rises = cert.rises(log).astype(int).tolist()
    for x in detector.rows:
        ix=round(x['t']*1000)
        x['table_N']=float(log['f_table'][ix]);x['external_N']=float(log['f_ext'][ix])
        x['tool_N']=float(log['f_tool'][ix])
    return {'rises':rises,'physics_sha256':cert.physical_hash(log),'rows':detector.rows}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=json.loads(held.PLAN.read_bytes());case=next(c for c in plan['negative'] if c['id']=='n23_detached_heavy_interp')
    out={}
    for name in ('stock','actuator_skew'):
        out[name]=measurements(case,plan['variants'][name],None)
    with a.out.open('x') as f:json.dump(out,f,allow_nan=False)
    for name,v in out.items():
        print(name,v['rises'])
        for row in v['rows']:
            if round(row['t']*1000) in [3857,3861,3864]:
                print({k:row[k] for k in ('t','r10','directional','normal_model','horizontal','yaw','grip','scores','yaw_score','normal_ema','wrench','table_N')})


if __name__=='__main__':
    main()
