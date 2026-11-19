"""A post-R24 measurement: independent physical signals, observed before looking at applied-force truth."""
import argparse
from pathlib import Path
import hashlib
import json


from sim import config as C
from experiments.r15_delayed_wrench.observer import NormalForce
from experiments.r24_composed.detector import Detector
from experiments.r24_holdout import evaluator as lab


class Probe(Detector):
    def reset(self):
        super().reset()
        self.values = []
        self.inverse = NormalForce(self.nom, self.nom_box, dt=C.DT,
                                   viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)

    def update(self, t, th, thd, tau_m, tau_s, grip):
        res, alarm = super().update(t, th, thd, tau_m, tau_s, grip)
        smp = self.inverse.update(t, th, tau_s, grip)
        x = self._shared
        n = self.normal_model
        force = (smp.force_n.tolist() if smp is not None else [None]*3)
        self.values.append([bool(alarm), bool(x.detected), bool(n and n.detected),
                            bool(self.r8_base_detect), bool(self.horizontal_detected), bool(self.yaw_active),
                            bool(grip), float(x.last_score), float(x.last_fast_score),
                            float(n.ema/n.threshold) if n else None,
                            list(map(float, x.last_residual_wrench)), list(map(float, x.last_wrench)),
                            n.last_error.tolist() if n else None, force,
                            None if self.yaw_value is None else float(self.yaw_value)])
        return res, alarm


def run(prior):
    if hashlib.sha256(prior.read_bytes()).hexdigest() != 'dfce0da5e2ded2dd0dfb3e02a168aca48f661f715df0647fb1b46c1c87872c08':
        raise ValueError('expected pinned R24 independent result')
    plan = json.loads(lab.PLAN.read_bytes())
    case = next(c for c in plan['paired'] if c['id'] == 'p24_transfer_near_cross')
    mech = 'actuator_delay_gain'
    archive = next(rr for rr in json.loads(prior.read_bytes())['rows']
                   if rr['id']==case['id'] and rr['variant']==mech)
    twins = {'none': (None,None), 'first': (case['impacts'][0],None),
             'second': (None,case['impacts'][1]), 'both': tuple(case['impacts'])}
    records,logs = {},{}
    for tw,impacts in twins.items():
        det = Probe()
        _,log = lab.run_episode(case,plan['variants'][mech],lambda: det,impacts)
        starts = [lab.certified.actual_start(log['t'],x)[0] for x in impacts if x]
        assert lab.record_common(log,impacts,starts,plan)==archive['records']['r24']['twins'][tw]
        rows=[]
        for i in range(1420,1940):
            rows.append([i,*det.values[i],log['f_ext_xyz_committed'][i].tolist(),float(log['f_tool'][i]),float(log['f_table'][i])])
        records[tw]={'rises':lab.certified.rises(log).astype(int).tolist(),'samples':rows}
        logs[tw]=log
    assert lab.certified.trace_signature(logs)==archive['r23_physics_sha256']
    return {'case':case,'mechanics':mech,'original_per_twin_physics_and_common_score_exact':True,
            'source':'post-disclosure measurement; no force truth provided to a detector',
            'fields':['tick','alarm','directional','normal_model','r10','horizontal','yaw','grip','score','fast','normal_ratio','residual_wrench','raw_wrench','normal_xy_error','inverse_force','yaw_Nm','applied_N','tool_N','table_N'],
            'twins':records}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    j=run(a.prior)
    with a.out.open('x') as f:json.dump(j,f,allow_nan=False);f.write('\n')
    print({k:r['rises'] for k,r in j['twins'].items()})
