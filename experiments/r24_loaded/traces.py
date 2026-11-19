"""Measured examples with matching no-force twins; never the original course demonstration."""
import argparse
import json
from pathlib import Path

from experiments.r23_detached.free import Detector as Free
from experiments.r23_holdout import evaluator as lab
from experiments.r24_loaded.studies import get_arm


def record(cls,case,variant,impact):
    class Measured(cls):
        def reset(self):
            super().reset();self.samples=[];self.motor_departure=None
        def update(self,t,th,thd,tau_m,tau_s,grip):
            result=super().update(t,th,thd,tau_m,tau_s,grip)
            if 3.78<=t<=3.98:
                y=self.yaw_value;raw=getattr(self.yaw,'raw_value',y)
                self.motor_departure=getattr(self,'changed_drive',None)
                self.samples.append([round(t*1000),bool(grip),None if raw is None else float(raw),None if y is None else float(y),
                    float(self.bounds[int(bool(grip))]),float(self._shared.last_table_wrench[2]),bool(result[1])])
            return result
    det=Measured();_,log=lab.run_episode(case,variant,lambda:det,(impact,None))
    cert=lab.certified
    if not cert.exact_force(log,(impact,None))['passed']:raise ValueError('force schedule differs')
    for s in det.samples:s.extend([float(log['f_table'][s[0]]),float(log['f_ext'][s[0]])])
    return {'samples':det.samples,'rises_ms':cert.rises(log).astype(int).tolist(),'physical':cert.physical_hash(log),
            'measured_drive_departure':det.motor_departure},log


def run(key):
    p=json.loads(lab.PLAN.read_bytes());case=next(c for c in p['negative'] if c['id']=='n23_detached_heavy_interp')
    pairs={};impact=(3.837,(14.,0.,0.))
    for name in ('stock','actuator_skew'):
        rec={};log={}
        for arm,cls in [('r23',Free),('candidate',get_arm(key))]:
            rec[arm]={};log[arm]={}
            for twin,x in [('none',None),('impact',impact)]:
                rec[arm][twin],log[arm][twin]=record(cls,case,p['variants'][name],x)
            start,_=lab.certified.actual_start(log[arm]['impact']['t'],impact)
            if not lab.certified.prefix(log[arm]['none'],log[arm]['impact'],start,start)['passed']:raise ValueError('different preforce history')
            rec[arm]['scored_force']=lab.event_record(log[arm]['impact'],[log[arm]['none']],start,start+p['event_window_ms'],p['quiet_ms'],p['reference_twin_ms'])
        if any(rec['r23'][tw]['physical']!=rec['candidate'][tw]['physical'] for tw in ('none','impact')):raise ValueError('physical trace changed')
        pairs[name]=rec
    return {'schema':'week09-r24-near-landing-twins.v1','classification':'disclosed research study on original simulator; not course demo',
            'episodes':8,'case':case,'force':impact,'candidate':key,'fields':['t_ms','grip','raw_yaw_Nm','used_yaw_Nm','original_bound_Nm','estimated_positive_support_N','reported','true_table_N','applied_N'],
            'mechanics':pairs}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    doc=run(a.arm)
    with a.out.open('x') as f:json.dump(doc,f,indent=1,allow_nan=False);f.write('\n')
    print({k:{arm:[rec['none']['rises_ms'],rec['scored_force']['delays_ms']] for arm,rec in v.items()} for k,v in doc['mechanics'].items()})
