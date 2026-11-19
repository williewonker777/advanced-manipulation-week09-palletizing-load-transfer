"""Calibrate the drive estimator on force-free nominal runs with different sensor noise, not collisions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import types

import numpy as np

from experiments.r23_detached.free import Detector as Free
from experiments.r23_holdout import evaluator as lab
from experiments.r24_loaded.drive import DriveEstimate, departure

HERE=Path(__file__).resolve().parent
PLAN=HERE/'drive_plan.json'


class Recorder(Free):
    def reset(self):
        super().reset();self.drive=DriveEstimate()
    def update(self,t,th,thd,tau_m,tau_s,grip):
        self.drive.feed(thd,tau_m,tau_s,grip)
        return super().update(t,th,thd,tau_m,tau_s,grip)


def capture(job):
    seed,variant=job
    def local(second, mechanics):
        parent=lab.local_plant(second,mechanics)
        class NoisePlant(parent):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs);self.rng=np.random.default_rng(seed)
        return NoisePlant
    orig=lab.run_episode; runner=types.FunctionType(orig.__code__,dict(orig.__globals__,local_plant=local))
    d=Recorder();_,log=runner({'mass':4.4,'com':[.013,-.017,.008]},variant,lambda:d,(None,None))
    if not lab.certified.exact_force(log,(None,None))['passed'] or d.drive.gain is None:raise ValueError('no authentic excited estimate')
    return {'seed':seed,'estimate':d.drive.gain.tolist(),'rms_command_excitation_Nm':d.drive.excitation.tolist(),
            'physics_sha256':lab.certified.physical_hash(log)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('fit','check'),required=True);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--model',type=Path);p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args()
    cfg=json.loads(PLAN.read_bytes());configs=json.loads(lab.PLAN.read_bytes())['variants']
    arms={'nominal':{'kind':'stock'}} if a.stage=='fit' else {'nominal':{'kind':'stock'},'actuator':configs['actuator_skew']}
    seeds=cfg['train_noise_seeds' if a.stage=='fit' else 'check_noise_seeds'];jobs=[(s,v) for v in arms.values() for s in seeds]
    with ProcessPoolExecutor(a.workers) as pool:rows=list(pool.map(capture,jobs))
    if a.stage=='fit':
        x=np.asarray([r['estimate'] for r in rows]);margin=np.maximum(.002,np.ptp(x,axis=0))
        doc={'schema':'week09-r24-motor-port-calibration.v1','low':(x.min(0)-margin).tolist(),'high':(x.max(0)+margin).tolist(),
             'normal_only':True,'fit_plan_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),'normal_seed_records':rows}
    else:
        m=json.loads(a.model.read_bytes())
        for i,r in enumerate(rows):r.update(mechanics=list(arms)[i//len(seeds)],departure=departure(r['estimate'],m['low'],m['high']))
        doc={'classification':'unused sensor seeds; this is a drive-port check not a collision holdout','rows':rows,
             'nominal_departures':sum(r['departure'] for r in rows if r['mechanics']=='nominal'),
             'actuator_departures':sum(r['departure'] for r in rows if r['mechanics']=='actuator')}
    with a.out.open('x') as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
    print(a.stage,{k:v for k,v in doc.items() if k not in ('rows','normal_seed_records')})


if __name__=='__main__':main()
