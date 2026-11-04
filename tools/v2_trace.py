"""Post-freeze plotting traces; all plant/log quantities are analysis-only."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from tools.check_integrity import check_integrity
from tools.v2_study import require_holdout_freeze,verify_freeze_sources
from tools.v2_resources import wait_for_capacity
from sim.runner import run_episode


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out.resolve()
    if out==ROOT or ROOT in out.parents:raise ValueError('raw plotting traces must stay outside the workcopy')
    verify_freeze_sources(require_holdout_freeze());check_integrity(ROOT);out.mkdir(parents=True,exist_ok=True)
    meta=[]
    for mode in ['servo','torque']:
        for version in ['base','v1','v2']:
            ctrl=mode+'_controller' if version=='base' else ('v1_'+mode if version=='v1' else 'my_'+mode)
            det='collision_detector' if version=='base' else ('v1_detector' if version=='v1' else 'my_detector')
            base=importlib.import_module(ctrl).Controller
            class Guarded(base):
                def compute(self,t,*signals):
                    if int(round(t*1000))%250==0:wait_for_capacity()
                    return super().compute(t,*signals)
            host=wait_for_capacity()
            r,log=run_episode(Guarded(),box_mass=8.,detector=importlib.import_module(det).Detector())
            p=out/f'{version}_{mode}.npz'
            # These are plotting/evaluation data, never imported by fitting.
            keys=['t','tau','e_mm','f_table','r','detect','q','q_d','f_ext']
            np.savez_compressed(p,**{k:log[k] for k in keys})
            meta.append(dict(file=p.name,controller=ctrl,detector=det,official=False,mass=8.,host=host,
                             sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            print(p.name,flush=True)
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');verify_freeze_sources(require_holdout_freeze())


if __name__=='__main__':main()
