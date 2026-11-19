"""Use unedited course evaluate.py for this explicit torque entrant, not the original demo."""
import argparse,json
from pathlib import Path
from tools import r31_event_replay as baseline
from experiments.r30_broad_checks.run import actual_official
from .check import ROOT,level,resolve

ENTRY='experiments.r33_vertical_composition.candidate:Detector'
def run(kit,out):
    d=ROOT/'experiments/r33_vertical_composition'
    reg=level.get(d/'protocol.json');pin=reg['candidate_sha256'];resolve(ENTRY,pin)
    raw,files,names=baseline.load()
    blobs={n:files[n] for n in names};pins={n:baseline.shared.digest(b) for n,b in blobs.items()}
    for n,h in {**reg['dependencies'],'experiments/r33_vertical_composition/__init__.py':level.sha(d/'__init__.py'),
                'experiments/r33_vertical_composition/candidate.py':pin}.items():
        f=ROOT/n
        if not n.startswith('experiments/r3') or f.is_symlink() or f.resolve().parent.parent!=ROOT/'experiments' or level.sha(f)!=h:
            raise ValueError('unexpected authored source')
        blobs[n]=f.read_bytes();pins[n]=h
    result=actual_official(kit,ENTRY,blobs,pins,out.parent)
    result.update(r31_closure=baseline.shared.digest(raw),candidate_sha256=pin,default_changed=False)
    level.save(out,result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():p.error('new output only')
    run(a.kit,a.out)
