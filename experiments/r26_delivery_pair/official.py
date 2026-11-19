"""Compare two landing research systems and the two untouched submission defaults."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from tools import r26_landing_replay as lab
from tools import r24_transfer_replay as shared
from tools.r20_official_compare import table,summarize


def main(kit,out):
    if out.is_symlink() or out.exists():raise ValueError('fresh JSON required')
    runtime=lab.runtime_gate(kit)
    closure=lab.closure(); root=lab.ROOT; blobs=lab.source_blobs(closure)
    default=json.loads((root/'results/r10/official_regression.json').read_bytes())
    for name in ('servo','torque'):
        path=f'my_{name}.py'; blobs[path]=(root/path).read_bytes()
        if lab.digest(blobs[path])!=default[name]['source_sha256'] or lab.digest(blobs['my_detector.py'])!=default[name]['detector_sha256']:
            raise ValueError('R10 submission pin changed')
    pins={k:lab.digest(v)for k,v in blobs.items()}
    mani=(root/shared.MANIFEST).read_bytes(); kit_pins=shared.authenticate_kit(kit,mani)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='disable')
    env.pop('PYTHONPATH',None)
    ctr='experiments.r26_delivery_pair.controller'
    arms={'r10_servo':('my_servo','my_detector'),'r10_torque':('my_torque','my_detector'),
          'landing_r24':(ctr,'experiments.r14_stock_handoff.detector'),
          'landing_pair':(ctr,'experiments.r26_delivery_pair.candidate')}
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        base=Path(tmp);shared.install(kit,base,mani,blobs,pins)
        def score(name):
            ctrl,det=arms[name]
            r=subprocess.run([sys.executable,'-B','evaluate.py','--ctrl',ctrl,'--detector',det,'--out',name+'.txt'],
                             cwd=base,capture_output=True,text=True,env=env)
            if r.returncode or sum(s.strip().startswith('실행 중: 박스') for s in r.stdout.splitlines())!=29:
                raise RuntimeError(r.stdout+'\n'+r.stderr)
            cells=table((base/(name+'.txt')).read_text())
            return name,{'modules':[ctrl,det],'episodes':29,'cells':cells,'counts':summarize(cells)}
        with ThreadPoolExecutor(2) as pool:records=dict(pool.map(score,arms))
        if (kit_pins!=shared.authenticate_kit(base,mani) or kit_pins!=shared.authenticate_kit(kit,mani)
                or any(lab.digest(base/k)!=h or lab.digest(root/k)!=h for k,h in pins.items())):
            raise ValueError('post-execution provenance changed')
    defaults={key:records['r10_'+key]['cells']==default[key]['control_and_collisions']for key in ('servo','torque')}
    doc={'schema':'r26-simultaneous-controller-detector-original-official.v1','runtime':runtime,'episodes':116,
         'stock_evaluator_sha256':shared.EVALUATOR_SHA256,'r10_relative_exact':defaults,
         'landing_same_66_cells':records['landing_r24']['cells']==records['landing_pair']['cells'],
         'records':records,'unauthorized_promotion':False}
    if not all(defaults.values()):raise ValueError('R10 official cells did not reproduce')
    with out.open('x')as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
    print(defaults,doc['landing_same_66_cells'],{k:v['counts'] for k,v in records.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.kit,a.out)
