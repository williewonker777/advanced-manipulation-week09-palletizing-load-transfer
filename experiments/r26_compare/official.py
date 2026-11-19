"""Execute the unchanged course evaluate.py for three optional yaw research systems."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from tools import r21_replay as shared
from tools import r26_signed_yaw_replay as replay
from tools.r20_official_compare import table,summarize


def main(kit,out):
    if out.exists() or out.is_symlink():raise ValueError('fresh output required')
    runtime=replay.runtime_gate(kit); source=replay.closure()
    blobs,pins=replay.source_blobs(source)
    for n in ('candidate.py','__init__.py'):
        rel='experiments/r26_yaw_acquisition/'+n
        blobs[rel]=(replay.ROOT/rel).read_bytes();pins[rel]=replay.digest(blobs[rel])
    if pins['experiments/r26_yaw_acquisition/candidate.py']!='212f4e79367cb636c0143a2d1f8aa7ba5584aaaca5ecde6b412d01a4985b6784':
        raise ValueError('unregistered source')
    m=(replay.ROOT/shared.KIT_MANIFEST).read_bytes(); kit_pins=shared.authenticate_kit(kit,m)
    regs={'r25':'experiments.r25_composed.detector','signed_level':'experiments.r26_long.candidate','signed_acquisition':'experiments.r26_yaw_acquisition.candidate'}
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1');env.pop('PYTHONPATH',None)
    rows={}
    with tempfile.TemporaryDirectory(dir=out.parent)as tmp:
        path=Path(tmp);shared.install_overlay(kit,path,m,blobs,pins)
        for a,src in regs.items():
            cmd=[sys.executable,'-B','evaluate.py','--ctrl','my_torque','--detector',src,'--out',a+'.txt']
            p=subprocess.run(cmd,cwd=path,env=env,check=True,text=True,capture_output=True)
            if sum(line.strip().startswith('실행 중: 박스') for line in p.stdout.splitlines())!=29:raise ValueError('incomplete official grid')
            cells=table((path/(a+'.txt')).read_text());rows[a]={'cells':cells,'counts':summarize(cells),'episodes':29}
        if kit_pins!=shared.authenticate_kit(kit,m) or kit_pins!=shared.authenticate_kit(path,m) or any(replay.digest(path/k)!=h or replay.digest(replay.ROOT/k)!=h for k,h in pins.items()):
            raise ValueError('stock or validated source changed')
    doc={'schema':'r26-yaw-unchanged-official-execution.v1','runtime':runtime,'episodes':87,'results':rows,
         'unchanged_evaluator_sha256':shared.EVALUATOR_SHA256,'all_66_exact':all(r['cells']==rows['r25']['cells']for r in rows.values())}
    with out.open('x')as f:json.dump(doc,f,indent=2);f.write('\n')
    print(doc['all_66_exact'],[(k,x['counts'])for k,x in rows.items()])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kit',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();main(a.kit,a.out)
