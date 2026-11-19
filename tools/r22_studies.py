"""Reproduce the R22 official-table and known four-twin studies in a temporary authenticated course kit."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools import r21_replay as replay
from tools.r20_official_compare import table, summarize

RUNTIME = (
    'experiments/r15_delayed_wrench/observer.py',
    'experiments/r22_normal/__init__.py','experiments/r22_normal/candidate.py',
    'experiments/r22_release/__init__.py','experiments/r22_release/state.py','experiments/r22_release/model.json',
    'experiments/r22_tangent/__init__.py','experiments/r22_tangent/state.py','experiments/r22_tangent/detector.py',
    'experiments/r22_tangent/frozen.py','experiments/r22_tangent/model.json',
    'experiments/r22_combined/__init__.py','experiments/r22_combined/core.py','experiments/r22_combined/candidate.py')
REVISED = ('experiments/r22_guarded/__init__.py','experiments/r22_guarded/candidate.py',
           'experiments/r22_light/__init__.py','experiments/r22_light/witness.py','experiments/r22_light/candidate.py')
MODULES = {'core':'experiments.r22_combined.core','augmented':'experiments.r22_combined.candidate'}


def sources(study, revised=False):
    baseline, _ = replay.snapshot('history' if study == 'pairs' else 'weak')
    extra = list(RUNTIME + (REVISED if revised else ()))
    if study == 'pairs':extra.append('experiments/r22_combined/known_pairs.py')
    if study == 'weak':extra.append('experiments/r22_light/weak_check.py')
    if study == 'timing':extra.extend('experiments/r22_timing/'+n for n in ('__init__.py','study.py','plan.json'))
    for n in extra:
        f=ROOT/n
        if not f.is_file() or f.is_symlink():raise ValueError('regular authored input required: '+n)
        baseline[n]=f.read_bytes()
    return baseline, {k:replay.digest(v) for k,v in baseline.items()}


def official(base, env, revised):
    output={}
    modules={'guarded':'experiments.r22_guarded.candidate','light':'experiments.r22_light.candidate'} if revised else MODULES
    for arm,mod in modules.items():
        report=base/(arm+'.txt')
        result=subprocess.run([sys.executable,'-B','evaluate.py','--ctrl','my_torque','--detector',mod,'--out',str(report)],
                              cwd=base,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        episodes=sum(line.strip().startswith('실행 중: 박스') for line in result.stdout.splitlines())
        if episodes!=29:raise ValueError('incomplete official evaluation')
        cells=table(report.read_text())
        output[arm]={'episodes':episodes,'cells':cells,'outcomes':summarize(cells)}
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--study',choices=('official','pairs','timing','weak'),required=True)
    p.add_argument('--revised',action='store_true');a=p.parse_args(argv)
    if a.study=='weak' and not a.revised:p.error('use --revised for the trusted-port weak comparison')
    kit,out=replay.validate_paths(a.kit,a.out)
    manifest=(ROOT/replay.KIT_MANIFEST).read_bytes()
    original=replay.authenticate_kit(kit,manifest)
    blobs,pins=sources(a.study,a.revised)
    with tempfile.TemporaryDirectory(dir=out.parent,prefix='week09-r22-') as temp:
        base=Path(temp)
        replay.install_overlay(kit,base,manifest,blobs,pins)
        if not (base/'evaluate.py').exists():shutil.copyfile(kit/'evaluate.py',base/'evaluate.py')
        if replay.authenticate_kit(base,manifest)!=original:raise ValueError('overlay differs')
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='disable');env.pop('PYTHONPATH',None)
        if a.study=='official':data=official(base,env,a.revised)
        else:
            mod={'pairs':'experiments.r22_combined.known_pairs','timing':'experiments.r22_timing.study','weak':'experiments.r22_light.weak_check'}[a.study]
            flags=['--revised'] if a.revised and a.study in ('pairs','timing') else []
            subprocess.run([sys.executable,'-B','-m',mod,'--out','result.json',*flags],cwd=base,env=env,check=True)
            data=json.loads((base/'result.json').read_bytes())
        if (replay.authenticate_kit(kit,manifest)!=original or replay.authenticate_kit(base,manifest)!=original or
                any(replay.digest(ROOT/n)!=h or replay.digest(base/n)!=h for n,h in pins.items())):
            raise ValueError('kit or authored inputs changed during replay')
        doc={'schema':'r22-authenticated-study.v1','study':a.study,'is_stock_demo':False,'original_evaluator_sha256':replay.EVALUATOR_SHA256,
             'source_sha256':pins,'kit_manifest_sha256':replay.digest(manifest),'data':data}
        with out.open('x') as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
    print(a.study,'written')


if __name__=='__main__':main()
