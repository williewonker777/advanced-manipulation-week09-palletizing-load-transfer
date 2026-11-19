"""Evaluate R23 on authenticated, separate copies of a local course kit; never a stock demo."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools import r21_replay as shared, r22_census, r22_challenge_replay as r22
from tools.r20_official_compare import table, summarize

OWN=('__init__.py','candidate.py','free.py','state.py','plan.json','diagnose.py','exposed.py','d4.py','weak.py')
MODES={'target':('exposed','--panel','target'), 'previous':('exposed','--panel','full'),
       'd4':('d4','--cohort','full'),'six':('d4','--cohort','six'),'diagnostic':('diagnose',),'traces':('response',),'tool':('contact_case',),'weak':('weak',),
       'pairs':('extra','--kind','pairs'),'timing':('extra','--kind','timing')}


def sources(study=None):
    _,lock=r22.read_lock(ROOT/'results/r22/holdout_selected.lock.json')
    inherited,_=r22.snapshot(lock)
    weak,_=shared.snapshot('weak')
    extra=()
    if study=='tool':extra=('experiments/r23_detached/contact_case.py','experiments/r23_tool_contact/__init__.py','experiments/r23_tool_contact/candidate.py','experiments/r23_tool_contact/model.json')
    elif study=='traces':extra=('experiments/r23_detached/response.py',)
    elif study=='pairs':
        hist,_=shared.snapshot('history'); extra=tuple(hist)+('experiments/r22_combined/known_pairs.py','experiments/r23_detached/extra.py')
    elif study=='timing':
        extra=('experiments/r22_timing/__init__.py','experiments/r22_timing/study.py','experiments/r22_timing/plan.json','experiments/r23_detached/extra.py')
    names=tuple(dict.fromkeys(tuple(inherited)+tuple(weak)+extra+r22_census.dependencies(True)+
                              tuple('experiments/r23_detached/'+n for n in OWN)))
    blobs={}
    for n in names:
        f=ROOT/n
        if not f.is_file() or f.is_symlink() or n.startswith(('sim/','model/')) or n=='evaluate.py':
            raise ValueError('expected authored project dependency: '+n)
        blobs[n]=f.read_bytes()
    return blobs,{n:shared.digest(blob) for n,blob in blobs.items()}


def official(base,env):
    results={}
    for k,mod in (('light','experiments.r22_light.candidate'),('free','experiments.r23_detached.free'),
                  ('onset','experiments.r23_detached.candidate')):
        report=base/(k+'.txt')
        run=subprocess.run([sys.executable,'-B','evaluate.py','--ctrl','my_torque','--detector',mod,'--out',report.name],
                           cwd=base,env=env,check=True,capture_output=True,text=True)
        if sum(line.strip().startswith('실행 중: 박스') for line in run.stdout.splitlines())!=29:
            raise ValueError('incomplete original evaluator run')
        cells=table(report.read_text());results[k]={'cells':cells,'outcomes':summarize(cells)}
    return {'episodes':87,'results':results}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--study',choices=tuple(MODES)+('official',),required=True)
    p.add_argument('--workers',choices=(1,2,3),type=int,default=2);a=p.parse_args(argv)
    kit,out=shared.validate_paths(a.kit,a.out);manifest=(ROOT/shared.KIT_MANIFEST).read_bytes()
    original=shared.authenticate_kit(kit,manifest);blobs,pins=sources(a.study)
    with tempfile.TemporaryDirectory(prefix='r23-study-',dir=out.parent) as tmp:
        base=Path(tmp);shared.install_overlay(kit,base,manifest,blobs,pins)
        if shared.authenticate_kit(base,manifest)!=original:raise ValueError('overlay changed')
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 OMP_NUM_THREADS='1',MUJOCO_GL='disable')
        env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
        if a.study=='official':data=official(base,env)
        else:
            module,*flags=MODES[a.study]
            opts=[] if a.study in ('diagnostic','traces','tool') else ['--workers',str(a.workers)]
            subprocess.run([sys.executable,'-B','-m','experiments.r23_detached.'+module,*flags,*opts,'--out','data.json'],
                           cwd=base,env=env,check=True)
            data=json.loads((base/'data.json').read_bytes())
        if (shared.authenticate_kit(kit,manifest)!=original or shared.authenticate_kit(base,manifest)!=original or
                any(shared.digest(ROOT/n)!=v or shared.digest(base/n)!=v for n,v in pins.items())):
            raise ValueError('protected inputs changed during replay')
        doc={'schema':'r23-authenticated-offline-study.v1','study':a.study,'original_demo':False,'unseen':False,
             'original_evaluator_sha256':shared.EVALUATOR_SHA256,'protected_manifest_sha256':shared.digest(manifest),
             'source_sha256':pins,'data':data}
        with out.open('x') as fp:json.dump(doc,fp,indent=2,allow_nan=False);fp.write('\n')
    print(a.study,'written')


if __name__=='__main__':main()
