"""Reproduce the gated R19 unsigned-impulse comparison with the local unmodified course kit."""
from __future__ import annotations

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
from tools import r19_event_audit as setup
from tools.r15_early_check import validate_paths, validate_early_calibration

STOCK_SHA='f101b6cede7afdc11cc4093381e6b7c592b97d5f9f0ae73148113a0b4677aa5d'
MODULE='experiments/r19_impulse_gate/replay.py'
PLAN='experiments/r19_impulse_gate/plan.json'
PIN_ALGORITHM={'experiments/r19_impulse_end/detector.py':'7568d2e883ff150d2473a6f17bebb760ba425a4887122324d564d80819bbe909','experiments/r19_impulse_end/state.py':'e3211c06eff4b3974ed8afd3af66fd68c8da933851abe971e170e19fbbbfeb05',
               MODULE:'cfe1149f003b5f746e8a38db94c0c00009d88038bab8c877d510ebabba91e709',PLAN:'05bc5ef4ddb6077026e464651968cb4ba208c21bec336958c1c4d25f54849b7c'}
OWN=('tools/r19_impulse_audit.py','experiments/r19_impulse_end/__init__.py','experiments/r19_impulse_end/plan.json',
     'experiments/r19_impulse_gate/__init__.py',*PIN_ALGORITHM)


def verify(out, old, plan):
    stages=out['stages']
    if (out['schema']!='r19-impulse-gate-result.v1' or out['promotion_eligible'] is not False
            or out['published_reference_sha256']!=STOCK_SHA or not 1<=len(stages)<=3
            or list(stages)!=['known','normal','novel'][:len(stages)]):
        raise ValueError('invalid independent protocol')
    count=0
    primary=[]
    for i,(name,doc) in enumerate(stages.items()):
        rows=doc['rows']; ids=old if name=='known' else [r['id'] for r in plan['new']]
        if [r['id'] for r in rows]!=ids or len(set(ids))!=len(ids):
            raise ValueError('missing or duplicate conditions')
        for r in rows:
            n={'known':8,'normal':5,'novel':20}[name]; count+=n
            g=r['gates']
            if (not g or any(type(v) is not bool for v in g.values()) or r['passed'] is not all(g.values()) or r['episodes']!=n):
                raise ValueError('incomplete row')
            if name=='known' and not (g.get('published_R18_score_exact') and g.get('published_R18_character_exact')):
                raise ValueError('immutable comparator not reproduced')
            if name=='novel':
                cg=r['arms']['candidate']['r18_gates']
                if 'no_added_unmatched_rise' not in cg or 'no_programmed_normal_event_overlap' not in cg:
                    raise ValueError('missing original assessment')
                primary.append({'id':r['id'],'all_original_twin_gates':all(cg.values()),
                                'failed_original_gates':[k for k,v in cg.items() if v is not True],
                                'passed':r['passed'] and all(cg.values())})
        if (doc['passed'] is not all(r['passed'] for r in rows) or (not doc['passed'] and i+1<len(stages))):
            raise ValueError('stage improperly opened')
    if (len(stages)<3 and list(stages.values())[-1]['passed']
            or out['episodes']!=count or out['novel_forces_opened']!=(6 if 'novel' in stages else 0)
            or out['aggregate_success'] is not (len(stages)==3 and stages['novel']['passed'])):
        raise ValueError('inconsistent gate summary')
    return primary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit',type=Path,required=True);p.add_argument('--height-calibration',type=Path,required=True)
    p.add_argument('--v2-audit',type=Path,required=True,help='complete verified output of tools/r19_event_audit.py')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--jobs',type=int,choices=(1,2),default=1)
    a=p.parse_args(argv)
    try:
        kit,hpath,out=validate_paths(a.kit,a.height_calibration,a.out)
        if a.v2_audit.is_symlink() or not a.v2_audit.is_file():raise ValueError('regular V2 report required')
        stock=a.v2_audit.read_bytes()
        if setup.sha(stock)!=STOCK_SHA:raise ValueError('expected the complete known V2 report, not its summary')
        blobs,pins=setup.snapshots()
        for n in OWN:
            rel=ROOT/n
            if not rel.is_file() or rel.is_symlink():raise ValueError(n)
            blobs[n]=rel.read_bytes();pins[n]=setup.sha(blobs[n])
        if any(pins[n]!=v for n,v in PIN_ALGORITHM.items()):raise ValueError('changed frozen candidate or scorer')
        h_blob=hpath.read_bytes();cal=json.loads(blobs[setup.BASE_PLAN]);setup.r14.validate_plan(cal)
        _,height=setup.r14.validate_calibration(h_blob,cal,{k:setup.sha((ROOT/v).read_bytes()) for k,v in setup.r14.SOURCE_PATHS.items()})
        early=validate_early_calibration(blobs[setup.EARLY])
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 MUJOCO_GL='disable');env.pop('PYTHONPATH',None)
        with tempfile.TemporaryDirectory(prefix='r19-impulse-',dir=out.parent) as tmp:
            base=Path(tmp);setup.prepare_overlay(kit,base,'my','torque')
            for n,blob in blobs.items():
                dst=base/n;dst.parent.mkdir(parents=True,exist_ok=True);setup.assembly.install_snapshot(blob,pins[n],dst)
            (base/'v2.json').write_bytes(stock)
            for n,data in (('height',height),('early',early)):
                (base/(n+'.json')).write_text(json.dumps(data,allow_nan=False))
            close='experiments/r19_close_pairs/plan.json'
            cmd=[sys.executable,'-B','-m','experiments.r19_impulse_gate.replay','--plan',PLAN,'--stock','v2.json',
                 '--prior-plan',setup.r18.PAIR_PLAN,'--close-plan',close,'--height','height.json','--early','early.json',
                 '--model',setup.r18.prior.MODEL,'--out','stages','--jobs',str(a.jobs)]
            subprocess.run(cmd,cwd=base,env=env,check=True)
            report=json.loads((base/'stages/report.json').read_bytes());src=json.loads(stock)['result']['stages']
            old=[r['id'] for st in ('known','new') for r in src[st]['rows']]
            cross=verify(report,old,json.loads(blobs[PLAN]))
        if any(setup.sha((ROOT/n).read_bytes())!=v for n,v in pins.items()):raise ValueError('source changed during run')
        doc={'schema':'r19-impulse-portable-audit.v1','research_only':True,'original_demo':False,
             'pins':pins,'height_sha256':setup.sha(h_blob),'result':report,'all_original_gates_crosscheck':cross}
        with out.open('x') as fp:json.dump(doc,fp,indent=2,allow_nan=False);fp.write('\n')
    except (OSError,ValueError,KeyError,subprocess.CalledProcessError) as exc:
        p.error(str(exc))
    print(json.dumps({'episodes':report['episodes'],'new_conditions':len(cross),
                      'strictly_passed':sum(row['passed'] for row in cross),'promoted':False}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
