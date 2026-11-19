"""Run the opt-in R19 event-separation protocol against a verified external course kit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from tools import r13_grasp_audit as assembly
from tools import r14_registered_check as r14
from tools import r18_recovery_audit as r18
from tools.r15_early_check import validate_paths, validate_early_calibration
from tools.validate_submission import prepare_overlay

OWN = ('tools/r19_event_audit.py','experiments/r19_close_pairs/__init__.py','experiments/r19_close_pairs/assay.py',
       'experiments/r19_close_pairs/protocol.py','experiments/r19_close_pairs/plan.json',
       'experiments/r19_event_state/__init__.py','experiments/r19_event_state/state.py',
       'experiments/r19_event_state/physical_detector.py','experiments/r19_event_state/physical_plan.json')
EARLY = 'results/r15/early_stock_calibration.json'
BASE_PLAN = 'experiments/r14_stock_handoff/plan.json'
FROZEN_PLAN = 'f771623fd41aec91754a2e39e2b376be21d8c4344a202ac42d4a1ad164636c94'
STAGES = ('known','normal','new')
ARMS = {'r10','raw_r15','D_coupled','single_pass','event'}


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def checked_report(out, plan, previous):
    stages = out.get('stages',{})
    if (out.get('schema')!='r19-close-separation.v1' or out.get('promotion_eligible') is not False
            or not stages or list(stages)!=list(STAGES[:len(stages)]) or out.get('plan_sha256')!=FROZEN_PLAN):
        raise ValueError('incomplete or unordered staged evidence')
    expected = {k:[c['id'] for c in rows] for k,rows in (
        ('known',previous['fresh_paired_cases']+previous['diagnostic_sequences']),
        ('normal',plan['new']),('new',plan['new']))}
    total = 0
    for i,(stage,doc) in enumerate(stages.items()):
        rows = doc.get('rows',[])
        if [r['id'] for r in rows]!=expected[stage] or len(rows)!=6:
            raise ValueError('missing or repeated case')
        for r in rows:
            n = 5 if stage=='normal' else 20
            g = r['normal']['gates'] if stage=='normal' else r['gates']
            if (not g or any(type(v) is not bool for v in g.values()) or r['passed'] is not all(g.values())
                    or r['episodes']!=n or set(r['normal']['events'])!=ARMS):
                raise ValueError('contradictory or incomplete case result')
            if stage!='normal':
                if set(r['arms'])!=ARMS or set(r['assessment_signatures'])!=ARMS:
                    raise ValueError('missing comparator')
                for arm in r['arms'].values():
                    if (set(arm['levels'])!={'first','second','none','both'} or
                            set(arm['slots'])!={'isolated_first','isolated_second','both_first','both_second'}):
                        raise ValueError('missing twin or scoring context')
                if stage=='known' and not g.get('sealed_reference_all_arms'):
                    raise ValueError('sealed R18 controls not reproduced')
            total += n
        if doc['passed'] is not all(r['passed'] for r in rows) or (not doc['passed'] and i+1<len(stages)):
            raise ValueError('inconsistent or reopened stage')
    if (len(stages)<3 and list(stages.values())[-1]['passed']
            or out['new_forces_opened']!=(6 if 'new' in stages else 0)
            or out['all_stages_passed'] is not (len(stages)==3 and stages['new']['passed'])
            or out['episodes']!=total):
        raise ValueError('incomplete study summary')
    return out


def snapshots():
    blobs,pins = r18.prior.snapshots()
    for n in r18.ADDITIONS+r18.PAIR_FILES+OWN+(EARLY,BASE_PLAN):
        p=ROOT/n
        if not p.is_file() or p.is_symlink():
            raise ValueError('expected regular source '+n)
        blobs[n]=p.read_bytes(); pins[n]=sha(blobs[n])
    r18.verify_source(blobs)
    if pins['experiments/r19_close_pairs/plan.json']!=FROZEN_PLAN:
        raise ValueError('close-pair plan differs from the pre-exposure plan')
    return blobs,pins


def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--kit',type=Path,required=True);cli.add_argument('--height-calibration',type=Path,required=True)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--jobs',type=int,choices=(1,2,3),default=1)
    a=cli.parse_args(argv)
    try:
        kit,hpath,out=validate_paths(a.kit,a.height_calibration,a.out)
        blobs,pins=snapshots(); hblob=hpath.read_bytes()
        calibration_plan=json.loads(blobs[BASE_PLAN]);r14.validate_plan(calibration_plan)
        _,h=r14.validate_calibration(hblob,calibration_plan,{k:sha((ROOT/v).read_bytes()) for k,v in r14.SOURCE_PATHS.items()})
        e=validate_early_calibration(blobs[EARLY])
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                 MKL_NUM_THREADS='1',MUJOCO_GL='disable')
        env.pop('PYTHONPATH',None)
        with tempfile.TemporaryDirectory(prefix='r19-event-',dir=out.parent) as tmp:
            base=Path(tmp); prepare_overlay(kit,base,'my','torque')
            for n,blob in blobs.items():
                target=base/n;target.parent.mkdir(parents=True,exist_ok=True)
                assembly.install_snapshot(blob,pins[n],target)
            for name,record in (('height',h),('early',e),('pins',pins)):
                (base/(name+'.json')).write_text(json.dumps(record,allow_nan=False))
            cmd=[sys.executable,'-B','-m','experiments.r19_close_pairs.protocol','--out','stages','--jobs',str(a.jobs)]
            subprocess.run(cmd,cwd=base,env=env,check=True)
            result=checked_report(json.loads((base/'stages/report.json').read_bytes()),
                                  json.loads(blobs['experiments/r19_close_pairs/plan.json']),json.loads(blobs[r18.PAIR_PLAN]))
        if any(sha((ROOT/n).read_bytes())!=v for n,v in pins.items()):
            raise ValueError('source changed during replay')
        record={'schema':'r19-event-audit.v1','research_only':True,'original_demo':False,
                'height_sha256':sha(hblob),'pins':pins,'result':result}
        with out.open('x') as stream:
            json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n')
    except (OSError,ValueError,KeyError,subprocess.CalledProcessError) as exc:
        cli.error(str(exc))
    print(json.dumps({k:v for k,v in result.items() if k!='stages'}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
