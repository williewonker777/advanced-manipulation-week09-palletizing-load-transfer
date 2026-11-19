"""Replay single- or double-impulse research outcomes on an authenticated external stock kit."""
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
    sys.path.insert(0, str(ROOT))

from tools import r13_grasp_audit as assembly
from tools import r14_registered_check as r14
from tools import r17_release_replay as prior
from tools.r15_early_check import validate_early_calibration, validate_paths
from tools.validate_submission import prepare_overlay

ADDITIONS = ('tools/r18_recovery_audit.py', 'experiments/r18_single_pass/__init__.py',
             'experiments/r18_single_pass/detector.py', 'experiments/r18_single_pass/recovery.py',
             'experiments/r18_single_pass/reference_recovery.py','experiments/r18_single_pass/reproduce.py')
PAIR_PLAN = 'experiments/r18_paired_impulses/plan.json'
PAIR_REF = 'results/r18/paired_reference.json'
PAIR_FILES = (PAIR_PLAN, PAIR_REF, 'experiments/r18_single_pass/paired_replay.py',
              'experiments/r18_paired_impulses/__init__.py','experiments/r18_paired_impulses/runner.py',
              'experiments/r18_paired_impulses/scoring.py')
PROGRESSIVE_FILES = (
    'experiments/r18_combined_gate/__init__.py','experiments/r18_combined_gate/policy.py',
    'experiments/r18_combined_gate/plan.json','experiments/r18_combined_gate/study.py',
    'experiments/r17_qualification/__init__.py','experiments/r17_qualification/qualification.py','experiments/r17_qualification/plan.json',
    'experiments/r17_candidate/__init__.py','experiments/r17_candidate/policy.py','experiments/r17_candidate/domain.py',
    'experiments/r17_candidate/timing.py','experiments/r17_candidate/observer.py',
    'experiments/r17_realtime/controller.py','experiments/r17_realtime/contract.py',
    'experiments/r17_staged_realtime/controller.py','experiments/r17_staged_realtime/contract.py','experiments/r17_staged_realtime/staged.py')
STAGE_COUNTS = (('known-normal',37),('exposed-impact',7),('fresh-normal',6),('fresh-impact',8))
FROZEN_R18 = '92d7df88382f40d2f839a8d7f635db245ed21675a1c7514e9e9485fb0177e76c'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def verify_source(blobs):
    key = 'experiments/r18_single_pass/reference_recovery.py'
    bound = b'from experiments.r17_release_coincidence.portable_detector import'
    if blobs[key].count(bound) != 1 or sha(blobs[key].replace(bound, bound.replace(b'.portable_detector ', b'.detector '))) != FROZEN_R18:
        raise ValueError('portable and sealed recovery algorithms differ')
    r17 = blobs['experiments/r17_release_coincidence/portable_detector.py']
    for annotated, original in ((b't: float, th,', b't, th,'), (b'grip) -> tuple[np.ndarray, bool]:', b'grip):')):
        if r17.count(annotated) != 1:
            raise ValueError('unexpected R17 public interface')
        r17 = r17.replace(annotated, original)
    if sha(r17) != 'f7d94c84943a6c4ac95337671cc4235c169461888ddfb638d6698fdd132f9d6c':
        raise ValueError('frozen coincidence source mismatch')
    plan = json.loads(blobs['experiments/r17_release_coincidence/plan.json'])
    if sha(blobs[prior.MODEL]) != plan['c_model_sha256']:
        raise ValueError('frozen release model mismatch')


def valid_report(out):
    gates = {'genuine_reference_signature', 'all_prefixes_and_realized_forces',
             'all_D_physics_identical', 'fast_exact_residual_alarm_and_state', 'preserve_tool'}
    values = out.get('gates', {})
    return (out.get('schema') == 'r18-known-sentinel-live.v1' and out.get('reproduced') is True
            and out.get('promotion_eligible') is False and set(values)==gates
            and all(x is True for x in values.values()))


def load_progress(folder):
    summary=json.loads((folder/'summary.json').read_bytes())
    stages=summary.get('stages',[])
    if (summary.get('schema')!='r18-integrated-outcome.v1' or summary.get('qualified_for_deployment') is not False
            or not 1<=len(stages)<=len(STAGE_COUNTS)):
        raise ValueError('missing progressive stage evidence')
    details={}
    for idx,stage in enumerate(stages):
        name,count=STAGE_COUNTS[idx]
        if stage.get('stage')!=name or stage.get('count')!=count:
            raise ValueError('stage order or size mismatch')
        doc=json.loads((folder/(name+'.json')).read_bytes()); rows=doc.get('rows',[])
        if doc.get('stage')!=name or len(rows)!=count or len({row['id'] for row in rows})!=count:
            raise ValueError('duplicate or missing stage cases')
        row_outcomes=[row['passed'] for row in rows]
        if any(type(v) is not bool for v in row_outcomes):
            raise ValueError('untyped outcomes')
        if any(row['passed']!=(all(row['gates'].values()) and all(row['extra_gates'].values())) for row in rows):
            raise ValueError('inconsistent case gates')
        extra=doc['extra_evidence']
        if (stage['passed_count']!=sum(row_outcomes) or doc['passed']!=(all(row_outcomes) and all(z['passed'] for z in extra))
                or stage['passed'] is not doc['passed'] or (not stage['passed'] and idx!=len(stages)-1)
                or (name=='fresh-impact' and (len(extra)!=1 or 'candidate_positive_nonvacuous' not in extra[0]['gates']))):
            raise ValueError('inconsistent stage gates')
        details[name]=doc
    if len(stages)<len(STAGE_COUNTS) and stages[-1]['passed']:
        raise ValueError('incomplete passing run')
    return {'summary':summary,'details':details,'all_stages_passed':len(stages)==4 and stages[-1]['passed'],
            'promotion_eligible':False,'scope':'local machine measurement; a failed stage stops the next one'}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--kit', type=Path, required=True)
    cli.add_argument('--height-calibration', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--study', choices=('sentinel','pairs','progressive'), default='sentinel')
    a = cli.parse_args(argv)
    try:
        kit, height_path, out = validate_paths(a.kit,a.height_calibration,a.out)
        h_blob=height_path.read_bytes()
        early_path = ROOT/'results/r15/early_stock_calibration.json'
        e_blob=early_path.read_bytes()
        plan=json.loads((ROOT/'experiments/r14_stock_handoff/plan.json').read_bytes())
        r14.validate_plan(plan)
        _,height=r14.validate_calibration(h_blob,plan,{k:sha((ROOT/v).read_bytes()) for k,v in r14.SOURCE_PATHS.items()})
        early=validate_early_calibration(e_blob)
        blobs,pins=prior.snapshots()
        for rel in ADDITIONS+{'pairs':PAIR_FILES,'progressive':PROGRESSIVE_FILES,'sentinel':()}[a.study]:
            path=ROOT/rel
            if not path.is_file() or path.is_symlink():raise ValueError('expected regular source file: '+rel)
            blobs[rel]=path.read_bytes();pins[rel]=sha(blobs[rel])
        verify_source(blobs)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                 MKL_NUM_THREADS='1',MUJOCO_GL='disable')
        env.pop('PYTHONPATH',None)
        with tempfile.TemporaryDirectory(prefix='r18-known-',dir=out.parent) as tmp:
            overlay=Path(tmp)
            prepare_overlay(kit,overlay,'my','torque')
            for rel,blob in blobs.items():
                dest=overlay/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                assembly.install_snapshot(blob,pins[rel],dest)
            for key,data in (('height',height),('early',early)):
                (overlay/(key+'.json')).write_text(json.dumps(data,allow_nan=False))
            args=[sys.executable,'-B','-m','experiments.r18_single_pass.reproduce',
                  '--height','height.json','--early','early.json','--model',prior.MODEL,'--out','study.json']
            p=subprocess.run(args,cwd=overlay,env=env,capture_output=True,text=True)
            if p.returncode:raise ValueError('stock replay did not qualify: '+(p.stderr or p.stdout).strip())
            result=json.loads((overlay/'study.json').read_bytes())
            if not valid_report(result):raise ValueError('invalid or incomplete study')
            if a.study=='pairs':
                args=[sys.executable,'-B','-m','experiments.r18_single_pass.paired_replay',
                      '--height','height.json','--early','early.json','--model',prior.MODEL,
                      '--plan',PAIR_PLAN,'--reference',PAIR_REF,'--out','pairs.json']
                p=subprocess.run(args,cwd=overlay,env=env,capture_output=True,text=True)
                if p.returncode:raise ValueError('two-pulse reproduction failed: '+(p.stderr or p.stdout).strip())
                pairs=json.loads((overlay/'pairs.json').read_bytes())
                if (pairs.get('schema')!='r18-known-four-twin-replay.v1' or pairs.get('reproduced') is not True
                        or pairs.get('promotion_eligible') is not False or pairs.get('episodes')!=120
                        or len(pairs.get('rows',[]))!=6
                        or not all(all(row.get(k) is True for k in ('reference_matched','fast_exact_all_twins','same_D_physics'))
                                   for row in pairs['rows'])):raise ValueError('incomplete four-twin reproduction')
            if a.study=='progressive':
                (overlay/'run_pins.json').write_text(json.dumps(pins,allow_nan=False))
                args=[sys.executable,'-B','-m','experiments.r18_combined_gate.study',
                      '--height','height.json','--early','early.json','--pins','run_pins.json','--out','progress']
                p=subprocess.run(args,cwd=overlay,env=env,capture_output=True,text=True)
                if p.returncode:raise ValueError('progressive protocol execution failed: '+(p.stderr or p.stdout).strip())
                progression=load_progress(overlay/'progress')
        if any(sha((ROOT/rel).read_bytes())!=pin for rel,pin in pins.items()):raise ValueError('source changed during replay')
        doc={'schema':'r18-portable-known-recovery.v1','pins':pins,'height_sha256':sha(h_blob),
             'early_sha256':sha(e_blob),'frozen_recovery_source_sha256':FROZEN_R18,
             'result':result,'scope':'six arms on the same previously exposed single pulse; not deployment qualification'}
        if a.study=='pairs':
            doc['paired']=pairs
            doc['scope']='known sentinel and reproduction of the complete six already reported four-twin cases; no promotion'
        if a.study=='progressive':
            doc['progressive']=progression
            doc['scope']='known sentinel followed by the predeclared local integrated staged qualification; timing depends on the computer'
        with out.open('x') as fp:json.dump(doc,fp,indent=2,allow_nan=False);fp.write('\n')
    except (KeyError,OSError,ValueError,RuntimeError) as exc:
        cli.error(str(exc))
    print(p.stdout.strip())
    return 0


if __name__=='__main__':
    raise SystemExit(main())
