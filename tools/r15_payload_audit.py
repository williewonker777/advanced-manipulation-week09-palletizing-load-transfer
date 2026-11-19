"""Portable ORIGINAL-stock-kit audit of the R15 horizontal payload-COM sensor.

The CLI runs 12 normal calibration and 20 evaluation episodes in an immutable
temporary overlay.  All public-signal predictions are sealed before retained
free-body scoring capsules are opened.  Only derived JSON is emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

try:
    from tools import r13_grasp_audit as base
except ImportError:
    import r13_grasp_audit as base


ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('none','explicit_zero','xy','up','down')
SOURCE_PATHS={
    'torque':'r1_torque.py','detector':'my_detector.py',
    'estimator':'tools/r13_grasp_estimator.py',
    'registration':'experiments/r14_airborne/registration.py',
    'r14_plan':'experiments/r14_airborne/plan.json',
    'sensing':'experiments/r15_com_model/sensing.py',
    'r15_plan':'experiments/r15_com_model/plan.json',
    'base_audit':'tools/r13_grasp_audit.py',
    'kit_manifest':'tools/kit_manifest.sha256',
}


def sha(blob):return hashlib.sha256(blob).hexdigest()


def validate_plans(r14,r15):
    if r14.get('schema')!='r14-airborne.v1' or len(r14.get('fit',()))!=12:
        raise ValueError('frozen twelve-case R14 normal plan required')
    if (r15.get('schema')!='r15-effective-com.v1' or len(r15.get('test',()))!=4
            or tuple(r15.get('variants',()))!=VARIANTS):
        raise ValueError('frozen R15 stock COM plan required')
    ids=[]
    for case in [*r14['fit'],*r15['test']]:
        ids.append(case.get('id'));mass=float(case['mass']);com=np.asarray(case['com'],float)
        if mass<=0 or not np.isfinite(mass) or com.shape!=(3,) or not np.isfinite(com).all():
            raise ValueError('invalid finite payload specification')
    if len(ids)!=len(set(ids)):
        raise ValueError('duplicate calibration/evaluation id')
    for case in r15['test']:
        shift=np.asarray(case['shift'],float)
        if shift.shape!=(3,) or not np.isfinite(shift).all():
            raise ValueError('finite 3-D authored pickup shift required')


def offset(case,variant):
    if variant=='none':return None
    if variant=='explicit_zero':return (0.,0.,0.)
    x,y,z=map(float,case['shift'])
    return (x,y,{'xy':0.,'up':z,'down':-z}[variant])


def physics_digest(log):
    return {key:hashlib.sha256(np.asarray(log[key]).tobytes()).hexdigest()
            for key in ('q','tau','f_tool','f_table','r','detect')}


def measures(rows):
    norms=lambda key:np.array([np.linalg.norm(row[key]) for row in rows])
    truth=norms('true_xy_mm');error=norms('point_error_mm');action=norms('action_error_mm')
    tiny=truth<1.;large=truth>=3.
    misses=sum(row['accept'] and not row['covered'] for row in rows)
    large_good=sum(bool(big and row['accept'] and
                        np.dot(row['true_xy_mm'],row['model_action_xy_mm'])>0)
                   for big,row in zip(large,rows))
    return {'count':len(rows),'vector_point_rmse_mm':float(np.sqrt(np.mean(error**2))),
            'max_point_error_mm':float(error.max()),'max_action_error_mm':float(action.max()),
            'zero_action_rmse_mm':float(np.sqrt(np.mean(truth**2))),
            'action_rmse_mm':float(np.sqrt(np.mean(action**2))),
            'accepted':int(sum(row['accept'] for row in rows)),
            'abstained':int(sum(not row['accept'] for row in rows)),
            'accepted_interval_misses':int(misses),'tiny_n':int(tiny.sum()),
            'false_tiny_actions':int(sum(bool(flag and row['accept']) for flag,row in zip(tiny,rows))),
            'big_n':int(large.sum()),'big_signed':int(large_good),
            'qualified':bool(float(np.sqrt(np.mean(error**2)))<=1.5 and action.max()<=2.
                             and misses==0 and not any(flag and row['accept'] for flag,row in zip(tiny,rows))
                             and large_good==int(large.sum()))}


def run_stock(r14_plan,r15_plan):
    """Run inside the authenticated overlay; no research fixture is imported."""
    from sim import config
    from sim.runner import run_episode
    import collision_detector
    from experiments.r15_com_model import sensing
    from tools import r13_grasp_estimator as old

    validate_plans(r14_plan,r15_plan)
    controller=importlib.import_module('torque_controller').Controller
    if getattr(controller,'MODE',None)!='torque':raise ValueError('R1 torque controller required')
    fit=[];fit_runs=[]
    for case in r14_plan['fit']:
        obj,samples=base.capture(controller)
        metrics,_=run_episode(obj,float(case['mass']),tuple(case['com']),collision_detector.Detector())
        if metrics['diverged'] or len(samples['t']) not in (3550,3551):
            raise ValueError('incomplete calibration episode')
        fit.append(base.public_features(old,samples,case))
        fit_runs.append({'id':case['id'],'mass':case['mass'],'com':case['com'],
                         'run':base.compact_metrics(metrics)})
    model=sensing.calibrate(fit,config.JTS_NOISE)

    rows=[];capsules=[]
    for case in r15_plan['test']:
        for variant in VARIANTS:
            obj,samples=base.capture(controller)
            metrics,log=run_episode(obj,float(case['mass']),tuple(case['com']),
                                    collision_detector.Detector(),pick_offset=offset(case,variant))
            if metrics['diverged'] or len(samples['t']) not in (3550,3551):
                raise ValueError('incomplete evaluation episode')
            feature=base.public_features(old,samples,case)
            prediction=sensing.decide(feature,model)
            rows.append({'id':case['id'],'variant':variant,**prediction,
                         'run':base.compact_metrics(metrics)})
            capsules.append({'geometry':base.score_capsule(log),'com':np.asarray(case['com']),
                             'digests':physics_digest(log),
                             'tool_peak_n':float(np.max(log['f_tool']))})

    # Hash every decision before the first free-body scoring function is called.
    prediction_blob=json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    prediction_seal=sha(prediction_blob)
    truths=[base.observed_geometry(item['geometry'],item['com']) for item in capsules]
    baselines={};explicit=[]
    for row,actual,capsule in zip(rows,truths,capsules,strict=True):
        identifier,variant=row['id'],row['variant'];com=np.asarray(actual['local_com_shift_mm'])[:2]
        if variant=='none':
            baselines[identifier]={'com':com,'digests':capsule['digests'],
                                   'tool_peak_n':capsule['tool_peak_n']}
        truth=com-baselines[identifier]['com'];point=np.asarray(row['relative_xy_mm'])-truth
        action=np.asarray(row['model_action_xy_mm'])-truth
        row.update(true_xy_mm=truth.tolist(),point_error_mm=point.tolist(),
                   action_error_mm=action.tolist(),covered=bool(np.linalg.norm(point)<=row['halfwidth_radius_mm']),
                   true_absolute_flange_com_xy_mm=com.tolist())
        if variant=='explicit_zero':
            initial=baselines[identifier]
            explicit.append({'id':identifier,'none_physics_digests':initial['digests'],
                'explicit_zero_physics_digests':capsule['digests'],
                'physics_digest_match':{key:initial['digests'][key]==value
                                        for key,value in capsule['digests'].items()},
                'none_tool_peak_n':initial['tool_peak_n'],
                'explicit_zero_tool_peak_n':capsule['tool_peak_n'],
                'true_xy_change_mm':truth.tolist()})
    return {'calibration':model,'calibration_runs':fit_runs,'rows':rows,
            'metrics':measures(rows),'explicit_zero_audit':explicit,
            'prediction_seal_sha256':prediction_seal,
            'predictions_sealed_before_truth':True}


def source_inputs():
    paths={key:ROOT/relative for key,relative in SOURCE_PATHS.items()}
    paths['audit']=Path(__file__)
    if not all(path.is_file() and not path.is_symlink() for path in paths.values()):
        raise ValueError('regular frozen public payload audit inputs required')
    return base.snapshot_files(paths)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit',type=Path,required=True,help='licensed ORIGINAL stock kit')
    parser.add_argument('--out',type=Path,required=True,help='NEW output directory')
    args=parser.parse_args(argv)
    kit,out=base.validate_paths(args.kit,args.out,parser)
    blobs,pins=source_inputs();r14=json.loads(blobs['r14_plan']);r15=json.loads(blobs['r15_plan'])
    validate_plans(r14,r15)

    sys.path.insert(0,str(ROOT))
    from tools.validate_submission import prepare_overlay
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',
                     OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    environment.pop('PYTHONPATH',None)
    with tempfile.TemporaryDirectory(prefix='r15-payload-audit-',dir=out.parent) as folder:
        overlay=Path(folder);prepare_overlay(kit,overlay,'my','torque')
        if base.digest(overlay/'tools/kit_manifest.sha256')!=pins['kit_manifest']:
            raise ValueError('kit manifest differs from frozen audit input')
        destinations={'torque':'torque_controller.py','detector':'collision_detector.py',
            'estimator':'tools/r13_grasp_estimator.py',
            'registration':'experiments/r14_airborne/registration.py',
            'r14_plan':'r14_plan.json','sensing':'experiments/r15_com_model/sensing.py',
            'r15_plan':'r15_plan.json','base_audit':'r13_grasp_audit.py','audit':'payload_audit.py'}
        for relative in destinations.values():(overlay/relative).parent.mkdir(parents=True,exist_ok=True)
        for package in ('tools','experiments','experiments/r14_airborne','experiments/r15_com_model'):
            path=overlay/package;path.mkdir(parents=True,exist_ok=True);(path/'__init__.py').write_text('')
        for key,relative in destinations.items():base.install_snapshot(blobs[key],pins[key],overlay/relative)
        # The isolated base helper uses its public fallback import path.
        base.install_snapshot(blobs['base_audit'],pins['base_audit'],overlay/'r13_grasp_audit.py')
        code=('import json,payload_audit;from pathlib import Path;'
              'json.dump(payload_audit.run_stock(json.loads(Path("r14_plan.json").read_text()),'
              'json.loads(Path("r15_plan.json").read_text())),sys.stdout,allow_nan=False)')
        process=subprocess.run([sys.executable,'-B','-c','import sys;'+code],cwd=overlay,env=environment,
                               text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if process.returncode:raise RuntimeError('stock payload audit failed:\n'+process.stderr)
        result=json.loads(process.stdout)

    scope={'mode':'torque','episodes':32,'calibration_episodes':12,'evaluation_episodes':20,
           'runner':'authenticated original stock sim.runner.run_episode',
           'inference_deadline_s':3.55,'predictions_sealed_before_truth':True,
           'retained_data':'derived JSON only; no raw sensor captures or kit copies',
           'research_perturbations_applied':False,'collision_episodes':0}
    calibration={'schema':'r15-payload-calibration-audit.v1','pins':pins,'scope':scope,
                 'model':result['calibration'],'runs':result['calibration_runs']}
    evaluation={'schema':'r15-payload-evaluation-audit.v1','pins':pins,'scope':scope,
                'prediction_seal_sha256':result['prediction_seal_sha256'],'rows':result['rows'],
                'explicit_zero_audit':result['explicit_zero_audit']}
    summary={'schema':'r15-payload-audit.v1','pins':pins,'scope':scope,
             'prediction_seal_sha256':result['prediction_seal_sha256'],
             'metrics':result['metrics'],'accepted':result['metrics']['accepted'],
             'abstained':result['metrics']['abstained'],
             'decision':'stock sensing qualification passed' if result['metrics']['qualified']
                        else 'stock sensing qualification failed'}
    out.mkdir()
    for name,document in (('calibration.json',calibration),('evaluation.json',evaluation),
                          ('summary.json',summary)):
        with (out/name).open('x') as stream:
            json.dump(document,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'episodes':32,'qualified':summary['metrics']['qualified'],
                      'vector_point_rmse_mm':summary['metrics']['vector_point_rmse_mm'],
                      'accepted':summary['accepted'],'abstained':summary['abstained']}))


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
