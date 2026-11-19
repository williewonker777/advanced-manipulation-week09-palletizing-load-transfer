"""Portable ORIGINAL-stock reproduction of the R17 2.943 s information-prefix observer.

The shipped pooled calibration was fitted from 24 prior normal public records
across stock and perturbed domains.  Stock simulation reproduces the 15 stock
observation rows but cannot regenerate the perturbed half of that calibration.
Predictions are completed before score-only free-box capsules are inspected;
no raw sensor record or kit source is emitted.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import subprocess,sys,tempfile
try:
    from tools import r13_grasp_audit as base
except ImportError:
    import r13_grasp_audit as base
ROOT=Path(__file__).resolve().parents[1]
SOURCE_PATHS={
 'torque':'r1_torque.py','detector':'my_detector.py','base_audit':'tools/r13_grasp_audit.py',
 'estimator':'tools/r13_grasp_estimator.py','online_stream':'experiments/r14_online/stream.py',
 'online_init':'experiments/r14_online/__init__.py','robust':'experiments/r16_robust_prefix/robust.py',
 'information_stream':'experiments/r17_information_prefix/stream.py',
 'estimate':'experiments/r17_information_prefix/estimate.py','portable':'experiments/r17_information_prefix/portable.py',
 'plan':'experiments/r17_early_interpolation/plan.json','calibration':'experiments/r17_information_prefix/portable_calibration.json',
 'reference':'experiments/r17_information_prefix/portable_stock_reference.json','kit_manifest':'tools/kit_manifest.sha256'}

def sha(blob): return hashlib.sha256(blob).hexdigest()

def inputs():
    paths={key:ROOT/value for key,value in SOURCE_PATHS.items()}; paths['audit']=Path(__file__)
    if not all(path.is_file() and not path.is_symlink() for path in paths.values()): raise ValueError('regular frozen public inputs required')
    return base.snapshot_files(paths)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--kit',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(argv)
    kit,out=base.validate_paths(a.kit,a.out,p)
    if out.is_relative_to(ROOT.resolve()): p.error('NEW output must be outside source and kit')
    blobs,pins=inputs(); plan=json.loads(blobs['plan']); calibration=json.loads(blobs['calibration']); reference=json.loads(blobs['reference'])
    sys.path.insert(0,str(ROOT)); from experiments.r17_information_prefix.portable import validate; validate(calibration,reference,plan)
    from tools.validate_submission import prepare_overlay
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='disable'); env.pop('PYTHONPATH',None)
    with tempfile.TemporaryDirectory(prefix='r17-payload-audit-',dir=out.parent) as folder:
        overlay=Path(folder); prepare_overlay(kit,overlay,'my','torque')
        if base.digest(overlay/'tools/kit_manifest.sha256')!=pins['kit_manifest']: raise ValueError('kit manifest differs from frozen audit input')
        destinations={'torque':'torque_controller.py','detector':'collision_detector.py','base_audit':'r13_grasp_audit.py',
          'estimator':'tools/r13_grasp_estimator.py','online_stream':'experiments/r14_online/stream.py','online_init':'experiments/r14_online/__init__.py',
          'robust':'experiments/r16_robust_prefix/robust.py',
          'information_stream':'experiments/r17_information_prefix/stream.py','estimate':'experiments/r17_information_prefix/estimate.py',
          'portable':'experiments/r17_information_prefix/portable.py','plan':'plan.json','calibration':'calibration.json','reference':'reference.json','audit':'payload_audit.py'}
        for package in ('tools','experiments','experiments/r14_online','experiments/r16_robust_prefix','experiments/r17_information_prefix'):
            path=overlay/package; path.mkdir(parents=True,exist_ok=True); (path/'__init__.py').write_text('')
        for key,relative in destinations.items():
            target=overlay/relative; target.parent.mkdir(parents=True,exist_ok=True); base.install_snapshot(blobs[key],pins[key],target)
        code=('import json,payload_audit,sys;from pathlib import Path;'
              'from experiments.r17_information_prefix.portable import run_stock;'
              'json.dump(run_stock(json.loads(Path("calibration.json").read_text()),json.loads(Path("reference.json").read_text()),json.loads(Path("plan.json").read_text())),sys.stdout,allow_nan=False)')
        process=subprocess.run([sys.executable,'-B','-c',code],cwd=overlay,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if process.returncode: raise RuntimeError('stock observation audit failed:\n'+process.stderr)
        result=json.loads(process.stdout)
    if any(sha((ROOT/path).read_bytes())!=pins[key] for key,path in SOURCE_PATHS.items()): raise ValueError('source mutated during execution')
    scope={'mode':'torque','episodes':15,'calibration_episodes':0,'evaluation_episodes':15,'runner':'authenticated ORIGINAL stock sim.runner.run_episode','decision_s':2.943,'raw_sensor_captures_emitted':False,'collision_episodes':0,'control_claim':False}
    evaluation={'schema':'r17-portable-information-prefix-evaluation.v1','pins':pins,'scope':scope,'prediction_seal_sha256':result['prediction_seal_sha256'],'rows':result['rows'],'research_row_parity':result['research_row_parity'],'research_metric_parity':result['research_metric_parity']}
    summary={'schema':'r17-portable-information-prefix-audit.v1','pins':pins,'scope':scope,'calibration_provenance':calibration['portable_provenance'],'metrics':result['metrics'],'research_row_parity':result['research_row_parity'],'research_metric_parity':result['research_metric_parity'],'prediction_seal_sha256':result['prediction_seal_sha256'],'decision':'stock observation reproduction passed' if result['metrics']['qualified'] else 'stock observation reproduction failed','limitations':['exposed stock reproduction, not new blind evidence','stock cannot regenerate pooled perturbed calibration','observation accuracy does not establish control or impact performance','controller timing is outside this sensing audit']}
    out.mkdir()
    for name,doc in (('evaluation.json',evaluation),('summary.json',summary)):
        with (out/name).open('x') as f: json.dump(doc,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'episodes':15,'decision_s':2.943,'qualified':result['metrics']['qualified'],'vector_rmse_mm':result['metrics']['vector_rmse_mm'],'accepted':result['metrics']['accepted'],'big_signed':result['metrics']['big_signed'],'research_row_parity':result['research_row_parity'],'research_metric_parity':result['research_metric_parity']},sort_keys=True))
    return 0
if __name__=='__main__': sys.dont_write_bytecode=True; raise SystemExit(main())
