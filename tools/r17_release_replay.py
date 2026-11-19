"""Opt-in authenticated replay of one already-exposed R17 release sentinel."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import r13_grasp_audit as assembly
from tools import r14_registered_check as r14
from tools.r15_early_check import validate_early_calibration,validate_paths
MODEL='results/r17/release_observation_model.json'
DEPENDENCIES=(
 'tools/r17_release_replay.py','r1_torque.py','my_detector.py','experiments/r12_hybrid/controller.py',
 'experiments/r14_online/__init__.py','experiments/r14_online/stream.py','experiments/r14_airborne/registration.py','experiments/r14_airborne/plan.json',
 'experiments/r14_stock_handoff/__init__.py','experiments/r14_stock_handoff/controller.py','experiments/r14_stock_handoff/detector.py','experiments/r14_stock_handoff/geometry.py','experiments/r14_stock_handoff/qualification.py',
 'experiments/r15_early_com/__init__.py','experiments/r15_early_com/stream.py','experiments/r15_early_com/sensing.py','experiments/r15_early_com/controller.py',
 'experiments/r16_payload_contract/__init__.py','experiments/r16_payload_contract/controller.py','experiments/r16_payload_contract/plan.json',
 'experiments/r16_balanced/controller.py','experiments/r16_balanced/coefficients.json','experiments/r16_balanced/plan.json',
 'experiments/r16_release/__init__.py','experiments/r16_release/controller.py','experiments/r16_release/plan.json',
 'experiments/r16_normal_phase/__init__.py','experiments/r16_normal_phase/controller.py','experiments/r16_normal_phase/plan.json',
 'experiments/r17_release_prediction/__init__.py','experiments/r17_release_prediction/detector.py','experiments/r17_release_prediction/plan.json',
 'experiments/r17_release_coincidence/__init__.py','experiments/r17_release_coincidence/portable_detector.py','experiments/r17_release_coincidence/plan.json','experiments/r17_release_coincidence/replay.py',MODEL,
 'tools/r13_grasp_estimator.py','tools/r11_placement.py','tools/r14_paired_events.py','tools/r5_metrics.py','models/r12/counterfactual_coefficients.json','tools/kit_manifest.sha256')

def digest(blob):return hashlib.sha256(blob).hexdigest()

def snapshots():
    if len(DEPENDENCIES)!=len(set(DEPENDENCIES)) or any(Path(x).name=='run.py' for x in DEPENDENCIES):raise ValueError('unique neutral dependencies required')
    paths={x:ROOT/x for x in DEPENDENCIES}
    if any(not p.is_file() or p.is_symlink() for p in paths.values()):raise ValueError('regular frozen replay dependencies required')
    blobs={x:p.read_bytes() for x,p in paths.items()};return blobs,{x:digest(b) for x,b in blobs.items()}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kit',type=Path,required=True);p.add_argument('--height-calibration',type=Path,required=True);p.add_argument('--early-calibration',type=Path,default=ROOT/'results/r15/early_stock_calibration.json');p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    try:
        kit,hpath,out=validate_paths(a.kit,a.height_calibration,a.out);epath=a.early_calibration
        if epath.is_symlink() or not epath.is_file():raise ValueError('regular immutable early calibration required')
        epath=epath.resolve()
        hb,eb=hpath.read_bytes(),epath.read_bytes();plan=json.loads((ROOT/'experiments/r14_stock_handoff/plan.json').read_text());r14.validate_plan(plan)
        r14pins={k:digest((ROOT/v).read_bytes()) for k,v in r14.SOURCE_PATHS.items()};_,height=r14.validate_calibration(hb,plan,r14pins);early=validate_early_calibration(eb);blobs,pins=snapshots()
        frozen_plan=json.loads(blobs['experiments/r17_release_coincidence/plan.json'])
        if pins[MODEL] != frozen_plan['c_model_sha256']:raise ValueError('unexpected release model')
    except (OSError,ValueError,KeyError,json.JSONDecodeError) as error:p.error(str(error))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='disable');env.pop('PYTHONPATH',None)
    try:
        with tempfile.TemporaryDirectory(prefix='r17-release-replay-',dir=out.parent) as folder:
            overlay=Path(folder);from tools.validate_submission import prepare_overlay;prepare_overlay(kit,overlay,'my','torque')
            if assembly.digest(overlay/'tools/kit_manifest.sha256')!=pins['tools/kit_manifest.sha256']:raise ValueError('ORIGINAL kit manifest mismatch')
            for rel,blob in blobs.items():
                destination=overlay/rel;destination.parent.mkdir(parents=True,exist_ok=True);assembly.install_snapshot(blob,pins[rel],destination)
            # Explicit research model is opt-in and is never installed into a default submission path.
            model=overlay/'experiments/r17_release_prediction/model.json';assembly.install_snapshot(blobs[MODEL],pins[MODEL],model)
            (overlay/'height.json').write_text(json.dumps(height,allow_nan=False));(overlay/'early.json').write_text(json.dumps(early,allow_nan=False))
            command=[sys.executable,'-B','-m','experiments.r17_release_coincidence.replay','--height','height.json','--early','early.json','--model',str(model.relative_to(overlay)),'--out','replay.json']
            process=subprocess.run(command,cwd=overlay,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if process.returncode:raise RuntimeError('negative replay failed:\n'+process.stderr)
            result=json.loads((overlay/'replay.json').read_text())
            if (result.get('replay_valid') is not True or
                    result.get('status')!='NEGATIVE_CONTROL_REPRODUCED' or
                    result.get('candidate_passed') is not False or
                    result.get('promotion_eligible') is not False):
                raise ValueError('complete valid negative replay disposition not reproduced')
        if any(digest((ROOT/rel).read_bytes())!=pin for rel,pin in pins.items()):raise ValueError('source mutated during replay')
        document={'schema':'r17-portable-release-negative-replay.v1','scope':'opt-in single already-exposed sentinel; never promotion evidence','pins':pins,'calibration_provenance':{'height_sha256':digest(hb),'early_sha256':digest(eb),'release_model_sha256':pins[MODEL],'release_model_default_installed':False},'runner_stdout':process.stdout.strip(),'result':result,'limitations':['negative control only','already exposed sentinel','no novel positive','no model fitting','no hard-coded CPU affinity','no promotion']}
        with out.open('x') as stream:json.dump(document,stream,indent=2,allow_nan=False);stream.write('\n')
    except (OSError,ValueError,RuntimeError,json.JSONDecodeError) as error:p.error(str(error))
    print(process.stdout.strip());return 0
if __name__=='__main__':sys.dont_write_bytecode=True;raise SystemExit(main())
