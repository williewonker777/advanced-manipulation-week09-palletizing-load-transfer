"""Run the V2 comparison on exactly the six impulse-end tuples, not a similar grid."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from tools import r19_impulse_audit as imp

# The previous portable computation is deterministic and includes every scored
# arm, the six matching controls and the authenticated original-kit inputs.
REFERENCE='05e4c56035a6a273474ea8e8325e0ee2e1a6b3c02e90cc8d84bf3b8891ee867f'
PROGRAM='experiments/r19_same_tuple/compare.py'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit',type=Path,required=True);p.add_argument('--height-calibration',type=Path,required=True)
    p.add_argument('--impulse-audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--jobs',type=int,choices=(1,2),default=1);a=p.parse_args(argv)
    try:
        kit,hpath,out=imp.validate_paths(a.kit,a.height_calibration,a.out)
        if not a.impulse_audit.is_file() or a.impulse_audit.is_symlink():raise ValueError('regular prior result required')
        data=a.impulse_audit.read_bytes()
        if imp.setup.sha(data)!=REFERENCE:raise ValueError('expected the complete verified impulse audit, not its summary')
        previous=json.loads(data);blobs,pins=imp.setup.snapshots()
        for n in (*imp.OWN,PROGRAM,'experiments/r19_same_tuple/__init__.py','tools/r19_same_tuple_audit.py'):
            f=ROOT/n
            if not f.is_file() or f.is_symlink():raise ValueError('expected regular source '+n)
            blobs[n]=f.read_bytes();pins[n]=imp.setup.sha(blobs[n])
        if any(pins.get(n)!=val for n,val in previous['pins'].items()):raise ValueError('previous source snapshot changed')
        hblob=hpath.read_bytes();plan=json.loads(blobs[imp.setup.BASE_PLAN]);imp.setup.r14.validate_plan(plan)
        _,height=imp.setup.r14.validate_calibration(hblob,plan,{k:imp.setup.sha((ROOT/v).read_bytes()) for k,v in imp.setup.r14.SOURCE_PATHS.items()})
        early=imp.validate_early_calibration(blobs[imp.setup.EARLY])
        with tempfile.TemporaryDirectory(prefix='r19-same-',dir=out.parent) as tmp:
            base=Path(tmp);imp.setup.prepare_overlay(kit,base,'my','torque')
            for n,blob in blobs.items():
                f=base/n;f.parent.mkdir(parents=True,exist_ok=True);imp.setup.assembly.install_snapshot(blob,pins[n],f)
            for name,value in (('height',height),('early',early)):
                (base/(name+'.json')).write_text(json.dumps(value,allow_nan=False))
            ref=base/'reference';ref.mkdir();(ref/'report.json').write_text(json.dumps(previous['result'],allow_nan=False))
            for stage in ('normal','novel'):
                dest=ref/stage;dest.mkdir()
                for row in previous['result']['stages'][stage]['rows']:
                    (dest/(row['id']+'.json')).write_text(json.dumps(row,allow_nan=False))
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     MUJOCO_GL='disable');env.pop('PYTHONPATH',None)
            subprocess.run([sys.executable,'-B','-m','experiments.r19_same_tuple.compare','--height','height.json','--early','early.json',
                            '--model',imp.setup.r18.prior.MODEL,'--plan',imp.PLAN,'--impulse-dir','reference','--out','comparison','--jobs',str(a.jobs)],
                           env=env,cwd=base,check=True)
            result=json.loads((base/'comparison/report.json').read_bytes());st=result['stages']
            if not (result['episodes']==48 and set(st)=={'normal','forced'} and all(v['passed'] and len(v['rows'])==6 and
                    all(row['passed'] and all(row['gates'].values()) for row in v['rows']) for v in st.values())):
                raise ValueError('control reproduction failed before outcomes could be interpreted')
        if any(imp.setup.sha((ROOT/n).read_bytes())!=v for n,v in pins.items()):raise ValueError('source changed during comparison')
        with out.open('x') as f:json.dump({'schema':'r19-same-tuple-audit.v1','reference_sha256':REFERENCE,'pins':pins,'result':result},f,indent=2,allow_nan=False);f.write('\n')
    except (ValueError,OSError,KeyError,subprocess.CalledProcessError) as err:p.error(str(err))
    print(json.dumps(result['strict_both_second']))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
