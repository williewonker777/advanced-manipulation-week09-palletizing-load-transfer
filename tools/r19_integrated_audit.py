"""Evaluate physical-floor R19 on the staged control policy, without installing a candidate."""
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

from experiments.r19_combined_gate import prepare_overlay as assembly
from tools import r18_recovery_audit as r18
from tools.r15_early_check import validate_paths


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit',type=Path,required=True)
    p.add_argument('--height-calibration',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        kit,height,out=validate_paths(a.kit,a.height_calibration,a.out)
        with tempfile.TemporaryDirectory(prefix='r19-integrated-',dir=out.parent) as tmp:
            base=Path(tmp); overlay=base/'overlay';auth_path=base/'auth.json'
            assembly.main(['--kit',str(kit),'--height',str(height),'--early',str(ROOT/'results/r15/early_stock_calibration.json'),
                           '--overlay',str(overlay),'--auth',str(auth_path)])
            auth=json.loads(auth_path.read_bytes());pins=auth['source_pins']
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     MUJOCO_GL='disable');env.pop('PYTHONPATH',None)
            subprocess.run([sys.executable,'-B','-m','experiments.r19_combined_gate.study',
                            '--height','height.json','--early','early.json','--pins','run_pins.json','--out','stages'],
                           cwd=overlay,env=env,check=True)
            result=r18.load_progress(overlay/'stages')
            if any(r18.sha((ROOT/n).read_bytes())!=v for n,v in pins.items()):
                raise ValueError('source changed during replay')
        with out.open('x') as stream:
            json.dump({'schema':'r19-integrated-regression.v1','fresh_independent_evidence':False,
                       'original_demo':False,'auth':auth,'result':result},stream,indent=2,allow_nan=False);stream.write('\n')
    except (OSError,ValueError,KeyError,subprocess.CalledProcessError) as exc:
        p.error(str(exc))
    print(json.dumps(result['summary']['stages']))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
