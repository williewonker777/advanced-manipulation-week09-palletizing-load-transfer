"""Fresh eight-combination official matrix, preserving legacy v1 artifacts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from tools.check_integrity import check_integrity
from tools.compare_baseline import official_cells
from tools.v2_resources import wait_for_capacity,snapshot
from tools.v2_study import verify_freeze_sources,require_holdout_freeze


def main():
    check_integrity(ROOT)
    verify_freeze_sources(require_holdout_freeze())
    combinations={}
    env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1'
    outdir=ROOT/'results/v2_protocol'
    for mode in ('servo','torque'):
        for ctrl_label,ctrl in [('base',mode+'_controller'),('v2','my_'+mode)]:
            for det_label,det in [('base','collision_detector'),('v2','my_detector')]:
                key=f'{mode}_{ctrl_label}_{det_label}';host=wait_for_capacity()
                target=outdir/f'evaluation_{key}.txt'
                command=[sys.executable,'tools/evaluate_checked.py','--ctrl',ctrl,'--detector',det,'--out',str(target.relative_to(ROOT))]
                result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
                raw=Path(os.environ['V2_EVIDENCE'])/'official';raw.mkdir(parents=True,exist_ok=True)
                (raw/f'{key}.log').write_text(result.stdout+result.stderr);result.check_returncode()
                control,collision=official_cells(target.read_text())
                combinations[key]=dict(controller=ctrl,detector=det,official=True,
                    control={str(m):dict(rms_mm=float(v[0]),max_mm=float(v[1]),saturation_pct=float(v[2]),pick_nm=float(v[3]),place_nm=float(v[4]),table_n=float(v[5]),false_alarms=[int(x) for x in v[6:9]],status=v[9]) for m,v in control.items()},
                    collisions=[dict(mass=m,time=t,force10=v[0],force15=v[1],force40=v[2],force100=v[3],false_max=v[4],minimum=v[5]) for (m,t),v in collision.items()],
                    host_before=host,host_after=snapshot(),evaluation_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
                print(key,'complete',flush=True)
    check_integrity(ROOT);verify_freeze_sources(require_holdout_freeze())
    (outdir/'S4_OFFICIAL.json').write_text(json.dumps(dict(official=True,source='unchanged evaluate.py',combinations=combinations),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
