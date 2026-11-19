"""Run the fixed D4 panel from an original licensed kit with three separately simulated observer arms."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools import r21_d4_check, r21_replay as shared
from tools.r22_studies import RUNTIME, REVISED

EXTRA=('experiments/r22_census/__init__.py','experiments/r22_census/check.py','experiments/r22_census/plan.json',
       'experiments/r21_yaw_continuity/candidate.py','experiments/r21_yaw_continuity/state.py',
       'experiments/r21_yaw_continuity/model.json','experiments/r21_yaw_impulse/coarse_calibration.py',
       'experiments/r21_yaw_impulse/calibrate.py','experiments/r21_yaw_impulse/plan.json')
POLICIES={'core':'core','augmented':'integrated','guarded':'guarded','light':'light'}


def dependencies(revised=False):
    return tuple(dict.fromkeys(r21_d4_check.DEPENDENCIES+RUNTIME+EXTRA+(REVISED if revised else ())))


# Cohorts include the normal twins, not just the four or six exposed incidents.
COHORT_IDS = {
    "four": (39, 84, 167, 174, 330, 339, 349),
    "six": (39, 84, 134, 167, 174, 175, 319, 330, 335, 339, 349),
    "full": tuple(range(385)),
}
COHORT_KINDS = {
    "four": {"reference": 4, "collision": 3},
    "six": {"reference": 6, "collision": 5},
    "full": {"reference": 189, "collision": 180, "offset": 16},
}


def validate_accounting(data, policy, cohort):
    """A recorded performance failure is valid; a missing run is not."""
    ids, kinds = COHORT_IDS[cohort], COHORT_KINDS[cohort]
    rows = data.get("rows")
    if (data.get("cohort") != cohort or data.get("arm") != POLICIES[policy]
            or data.get("cases") != len(ids) or data.get("collisions") != kinds["collision"]
            or data.get("reference_cache_sha256", "") is not None
            or data.get("simulated_episodes") != 3 * len(ids)
            or not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows)
            or tuple(row.get("id") for row in rows) != ids
            or Counter(row.get("kind") for row in rows) != kinds):
        raise ValueError("incomplete, cached or mismatched public D4 cohort")
    for flag, failures in (("input_certified", "input_failures"), ("protection", "protection_failures")):
        if (any(type(row.get(flag)) is not bool for row in rows)
                or data.get(failures) != [row["id"] for row in rows if not row[flag]]):
            raise ValueError("inconsistent D4 accounting: " + flag)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--policy',choices=tuple(POLICIES),required=True)
    p.add_argument('--cohort',choices=('four','six','full'),default='full')
    p.add_argument('--workers',type=int,choices=(1,2,3),default=2)
    a=p.parse_args(argv)
    kit,out=shared.validate_paths(a.kit,a.out)
    manifest=(ROOT/shared.KIT_MANIFEST).read_bytes()
    original=shared.authenticate_kit(kit,manifest)
    blobs={}
    for n in dependencies(a.policy in ('guarded','light')):
        f=ROOT/n
        if f.is_symlink() or not f.is_file():raise ValueError('regular authored input required: '+n)
        blobs[n]=f.read_bytes()
    pins={k:shared.digest(v) for k,v in blobs.items()}
    with tempfile.TemporaryDirectory(dir=out.parent,prefix='r22-d4-') as temp:
        base=Path(temp)
        shared.install_overlay(kit,base,manifest,blobs,pins)
        if shared.authenticate_kit(base,manifest)!=original:raise ValueError('materialized kit differs')
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='disable');env.pop('PYTHONPATH',None)
        subprocess.run([sys.executable,'-B','-m','experiments.r22_census.check','--arm',POLICIES[a.policy],
                       '--cohort',a.cohort,'--workers',str(a.workers),'--out','result.json'],cwd=base,env=env,check=True)
        data=json.loads((base/'result.json').read_bytes())
        validate_accounting(data,a.policy,a.cohort)
        if (shared.authenticate_kit(base,manifest)!=original or shared.authenticate_kit(kit,manifest)!=original
                or any(shared.digest(ROOT/n)!=v or shared.digest(base/n)!=v for n,v in pins.items())):
            raise ValueError('unverified replay or changed source')
        doc={'schema':'r22-public-d4.v1','policy':a.policy,'cohort':a.cohort,'source_sha256':pins,
             'protected_manifest_sha256':shared.digest(manifest),'official_table':False,
             'all_input_certified':not data['input_failures'],'all_regression_checks_passed':not data['protection_failures'],
             'data':data}
        with out.open('x') as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
    print({k:doc[k] for k in ('policy','cohort','all_input_certified','all_regression_checks_passed')})


if __name__=='__main__':main()
