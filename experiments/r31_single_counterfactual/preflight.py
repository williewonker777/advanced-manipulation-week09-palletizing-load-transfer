"""Validate generic prefix scoring on disclosed D4 case 330/reference 39."""
import argparse,hashlib,json
from pathlib import Path
from experiments.r21_full_protection import runner_v2 as raw
from experiments.r22_census import check
from experiments.r26_yaw_acquisition.candidate import Detector as C3
from experiments.r30_yaw_headroom.candidate import Detector as Headroom
from experiments.r31_single_counterfactual.scorer import calm_record
from tools.d4_robustness import build_cases

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    if a.out.exists() or a.out.is_symlink():p.error('fresh output required')
    cases=build_cases('my_torque');forced=cases[330];calm=cases[39]
    assert forced['collision'][0]==3.83 and calm['collision'] is None
    rows={}
    for name,detector in {'c3':C3,'r30_headroom':Headroom}.items():
        native=raw.episode_record(detector,calm)
        generic=calm_record(detector,calm,3830)
        event=raw.episode_record(detector,forced)
        a_score=check.score(forced,event,native);b_score=check.score(forced,event,generic)
        rows[name]={'native_score':a_score,'generic_score':b_score,
                    'prefix_equal':native['prefix_sha256']['3830']==generic['prefix_sha256']['3830'],
                    'physics_equal':native['physical_sha256']==generic['physical_sha256']}
    passed=all(r['native_score']==r['generic_score'] and r['prefix_equal'] and r['physics_equal'] for r in rows.values())
    out={'schema':'r31-single-counterfactual-preflight.v1','published_cases':[39,330],
         'scorer_sha256':sha(Path(__file__).with_name('scorer.py')),'rows':rows,'passed':passed}
    with a.out.open('x') as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    if not passed:raise RuntimeError('generic scorer differs from historical scorer')
    print('PASS published parity',list(rows))

if __name__=='__main__':main()
