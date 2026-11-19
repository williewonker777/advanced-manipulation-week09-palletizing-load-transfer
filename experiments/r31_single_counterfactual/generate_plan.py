"""Create the one immutable seed-derived R31 single-force plan."""
import argparse,hashlib,json,math,random
from pathlib import Path

SEED=310031
DIRECTIONS=((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight',type=Path,required=True);a=p.parse_args(argv)
    if a.out.exists() or a.out.is_symlink():p.error('fresh output required')
    pre=json.loads(a.preflight.read_bytes())
    if not pre.get('passed'):p.error('generic scorer preflight did not pass')
    rng=random.Random(SEED);cases=[]
    for index,(dx,dy) in enumerate(DIRECTIONS,1):
        magnitude=round(rng.uniform(9.0,22.0),3);norm=math.hypot(dx,dy)
        force=[round(magnitude*dx/norm,6),round(magnitude*dy/norm,6),0.0]
        cases.append({'id':f's{index:02d}','mass':round(rng.uniform(.75,7.75),3),
                      'com':[round(rng.uniform(-.028,.028),4),round(rng.uniform(-.028,.028),4),round(rng.uniform(-.012,.042),4)],
                      'time':round(rng.uniform(2.12,3.42),3),'force':force})
    doc={'schema':'r31-single-counterfactual-plan.v1','status':'FROZEN_BEFORE_ANY_NOVEL_OUTPUT','seed':SEED,
         'classification':'fresh deterministic single-force/calm-twin loaded-transport panel; disjoint from invalid R31 panel and all double-force panels',
         'invalid_predecessor':{'status':'INVALID_INFRASTRUCTURE','plan_sha256':'101e960bc5bfa86a534cf27cf9af6a6404e53c3743a2c9bd09282356aa73a0a0','released_case_outputs':0,'reused_tuples':0},
         'preflight':{'published_cases':[39,330],'receipt_sha256':sha(a.preflight),'scorer_sha256':pre['scorer_sha256'],'passed':True},
         'arms':{'c3':{'entrypoint':'experiments.r26_yaw_acquisition.candidate:Detector','sha256':'212f4e79367cb636c0143a2d1f8aa7ba5584aaaca5ecde6b412d01a4985b6784'},
                 'r30_headroom':{'entrypoint':'experiments.r30_yaw_headroom.candidate:Detector','sha256':'869ec502c5538e4b9e2de501a0cc501071ce7249b9b902658657e57bd3b4e5b1'},
                 'r31_held':{'entrypoint':'experiments.r31_held_identity.candidate:Detector','sha256':'1e33b4698ef3678afe72cc4e286fcab5d27b2232e7bf087fdef7500ccc674764'}},
         'cases':cases,'execution':'12 collision inputs plus exact calm twins for all three arms; 72 episodes once; retain every row',
         'input_contracts':['complete finite 6000-sample trace','exact signed vector and scalar norm','generic arbitrary-onset calm/forced prefix exact','same-input physical arrays byte-identical across arms'],
         'protection':['R31 retains every C3/R30 strict detection','no new false/delayed false/duplicate/mask/tool prealarm/stuck tail versus either parent','first attributable rise no more than 5ms later','primary attributable interval no more than 5 samples shorter','all R31 calm twins have zero false rise'],
         'success':'all input/protection gates pass and at least one actual R30 duplicate is removed while R31 remains strict',
         'inconclusive':'all gates pass but no R30 duplicate is exercised','failure':'any gate fails or duplicate removal loses strict detection','no_third_panel':True}
    with a.out.open('x') as f:json.dump(doc,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print('seed',SEED,'cases',len(cases))

if __name__=='__main__':main()
