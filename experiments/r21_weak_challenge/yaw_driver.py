"""Apply the frozen weak challenge with the same scorer to a separately pinned yaw arm."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
from experiments.r21_weak_challenge import evaluator as common
from experiments.r21_yaw_impulse.coarse import Detector as Yaw

ROOT=Path(__file__).resolve().parents[2]
PINS={
 'experiments/r21_yaw_impulse/state.py':'f04d46e93c55e04aa869050fe43c895951f3a970e3729b01e8d94744532f3a9f',
 'experiments/r21_yaw_impulse/detector.py':'69e1e884f393a17751546165554f994c19b0cb127830233fba1f2329481c87cb',
 'experiments/r21_yaw_impulse/frozen.py':'71a75f131ea6d3aae94f9acf40c42abe92d9123000d86e685f8bb5f4a00a4342',
 'experiments/r21_yaw_impulse/coarse.py':'f3a8a4202e638346dfa6a7994b056cbf509be4f335fb09c76d6657d47abcafe9',
 'experiments/r21_yaw_impulse/normal_bins.json':'5f22f640eb8824f764ffeec7ecbef7ba23df02a874547b93b8ddc72925e25c15',
 'experiments/r21_weak_challenge/evaluator.py':'690d77d962c74907aab2123f5b2d1cd55e562fb0c9db9eb699e644cb7f3ac6f7',
}
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
 if os.path.lexists(a.out):p.error('new output path required')
 for rel,pin in PINS.items():
  if common.sha(ROOT/rel)!=pin:p.error('source pin differs: '+rel)
 raw=a.plan.read_bytes();plan=json.loads(raw);common.validate(plan);arms={'r10':common.STATIC['r10'],'r20':common.STATIC['r20'],'r21_yaw':Yaw}
 forced=[common.forced_row(c,arms,int(plan['reference_twin_ms'])) for c in plan['forced']]
 negatives=[common.negative_row(c,arms) for c in plan['negative_only']]
 if any(common.sha(ROOT/rel)!=pin for rel,pin in PINS.items()):raise ValueError('source changed during run')
 doc={'schema':'r21-weak-yaw-result.v1','plan_sha256':hashlib.sha256(raw).hexdigest(),'pins':PINS,
      'arms':list(arms),'episodes':sum(r['episodes'] for r in forced+negatives),'forced':forced,
      'negative_only':negatives,'passed':all(r['passed'] for r in forced+negatives)}
 with a.out.open('x') as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'episodes':doc['episodes'],'forced':len(forced),'negative_only':len(negatives),'passed':doc['passed']}))
if __name__=='__main__':main()
