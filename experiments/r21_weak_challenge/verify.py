"""Fail-closed crosscheck of the two frozen-policy weak-challenge results."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path

PLAN_SHA='7f2e9f9659a6b56a0982b5a19b0b04b4fd0e241ca7403b7ea5c5c3cf92c74a95'
PICK_PINS={'detector':'30e56f7ac752c6072ef88869378147be3534484b097ee23044c328f787f50364',
 'candidate':'1f03ed4918fb279783813c8f1f769ee3a4476b1d973a765b2c5d81ce81e9e3e8',
 'model':'499b2577f659dff86bcc665fffc385adddadfe21824ef0e06c7d848caa49c6bf'}
YAW_PINS={'state':'f04d46e93c55e04aa869050fe43c895951f3a970e3729b01e8d94744532f3a9f',
 'detector':'69e1e884f393a17751546165554f994c19b0cb127830233fba1f2329481c87cb',
 'frozen':'71a75f131ea6d3aae94f9acf40c42abe92d9123000d86e685f8bb5f4a00a4342',
 'coarse':'f3a8a4202e638346dfa6a7994b056cbf509be4f335fb09c76d6657d47abcafe9',
 'normal_bins':'5f22f640eb8824f764ffeec7ecbef7ba23df02a874547b93b8ddc72925e25c15'}
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
FORCED_CHECKS = {'no_added_no_force_false', 'no_added_force_false', 'no_new_prealarm',
                 'preserve_tool', 'no_new_stuck_tail', 'no_strict_r10_loss',
                 'preimpact_prefix_exact', 'applied_force_exact', 'physical_equal'}
NEGATIVE_CHECKS = {'no_added_false', 'preserve_tool', 'no_new_prealarm',
                   'no_new_stuck_tail', 'physical_equal'}


def paired_rows(plan, pick, yaw):
    """Check recorded cohort, counts and repeated reference arms before aggregating."""
    for result, candidate in ((pick, 'r21_pick'), (yaw, 'r21_yaw')):
        if (result['episodes'] != 99 or result['passed'] is not True or
                result['arms'] != ['r10', 'r20', candidate]):
            raise ValueError('incomplete run or unexpected arms')
        for kind, size, gates in (('forced', 6, FORCED_CHECKS), ('negative_only', 3, NEGATIVE_CHECKS)):
            cases = plan[kind]
            rows = result[kind]
            ids = [c['id'] for c in cases]
            if ids != [r['id'] for r in rows] or len(ids) != len(set(ids)):
                raise ValueError('cohort or order differs: ' + kind)
            for case, row in zip(cases, rows):
                if (row['episodes'] != size or row['case'] != case or row['role'] != case['role']
                    or row['passed'] is not True or set(row['arms']) != set(result['arms'])
                    or set(row['comparisons']) != {'r20', candidate}):
                    raise ValueError('incomplete case: ' + row['id'])
                for comp in row['comparisons'].values():
                    if (comp['passed'] is not True or set(comp['checks']) != gates
                            or any(v is not True for v in comp['checks'].values())):
                        raise ValueError('failed or missing gate: ' + row['id'])
        if sum(r['episodes'] for k in ('forced', 'negative_only') for r in result[k]) != result['episodes']:
            raise ValueError('episode sum differs')
    for kind in ('forced', 'negative_only'):
        for left, right in zip(pick[kind], yaw[kind]):
            if any(left['arms'][arm] != right['arms'][arm] for arm in ('r10', 'r20')):
                raise ValueError('repeat comparators differ: ' + left['id'])
    return ({r['id']: r for r in result['forced']} for result in (pick, yaw))

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
 p.add_argument('--pick',type=Path,required=True);p.add_argument('--yaw',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
 if os.path.lexists(a.out):p.error('new output required')
 if sha(a.plan)!=PLAN_SHA:raise ValueError('plan pin differs')
 for rel,pin in {
  'experiments/r21_pick_observer/detector.py':PICK_PINS['detector'],
  'experiments/r21_pick_observer/candidate.py':PICK_PINS['candidate'],
  'experiments/r21_pick_observer/model.json':PICK_PINS['model'],
  'experiments/r21_yaw_impulse/state.py':YAW_PINS['state'],
  'experiments/r21_yaw_impulse/detector.py':YAW_PINS['detector'],
  'experiments/r21_yaw_impulse/frozen.py':YAW_PINS['frozen'],
  'experiments/r21_yaw_impulse/coarse.py':YAW_PINS['coarse'],
  'experiments/r21_yaw_impulse/normal_bins.json':YAW_PINS['normal_bins']}.items():
  if sha(ROOT/rel)!=pin:raise ValueError('policy pin differs: '+rel)
 plan=json.loads(a.plan.read_bytes());pick=json.loads(a.pick.read_bytes());yaw=json.loads(a.yaw.read_bytes())
 if (pick['schema']!='r21-weak-horizontal-result.v1' or yaw['schema']!='r21-weak-yaw-result.v1'
   or pick['plan_sha256']!=PLAN_SHA or yaw['plan_sha256']!=PLAN_SHA
   or pick['candidate_pin']!=PICK_PINS['detector'] or yaw['pins']['experiments/r21_yaw_impulse/coarse.py']!=YAW_PINS['coarse']):
  raise ValueError('unexpected result identity')
 pp,yy=paired_rows(plan,pick,yaw)
 cases=[]
 for case in plan['forced']:
  i=case['id'];p1,y1=pp[i],yy[i]
  cases.append({'id':i,'role':case['role'],'mass':case['mass'],'time':case['time'],'force':case['force'],
   'r10':p1['arms']['r10']['force'],'r20':p1['arms']['r20']['force'],
   'r21_pick':p1['arms']['r21_pick']['force'],'r21_yaw':y1['arms']['r21_yaw']['force']})
 weak=[c for c in cases if c['role']=='fresh_weak_early']
 totals={}
 for arm in ('r10','r20','r21_pick','r21_yaw'):
  totals[arm]={'weak_strict':sum(c[arm]['strict_true_positive'] for c in weak),
               'all_forced_strict':sum(c[arm]['strict_true_positive'] for c in cases)}
 totals['r21_pick']['candidate_only_weak_vs_r20']=sum(c['r21_pick']['strict_true_positive'] and not c['r20']['strict_true_positive'] for c in weak)
 totals['r21_yaw']['candidate_only_weak_vs_r20']=sum(c['r21_yaw']['strict_true_positive'] and not c['r20']['strict_true_positive'] for c in weak)
 summary={'schema':'r21-independent-weak-summary.v1','plan_sha256':PLAN_SHA,
  'result_sha256':{'pick':sha(a.pick),'yaw':sha(a.yaw)},'episodes':pick['episodes']+yaw['episodes'],
  'policies_frozen_before_hidden_outcomes':True,'comparator_repeat_exact':True,
  'all_pairwise_gates_passed':True,'pins':{'pick':PICK_PINS,'yaw':YAW_PINS},
  'totals':totals,'negative_only':pick['negative_only'],'cases':cases}
 with a.out.open('x') as f:json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'episodes':summary['episodes'],'totals':totals}))
if __name__=='__main__':main()
