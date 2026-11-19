"""Check remaining disclosed R25 negatives and singles against frozen R25."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from experiments.r25_holdout import evaluator
from experiments.r26_long.candidate import Detector

ARCHIVE_SHA="83987dba8ce14fb6cc0283021d6bd062ac2fedb317ceb8da67c3d4b2b8c4f4fe"
TARGET="s25_long_transfer_signed"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def negative(case,variant,plan,old):
 _,log=evaluator.run_episode(case,variant,Detector,(None,None));common=evaluator.common(log,(None,None),[],plan);frozen=old['records']['r25']
 gates={'zero_false':not common['false_times_s'],'no_stuck_tail':not common['stuck_tail'],
        'force_exact':common['force_exact']['passed'],
        'tool_floor':evaluator.prior.prior.tool_floor(frozen['tool'],common['tool'],plan['preservation_slack_ms']),
        'physical_exact':evaluator.certified.trace_signature({'none':log})==old['r24_physics_sha256']}
 return {'id':case['id'],'kind':'negative','variant':variant['kind'],'common':common,'gates':gates,'passed':all(gates.values())}

def single(case,variant,plan,old):
 _,none=evaluator.run_episode(case,variant,Detector,(None,None));_,force=evaluator.run_episode(case,variant,Detector,(case['impact'],None));start,boundary=evaluator.actual_start(force['t'],case['impact']);event=evaluator.event(force,[none],start,min(6000,start+plan['event_window_ms']),plan);frozen=old['records']['r25'];previous=frozen['event'];commons={'none':evaluator.common(none,(None,None),[],plan),'single':evaluator.common(force,(case['impact'],None),[start],plan)}
 gates={'strict_floor':not previous['strict'] or event['strict'],
        'latency_plus_5_ms':not previous['strict'] or event['delays_ms'][0]<=previous['delays_ms'][0]+5,
        'duration_plus_5_samples':not previous['strict'] or event['active_samples_in_slot']<=previous['active_samples_in_slot']+5,
        'zero_duplicate':event['duplicate_rises']==0,
        'zero_false':all(not x['false_times_s'] for x in commons.values()),
        'no_stuck_tail':all(not x['stuck_tail'] for x in commons.values()),
        'force_exact':all(x['force_exact']['passed'] for x in commons.values()),
        'tool_floor':all(evaluator.prior.prior.tool_floor(frozen['twins'][tw]['tool'],commons[tw]['tool'],plan['preservation_slack_ms']) for tw in commons),
        'physical_exact':evaluator.certified.trace_signature({'none':none,'single':force})==old['r24_physics_sha256'],
        'boundary_exact':boundary==frozen['boundary']}
 return {'id':case['id'],'kind':'single','variant':variant['kind'],'frozen':previous,'candidate':event,'common':commons,'gates':gates,'passed':all(gates.values())}

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args(argv)
 if a.out.exists() or a.out.is_symlink():p.error('fresh --out required')
 if sha(a.archive)!=ARCHIVE_SHA:raise ValueError('unexpected disclosed R25 archive')
 archive=json.loads(a.archive.read_bytes());plan=json.loads(evaluator.PLAN.read_bytes());old={(r['kind'],r['id'],r['variant']):r for r in archive['rows']};rows=[]
 for case in plan['negative']:
  for name,variant in plan['variants'].items():rows.append(negative(case,variant,plan,old[('negative',case['id'],name)]))
 for case in plan['single']:
  if case['id']==TARGET:continue
  for name,variant in plan['variants'].items():rows.append(single(case,variant,plan,old[('single',case['id'],name)]))
 doc={'schema':'r26-long-remaining-r25-protection.v1','archive_sha256':sha(a.archive),'episodes':9+24,'rows':rows,'passed':all(r['passed'] for r in rows)}
 a.out.write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n');print(json.dumps({'episodes':doc['episodes'],'failed':[r['variant']+':'+r['id'] for r in rows if not r['passed']],'passed':doc['passed']},indent=2))
 if not doc['passed']:raise SystemExit(1)
if __name__=='__main__':main()
