"""Retest previously exposed R21 four-twin cases; do not count as fresh evidence."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

import my_detector
from experiments.r18_single_pass import paired_replay
from experiments.r19_close_pairs import assay
from experiments.r20_force_certificate import certificate
from experiments.r21_surface_pairs.study import KEYS, local_episode
from experiments.r21_yaw_impulse.coarse import Detector as Yaw
from experiments.r22_combined.core import Detector as Core
from experiments.r22_combined.candidate import Detector as Augmented

ARMS = {'r10':my_detector.Detector, 'yaw':Yaw,'core':Core,'augmented':Augmented}
ROOT = Path(__file__).resolve().parents[1]


def episode_group(task):
    case, stage, mode, revised=task
    if revised:
        from experiments.r22_guarded.candidate import Detector as Guarded
        from experiments.r22_light.candidate import Detector as Light
        arms={'r10':my_detector.Detector,'yaw':Yaw,'guarded':Guarded,'light':Light}
    else:arms=ARMS
    first,second=case['impacts']
    suite={'none':(None,None),'first':(first,None),'second':(None,second),'both':(first,second)}
    result={};ref=None
    expected={k:certificate.exact_trace(case,k) for k in suite}
    for name,det in arms.items():
        logs={k:local_episode(case,mode,det,impacts) for k,impacts in suite.items()}
        if ref is None:ref=logs
        physical=all(np.array_equal(ref[tw][key],log[key]) for tw,log in logs.items() for key in paired_replay.PHYSICAL+('r',))
        signed=all(certificate.exact_equal({k:log[f] for k,f in KEYS.items()},expected[tw]) and np.array_equal(log['t'],expected[tw]['t']) for tw,log in logs.items())
        if not (signed and physical):raise ValueError('force or physical trace differs: '+case['id'])
        rec, raw=assay.characterize(case,logs,20,320)
        strict=assay.one_per_pulse(rec) and all(v for k,v in rec['r18_gates'].items() if k!='signed_profiles_exact')
        result[name]={'record':rec,'score_sha256':paired_replay.signature(raw),'strict':strict, 'exact_force_and_physics':True}
    public={}
    for arm,v in result.items():
        rec=v['record']
        vs={k:assay.all_comparisons([assay.compare(rec,result[k]['record'])]) for k in ('r10','yaw')}
        public[arm]={k:v[k] for k in ('score_sha256','strict','exact_force_and_physics')} | {
            'preserve_genuine_quiet_false_tool_tail':vs, 'slots':rec['slots'],
            'missed_checks':[k for k, passed in rec['r18_gates'].items() if k!='signed_profiles_exact' and not passed]}
    return {'id':case['id'],'stage':stage,'surface':mode,'episodes':16,'arms':public}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--jobs',type=int,choices=(1,2,3),default=2);p.add_argument('--revised',action='store_true');a=p.parse_args(argv)
    old=json.loads((ROOT/'r21_surface_pairs/plan.json').read_bytes())['cases']
    later=json.loads((ROOT/'r21_yaw_continuity/plan.json').read_bytes())['fresh']
    tasks=[(case,label,mode,a.revised) for label,cases in [('previous_surface',old),('previous_history',later)]
           for case in cases for mode in ('stock','changed')]
    with ProcessPoolExecutor(a.jobs) as pool:rows=list(pool.map(episode_group,tasks))
    summary={label:{arm:sum(r['arms'][arm]['strict'] for r in rows if r['stage']==label) for arm in rows[0]['arms']}
             for label in ('previous_surface','previous_history')}
    extra={arm:[(r['id'],r['surface']) for r in rows if not all(r['arms'][arm]['preserve_genuine_quiet_false_tool_tail'].values())
                or (r['arms']['yaw']['strict'] and not r['arms'][arm]['strict'])] for arm in (('guarded','light') if a.revised else ('core','augmented'))}
    doc={'schema':'r22-previous-four-twin-study.v1','unseen':False,'episodes':sum(r['episodes'] for r in rows),
         'summary':summary,'protection_failures':extra,'rows':rows}
    with a.out.open('x') as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
    print(summary,extra)


if __name__=='__main__':
    main()
