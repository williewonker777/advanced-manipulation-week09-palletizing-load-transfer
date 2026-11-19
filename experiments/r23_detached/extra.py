"""Legacy four-twin protection or real-cycle timing, separate from independent testing."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from types import FunctionType

import my_detector
from experiments.r22_light.candidate import Detector as Light
from experiments.r23_detached.candidate import Detector, Free


def pair(task):
    from experiments.r22_combined import known_pairs as known
    aliases={'r10':my_detector.Detector,'yaw':Light,'core':Free,'augmented':Detector}
    call=FunctionType(known.episode_group.__code__,dict(known.episode_group.__globals__,ARMS=aliases))
    row=call(task)
    row['arms']={name:row['arms'][key] for name,key in
                 [('r10','r10'),('light','yaw'),('free','core'),('onset','augmented')]}
    for r in row['arms'].values():
        refs=r['preserve_genuine_quiet_false_tool_tail']
        r['preserve_genuine_quiet_false_tool_tail']={'r10':refs['r10'],'light':refs['yaw']}
    return row


def pairs(workers):
    from experiments.r22_combined import known_pairs as old
    earlier=json.loads((old.ROOT/'r21_surface_pairs/plan.json').read_bytes())['cases']
    more=json.loads((old.ROOT/'r21_yaw_continuity/plan.json').read_bytes())['fresh']
    jobs=[(c,stage,mode,False) for stage,cases in [('previous_surface',earlier),('previous_history',more)]
          for c in cases for mode in ('stock','changed')]
    with ProcessPoolExecutor(workers) as pool:rows=list(pool.map(pair,jobs))
    if len(rows)!=14 or sum(r['episodes'] for r in rows)!=224:raise ValueError('incomplete legacy twin matrix')
    return {'schema':'r23-legacy-pairs.v1','episodes':224,'unseen':False,'rows':rows,
            'totals':{k:{s:sum(r['arms'][k]['strict'] for r in rows if r['stage']==s)
                         for s in ('previous_surface','previous_history')} for k in ('r10','light','free','onset')},
            'regressions':{k:[(r['id'],r['surface']) for r in rows if not
                              r['arms'][k]['preserve_genuine_quiet_false_tool_tail']['light']
                              or (r['arms']['light']['strict'] and not r['arms'][k]['strict'])]
                           for k in ('free','onset')}}


def timing():
    from experiments.r22_timing import study as old
    types={'light':Light,'free':Free,'onset':Detector}
    trial=FunctionType(old.trial.__code__,dict(old.trial.__globals__,TYPES=types))
    cases=json.loads(Path(old.__file__).with_name('plan.json').read_bytes())['cases']
    cases=cases+[{'id':'detached','mass':4.83,'com':[-.021,.018,.007],'collision':[3.927,[-7.7,-7.1,0]]}]
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    rows=[trial(case,k,i) for i in range(2) for case in cases for k in (tuple(types) if i==0 else tuple(reversed(types)))]
    for case in cases:
        group=[r for r in rows if r['case']==case['id']]
        if len(group)!=6 or any(len({r[field] for r in group})!=1 for field in ('physical','residual')):
            raise ValueError('timing trials changed protected data')
        for arm in types:
            if len({r['detected'] for r in group if r['arm']==arm})!=1:raise ValueError('nonrepeatable timing trial')
    return {'schema':'r23-timing.v1','episodes':len(rows),'isolated_machine':False,'hard_realtime_proven':False,
            'summary':{k:{'cycles':sum(r['cycles'] for r in rows if r['arm']==k),
                          'worst_episode_wall_p99_us':max(r['sum_wall']['p99_us'] for r in rows if r['arm']==k),
                          'worst_episode_cpu_p99_us':max(r['sum_cpu']['p99_us'] for r in rows if r['arm']==k),
                          'worst_wall_us':max(r['sum_wall']['max_us'] for r in rows if r['arm']==k),
                          'wall_cycles_at_least_1ms':sum(r['sum_wall']['at_least_1ms'] for r in rows if r['arm']==k)}
                       for k in types},'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=('pairs','timing'),required=True)
    p.add_argument('--workers',type=int,choices=(1,2,3),default=2);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();doc=pairs(a.workers) if a.kind=='pairs' else timing()
    with a.out.open('x') as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
