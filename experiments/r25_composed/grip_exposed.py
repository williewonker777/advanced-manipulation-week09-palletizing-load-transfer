"""Prospective pickup-straddling same/opposite second-force guards, scored by the unchanged R24 event scorer."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from experiments.r24_holdout import evaluator
from experiments.r25_fragment.study import capture,summarize


def run(arms,workers):
    src=json.loads(Path(__file__).with_name('plan.json').read_bytes())['supplemental_grip_pairs']
    p=json.loads(evaluator.PLAN.read_bytes());tasks=[]
    for i,t in enumerate(src['first_times']):
        for j,gap in enumerate(src['separations_s']):
            for k,f in enumerate(src['second_N']):
                case={'id':f'grip-{i}{j}{k}','phase':'pickup','mass':src['mass_kg'],'com':src['com_m'],
                      'offset':None,'impacts':[[t,src['first_N']],[t+gap,f]]}
                tasks.extend(('grip','paired',case,p['variants'][m],arms,p) for m in src['mechanics'])
    with ProcessPoolExecutor(workers) as pool:rows=list(pool.map(capture,tasks))
    metrics,reg=summarize(rows,arms)
    return {'schema':'r25-disclosed-grip-second.v1','independent':False,'definition':src,'episodes':sum(x['episodes'] for x in rows),
            'arms':arms,'totals':metrics,'regressions':reg,'rows':rows}


if __name__=='__main__':
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--arm',action='append',required=True)
    q.add_argument('--workers',type=int,choices=(1,2),default=2);q.add_argument('--out',type=Path,required=True);a=q.parse_args()
    if a.out.exists():raise ValueError('fresh output required')
    j=run(a.arm,a.workers)
    with a.out.open('x') as f:json.dump(j,f,allow_nan=False);f.write('\n')
    print(j['episodes'],j['totals'],j['regressions'])
