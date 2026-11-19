"""Reproduce the already-disclosed false rises; physical force truth is used only by the scorer."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from experiments.r23_holdout import evaluator as lab
from experiments.r24_loaded.probe import Probe

PINS='8b55ed07f64128d5b957630019576b8477b47f57447989668fd284bee7abd022'
PORTS=('r10','directional','normal_model','horizontal','yaw','light','detached')


def one(job):
    case,variant,twin,impacts,want=job;probe=Probe()
    _,log=lab.run_episode(case,variant,lambda:probe,impacts)
    if not lab.certified.exact_force(log,impacts)['passed']:raise ValueError('force differs')
    starts=[None if impact is None else lab.certified.actual_start(log['t'],impact)[0] for impact in impacts]
    actual=lab.certified.false_record(log,starts,280)
    if actual!=want:raise ValueError('archived event did not reproduce')
    rows=[r for r in probe.rows if r['t'] in want]
    if len(rows)!=len(want):raise ValueError('not captured')
    return [{'id':case['id'],'mechanics':variant['kind'],'twin':twin,'t':r['t'],'active_ports':[n for n in PORTS if r[n]],
             'yaw_ratio':r['yaw_score'],'largest_R10_directional_ratio':max(r['scores']),
             'tool_N':float(log['f_tool'][round(r['t']*1000)]),'applied_N':float(log['f_ext'][round(r['t']*1000)]),
             'physical_sha256':lab.certified.physical_hash(log)}for r in rows]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prior',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args()
    raw=a.prior.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PINS:raise ValueError('expected exact previously disclosed R23 full report')
    plan=json.loads(lab.PLAN.read_bytes()); prior=json.loads(raw);jobs=[]
    for row in prior['rows']:
        c=next(c for c in plan[row['kind']] if c['id']==row['id']);v=plan['variants'][row['variant']];rec=row['records']['r23']
        assign=({'none':(None,None)} if row['kind']=='negative' else
                {'none':(None,None),'force':(c['impact'],None)} if row['kind']=='single' else
                {'none':(None,None),'first':(c['impacts'][0],None),'second':(None,c['impacts'][1]),'both':tuple(c['impacts'])})
        for tw,tr in rec.get('twins',{'none':rec}).items():
            if tr['false_times_s']:jobs.append((c,v,tw,assign[tw],tr['false_times_s']))
    with ProcessPoolExecutor(a.workers) as pool:results=[r for rows in pool.map(one,jobs) for r in rows]
    data={'schema':'r24-inherited-false-port-attribution.v1','previous_result_sha256':PINS,'episodes':len(jobs),'rises':results,'all_yaw_only':all(r['active_ports']==['yaw'] for r in results)}
    with a.out.open('x') as fp:json.dump(data,fp,indent=2);fp.write('\n')
    print('reproduced',len(results),'false rises in',len(jobs),'episodes; all yaw only:',data['all_yaw_only'])


if __name__=='__main__':main()
