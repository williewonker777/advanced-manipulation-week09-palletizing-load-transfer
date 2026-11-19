"""Fresh R22/Free/Onset simulations on the R22 fixed 385-control/collision/offset matrix."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from experiments.r22_census import check
from experiments.r22_light.candidate import Detector as Light
from experiments.r23_detached.candidate import Detector, Free

ARMS = {'light':Light,'free':Free,'onset':Detector}


def episode(case):
    return {k:check.old.episode_record(cls,case) for k,cls in ARMS.items()}


def analyze(cases, raw):
    normal = {check.old.reference_key(c):r for c,r in zip(cases,raw,strict=True) if c['kind']=='reference'}
    rows = []
    for c,r in zip(cases,raw,strict=True):
        ref = normal[check.old.reference_key(c)] if c['kind']=='collision' else r
        score = {k:check.score(c, record, ref[k]) for k,record in r.items()}
        physical = all(record['physical_sha256']==r['light']['physical_sha256'] for record in r.values())
        rows.append({'id':c['case_id'],'kind':c['kind'],'scores':score,
                     'physics_sha256':r['light']['physical_sha256'],'input_certified':physical
                     and all(all(a['valid'].values()) for a in score.values()),
                     'vs_light':{key:check.compare(score[key],score['light']) for key in ('free','onset')}})
    collisions = [r for r in rows if r['kind']=='collision']
    return {'summary':{k:{'strict':sum(r['scores'][k]['strict'] for r in collisions),
                          'false':sum(len(r['scores'][k]['false']) for r in rows),
                          'duplicate':sum(r['scores'][k]['duplicate'] for r in collisions),
                          'tool_detected':sum(r['scores'][k]['tool']['detected'] for r in rows)}
                       for k in ARMS},
            'invalid':[r['id'] for r in rows if not r['input_certified']],
            'regressions':{k:[(r['id'],[n for n,v in r['vs_light'][k].items() if not v])
                              for r in rows if not all(r['vs_light'][k].values())] for k in ('free','onset')},
            'latency_change_ms':{k:{r['id']:r['scores'][k]['delay_ms']-r['scores']['light']['delay_ms']
                                    for r in collisions if r['scores'][k]['strict'] and r['scores']['light']['strict']}
                                 for k in ('free','onset')},
            'rows':rows}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,choices=(1,2,3),default=2)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--cohort',choices=('six','full'),required=True)
    a=p.parse_args(argv)
    if a.out.exists() or a.out.is_symlink():
        raise ValueError('fresh output required')
    kit=check.check_integrity(); cases=check.build_cases('my_torque')
    if hashlib.sha256(check.old.canonical_cases(cases)).hexdigest()!=check.MATRIX:
        raise ValueError('matrix changed')
    ids=[39,84,134,167,174,175,319,330,335,339,349] if a.cohort=='six' else list(range(385))
    cases=[c for c in cases if c['case_id'] in ids]
    if [c['case_id'] for c in cases]!=ids: raise ValueError('missing or duplicate case')
    with ProcessPoolExecutor(a.workers) as pool:
        raw=list(pool.map(episode,cases))
    data=analyze(cases,raw)
    doc={'schema':'r23-d4-exposed.v1','unseen':False,'cohort':a.cohort,'matrix_sha256':check.MATRIX,
         'protected_kit':kit,'episodes':len(cases)*len(ARMS),'cases':len(cases),**data}
    with a.out.open('x') as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
    print({k:doc[k] for k in ('cohort','episodes','summary','invalid','regressions')})


if __name__=='__main__':main()
