"""Sharded fresh R24 traces, counterfactual scoring, and comparison with the pinned 385-case R23 census."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from experiments.r22_census import check
from experiments.r24_loaded.studies import get_arm

PRIOR='f182141ef712101b005ba9349b5f2d2137650ccf8310e1a89fb8bdae75972eec'


def capture(job):
    case,keys=job
    return {'id':case['case_id'],'records':{arm:check.old.episode_record(get_arm(arm),case) for arm in keys}}


def load_cases():
    cases=check.build_cases('my_torque')
    if hashlib.sha256(check.old.canonical_cases(cases)).hexdigest()!=check.MATRIX:raise ValueError('altered matrix')
    return cases


def study_chunk(keys,idx,n,workers):
    stock=check.check_integrity();cases=load_cases()
    if not (0<=idx<n<=385):raise ValueError('invalid shard')
    take=cases[idx::n]
    with ProcessPoolExecutor(workers) as pool: rows=list(pool.map(capture,[(c,tuple(keys)) for c in take]))
    return {'schema':'r24-loaded-d4-shard.v1','matrix_sha256':check.MATRIX,'kit':stock,'shard':[idx,n],'arms':keys,
            'episodes':len(take)*len(keys),'rows':rows}


def summarize(chunks, prior):
    if hashlib.sha256(prior.read_bytes()).hexdigest()!=PRIOR:raise ValueError('expected authentic R23 original scores')
    old=json.loads(prior.read_bytes())['data'];cases=load_cases();pool={};keys=chunks[0]['arms']
    total=chunks[0]['shard'][1];kinds={s['shard'][0] for s in chunks}
    if len(chunks)!=total or kinds!=set(range(total)):raise ValueError('missing or duplicate shards')
    for s in chunks:
        if s['arms']!=keys or s['matrix_sha256']!=check.MATRIX or s['shard'][1]!=total:raise ValueError('mixed shards')
        for r in s['rows']:
            if r['id'] in pool or r['id']%total!=s['shard'][0]:raise ValueError('misassigned row')
            pool[r['id']]=r['records']
    if set(pool)!=set(range(385)):raise ValueError('incomplete')
    normal={check.old.reference_key(c):pool[c['case_id']] for c in cases if c['kind']=='reference'}
    rows=[]
    for case,base in zip(cases,old['rows'],strict=True):
        i=case['case_id']; assert base['id']==i
        prim=pool[i]; ref=normal[check.old.reference_key(case)] if case['kind']=='collision' else prim
        scores={'r23_free':base['scores']['free']}
        valid=base['input_certified'];gates={}
        for arm,r in prim.items():
            s=check.score(case,r,ref[arm]); scores[arm]=s
            valid=valid and r['physical_sha256']==base['physics_sha256'] and all(s['valid'].values())
            g=check.compare(s,scores['r23_free']);b=scores['r23_free']
            g['latency_no_more_than_5ms_later']=not b['strict'] or (s['strict'] and s['delay_ms']<=b['delay_ms']+5)
            gates[arm]=g
        rows.append({'id':i,'kind':case['kind'],'input_certified':valid,'scores':scores,'gates':gates})
    collisions=[r for r in rows if r['kind']=='collision']
    data={'schema':'r24-loaded-d4-after-r23.v1','prior_full_sha256':PRIOR,'fresh_episodes':sum(s['episodes'] for s in chunks),
          'archive_scope':'Only baseline R23 FREE; all reported R24 arms, including matching normal motions, were newly simulated.',
          'summary':{k:{'strict':sum(r['scores'][k]['strict'] for r in collisions),'false':sum(len(r['scores'][k]['false']) for r in rows),
              'duplicate':sum(r['scores'][k]['duplicate'] for r in collisions),'tool_detected':sum(r['scores'][k]['tool']['detected'] for r in rows)}
              for k in ['r23_free',*keys]},'invalid':[r['id'] for r in rows if not r['input_certified']],
          'regressions':{k:[[r['id'],[flag for flag,v in r['gates'][k].items() if not v]] for r in rows if not all(r['gates'][k].values())] for k in keys},'rows':rows}
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('shard');a.add_argument('--index',type=int,required=True);a.add_argument('--count',type=int,required=True)
    a.add_argument('--workers',type=int,choices=(1,2,3),default=2);a.add_argument('--arm',action='append',required=True)
    b=sub.add_parser('verify');b.add_argument('--shards',type=Path,nargs='+',required=True);b.add_argument('--prior',type=Path,required=True)
    for parser in (a,b):parser.add_argument('--out',type=Path,required=True)
    q=p.parse_args()
    if q.out.exists() or q.out.is_symlink():raise ValueError('fresh output required')
    d=study_chunk(q.arm,q.index,q.count,q.workers) if q.action=='shard' else summarize([json.loads(x.read_bytes()) for x in q.shards],q.prior)
    with q.out.open('x') as f:json.dump(d,f,allow_nan=False);f.write('\n')
    print(q.action, d.get('shard',d.get('summary')),d.get('regressions'))
