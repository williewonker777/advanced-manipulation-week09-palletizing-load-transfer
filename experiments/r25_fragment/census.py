"""New candidate D4 physics vs the authenticated, complete earlier R24 original-physics census."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from experiments.r22_census import check
from experiments.r24_loaded.d4 import load_cases
from experiments.r25_fragment.study import constructor

RAW=Path(__file__).with_name('d4_archive.json')
OLD='experiments.r24_composed.detector:Detector'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def occupied(rec):
    lo,hi=rec['true_window_first_index'],rec['true_window_last_index']
    if lo is None or hi is None:return 0
    return sum(max(0,min(b,hi+1)-max(a,lo)) for a,b in rec['detect_intervals'])


def capture(job):
    c,arm=job
    return check.old.episode_record(constructor(arm),c)


def summarize(archive,chunks,arm):
    pin=read(RAW)
    if any(sha(archive/n)!=h for n,h in pin.items()):raise ValueError('altered R24 archive')
    prior=read(archive/'result.json')
    if prior['invalid'] or prior['regressions'][OLD]:raise ValueError('earlier R24 census not certified')
    old={r['id']:r['records'][OLD] for n in pin if n.startswith('shard-') for r in read(archive/n)['rows']}
    cases=load_cases();new={}
    for chunk in chunks:
        if chunk['candidate']!=arm or chunk['matrix_sha256']!=check.MATRIX:raise ValueError('mixed study')
        for k,v in chunk['records'].items():
            i=int(k)
            if i in new:raise ValueError('repeated case')
            new[i]=v
    if set(new)!=set(old) or set(new)!=set(range(len(cases))):raise ValueError('incomplete census')
    norm={check.old.reference_key(c):c['case_id'] for c in cases if c['kind']=='reference'}
    out=[]
    for c,w in zip(cases,prior['rows'],strict=True):
        i=c['case_id'];assert i==w['id']
        n=norm[check.old.reference_key(c)] if c['kind']=='collision' else i
        x=check.score(c,old[i],old[n]);y=check.score(c,new[i],new[n])
        assert x==w['scores'][OLD] and all(x['valid'].values())
        g=check.compare(y,x)
        g['time']=not x['strict'] or (y['strict'] and y['delay_ms']<=x['delay_ms']+5)
        g['alarm_cost']=not x['strict'] or occupied(new[i])<=occupied(old[i])+5
        # Old D4 stores the tool-timed rise indices, not per-score delay_ms.
        # The first true tool onset/sample is identical across the two physics traces.
        g['tool_time']=not x['tool']['detected'] or (y['tool']['detected'] and
                     min(new[i]['tool_rise_indices'])<=min(old[i]['tool_rise_indices'])+5)
        g['physics_force_prefix']=all(y['valid'].values()) and new[i]['physical_sha256']==old[i]['physical_sha256']
        out.append({'id':i,'r24':x,'candidate':y,'r24_cost':occupied(old[i]),'candidate_cost':occupied(new[i]),
                    'failures':[k for k,v in g.items() if not v]})
    totals={k:{'strict':sum(r[k]['strict'] for r in out),'false_rises':sum(len(r[k]['false']) for r in out),
               'extra_edges':sum(r[k]['duplicate'] for r in out),'tool':sum(r[k]['tool']['detected'] for r in out)} for k in ('r24','candidate')}
    return {'schema':'r25-d4-certified-comparison.v1','candidate':arm,'new_physics_episodes':len(new),'census':len(out),
            'scope':'All candidate cases including reference are newly simulated; earlier R24 records authenticated, recomputed, and tied to the archived scores. Occupancy within the actual registered collision window in samples.',
            'summary':totals,'regressions':[[r['id'],r['failures']] for r in out if r['failures']],'rows':out}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',required=True)
    p.add_argument('--shard',nargs=2,type=int);p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--prior',type=Path);p.add_argument('--chunks',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('fresh output required')
    if a.shard:
        i,n=a.shard
        if not(0<=i<n<=385) or not check.check_integrity():raise ValueError('invalid shard')
        selected=load_cases()[i::n]
        with ProcessPoolExecutor(a.workers) as pool:
            rec=dict(zip([c['case_id'] for c in selected],pool.map(capture,[(c,a.arm)for c in selected]),strict=True))
        doc={'candidate':a.arm,'matrix_sha256':check.MATRIX,'records':rec,'shard':[i,n]}
    else:doc=summarize(a.prior,[read(x) for x in a.chunks],a.arm)
    with a.out.open('x')as f:json.dump(doc,f,allow_nan=False);f.write('\n')
    print(doc.get('summary',doc.get('shard')),doc.get('regressions',[]))
