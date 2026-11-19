"""Predeclared multi-pulse checks on the actual calibrated R24 physical controller."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from types import FunctionType

import mujoco

from experiments.r14_stock_handoff.detector import Detector as Coupled
from experiments.r25_holdout import evaluator as pulse
from experiments.r26_delivery_pair import study
from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r26_delivery_pair.controller import Controller


HERE=Path(__file__).resolve().parent


def fork(fn, **override):
    out=FunctionType(fn.__code__,dict(fn.__globals__,**override),fn.__name__,fn.__defaults__,fn.__closure__)
    out.__kwdefaults__=fn.__kwdefaults__
    return out


def select(arm):
    if arm=='c1':
        return C1
    if arm=='c2':
        from experiments.r26_impulse_onset.candidate import Detector
        return Detector
    raise ValueError(arm)


def paired(c,v,a,p):
    # R25's explicit signed injector and strict twin scorer with a single
    # substitution: the actual physical controller used on every branch.
    run=fork(pulse.run_episode,Controller=Controller)
    call=fork(pulse.paired_case,run_episode=run)
    row=call(c,v,{'r24_joint':Coupled,'r25':select(a)},p)
    row['records']={'coupled':row['records']['r24_joint'],a:row['records']['r25']}
    row['baseline_physics_sha256']=row.pop('r24_physics_sha256')
    return row


def paired_job(x):
    return paired(*x)


def singles_case(x):
    case,forces=x
    def lookup(k):
        return (select('c2'),Controller) if k=='c2' else study.factory(k)
    bundle=fork(study.bundle,factory=lookup)
    arms={k:bundle(case,forces,k) for k in ('r10','r19','r24','pair','c2')}
    return study.result(case,forces,arms)


def read(p):
    return json.loads(Path(p).read_bytes())


def run(kind,arm,workers,admission):
    if mujoco.__version__!='3.17.0':
        raise ValueError('authentic course MuJoCo 3.17.0 required')
    if arm=='c2':
        if admission is None:
            raise ValueError('a frozen and fully verified known C2 admission is mandatory')
        j=read(admission)
        if not (j['all_known_guards'] and j['known_attributable']==4 and j['real_gain'] and
                all(hashlib.sha256((HERE.parents[1]/file).read_bytes()).hexdigest()==pin for file,pin in j['candidate_pins'].items())
                and 'experiments/r26_impulse_onset/candidate.py' in j['candidate_pins']):
            raise ValueError('C2 not admitted; do not open this panel')
    p=read(HERE/'plan.json')
    if kind=='paired':
        jobs=[(c,v,arm,p['paired_adjudication']) for c in p['paired_c1_and_c2'] for v in p['variants']]
        with ProcessPoolExecutor(workers) as pool: rows=list(pool.map(paired_job,jobs))
        counts={}
        for name in ('coupled',arm):
            ev=[x for r in rows for x in r['records'][name]['events'].values()]
            tw=[x for r in rows for x in r['records'][name]['twins'].values()]
            counts[name]={'strict_events':sum(x['strict'] for x in ev),'slots':len(ev),
                'both_second':sum(r['records'][name]['events']['both_second']['strict']for r in rows),
                'rows':len(rows),'duplicate_edges':sum(x['duplicate_rises'] for x in ev),
                'out_of_window_false_rises':sum(len(x['false_times_s'])for x in tw),
                'genuine_tool_hits':sum(x['tool']['detected']for x in tw)}
    else:
        if arm!='c2':raise ValueError('reserved separate singles require admitted C2')
        jobs=[({'id':f'onset_check_{i}','mass':m,'com':c,'offset':o,'mechanics':{'table_solref_time_scale':s}},
               [{'id':f'force_{j}','time_s':t,'force_n':f} for j,f in enumerate(p['c2_forces'])])
              for i,(m,c,o,s,t) in enumerate(p['c2_singles'])]
        with ProcessPoolExecutor(workers) as pool:rows=list(pool.map(singles_case,jobs))
        counts={a:{'attributed':sum(r['comparisons'][a]['strict_genuine'] for r in rows),
                   'slots':sum(r['comparisons'][a]['total_impact_slots']for r in rows),
                   'all_guards':all(r['comparisons'][a]['guards_passed']for r in rows)}for a in ('pair','c2')}
    return {'schema':'r26-control-detector-paired-stress.v1','kind':kind,'arm':arm,'counts':counts,
            'episodes':sum(r['episodes']for r in rows),'rows':rows,'promotable':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=('paired','singles'),required=True)
    p.add_argument('--arm',choices=('c1','c2'),required=True);p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--admission',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():p.error('fresh output path required')
    j=run(a.kind,a.arm,a.workers,a.admission)
    with a.out.open('x')as f:json.dump(j,f,allow_nan=False);f.write('\n')
    print(j['counts'])
