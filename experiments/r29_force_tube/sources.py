"""Disclosed R26 reference inputs and normal-only splits, separate from any R27/R28 fresh case."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(n):return json.loads((ROOT/n).read_bytes())
def inputs():
    p=read('experiments/r26_holdout/plan.json')
    r=read('experiments/r27_load_change/registration.json')
    s=read('experiments/r27_load_change/settled_registration.json')
    m=read('experiments/r27_load_change/manifold_registration.json')
    groups={'fit':r['fit_normals'],'cal':m['extra_cal_normals'],'audit15':r['audit_normals'],'audit18':s['additional_future_normals']}
    rows=[]
    for split,arr in groups.items():
        for c in arr:
            for mech in p['variants'].values():rows.append((split,c,mech,'none',(None,None)))
    for c in p['paired']:
        a,b=c['impacts']
        for mech in p['variants'].values():
            for twin,imp in {'none':(None,None),'first':(a,None),'second':(None,b),'both':(a,b),'long':(c['long'],None)}.items():
                rows.append(('paired',c,mech,twin,imp))
    for c in p['single']:
        for mech in p['variants'].values():
            for twin,imp in [('none',(None,None)),('single',(c['impact'],None))]:rows.append(('single',c,mech,twin,imp))
    return rows
