"""Necessary lower bound on occupied samples for one row, independent of any new detector."""
import argparse
from bisect import bisect_left
import hashlib
import json
from pathlib import Path
from tools.r31_notice_duration import GRID,primary

def lower_in_window(required_width, latest_start, window_stop):
    return min(required_width,max(0,window_stop-latest_start))


def analyze(old):
    conflicts={};rows=[]
    for r in old['rows']:
        c=r['input']
        if c['kind']!='collision':continue
        start=c['collision'][0];on=bisect_left(GRID,start);end=bisect_left(GRID,start+.32)
        src=r['arms'];base=primary(src['c3'],start)
        if base is None:continue
        for name in ('headroom','identity'):
            w=primary(src[name],start)
            if w is None:continue
            latest=w['start_tick']+5;req=max(0,base['width_ms']-5)
            old_occupied=sum(max(0,min(b,end)-max(s,on)) for s,b in src[name]['alarm_intervals'])
            budget=old_occupied+5
            low=lower_in_window(req,latest,end)
            if low>budget:
                rows.append(dict(id=r['id'],anchor=name,c3_width_ms=base['width_ms'],required_width_ms=req,
                     latest_allowed_tick=latest,window_stop_exclusive=end,minimum_occupied=low,maximum_occupied=budget,
                     impossible_by_samples=low-budget))
                conflicts.setdefault(name,[]).append(r['id'])
    return dict(schema='r32-single-boolean-constraint-lower-bound.v1',scope='Mathematical feasibility, not another replay.',
       constraints=['first reported interval at least C3 primary width minus 5 ms','first retained onset no later than anchor onset plus 5 ms',
                    'total on time within the existing 320 ms attribution window at most the anchor total plus 5 samples'],
       method='An interval starting no later than latest_start uses at least min(required_width, window_stop-latest_start) in-window samples. Any earlier start at or after forcing makes the lower bound no smaller.',
       impossible=conflicts,rows=rows)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();raw=a.previous.read_bytes();h=hashlib.sha256(raw).hexdigest()
    if h!='7070f5f1651e07faa8701f28421082b7eaf32a16cb866dbf8e5d9ffa07018d43' or a.out.exists() or a.out.is_symlink():p.error('authenticated full matrix and fresh output required')
    s=analyze(json.loads(raw));s['input_sha256']=h
    with a.out.open('x')as f:json.dump(s,f,sort_keys=True,indent=2);f.write('\n')
    print({k:len(v) for k,v in s['impossible'].items()},next(row for row in s['rows'] if row['id']==266 and row['anchor']=='identity'))
