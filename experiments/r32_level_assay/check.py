"""Run authenticated historical physics; save each complete case before cross-case attribution."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from experiments.r22_census import check
from experiments.r31_event_replay.runner import audit
from tools.r31_notice_duration import primary

ROOT=Path(__file__).resolve().parents[2]
OLD='7070f5f1651e07faa8701f28421082b7eaf32a16cb866dbf8e5d9ffa07018d43'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def get(f):return json.loads(f.read_bytes())
def save(f,obj):
    part=f.with_suffix('.tmp')
    with part.open('x')as fp:json.dump(obj,fp,separators=(',',':'),sort_keys=True,allow_nan=False);fp.write('\n')
    if f.exists():raise FileExistsError(str(f))
    part.rename(f)
def resolve(entry,digest):
    mod,kind=entry.split(':');parts=mod.split('.')
    if len(parts)!=3 or parts[0]!='experiments' or not parts[1].startswith('r32_') or kind!='Detector' or not all(n.isidentifier() for n in parts):
        raise ValueError('invalid registered source')
    f=ROOT/Path(*parts).with_suffix('.py')
    if sha(f)!=digest or get(f.parent/'protocol.json')['candidate_sha256']!=digest:raise ValueError('frozen source mismatch')
    return getattr(importlib.import_module(mod),kind)
def one(job):
    c,entry,digest,old,out=job;base=resolve(entry,digest)
    class P(base):
        def reset(self):
            super().reset();self.event_bits=[];self.start_bits=[];self.out_bits=[];self.load_bits=[];self.native_bits=[];P.instance=self
        def update(self,*args):
            result=super().update(*args)
            flag=getattr(self,'decision',None)
            self.out_bits.append(bool(result[1]))
            if flag is not None:
                self.event_bits.append(bool(flag.event_output));self.start_bits.append(bool(flag.event_started))
                if flag.alarm_level!=result[1]:raise ValueError('public output differs from decision')
                if hasattr(flag,'linked_native'):
                    self.load_bits.append(bool(flag.measured_load));self.native_bits.append(bool(flag.linked_native))
                    if bool(flag.measured_load or flag.linked_native or flag.event_output)!=bool(result[1]):
                        raise ValueError('Boolean must follow sample provenance')
            return result
    rec=check.old.episode_record(P,c);prof=P.instance;i=c['case_id']
    if rec['physical_sha256']!=old['physics_sha256'] or rec['detect_intervals']!=check.old.intervals(prof.out_bits):
        raise ValueError('not an authenticated passive replay')
    side=None
    if prof.event_bits:
        side=check.old.intervals(prof.event_bits)
        if side!=old['arms']['identity']['alarm_intervals'] or [s for s,_ in side]!=np.flatnonzero(prof.start_bits).tolist():
            raise ValueError('separate event channel is not R31')
    channels={}
    if prof.load_bits:
        channels=dict(measured_load=check.old.intervals(prof.load_bits),linked_native=check.old.intervals(prof.native_bits))
    save(out/f'{i:03d}.json',dict(id=i,entry=entry,source=digest,rec=rec,side=side,channels=channels))
    return i
def occupied(spans,lo,hi):
    return 0 if lo is None else sum(max(0,min(b,hi+1)-max(a,lo)) for a,b in spans)
def cell(c,rec,ref):
    s=check.score(c,rec,ref)
    if not all(s['valid'].values()):raise ValueError('invalid force, grid or causal prefix')
    return dict(score=s,alarm_intervals=rec['detect_intervals'],first_tool_rise=min(rec['tool_rise_indices'],default=None))
def describe(obj,lo,hi,on,off,stamp):
    w=primary(obj,stamp) if stamp is not None else None
    result={**obj,'active_samples':occupied(obj['alarm_intervals'],lo,hi),'width_ms':w['width_ms'] if w else None}
    if on is not None:
        inside=occupied(obj['alarm_intervals'],on,off)
        result.update(actual_force_samples=off-on+1,overlap_with_real_input=inside,
                      after_externally_applied_force=occupied(obj['alarm_intervals'],off+1,hi))
    return result
def d4(a):
    if sha(a.previous)!=OLD or not a.rows.is_dir():raise ValueError('authentic prior and fresh raw folder required')
    resolve(a.entry,a.sha)
    prior=get(a.previous);cases=check.build_cases('my_torque')
    if len(cases)!=385 or len(prior['rows'])!=385 or hashlib.sha256(check.old.canonical_cases(cases)).hexdigest()!=check.MATRIX:raise ValueError('bad matrix')
    if not a.score_only:
        if list(a.rows.iterdir()):raise ValueError('would overwrite previous rows')
        with ProcessPoolExecutor(a.jobs) as pool:
            completed=list(pool.map(one,[(c,a.entry,a.sha,old,a.rows) for c,old in zip(cases,prior['rows'],strict=True)]))
            if completed!=list(range(385)):raise ValueError('missing physical execution')
    rs={k:get(a.rows/f'{k:03d}.json') for k in range(385)}
    if any(r['id']!=k for k,r in rs.items()):raise ValueError('wrong row index')
    if any(r.get('source')!=a.sha or r.get('entry')!=a.entry for r in rs.values()):raise ValueError('unauthenticated raw candidate identity')
    normal={check.old.reference_key(c):c['case_id'] for c in cases if c['kind']=='reference'}
    rows=[];flags={n:[] for n in ('c3','headroom','identity')};width={n:[] for n in flags}
    compare_counts={}; gains=[]
    for c,old in zip(cases,prior['rows'],strict=True):
        i=c['case_id'];r=rs[i]['rec'];k=normal[check.old.reference_key(c)] if c['kind']=='collision' else i
        if r['physical_sha256']!=old['physics_sha256']:raise ValueError('mismatched physical record')
        g=cell(c,r,rs[k]['rec'])
        args=(r['true_window_first_index'],r['true_window_last_index'],r['force_first_nonzero_index'],r['force_last_nonzero_index'],
              None if c['collision'] is None else c['collision'][0])
        arms={n:describe(cell_,*args) for n,cell_ in {**old['arms'],'new':g}.items()}
        for n in flags:
            fail=audit(arms['new'],arms[n])
            if fail:flags[n].append(dict(id=i,failed=fail))
            neww,oldw=arms['new']['width_ms'],arms[n]['width_ms']
            if oldw is not None and (neww is None or neww+5<oldw):width[n].append(i)
        if arms['identity']['width_ms']==4 and arms['new']['width_ms'] is not None and arms['new']['width_ms']>4:gains.append(i)
        rows.append(dict(id=i,case=c,arms=arms,channels=rs[i]['channels']))
    for n in ('c3','headroom','identity','new'):
        arm=[r['arms'][n] for r in rows]
        compare_counts[n]=dict(strict=sum(v['score']['strict'] for v in arm), duplicates=sum(v['score']['duplicate'] for v in arm),
             false_rises=sum(len(v['score']['false']) for v in arm), four_ms=sum(v['width_ms']==4 for v in arm),
             raw_duplicate=sum(v['score']['raw_duplicate'] for v in arm), physically_overlapping_samples=sum(v.get('overlap_with_real_input',0) for v in arm))
    return dict(schema='r32-d4-physical-level-audit.v1',candidate=a.entry,candidate_sha256=a.sha,prior_sha256=OLD,
          unique_physical_conditions=385,episodes=385,matched_force_physics_prefix=385,
          parent_event_channel_exact=sum(rs[i]['side'] is not None for i in rs),
          counts=compare_counts,upper_and_existing_guard_failures=flags,minimum_primary_width_failures=width,
          prior_four_ms_lengthened=gains,rows=rows)
def r26_one(job):
    from experiments.r26_compare.legacy import fork
    from experiments.r29_force_tube import known
    return fork(known.one,resolve=resolve)(job)
def guard_one(job):
    from experiments.r26_compare.legacy import fork
    from experiments.r29_force_tube import panels
    return fork(panels.one,resolve=resolve)(job)
def main():
    cmd=sys.argv[1];argv=sys.argv[2:]
    if cmd in ('r26','guards','cross','r25'):
        from experiments.r26_compare.legacy import fork
    if cmd=='r26':
        from experiments.r29_force_tube import known
        return fork(known.main,resolve=resolve,one=r26_one)(argv)
    if cmd=='guards':
        from experiments.r29_force_tube import panels
        return fork(panels.main,resolve=resolve,one=guard_one)(argv)
    if cmd in ('cross','r25'):
        from experiments.r30_source_association import history
        sys.argv=[sys.argv[0],'--panel',cmd,*argv];return fork(history.main,resolve=resolve)()
    if cmd!='d4':raise ValueError('invalid action')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--entry',required=True);p.add_argument('--sha',required=True)
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--jobs',type=int,choices=range(1,9),default=4)
    p.add_argument('--score-only',action='store_true');a=p.parse_args(argv)
    if a.out.exists() or a.out.is_symlink():p.error('new report required')
    result=d4(a);save(a.out,result)
    print(result['counts'],{k:len(v) for k,v in result['upper_and_existing_guard_failures'].items()},
          {k:len(v) for k,v in result['minimum_primary_width_failures'].items()})
if __name__=='__main__':main()
