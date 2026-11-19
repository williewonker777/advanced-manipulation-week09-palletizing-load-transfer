"""Authenticate R33 public simulation rows against the frozen R31 input and physical hashes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
from pathlib import Path
import sys
from experiments.r22_census import check as census
from experiments.r32_level_assay import check as level
from experiments.r26_compare.legacy import fork
from experiments.r31_event_replay.runner import audit

ROOT=Path(__file__).resolve().parents[2]
def resolve(entry,digest):
    mod,kind=entry.split(':');parts=mod.split('.')
    if len(parts)!=3 or parts[0]!='experiments' or not parts[1].startswith('r33_') or parts[2]!='candidate' or kind!='Detector' or not all(n.isidentifier() for n in parts):
        raise ValueError('invalid R33 historical entry')
    src=ROOT/Path(*parts).with_suffix('.py');protocol=level.get(src.parent/'protocol.json')
    if level.sha(src)!=digest or protocol['candidate_sha256']!=digest:raise ValueError('unfrozen R33 source')
    for name,pin in protocol.get('pins',{}).items():
        f=src.parent/name
        if f.resolve().parent!=src.parent.resolve() or level.sha(f)!=pin:raise ValueError('changed R33 dependency')
    return getattr(importlib.import_module(mod),kind)
def one(job):
    c,entry,digest,old,out=job;detector=resolve(entry,digest)
    rec=census.old.episode_record(detector,c);i=c['case_id']
    if rec['physical_sha256']!=old['physics_sha256']:raise ValueError('passive detector changed plant')
    level.save(out/f'{i:03d}.json',dict(id=i,entry=entry,source=digest,rec=rec,side=None,channels={}))
    return i
def d4(a):
    if level.sha(a.previous)!=level.OLD:raise ValueError('unauthenticated R31 reference')
    cases=census.build_cases('my_torque')
    import hashlib
    if hashlib.sha256(census.old.canonical_cases(cases)).hexdigest()!=census.MATRIX:raise ValueError('changed D4')
    cs={c['case_id']:c for c in cases}; refs={census.old.reference_key(c):c['case_id'] for c in cases if c['kind']=='reference'}
    subset=set(a.ids or cs)
    if not subset<=set(cs):raise ValueError('unknown case')
    subset|={refs[census.old.reference_key(cs[i])] for i in subset if cs[i]['kind']=='collision'}
    prior=level.get(a.previous)['rows']
    if not a.rows.is_dir() or list(a.rows.iterdir()):raise ValueError('new rows directory required')
    resolve(a.entry,a.sha)
    with ProcessPoolExecutor(a.jobs) as pool:
        completed=list(pool.map(one,[(cs[i],a.entry,a.sha,prior[i],a.rows) for i in sorted(subset)]))
    if completed!=sorted(subset):raise ValueError('missing physical run')
    rs={i:level.get(a.rows/f'{i:03d}.json')['rec'] for i in subset}
    rows=[]
    for i in sorted(subset):
        c=cs[i];rec=rs[i];k=refs[census.old.reference_key(c)] if c['kind']=='collision' else i
        cell=level.cell(c,rec,rs[k]);args=(rec['true_window_first_index'],rec['true_window_last_index'],
            rec['force_first_nonzero_index'],rec['force_last_nonzero_index'],c['collision'][0] if c['collision'] else None)
        new=level.describe(cell,*args);old=level.describe(prior[i]['arms']['identity'],*args)
        s=new['score'];first=s['delay_ms'] if s['strict'] else None
        tick=round(c['collision'][0]*1000)+first if first is not None else None
        on,off=args[2:4]
        rows.append(dict(id=i,case=c,reference=k,physically_verified=True,old=old,new=new,
            in_force_first=tick is not None and on<=tick<=off,fail=audit(new,old)))
    counts={n:dict(strict=sum(r[n]['score']['strict'] for r in rows),duplicates=sum(r[n]['score']['duplicate'] for r in rows),
        false_rises=sum(len(r[n]['score']['false']) for r in rows)) for n in ('old','new')}
    return dict(schema='r33-authenticated-axial-events.v1',entry=a.entry,candidate_sha256=a.sha,previous=level.OLD,
        actual_episodes=len(rows),counts=counts,gains=[r['id'] for r in rows if r['new']['score']['strict'] and not r['old']['score']['strict']],
        losses=[r['id'] for r in rows if not r['new']['score']['strict'] and r['old']['score']['strict']],
        failures=[(r['id'],r['fail']) for r in rows if r['fail']],rows=rows)
def r26_one(job):
    from experiments.r29_force_tube import known
    return fork(known.one,resolve=resolve)(job)
def guard_one(job):
    from experiments.r29_force_tube import panels
    return fork(panels.one,resolve=resolve)(job)
def main():
    cmd=sys.argv[1];argv=sys.argv[2:]
    if cmd=='r26':
        from experiments.r29_force_tube import known
        return fork(known.main,one=r26_one,resolve=resolve)(argv)
    if cmd=='guards':
        from experiments.r29_force_tube import panels
        return fork(panels.main,one=guard_one,resolve=resolve)(argv)
    if cmd in ('cross','r25'):
        from experiments.r30_source_association import history
        sys.argv=[sys.argv[0],'--panel',cmd,*argv];return fork(history.main,resolve=resolve)()
    if cmd!='d4':raise ValueError('unknown historical panel')
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);p.add_argument('--sha',required=True)
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--jobs',type=int,choices=range(1,9),default=4)
    p.add_argument('--ids',nargs='+',type=int);a=p.parse_args(argv)
    if a.out.exists():p.error('new report required')
    s=d4(a);level.save(a.out,s);print(s['actual_episodes'],s['counts'],s['gains'],s['losses'],s['failures'])
if __name__=='__main__':main()
