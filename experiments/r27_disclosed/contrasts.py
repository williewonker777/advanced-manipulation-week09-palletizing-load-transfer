"""One real no-force twin and four signed forces per pose/mechanics; no thresholds are fit."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.r26_holdout import evaluator as lab
from experiments.r26_yaw_acquisition.candidate import Detector as Baseline

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = 'experiments.r26_yaw_acquisition.candidate:Detector'
LABELS = ('registered', 'reverse', 'perpendicular_xy', 'perpendicular_tilt')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(arm):
    if arm == BASE:
        return Baseline
    mod, cls = arm.split(':')
    if (not mod.startswith(('experiments.r27_load_change.', 'experiments.r27_loaded_observer.'))
            or cls != 'Detector' or not all(token.isidentifier() for token in mod.split('.'))):
        raise ValueError('explicit opt-in loaded observer required')
    return getattr(importlib.import_module(mod), cls)


def inputs(impact):
    t, duration, v = impact
    v = np.asarray(v, float)
    if v.shape != (3,) or not np.isfinite(v).all() or np.hypot(*v[:2]) == 0:
        raise ValueError('nonzero horizontal component required for signed basis')
    xy = np.array([-v[1], v[0], 0.])
    xy *= np.linalg.norm(v) / np.linalg.norm(xy)
    tilt = np.cross(v, xy) / np.linalg.norm(v)
    return {'registered': impact, 'reverse': [t,duration,(-v).tolist()],
            'perpendicular_xy': [t,duration,xy.tolist()], 'perpendicular_tilt': [t,duration,tilt.tolist()]}


def acquisition_in_force(event, boundary):
    return bool(event['strict'] and boundary['first_nonzero_index'] <= event['attributable_rise_indices'][0] <= boundary['last_nonzero_index'])


def capture(job):
    case, mech, arm, expected, expected_sig, plan = job
    c = resolve(arm)
    _, zero = lab.run_episode(case, mech, c, (None,None))
    normal = lab.common(zero,(None,None),[],plan)
    force_inputs = inputs(case['impact'])
    measured = {}
    for label in LABELS:
        impact = force_inputs[label]
        _, log = lab.run_episode(case, mech, c, (impact,None))
        start, boundary = lab.actual_start(log['t'],impact)
        event = lab.event(log,[zero],start,min(6000,start+plan['event_window_ms']),plan)
        common = lab.common(log,(impact,None),[start],plan)
        prefix = lab.certified.prefix(zero,log,start,start)
        physics = lab.certified.trace_signature({'none':zero,'single':log})
        if not (prefix['passed'] and normal['force_exact']['passed'] and common['force_exact']['passed']):
            raise ValueError('invalid input or preforce twin')
        if label == 'registered':
            if physics != expected_sig or boundary != expected['boundary']:
                raise ValueError('baseline R26 physics did not reproduce')
            if arm == BASE and not (event == expected['event'] and normal == expected['twins']['none']
                                    and common == expected['twins']['single'] and prefix == expected['prefix']):
                raise ValueError('registered R26 reports differ')
        measured[label] = dict(impact=impact,boundary=boundary,event=event,in_force=acquisition_in_force(event,boundary),
                               common=common,physical_signature=physics,prefix=prefix)
    return dict(case=case['id'],mechanics=mech['kind'],normal=normal,forced=measured,episodes=5)


def compare(rows, before):
    if len(rows) != len(before):
        raise ValueError('mismatched comparison')
    reg=[]; gains=[]; first_in_force=[]; baseline_in_force=[]
    for row, old in zip(rows,before):
        key=[row['case'],row['mechanics']]
        if key != [old['case'],old['mechanics']]:
            raise ValueError('mismatched row order')
        for label in [*LABELS,'none']:
            if label=='none':
                common=row['normal']; anchor=old['normal']
            else:
                cur,prev=row['forced'][label],old['forced'][label]
                if cur['physical_signature'] != prev['physical_signature'] or not (cur['prefix']['passed'] and prev['prefix']['passed']):
                    raise ValueError('change in physical traces or prefix')
                e,ref=cur['event'],prev['event']
                floor=lab.preserved(ref,e,{'preservation_slack_ms':5,'duration_cost_slack_samples':5})
                checks={**floor,'duplicated':e['duplicate_rises'] <= ref['duplicate_rises'],
                        'quiet':not ref['quiet_before'] or e['quiet_before']}
                reg.extend([*key,label,k] for k,ok in checks.items() if not ok)
                if e['strict'] and not ref['strict']:
                    gains.append([*key,label,e['delays_ms'][0]])
                if cur['in_force']:first_in_force.append([*key,label])
                if prev['in_force']:baseline_in_force.append([*key,label])
                common,anchor=cur['common'],prev['common']
            if common['false_times_s'] != anchor['false_times_s']:
                # Changes reducing false output are not a loss; timestamps must match one-to-one within one native tick.
                allowed=list(anchor['false_times_s'])
                for rise in common['false_times_s']:
                    match=next((i for i,v in enumerate(allowed) if abs(v-rise)<.001000001),None)
                    if match is None:reg.append([*key,label,'new_false_rise']);break
                    allowed.pop(match)
            if not lab.tool_floor(anchor['tool'],common['tool'],5) or (common['tool']['prealarm'] and not anchor['tool']['prealarm']):
                reg.append([*key,label,'tool'])
            if common['stuck_tail'] and not anchor['stuck_tail']:reg.append([*key,label,'stuck_tail'])
    return dict(regressions=reg,strict_gains=gains,in_force=first_in_force,baseline_in_force=baseline_in_force)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',default=BASE)
    p.add_argument('--baseline',type=Path);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(1,2),default=2);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    if os.path.lexists(a.out):p.error('fresh file required')
    reg=json.loads((HERE/'policy.json').read_bytes()); arm=a.arm;resolve(arm)
    baseline_source=ROOT/'experiments/r26_yaw_acquisition/candidate.py'
    if sha(a.archive)!=reg['prior_archive_sha256'] or sha(baseline_source)!=reg['baseline_sha256']:
        raise ValueError('immutable original baseline and previous archive required')
    old=json.loads(a.archive.read_bytes());old_rows={(r['id'],r['variant']):r for r in old['rows']}
    plan=json.loads(lab.PLAN.read_bytes()); specs=reg['disclosed_contrasts']; tasks=[]
    for ident in specs['cases']:
        case=next(r for r in plan['single'] if r['id']==ident)
        for m in specs['mechanics']:
            ex=old_rows[ident,m];tasks.append((case,plan['variants'][m],arm,ex['records']['r26'],ex['r25_physics_sha256'],plan))
    if arm!=BASE and not a.baseline:raise ValueError('authentic baseline contrast required')
    with ProcessPoolExecutor(a.workers) as pool:rows=list(pool.map(capture,tasks))
    anchor=rows
    if a.baseline:
        earlier=json.loads(a.baseline.read_bytes())
        if earlier['arm']!=BASE or earlier['archive_sha256']!=sha(a.archive) or earlier['registration_sha256']!=sha(HERE/'policy.json'):
            raise ValueError('wrong contrast baseline')
        anchor=earlier['rows']
    report={'schema':'r27-physical-signed-contrast.v1','disclosed':True,'arm':arm,'baseline_reference_sha256':sha(a.baseline) if a.baseline else None,
            'registration_sha256':sha(HERE/'policy.json'),'archive_sha256':sha(a.archive),'episodes':sum(r['episodes'] for r in rows),
            'strict':sum(e['event']['strict'] for r in rows for e in r['forced'].values()),
            'normals':len(rows),'force_slots':sum(len(r['forced']) for r in rows),'comparison':compare(rows,anchor),'rows':rows}
    with a.out.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(report['arm'],report['episodes'],report['strict'],report['comparison']['strict_gains'],report['comparison']['regressions'])


if __name__=='__main__':
    main()
