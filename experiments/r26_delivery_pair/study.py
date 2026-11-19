"""Run registered systems on matched full original-physics landing episodes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from experiments.r14_stock_handoff.detector import Detector as Coupled
from experiments.r14_stock_handoff.qualification import compact
from experiments.r19_load_transfer.controller import Controller as Informed
from experiments.r24_transfer_replay import evaluator as replay
from experiments.r26_delivery_pair.candidate import Detector as Pair
from experiments.r26_delivery_pair.controller import Controller
from experiments.r26_delivery_pair.probe import HERE, ORIGINAL, cases, run
from tools.r11_placement import score
import r1_torque


def factory(key):
    if key == 'r10':
        # The historical R24 landing study compared controllers under one
        # coupled detector. This is its matched comparator, NOT the R10 default.
        return Coupled, r1_torque.Controller
    if key == 'r19':
        cal = replay.read(ORIGINAL/'calibration.json')
        return Coupled, lambda: Informed(cal['height'], cal['information'])
    if key == 'r24':
        return Coupled, Controller
    if key == 'r25':
        from experiments.r25_composed.detector import Detector
        return Detector, Controller
    if key == 'xy':
        from experiments.r26_delivery_pair.horizontal import Detector
        return Detector, Controller
    if key == 'pair':
        return Pair, Controller
    raise ValueError('expected a registered laboratory system')


def samples(log, interval):
    return int(np.count_nonzero(np.asarray(log['detect'], bool)[interval]))


def tool_delay_ok(old, new):
    return old is None or new is not None and new <= old+5.00000001


def measures(metrics, log, ctrl, mass):
    data=compact(metrics,log,ctrl,mass)
    # Runtime timestamps are not reproducible physical results.
    data.pop('ctrl_p99_us')
    if data.get('registration'):
        data['registration'].pop('stream_timing_us',None)
    return data


def bundle(case, impacts, arm):
    cls, control = factory(arm)
    _, ctrl, m, twin = run(case,cls,control=control)
    normal=measures(m,twin,ctrl,case['mass'])
    normal.update(physical_sha256=replay.physical_signature(twin),rises_s=replay.rise_times(twin),
                  active_samples=samples(twin,slice(None)))
    rows={}
    plan=replay.read(ORIGINAL/'plan.json')
    for spec in impacts:
        inp=[spec['time_s'],spec['force_n']]
        _, ctr, m, log = run(case,cls,inp,control)
        rec=replay.force_assessment(log,twin,inp,plan)
        t=np.asarray(log['t']); interval=(t>inp[0])&(t<inp[0]+plan['event_window_s'])
        rec.update(active_samples=samples(log,interval),outside_active_samples=samples(log,~interval),
                   duplicates=max(0,len(rec['genuine_rises_s'])-1),
                   prealarm=bool(np.any(np.asarray(log['detect'])[(t>=inp[0]-.020)&(t<=inp[0])])),
                   tail=bool(np.any(np.asarray(log['detect'])[-20:])),
                   detect_prefix_exact=bool(np.array_equal(np.asarray(twin['detect'])[t<=inp[0]],
                                                           np.asarray(log['detect'])[t<=inp[0]])),
                   tool_delay_ms=m.get('tool_contact_delay_ms'),official_false_s=m['false_alarm_times'],
                   physical_sha256=replay.physical_signature(log),
                   first_committed_nonzero_s=float(t[np.flatnonzero(np.linalg.norm(log['signed_force'],axis=1))[0]]))
        rows[spec['id']]=rec
    return {'normal':normal,'impacts':rows,'episodes':1+len(rows)}


def gate_impact(ref, c):
    old=ref['first_genuine_delay_ms']
    now=c['first_genuine_delay_ms']
    strict=old is not None
    return {'attributable_nonloss':not strict or now is not None,
            'delay_within_5ms':not strict or now is not None and now<=old+5.00000001,
            'active_within_5_samples':not strict or c['active_samples']<=ref['active_samples']+5,
            'no_new_duplicates':c['duplicates']<=ref['duplicates'],
            'no_added_official_false':replay.all_matched(c['official_false_s'],ref['official_false_s'],2.),
            'no_new_prealarm':not c['prealarm'] or ref['prealarm'],
            'no_new_tail':not c['tail'] or ref['tail'],
            'no_new_tool_loss':tool_delay_ok(ref['tool_delay_ms'],c['tool_delay_ms']),
            'force_and_prefix_exact':c['signed_force_exact'] and c['force_free_prefix_exact'] and c['detect_prefix_exact']}


def result(case, impacts, arms):
    data={'id':case['id'],'case':case,'signed_inputs':impacts,'systems':arms,
          'episodes':sum(x['episodes'] for x in arms.values()),'comparisons':{}}
    base=arms.get('r24')
    if not base:
        return data
    for name,value in arms.items():
        if name in ('r10','r19','r24'):
            continue
        n=value['normal']; ref=base['normal']
        g={'normal_same_mechanics_and_trajectory':ref['physical_sha256']==n['physical_sha256'],
           'all_deterministic_costs_and_saturation_equal':n['cost']==ref['cost'] and n['saturation_pct']==ref['saturation_pct'],
           'no_extra_normal_false':replay.all_matched(n['false_alarm_times'],ref['false_alarm_times'],2.),
           'normal_alarm_cost_not_higher':n['active_samples']<=ref['active_samples'],
           'normal_tool_nonloss':tool_delay_ok(ref['tool_delay_ms'],n['tool_delay_ms']),
           'same_support_and_physical_status':n.get('support')==ref.get('support') and n['status']==ref['status']=='ok'}
        if 'expected' in case:
            g['earlier_R24_physics_and_false_times_reproduce']=(
                    ref['physical_sha256']==case['expected']['r24_physical_sha256']
                    and ref['false_alarm_times']==case['expected']['r24_false'])
        floors={}; g4=None
        if 'r10' in arms:
            floor=arms['r10']['normal']
            g4={'candidate':score(floor,n),'r24':score(floor,ref)}
            g['g4_same_as_r24']=g4['candidate']==g4['r24']
        for b in ('r10','r19'):
            if b in arms:
                r=arms[b]['normal']
                floors[b]={'cost_ratios':replay.max_ratios(n,r),'saturation_delta':n['saturation_pct']-r['saturation_pct'],
                           'tool_delay_ms':r['tool_delay_ms'],'official_false_s':r['false_alarm_times']}
        second={}
        for k,c in value['impacts'].items():
            checks={}
            for b in ('r10','r19','r24'):
                if b not in arms:continue
                r=arms[b]['impacts'][k]; s=gate_impact(r,c)
                if b=='r24':
                    s.update(same_physics=c['physical_sha256']==r['physical_sha256'],
                             outside_occupancy_no_worse=c['outside_active_samples']<=r['outside_active_samples'],
                             outside_rises_no_added=replay.all_matched(c['forced_false_rises_s'],r['forced_false_rises_s'],2.))
                checks[b]=s
            second[k]=checks
        data['comparisons'][name]={'normal':g,'g4':g4,'control_floors':floors,'impact_gates':second,
                'guards_passed':all(g.values()) and all(all(v.values()) for s in second.values() for v in s.values()),
                'strict_genuine':sum(bool(r['genuine_rises_s']) for r in value['impacts'].values()),
                'total_impact_slots':len(impacts)}
    return data


def evaluate(case,impacts,arms):
    """One matched no-input twin per arm shared across its signed-input episodes."""
    if not arms or len(arms)!=len(set(arms)) or len({i['id'] for i in impacts})!=len(impacts):
        raise ValueError('unique named arms and signed inputs required')
    return result(case,impacts,{arm:bundle(case,impacts,arm) for arm in arms})


def panel(which):
    plan=replay.read(HERE/'plan.json'); old=replay.read(ORIGINAL/'plan.json')
    if which=='target':
        return [(dict(old['impulse_base'],id='known_late'),old['late_impulses'])]
    if which=='normal':
        allcases,parts=cases()
        return [(allcases[key],[]) for key in parts['fit']+parts['audit']]
    if which=='generalization':
        general=plan['separate_generalization']
        return [({'id':f'separate_{i}','mass':m,'com':c,'offset':o,'mechanics':{'table_solref_time_scale':s}},
                 [{'id':f'force_{j}','force_n':f,'time_s':t} for j,f in enumerate(general['forces'])])
                for i,(m,c,o,s,t) in enumerate(general['cases'])]
    raise ValueError(which)


def job(x):
    c,im,a=x
    return c['id'],a,bundle(c,im,a)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',choices=('target','normal','generalization'),required=True)
    p.add_argument('--arms',nargs='+',choices=('r10','r19','r24','xy','r25','pair'),required=True)
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists() or args.out.is_symlink() or len(set(args.arms))!=len(args.arms):
        p.error('fresh output and unique arms required')
    panel_cases=panel(args.panel)
    jobs=[(case,im,a) for case,im in panel_cases for a in args.arms]
    ar={}
    with ProcessPoolExecutor(args.workers) as pool:
        for k,a,pack in pool.map(job,jobs):
            ar.setdefault(k,{})[a]=pack
    rows=[result(c,im,ar[c['id']])for c,im in panel_cases]
    doc={'schema':'r26-combined-original-physics-landing-study.v1','panel':args.panel,'detector_input':'public sensors only',
         'classification':'Modified laboratory controllers/detectors, not the course original demo or a hardware qualification.',
         'rows':rows,'episodes':sum(x['episodes']for x in rows),'promotion_eligible':False}
    with args.out.open('x')as stream:json.dump(doc,stream,allow_nan=False,indent=2);stream.write('\n')
    for row in rows:
        print(row['id'],[(a,v['guards_passed'],v['strict_genuine'],v['total_impact_slots'])
                         for a,v in row['comparisons'].items()])
    print('episodes',doc['episodes'])


if __name__=='__main__':
    main()
