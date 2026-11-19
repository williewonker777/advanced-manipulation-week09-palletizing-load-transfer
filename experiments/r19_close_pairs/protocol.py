"""Run four controlled twins inside an authenticated, isolated course-kit overlay."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.r16_payload_contract.controller import Controller as D
from experiments.r18_paired_impulses import runner, scoring
from experiments.r18_single_pass import paired_replay as replay
from experiments.r19_close_pairs import assay
from experiments.r19_event_state.physical_detector import Detector
from tools.r14_paired_events import match_false_events, preserve_tool
from tools.r5_metrics import event_result

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('r10','raw_r15','D_coupled','single_pass','event')
PARENT = 'single_pass'
MODEL = 'results/r17/release_observation_model.json'
PLAN = 'experiments/r19_close_pairs/plan.json'
REF = 'results/r18/paired_reference.json'


def read(p):
    return json.loads(Path(p).read_bytes())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, record):
    p = Path(p)
    with p.open('x') as out:
        json.dump(record,out,indent=2,allow_nan=False)
        out.write('\n')


def episode(arm, case, height, early, impacts):
    if arm != 'event':
        return replay.episode(arm,case,height,early,MODEL,impacts)
    detect = Detector()
    detect.model_path = Path(MODEL)
    metrics, log = runner.run_episode(D(height,early),case['mass'],tuple(case['com']),detect,
                                      pick_offset=case['offset'],impacts=impacts)
    log['r18_diverged'] = bool(metrics['diverged'])
    if not scoring.finite_grid(log):
        raise ValueError('incomplete or nonfinite candidate trajectory')
    return log


def normal_scores(logs):
    records = {k:event_result(v['t'],v['detect'],{},v['f_tool']) for k,v in logs.items()}
    parent, trial = logs[PARENT], logs['event']
    g = {'complete_finite':all(scoring.finite_grid(v) for v in logs.values()),
         'no_applied_force':all(not np.any(v['f_ext_xyz_committed']) for v in logs.values()),
         'R18_R19_D_physical_parity':all(np.array_equal(trial[n],logs[k][n])
             for k in (PARENT,'D_coupled') for n in replay.PHYSICAL),
         'R18_normal_residual_and_bit_parity':all(np.array_equal(trial[n],parent[n]) for n in ('detect','r')),
         'no_added_false':all(match_false_events(records['event']['false_alarm_times'],
                               records[k]['false_alarm_times'])['passed'] for k in ARMS[:-1]),
         'no_lost_tool':all(preserve_tool(records['event'],records[k]) for k in ARMS[:-1])}
    return {'gates':g, 'events':records, 'passed':all(g.values())}


def authenticity(case, signatures):
    reference = read(REF)
    plan = read(read(PLAN)['prior_protocol'])
    if (reference.get('schema') != 'r18-paired-reference.v1' or
            reference['plan_signature'] != replay.signature(plan)):
        raise ValueError('untrusted prior case/score pairing')
    saved = reference['case_assessment_sha256'][case['id']]
    return all(signatures[k] == saved['R18_recovery' if k==PARENT else k] for k in ARMS[:-1])


def score_case(stage, case, h, e):
    plan = read(PLAN)
    if stage == 'normal':
        logs = {k:episode(k,case,h,e,(None,None)) for k in ARMS}
        norm = normal_scores(logs)
        return {'id':case['id'], 'episodes':len(ARMS), 'normal':norm, 'passed':norm['passed']}
    first, second = case['impacts']
    assignments = {'none':(None,None),'first':(first,None),'second':(None,second),'both':(first,second)}
    # Keep only the incumbent, then compare directly while each other arm is available.
    arms = {}; signatures = {}; baseline = None; physics = {}
    normal = {}
    for kind in (PARENT,'event','D_coupled','raw_r15','r10'):
        twins = {tw:episode(kind,case,h,e,imp) for tw,imp in assignments.items()}
        normal[kind] = twins['none']
        arms[kind], raw = assay.characterize(case,twins,plan['quiet_ms'],plan['window_ms'])
        signatures[kind] = replay.signature(raw)
        if kind==PARENT:
            baseline = twins
        elif kind in ('D_coupled','event'):
            names = replay.PHYSICAL+(('r',) if kind=='event' else ())
            physics[kind] = {tw:all(np.array_equal(v[n],baseline[tw][n]) for n in names)
                             for tw,v in twins.items()}
    comparisons = {k:assay.compare(arms['event'],arms[k]) for k in ARMS[:-1]}
    norm = normal_scores(normal)
    common = {'normal':norm['passed'],
              'exact_D_and_R18_physics_R18_residual':all(all(rows.values()) for rows in physics.values()),
              'full_force_and_counterfactual_prefix':all(a['signed_and_scalar_exact']
                        and all(a['r18_gates'][n] for n in ('complete_finite_strict_grid',
                         'force_free_prefixes','all_required_prefixes_exact','distinct_non_noop_impacts',
                         'no_sign_inversion')) for a in arms.values())}
    if stage == 'known':
        # Negative and missed rows reproduced exactly are still negative; this stage protects
        # what R18 actually did, without granting it positives from the other detectors.
        trust = authenticity(case,signatures)
        if not trust or not all(common[k] for k in common if k!='normal'):
            raise ValueError('invalid matched physical/reference experiment: '+case['id'])
        gates = dict(common, sealed_reference_all_arms=trust,
                     retain_incumbent=assay.all_comparisons([comparisons[PARENT]]))
        if case['id']=='pair_return_opposed':
            old, new = (arms[k]['slots']['both_second'] for k in (PARENT,'event'))
            gates['reduce_fragmentation_without_missing'] = (old['credited_delays_ms']==[33,44,57] and
                    new['credited_delays_ms']==new['raw_delays_ms'] and len(new['credited_delays_ms'])==1)
        if case['id'].startswith('diag_known_'):
            gates['preserve_two_verified_diagnostics'] = assay.one_per_pulse(arms['event'])
    elif stage=='new':
        gates = dict(common, exactly_one_genuine_rise_in_each_context=assay.one_per_pulse(arms['event']),
                     original_candidate_gates=all(arms['event']['r18_gates'].values()),
                     all_reference_requirements=assay.all_comparisons(comparisons.values()))
    else:
        raise ValueError('unknown stage')
    return {'id':case['id'],'episodes':len(ARMS)*len(assay.TWINS),'case':case,
            'normal':norm,'arms':arms,'assessment_signatures':signatures,'comparisons':comparisons,
            'physics':physics,'gates':gates,'passed':all(gates.values())}


def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--jobs',type=int,default=1,choices=(1,2,3))
    a=cli.parse_args(argv)
    if a.out.exists():cli.error('new output directory required')
    a.out.mkdir(); manifests=read('pins.json')
    def verify():
        if not manifests or any(sha(ROOT/n)!=pin for n,pin in manifests.items()):
            raise ValueError('missing or mutated input snapshot')
    verify(); plan=read(PLAN); before=plan['prior_protocol']
    if sha(before)!=plan['prior_plan_sha256']:raise ValueError('upstream case definitions changed')
    previous=read(before); known=previous['fresh_paired_cases']+previous['diagnostic_sequences']; fresh=plan['new']
    if len(known)!=6 or len(fresh)!=6 or len({c['id'] for c in known+fresh})!=12:
        raise ValueError('unexpected or duplicate case rows')
    for c in fresh:
        if len(c['impacts'])!=2 or round(1000*(c['impacts'][1][0]-c['impacts'][0][0]))!=120+c['gap_ms']:
            raise ValueError('incorrect predeclared physical gap')
    h,e=read('height.json'),read('early.json')
    stages={}
    for stage,cases in (('known',known),('normal',fresh),('new',fresh)):
        folder=a.out/stage;folder.mkdir()
        results={}
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            futures={pool.submit(score_case,stage,c,h,e):c for c in cases}
            for f in as_completed(futures):
                row=f.result()
                write(folder/(row['id']+'.json'),row);results[row['id']]=row
                print(stage,row['id'],row['passed'],flush=True)
        rows=[results[c['id']] for c in cases];verify()
        stages[stage]={'rows':rows,'passed':all(row['passed'] for row in rows)}
        if not stages[stage]['passed']:break
    report={'schema':'r19-close-separation.v1','stages':stages,'plan_sha256':sha(PLAN),
            'reference_sha256':sha(REF),'promotion_eligible':False,
            'episodes':sum(row['episodes'] for st in stages.values() for row in st['rows']),
            'all_stages_passed':len(stages)==3 and stages['new']['passed'],
            'new_forces_opened':len(stages.get('new',{}).get('rows',[]))}
    write(a.out/'report.json',report)


if __name__=='__main__':
    main()
