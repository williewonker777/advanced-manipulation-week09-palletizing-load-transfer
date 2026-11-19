"""Reduce a complete R19 audit to derived case results, keeping missing stages explicit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SLOTS = ('isolated_first','isolated_second','both_first','both_second')


def public_summary(document):
    result=document['result']
    if (document['schema']!='r19-event-audit.v1' or not document['research_only']
            or result['schema']!='r19-close-separation.v1' or result['promotion_eligible'] is not False):
        raise ValueError('expected a research-only R19 event audit')
    stages=result['stages']
    report={'schema':'r19-event-separation-summary.v1','research_only':True,'submitted_candidate':False,
            'plan_sha256':result['plan_sha256'],'source_sha256':document['pins']['experiments/r19_event_state/physical_detector.py'],
            'episodes':result['episodes'],'stages':{k:{'passed':v['passed'],'conditions':len(v['rows']),
            'passed_conditions':sum(r['passed'] for r in v['rows'])} for k,v in stages.items()},'by_stage':{}}
    for name in ('known','new'):
        if name not in stages:
            report['by_stage'][name]={'status':'not_opened','rows':[]}
            continue
        rows=[]
        for row in stages[name]['rows']:
            case=row['case']; arms={}
            onsets=[int(round(1000*v[0])) for v in case['impacts']]
            for arm,obs in row['arms'].items():
                delays={s:obs['slots'][s]['credited_delays_ms'] for s in SLOTS}
                raw={s:obs['slots'][s]['raw_delays_ms'] for s in SLOTS}
                per={s:[[int(a),int(b)] for a,b in obs['levels'][s]['on_intervals_ms']] for s in ('first','second','both')}
                responses={}
                for slot,tw,idx in zip(SLOTS,('first','second','both','both'),(0,1,0,1)):
                    start=onsets[idx]; attempts=delays[slot]
                    segments=[v for v in per[tw] if attempts and v[0]==start+attempts[0]]
                    end=None if not segments else segments[0][1]
                    responses[slot]={
                      'first_end_ms':end,
                      'end_minus_nominal_force_end_ms':None if end is None else end-(start+120),
                      'max_active_run_ms':max((b-a for a,b in per[tw] if a-start in attempts),default=None),
                      'high_before_onset':obs['slots'][slot]['high_at_onset'],
                      'quiet_previous_20':obs['slots'][slot]['quiet_before_onset']}
                arms[arm]={'credited_ms':delays,'raw_ms':raw,'count_all_four_as_one':all(
                               len(delays[s])==len(raw[s])==1 for s in SLOTS),
                           'response':responses,'signed_scalar_certified':obs['signed_and_scalar_exact'],
                           'both_first_and_second_credited':bool(delays['both_first'] and delays['both_second']),
                           'outside_programmed_windows_rises_s':obs['false_s'],
                           'original_twin_gate':all(obs['r18_gates'].values())}
            rows.append({'id':row['id'],'mass_kg':case['mass'],'impact_onsets_ms':onsets,
                         'applied_forces_N':[x[1] for x in case['impacts']],
                         'gap_ms':onsets[1]-onsets[0]-120,
                         'observed_zero_force_samples':row['arms']['event']['actual_quiescent_force_samples'],
                         'passed':row['passed'],'failed_gates':[k for k,v in row['gates'].items() if not v],
                         'failed_comparisons':{arm:{k:[w for w,v in slots.items() if not v] for k,slots in cmp.items()
                                                if not all(slots.values())} for arm,cmp in row['comparisons'].items()},
                         'arms':arms})
        by_arm={}
        for a in rows[0]['arms']:
            vals=[r['arms'][a] for r in rows]
            by_arm[a]={'both_first':sum(bool(x['credited_ms']['both_first']) for x in vals),
                       'both_second':sum(bool(x['credited_ms']['both_second']) for x in vals),
                       'both_hits_on_same_trace':sum(x['both_first_and_second_credited'] for x in vals),
                       'one_genuine_in_all_four_contexts':sum(x['count_all_four_as_one'] for x in vals),
                       'excess_raw_rises_all_four_contexts':sum(max(0,len(times)-1) for x in vals for times in x['raw_ms'].values()),
                       'second_onsets_masked_preexisting':sum(x['response']['both_second']['high_before_onset'] for x in vals),
                       'second_onsets_not_quiet_20ms':sum(not x['response']['both_second']['quiet_previous_20'] for x in vals)}
        report['by_stage'][name]={'status':'completed','rows':rows,'by_arm':by_arm}
    return report


def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('audit',type=Path);cli.add_argument('out',type=Path);a=cli.parse_args(argv)
    blob=a.audit.read_bytes()
    summary=public_summary(json.loads(blob))
    summary['audit_sha256']=hashlib.sha256(blob).hexdigest()
    with a.out.open('x') as f:
        json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':
    main()
