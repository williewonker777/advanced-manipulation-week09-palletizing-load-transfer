"""Publish scored onset, ablation and duplicate counts; do not ship raw trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

SLOTS=('isolated_first','isolated_second','both_first','both_second')


def summarize(doc):
    out=doc.get('result',doc); stages=out['stages']; rows=[]
    if out['schema']!='r19-impulse-gate-result.v1' or out['promotion_eligible'] is not False:
        raise ValueError('an executed impulse gate is required')
    for r in stages.get('novel',{}).get('rows',[]):
        scores={}
        for name,a in r['arms'].items():
            slots=a['slots']
            scores[name]={'credited_ms':{k:slots[k]['credited_delays_ms'] for k in SLOTS},
                          'raw_ms':{k:slots[k]['raw_delays_ms'] for k in SLOTS},
                          'second_start_quiet_20ms':slots['both_second']['quiet_before_onset'],
                          'second_start_masked':slots['both_second']['high_at_onset'],
                          'all_four_exact':all(len(slots[k]['credited_delays_ms'])==len(slots[k]['raw_delays_ms'])==1 for k in SLOTS),
                          'paired_high_ms':a['levels']['both']['on_intervals_ms']}
        failed_r18=[k for k,v in r['arms']['candidate']['r18_gates'].items() if not v]
        rows.append({'id':r['id'],'mass_kg':r['case']['mass'],'nominal_gap_ms':r['case']['gap_ms'],
                     'impacts':r['case']['impacts'],'arms':scores,'failed_original_gates':failed_r18,
                     'failed_gates':[k for k,v in r['gates'].items() if not v],
                     'failed_comparisons':{arm:{k:[slot for slot,v in part.items() if not v] for k,part in vals.items()
                                            if not all(part.values())} for arm,vals in r['comparisons'].items()},
                     'passed_strict':bool(r['passed'] and not failed_r18)})
    stats={}
    if rows:
        for arm in rows[0]['arms']:
            group=[r['arms'][arm] for r in rows]
            stats[arm]={'first_credited':sum(bool(v['credited_ms']['both_first']) for v in group),
                        'second_credited':sum(bool(v['credited_ms']['both_second']) for v in group),
                        'both_in_same_run':sum(all(v['credited_ms'][s] for s in ('both_first','both_second')) for v in group),
                        'both_exactly_one':sum(all(len(v['credited_ms'][s])==len(v['raw_ms'][s])==1 for s in ('both_first','both_second')) for v in group),
                        'all_four_exact':sum(v['all_four_exact'] for v in group),
                        'excess_raw_rises_four_contexts':sum(max(0,len(t)-1) for v in group for t in v['raw_ms'].values())}
    return {'schema':'r19-unsigned-impulse-summary.v1','research_only':True,'promotion_eligible':False,
            'previously_exposed_controls':12,'episodes':out['episodes'],'new_forces_opened':out['novel_forces_opened'],
            'stage_counts':{s:{'count':len(v['rows']),'passed':sum(bool(row['passed']) for row in v['rows'])} for s,v in stages.items()},
            'strict_new_count':sum(row['passed_strict'] for row in rows),'new_results':stats,'rows':rows,
            'method':'new qualifications keep the original quiet/prefix/force/false/timing-of-onset criteria; no independent result is used to tune the model'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args(); data=a.input.read_bytes();result=summarize(json.loads(data))
    result['source_sha256']=hashlib.sha256(data).hexdigest()
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
