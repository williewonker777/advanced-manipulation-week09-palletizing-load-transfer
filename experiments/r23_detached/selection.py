"""Select for independent evaluation, not for deployment, using only authenticated exposed studies."""
import argparse
import hashlib
import json
from pathlib import Path

EXPOSED=('target','previous','d4','weak','official','pairs')
CANDIDATES=('onset','free')


def protect_official(new, ref):
    a,b=new['cells'],ref['cells']
    if a['control']!=b['control'] or len(a['collisions'])!=len(b['collisions']) or len(b['collisions'])!=6:
        return False
    for x,y in zip(a['collisions'],b['collisions'],strict=True):
        if (x['mass'],x['time'])!=(y['mass'],y['time']) or len(x['cells'])!=6 or x['cells'][4:]!=y['cells'][4:]:
            return False
        if not all(a.isdigit() and b.isdigit() and int(a)<=int(b)+5 for a,b in
                   zip(x['cells'][:4],y['cells'][:4],strict=True)):
            return False
    return new['outcomes']['impact_outcomes']['detected']==24 and new['outcomes']['no_impact_false_rises']==0


def criteria(d,k):
    target,old,d4,weak,official,pairs=(d[n] for n in EXPOSED)
    return {
        'causal_after_release_all_three':target['all_input_certified'] and target['episodes']==18
        and target['totals'][k]['single']==3 and target['totals']['light']['single']==0 and not target['regressions'][k],
        'old_challenge_full_retain':old['episodes']==486 and len(old['rows'])==66 and old['all_input_certified']
        and not old['regressions'][k] and old['totals'][k]['single']>old['totals']['light']['single']
        and old['totals'][k]['pair_slots']>=old['totals']['light']['pair_slots'],
        'd4_full_retain':d4['episodes']==1155 and [x['id'] for x in d4['rows']]==list(range(385))
        and not d4['invalid'] and not d4['regressions'][k] and d4['summary'][k]['strict']>=137,
        'all_15_small_and_control_twins':weak['episodes']==90 and len(weak['rows'])==15 and all(
            row['arms'][k]['input_certified'] and row['arms'][k]['strict']
            and not row['arms'][k]['none_false'] and not row['arms'][k]['false'] and not row['arms'][k]['extra']
            for row in weak['rows']),
        'official_original_29_each':official['episodes']==87 and protect_official(official['results'][k],official['results']['light']),
        'all_legacy_pair_protection':pairs['episodes']==224 and len(pairs['rows'])==14 and not pairs['regressions'][k]
        and all(r['arms'][k]['exact_force_and_physics'] for r in pairs['rows']),
    }


def footprint(d,k):
    wins={('D4',r['id']) for r in d['d4']['rows'] if r['kind']=='collision' and r['scores'][k]['strict']}
    for r in d['previous']['rows']:
        rec=r['records'][k]
        ev={'single':rec['event']} if r['kind']=='single' else rec.get('events',{})
        wins.update(('previous',r['variant'],r['id'],s) for s,e in ev.items() if e['exactly_one'] and e['quiet_before'])
    wins.update(('pairs',r['id'],r['surface']) for r in d['pairs']['rows'] if r['arms'][k]['strict'])
    return wins


def pick(eligible, footprints):
    if len(eligible)<2:return eligible[0] if eligible else None
    for k in eligible:
        if all(footprints[k]>footprints[q] for q in eligible if q!=k):return k
    return eligible[0]


def selection(paths):
    docs={};pins={}
    for name in EXPOSED:
        raw=paths[name].read_bytes(); doc=json.loads(raw)
        if (doc.get('schema')!='r23-authenticated-offline-study.v1' or doc.get('study')!=name
                or doc.get('unseen') is not False or doc.get('original_demo') is not False):
            raise ValueError('invalid exposed input '+name)
        docs[name]=doc['data'];pins[name]=hashlib.sha256(raw).hexdigest()
    checks={k:criteria(docs,k) for k in CANDIDATES}
    eligible=[k for k in CANDIDATES if all(checks[k].values())]
    footprints={k:footprint(docs,k) for k in CANDIDATES};chosen=pick(eligible,footprints)
    return {'schema':'r23-exposed-only-choice.v1','candidates_in_predeclared_order':list(CANDIDATES),
            'selected_for_independent':chosen,'default_promoted':False,'gates':checks,'exposed_inputs_sha256':pins,
            'only_free_strict':sorted(footprints['free']-footprints['onset']),
            'only_onset_strict':sorted(footprints['onset']-footprints['free']),
            'selection_policy_sha256':hashlib.sha256(Path(__file__).with_name('selection_policy.json').read_bytes()).hexdigest()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in EXPOSED:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    doc=selection(vars(a))
    with a.out.open('x') as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
    print(doc['selected_for_independent'],doc['gates'])
