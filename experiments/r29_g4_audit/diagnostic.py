"""Apply only the two corrected reporting contracts; do not rewrite the locked study verdict."""
import argparse
import hashlib
import json
from pathlib import Path
from experiments.r29_g4_audit.contract import physical_view, no_force_edges


def audit(doc):
    rows=[]
    for row in doc['rows']:
        guards=dict(row['gates']);records=row['records']
        twins=[k for k in records['memory'] if k not in ('prefix','prefixes','event','events')]
        names=('c1','v3','memory')
        compare=all(physical_view(records[arm][tw])==physical_view(records['memory'][tw])
                    for arm in names for tw in twins)
        extra={}
        if row['kind']=='negative':
            extra={arm:no_force_edges(records[arm]['none'],records['memory']['none']) for arm in names[:-1]}
            if not {'no_added_false','no_added_edge','no_lost_edge'}<=set(guards):
                raise ValueError('negative row contract differs')
            for key in ('no_added_false','no_added_edge','no_lost_edge'):guards.pop(key)
            guards['preserve_nonfalse_no_new_alarm']=all(v['passed'] for v in extra.values())
        if 'same_g4_support' not in guards:raise ValueError('unknown source schema')
        guards['same_g4_support']=compare
        rows.append(dict(id=row['id'],locked_result=row['passed'],physical_support=compare,edge_checks=extra,
                         revised_relative_protections=all(guards.values()),failed_guards=[k for k,v in guards.items()if not v]))
    return dict(schema='r29-post-run-g4-guard-diagnostic.v1',preregistered_verdict=doc['primary_protection_passed'],
                reran_physics=False,retroactive_qualification=False,absolute=doc['absolute_counts'],
                revised_relative_protections=all(r['revised_relative_protections']for r in rows),rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():p.error('fresh output required')
    raw=a.input.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a.sha:raise ValueError('pinned input report required')
    out=audit(json.loads(raw));out['source_sha256']=a.sha
    with a.out.open('x')as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(out['preregistered_verdict'],out['revised_relative_protections'])


if __name__=='__main__':main()
