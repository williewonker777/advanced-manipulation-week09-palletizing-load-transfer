"""Verify the advisory adapter on disclosed motion only, never on the spent prospective panel."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from experiments.r21_full_protection import runner_v2 as raw
from experiments.r24_loaded.d4 import load_cases
from .client import ObserveOnly

# Includes every recovered condition, true table re-contact, corrected rebound,
# and weak release misses so advisory zeros are checked as well.
IDS=(193,194,197,203,204,213,214,223,248,253,254,263,264,273,274,283,304,313,334)
def one(case):
    class Logged(ObserveOnly):
        def reset(self):
            super().reset();self.combined=[];self.advisories=[];self.edges=[];Logged.instance=self
        def update(self,*a):
            out=super().update(*a);o=self.observation
            self.advisories.append(o.pickup_evidence or o.carry_evidence)
            self.combined.append(bool(out[1] or self.advisories[-1]))
            self.edges.append(o.advisory_started)
            return out
    r=raw.episode_record(Logged,case);d=Logged.instance
    return case['case_id'],r,raw.intervals(d.combined),raw.intervals(d.advisories),[i for i,v in enumerate(d.edges) if v]
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--jobs',type=int,choices=range(1,9),default=4);a=p.parse_args()
    if a.out.exists():p.error('new output required')
    data=a.reference.read_bytes();pin=hashlib.sha256(data).hexdigest();prior=json.loads(data);rows=prior['rows']
    if (prior['scope']!='full-d4' or prior['episodes']!=770 or len(rows)!=385 or prior['summary']['r33']['strict']!=168
        or prior['candidate']['sha256']!='caef8fe6603ff658660c2ea2e3031946291b3028a73f51871aa082a532e878a0'):
        raise ValueError('requires the frozen full public independent R31/R33 physical replay')
    cases=load_cases();refs={raw.reference_key(c):c for c in cases if c['kind']=='reference'}
    ids=sorted(set(IDS)|{refs[raw.reference_key(cases[i])]['case_id'] for i in IDS});notes={}
    with ProcessPoolExecutor(a.jobs) as pool:
        for i,r,combined,adv,edges in pool.map(one,[cases[i] for i in ids]):
            old,new=(rows[i]['records'][k] for k in ('r31','r33'))
            if (r['physical_sha256']!=old['physical_sha256'] or r['physical_sha256']!=new['physical_sha256']
                or r['detect_intervals']!=old['detect_intervals'] or combined!=new['detect_intervals']
                or [t for t,_ in adv]!=edges):
                raise ValueError(f'public physical-channel mismatch: {i}')
            if adv:notes[str(i)]=dict(advisory_intervals=adv,starts=edges)
    result=dict(schema='r33-observe-only-public-parity.v1',episodes=len(ids),ticks=len(ids)*6000,alarms_equal_to_independently_run_r31=len(ids),
                combined_telemetry_equals_frozen_r33=len(ids),reference_sha256=pin,rows=ids,advisories=notes,
                candidate_reenrolled=False,prospective_executions=0,improvement_to_primary_alarm_claimed=False)
    with a.out.open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(len(ids),'real cases,',len(notes),'with supplemental observations; all R31 Booleans conserved')
if __name__=='__main__':main()
