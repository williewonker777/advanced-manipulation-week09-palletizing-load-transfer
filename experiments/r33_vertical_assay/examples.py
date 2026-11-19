"""Save signed external force and measured alarms from disclosed, matched physical tests only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .check import ROOT,level
from experiments.r21_full_protection import runner_v2 as raw
from experiments.r24_loaded.d4 import load_cases
from experiments.r33_vertical_composition.candidate import Detector as Entrant

IDS=(194,253,313,248)
class Sensor(Entrant):
    def reset(self):
        super().reset();self.events=[];Sensor.instance=self
    def update(self,*args):
        result=super().update(*args);self.events.append((self.parent_event,self.pick_event,self.carry_event))
        return result
def capture(c):
    _,log,f=raw.execute(Sensor,c);events=np.array(Sensor.instance.events,bool)
    if len(f)!=6000 or not np.array_equal(f,raw.expected_signed(log['t'],c['collision'])) or np.any(np.any(events,1)!=log['detect']):
        raise ValueError('incomplete physical cause or channel')
    return dict(case=c,physics={n:raw.array_sha(log[n]) for n in raw.PHYSICAL},source_intervals={k:raw.intervals(events[:,i]) for i,k in enumerate(('r31','pick','carry'))},
          new=raw.intervals(log['detect']),trace=(f,log['f_tool'],log['f_table'],np.array(log['t'])))
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if level.sha(a.reference)!=level.OLD:raise ValueError('authentic R31 required')
    archive=level.get(a.reference)['rows'];cs=load_cases();refs={raw.reference_key(c):c for c in cs if c['kind']=='reference'}
    subset={c['case_id']:c for i in IDS for c in (cs[i],refs[raw.reference_key(cs[i])])}
    with ProcessPoolExecutor(2) as pool:records={r['case']['case_id']:r for r in pool.map(capture,subset.values())}
    for i,r in records.items():
        a31=archive[i]
        if r['physics']!=a31['physics_sha256'] or r['source_intervals']['r31']!=a31['arms']['identity']['alarm_intervals']:
            raise ValueError('physical trajectory or parent changed')
    rows=[]
    for i in IDS:
        c=cs[i];ref=records[refs[raw.reference_key(c)]['case_id']];r=records[i]
        tick=round(c['collision'][0]*1000);f,tool,table,t=r['trace'];ff,nt,nf,tt=ref['trace'];last=tick+290
        if ff.any() or not np.array_equal(t[:tick],tt[:tick]):raise ValueError('bad control')
        clip=slice(tick-20,last)
        rows.append(dict(id=i,reference_id=ref['case']['case_id'],phase=c['phase'],axis=c['direction'],onset=tick,force_N=c['force_n'],
             t_ms=list(range(tick-20,last)),actual_world_z_N=np.round(f[clip,2],4).tolist(),tool_force_N=np.round(tool[clip],3).tolist(),
             table_force_N=np.round(table[clip],3).tolist(),matched_table_force_N=np.round(nf[clip],3).tolist(),
             r31=r['source_intervals']['r31'],r33=r['new'],pickup=r['source_intervals']['pick'],carry=r['source_intervals']['carry'],
             control_r33=ref['new']))
    level.save(a.out,dict(schema='r33-signed-force-traces.v1',actual_episodes=len(records),reference_sha256=level.OLD,
        candidate_sha256=level.sha(ROOT/'experiments/r33_vertical_composition/candidate.py'),rows=rows))
if __name__=='__main__':main()
