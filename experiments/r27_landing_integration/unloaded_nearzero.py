"""Run the preregistered near-zero normal and signed-Y panel after all guards."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from experiments.r26_delivery_pair import probe
from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r27_landing_integration.unloaded_aware import Detector

HERE=Path(__file__).resolve().parent
def rises(log):
 b=np.asarray(log['detect'],bool);return np.flatnonzero(b&~np.r_[False,b[:-1]]).astype(int).tolist()
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists() or a.out.is_symlink():p.error('fresh --out required')
 plan=json.loads((HERE/'unloaded_plan.json').read_bytes());rows=[]
 for raw in plan['near_zero_panel']:
  case=dict(raw,mechanics={'table_solref_time_scale':1.0});systems={}
  for name,cls in (('c1',C1),('candidate',Detector)):
   _,_,mn,ln=probe.run(case,cls);rec={'normal_rises':rises(ln),'normal_false':mn['false_alarm_times'],'normal_physics':probe.replay.physical_signature(ln),'impacts':{}}
   for j,inp in enumerate(plan['near_zero_inputs']):
    _,_,m,l=probe.run(case,cls,inp);assessment=probe.replay.force_assessment(l,ln,inp,{'event_window_s':.32,'twin_guard_s':.02})
    rec['impacts'][str(j)]={'rises':rises(l),'delay_ms':assessment['first_genuine_delay_ms'],'duplicates':max(0,len(assessment['genuine_rises_s'])-1),'signed_exact':assessment['signed_force_exact'],'prefix_exact':assessment['force_free_prefix_exact'],'physics':probe.replay.physical_signature(l)}
   systems[name]=rec
  gates={'same_normal':systems['c1']['normal_rises']==systems['candidate']['normal_rises'] and systems['c1']['normal_false']==systems['candidate']['normal_false'],'same_normal_physics':systems['c1']['normal_physics']==systems['candidate']['normal_physics'],'impact_physics_exact':all(systems['c1']['impacts'][k]['physics']==systems['candidate']['impacts'][k]['physics'] and systems['candidate']['impacts'][k]['signed_exact'] and systems['candidate']['impacts'][k]['prefix_exact'] for k in ('0','1')),'no_duplicates':all(systems['candidate']['impacts'][k]['duplicates']==0 for k in ('0','1'))}
  rows.append({'id':case['id'],'case':raw,'systems':systems,'gates':gates,'passed':all(gates.values())})
 doc={'schema':'r27-landing-unresolved-nearzero.v1','classification':'Preregistered after-freeze near-zero panel; not blind holdout.','episodes':len(rows)*6,'rows':rows,'passed':all(r['passed'] for r in rows)}
 a.out.write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
 compact=[{'id':r['id'],
           'c1':[r['systems']['c1']['impacts'][k]['delay_ms'] for k in ('0','1')],
           'candidate':[r['systems']['candidate']['impacts'][k]['delay_ms'] for k in ('0','1')]}
          for r in rows]
 print(json.dumps({'episodes':doc['episodes'],'passed':doc['passed'],'rows':compact},indent=2))
if __name__=='__main__':main()
