"""Measure two public vertical force estimates on disclosed D4 misses and twins."""
import argparse,json
from pathlib import Path
import numpy as np
from experiments.r21_full_protection import runner_v2 as raw
from experiments.r22_census import check
from experiments.r31_held_identity.candidate import Detector as Parent
IDS=(253,254,263,264,273,274,283)
class Probe(Parent):
 def reset(self):super().reset();self.rows=[];Probe.instance=self
 def update(self,*a):
  result=super().update(*a);p=self.r27_projection.last_force;w=self._shared.last_residual_wrench
  self.rows.append([None if p is None else float(p[2]),None if w is None else float(w[2]),bool(a[-1]),bool(self._shared.last_table_allowed)]);return result

def changes(rows,width=40):
 out=np.zeros(len(rows));direction=np.zeros(len(rows));values=[]
 for k,(p,w,grip,table) in enumerate(rows):
  if p is None or w is None or not grip or table:
   values=[];continue
  values.append(np.array([p,w],float))
  if len(values)>=2*width:
   before=np.mean(values[-2*width:-width],axis=0);now=np.mean(values[-width:],axis=0);d=now-before
   if d[0]*d[1]>0:out[k]=min(abs(d[0]),abs(d[1]));direction[k]=np.sign(d[0])
 return out,direction
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv);cases=check.build_cases('my_torque');refs={check.old.reference_key(c):c for c in cases if c['kind']=='reference'};rows=[]
 for i in IDS:
  for role,c in [('normal',refs[check.old.reference_key(cases[i])]),('force',cases[i])]:
   _,log,signed=raw.execute(Probe,c);x=np.asarray([[np.nan if v is None else v for v in r[:2]] for r in Probe.instance.rows]);on=int(round(cases[i]['collision'][0]*1000));sl=slice(on,on+401)
   z=x[sl];aligned=np.where(z[:,0]*z[:,1]>0,np.minimum(np.abs(z[:,0]),np.abs(z[:,1])),0.);j=int(np.nanargmax(aligned))
   ch,sgn=changes(Probe.instance.rows);cj=int(np.argmax(ch[sl]));rows.append({'id':i,'role':role,'case_id':c['case_id'],'max_abs_projection':float(np.nanmax(np.abs(x[sl,0]))),'max_abs_shared':float(np.nanmax(np.abs(x[sl,1]))),'max_aligned_min':float(aligned[j]),'aligned_tick':on+j,'aligned_projection':float(z[j,0]),'aligned_shared':float(z[j,1]),'max_change_concurrence':float(ch[sl][cj]),'change_tick':on+cj,'change_sign':int(sgn[sl][cj]),'projection_at_max':float(x[sl,0][np.nanargmax(np.abs(x[sl,0]))]),'shared_at_projection_max':float(x[sl,1][np.nanargmax(np.abs(x[sl,0]))])})
 out={'schema':'r33-vertical-diagnosis.v1','rows':rows};a.out.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
