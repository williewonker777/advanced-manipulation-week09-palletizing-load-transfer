"""Fit time-indexed horizontal normal wrench using normal-only course runs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from experiments.r20_horizontal.trace_detector import Detector
from my_torque import Controller
from sim.runner import run_episode

MASSES=(0.0,2.3,5.7,8.0)
COMS=((0.,0.,0.),(.02,-.01,.03),(.025,-.02,.03),(-.025,.02,.03),(.015,.015,-.01))
START,STOP=1.30,1.78
RIDGE=1e-7

def features(mass,com):
 m=float(mass);c=np.asarray(com,float)
 return np.r_[1.,m,m*m,c,m*c]

def trace(mass,com):
 d=Detector();result,_=run_episode(Controller(),box_mass=mass,box_com=com,detector=d,collision=None)
 t=np.asarray([r['t'] for r in d.trace]);w=np.vstack([r['residual'][:2] for r in d.trace])
 sel=(t>=START)&(t<STOP)
 return result,t[sel],w[sel]

def solve(x,y):
 scale=np.maximum(np.std(x,axis=0),1e-9);scale[0]=1.
 z=x/scale
 inv=np.linalg.inv(z.T@z+RIDGE*np.eye(z.shape[1]))@z.T
 # y: episodes,ticks,2; coefficients: ticks,feature,axis
 return scale,np.einsum('fe,eta->tfa',inv,y)

def predict(x,scale,coef):return np.einsum('ef,tfa->eta',x/scale,coef)

def main(out,raw_out):
 rows=[];signals=[];t0=None
 for ci,com in enumerate(COMS):
  for mass in MASSES:
   result,t,w=trace(mass,com)
   if result['false_alarms']!=0:raise ValueError('normal baseline false alarm')
   if t0 is None:t0=t
   elif not np.array_equal(t,t0):raise ValueError('time grid differs')
   rows.append({'mass':mass,'com':list(com),'com_group':ci});signals.append(w)
 x=np.vstack([features(r['mass'],r['com']) for r in rows]);y=np.stack(signals)
 # Truly out-of-group: every COM group is predicted from the other four groups.
 oof=np.empty_like(y)
 for group in range(len(COMS)):
  test=np.array([r['com_group']==group for r in rows]);scale,coef=solve(x[~test],y[~test]);oof[test]=y[test]-predict(x[test],scale,coef)
 scale,coef=solve(x,y)
 # Causal four-sample mean; threshold uses every OOF normal sample, axis neutral.
 kernel=np.ones(4)/4
 smooth=np.stack([np.apply_along_axis(lambda v:np.convolve(v,kernel,'full')[:len(v)],0,row) for row in oof])
 start_guard=np.arange(len(t0))>=3
 bound=float(np.max(np.abs(smooth[:,start_guard])))
 envelope=1.05*bound
 doc={'schema':'r21-pick-normal-model.v1','normal_only':True,'window_s':[START,STOP],
      'dt_s':float(t0[1]-t0[0]),'masses':list(MASSES),'coms':[list(x) for x in COMS],
      'feature':'[1,m,m^2,cx,cy,cz,m*cx,m*cy,m*cz]','ridge':RIDGE,
      'grouping':'leave-one-COM-group-out','smoothing_samples':4,'persistence_samples':3,
      'oof_max_abs_smoothed_N':bound,'threshold_N':envelope,
      'scale':scale.tolist(),'coefficients':coef.tolist()}
 Path(out).write_text(json.dumps(doc,separators=(',',':'),allow_nan=False)+'\n')
 np.savez_compressed(raw_out,t=t0,x=x,y=y,oof=oof,smooth=smooth)
 print(json.dumps({'normal_episodes':len(rows),'ticks':len(t0),'oof_max':bound,'threshold':envelope,
                   'in_sample_max':float(np.max(np.abs(y-predict(x,scale,coef))))}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--raw-out',required=True);a=p.parse_args();main(a.out,a.raw_out)
