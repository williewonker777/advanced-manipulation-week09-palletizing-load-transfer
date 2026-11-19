"""Public empty-approach nuisance fingerprint, complete at 1.252 seconds."""
from __future__ import annotations
import numpy as np
from sim import config as C,trajectory as T
from sim.nominal import Nominal
from tools.r13_grasp_estimator import flange

DT=.001
CAPACITY=1253
CENTRES=np.arange(1180,1241,10)
HALF=5
U=np.linspace(-1.,1.,2*HALF+1);WEIGHT=(1.-U**2)**2;WEIGHT/=WEIGHT.sum()

def _kin(table,c):
 ix=c+np.arange(-HALF,HALF+1);v=table[ix];avg=lambda a:np.einsum('s,sj->j',WEIGHT,a)
 return avg(v),avg(table[ix+1]-table[ix-1])/(2*DT),avg((table[ix+1]-v)-(v-table[ix-1]))/DT**2

class EmptyReference:
 def __init__(self,mass,com):
  self.mass=float(mass);self.com=np.asarray(com,float).copy()
  if not np.isfinite(self.mass) or self.mass<=0 or self.com.shape!=(3,) or not np.isfinite(self.com).all():raise ValueError('finite positive payload metadata required')
  self.nom=Nominal();self.encoder=np.zeros((CAPACITY,6));self.jts=np.zeros_like(self.encoder);self.count=0;self._result=None
 def observe(self,t,encoder,jts,grip):
  k=self.count;e=np.asarray(encoder,float);j=np.asarray(jts,float)
  if k>=CAPACITY or abs(float(t)-k*DT)>1e-12 or e.shape!=(6,) or j.shape!=(6,) or not np.isfinite(e).all() or not np.isfinite(j).all() or grip not in (0,1,False,True):raise ValueError('sequential finite public samples required')
  if bool(grip):raise ValueError('empty pre-grasp samples required')
  self.encoder[k]=e;self.jts[k]=j;self.count+=1
 def finish(self):
  if self.count<CAPACITY:raise ValueError('empty approach incomplete before 1.252s')
  if self._result is not None:return dict(self._result)
  theta=self.encoder;sensed=self.jts;elastic=np.asarray(self.nom.elastic,bool);inv=1./np.expm1(2*np.pi*C.JTS_CUTOFF*DT)
  sensor=sensed[1:]+inv*(sensed[1:]-sensed[:-1]);tau=sensor[:-1].copy();tau[:,~elastic]=.5*(sensor[:-1,~elastic]+sensor[1:,~elastic])
  tc=np.asarray(C.D)[elastic]/np.asarray(self.nom.K)[elastic];mem=np.exp(-DT/tc);ramp=1.-tc/DT*(1.-mem);spring=tau[:,elastic]/np.asarray(self.nom.K)[elastic];disp=np.zeros_like(tau);state=spring[0].copy();disp[0,elastic]=state
  for i in range(1,len(spring)):
   prev=spring[i-1];state=mem*state+(1.-mem)*prev+ramp*(spring[i]-prev);disp[i,elastic]=state
  enc=theta[:len(tau)];q=enc-disp
  # Direct public vertical tracking fingerprint, averaged only before 1.25s.
  errors=[]
  for k in range(1220,1251):
   measured=flange(self.nom,q[k])[0];reference=flange(self.nom,T.reference(k*DT)[0])[0];errors.append(measured[2]-reference[2])
  vertical_mm=1000*float(np.mean(errors))
  # One physics-fitted elastic scaling coordinate with fixed physical ridge.
  rows=[];scales=[]
  for c in CENTRES:
   q0,v0,a0=_kin(q,c);dq,dv,da=_kin(disp,c);tm=_kin(tau,c)[0];base=self.nom.inverse_dynamics(q0,v0,a0);plus=self.nom.inverse_dynamics(q0-.05*dq,v0-.05*dv,a0-.05*da);minus=self.nom.inverse_dynamics(q0+.05*dq,v0+.05*dv,a0+.05*da);rows.append(tm-base);scales.append((plus-minus)/.10)
  y=np.asarray(rows).ravel();s=np.asarray(scales).ravel();rho=float((s@y)/(s@s+1/.065**2))
  initial=float(np.linalg.norm(np.mean(disp[40:81,elastic],axis=0))*1000)
  self._result={'mass':self.mass,'empty_vector':[self.mass,vertical_mm,rho,initial],'vertical_tracking_mm':vertical_mm,'compliance_rho':rho,'initial_elastic_deflection_mm':initial,'completed_at_s':1.252}
  return dict(self._result)
 @property
 def feature(self):return self.finish()

def from_public(t,encoder,jts,grip,mass,com):
 s=EmptyReference(mass,com)
 for row in zip(t[:CAPACITY],encoder[:CAPACITY],jts[:CAPACITY],grip[:CAPACITY]):s.observe(float(row[0]),row[1],row[2],bool(row[3]))
 return s.finish()
