"""Install accepted early horizontal COM before final transport; retain R14 height."""
import numpy as np
from sim import config as C
from experiments.r14_stock_handoff.controller import Controller as Height
from experiments.r15_early_com.stream import Stream
from experiments.r15_early_com.sensing import decide

SENSE=2.463;START=2.470;END=2.670
def weight(t):
 if t<=START:return 0.
 if t>=END:return 1.
 s=(t-START)/(END-START);return s**3*(10+s*(-15+6*s))
class Controller(Height):
 def __init__(self,height_calibration,xy_calibration):super().__init__(height_calibration,stream_factory=Stream);self.early_calibration=xy_calibration
 def reset(self):
  super().reset();self.early_decided=self.box_mass<=0;self.early_result=None;self.early_offset=np.zeros(3);self.early_decided_at_s=None;self.early_updates=0;self._early_weight=0.;self._provided_com=np.asarray(self.box_com,float).copy()
 def compute(self,t,th,thd,tau_s,grip,q_d,qd_d,qdd_d):
  command=super().compute(t,th,thd,tau_s,grip,q_d,qd_d,qdd_d)
  if not self.early_decided and t>=SENSE:
   row=decide(self.registration_stream.finish_prefix(),self.early_calibration);action=np.asarray(row['model_action_xy_mm'],float)
   if action.shape!=(2,) or np.linalg.norm(action)>15.000001 or (not row['accept'] and np.any(action)):raise ValueError('early decision')
   self.early_offset[:2]=.001*action;self.early_result=row;self.early_decided=True;self.early_decided_at_s=float(t)
  if self.early_decided and np.any(self.early_offset):
   w=weight(float(t)+C.DT)
   if w!=self._early_weight:self.nom_box.set_payload(self.box_mass,self._provided_com+w*self.early_offset);self._early_weight=w;self.early_updates+=1
  return command
