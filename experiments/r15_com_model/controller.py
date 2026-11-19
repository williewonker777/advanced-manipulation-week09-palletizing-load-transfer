"""Experimental bounded horizontal gravity/inertia registration, no commanded XY change."""
import numpy as np
from sim import config as C
from experiments.r14_stock_handoff.controller import Controller as HeightController
from experiments.r15_com_model.sensing import decide as decide_xy

# Features are computed by 3.527 s, but Stream.finish validates 3550
# public samples. At index 3549 (3.549 s) that independent contract holds.
SENSE=3.549
START=3.550
END=3.710


def weight(t):
    if t<=START:return 0.
    if t>=END:return 1.
    s=(t-START)/(END-START)
    return s**3*(10.+s*(-15.+6.*s))


class Controller(HeightController):
    def __init__(self,calibration=None,xy_calibration=None,stream_factory=None,decide_fn=None,xy_decide_fn=None):
        super().__init__(calibration,stream_factory,decide_fn)
        self.xy_calibration=xy_calibration
        self.xy_decide_fn=xy_decide_fn or decide_xy

    def reset(self):
        super().reset()
        self.xy_decided=self.box_mass<=0
        self.xy_result=None
        self.xy_offset_m=np.zeros(3)
        self.xy_updates=0
        self.xy_decided_at_s=None
        self._last_weight=0.
        self._provided_com=np.asarray(self.box_com,float).copy()
        if self.box_mass>0 and not isinstance(self.xy_calibration,dict):
            raise ValueError('need a horizontal calibration for this domain')

    def compute(self,t,th,thd,tau_s,grip,q_d,qd_d,qdd_d):
        command=super().compute(t,th,thd,tau_s,grip,q_d,qd_d,qdd_d)
        if not self.xy_decided and t>=SENSE:
            row=self.xy_decide_fn(self.registration_stream.finish(),self.xy_calibration)
            action=np.asarray(row['model_action_xy_mm'],float)
            if (action.shape!=(2,) or not np.isfinite(action).all() or np.linalg.norm(action)>15.0000001
                    or (not row['accept'] and np.any(action))):
                raise ValueError('invalid horizontal COM decision')
            self.xy_offset_m[:2]=.001*action
            self.xy_result=dict(row);self.xy_decided=True;self.xy_decided_at_s=float(t)
        if self.xy_decided and np.any(self.xy_offset_m):
            # The original runner calls the detector before the controller. Set
            # the PUBLIC nominal model for the next sample only *after* both
            # consumers finished the current sample. The simulated plant is separate.
            w=weight(float(t)+C.DT)
            if w!=self._last_weight:
                self.nom_box.set_payload(self.box_mass,self._provided_com+w*self.xy_offset_m)
                self.xy_updates+=1;self._last_weight=w
        return command
