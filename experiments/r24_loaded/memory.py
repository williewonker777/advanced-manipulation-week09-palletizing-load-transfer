"""Physically scoped support memory for the cyclic yaw evidence. Full R23 FREE otherwise unchanged."""
import json
import math
from pathlib import Path

from experiments.r21_yaw_impulse.state import AVERAGE
from experiments.r22_guarded.candidate import GuardedReleaseBalance
from experiments.r23_detached.free import Detector as Free
from experiments.r24_loaded.control import subtract_aligned
from experiments.r24_loaded.response_state import ConeHistory, SupportHistory

MODEL=Path(__file__).with_name('normal_response.json')


class MemoryYaw(GuardedReleaseBalance):
    def __init__(self,mass,packet,weights):
        super().__init__(mass)
        self.packet=packet;self.weights=weights;self.away=False
        self.contact_history=SupportHistory() if weights is not None else ConeHistory(AVERAGE)
        self.raw_value=None;self.reaction=0.

    def feed(self,th,tau,mass_row,grip):
        value=super().feed(th,tau,mass_row,grip);s=self.packet
        self.raw_value=value;self.reaction=0.
        if not grip:
            if self.away:self.contact_history=SupportHistory() if self.weights is not None else ConeHistory(AVERAGE)
            self.away=False
            return value
        if not s.last_table_allowed:self.away=True
        if not self.away:return value
        n=max(0.,float(s.last_table_wrench[2])) if s.last_table_allowed else 0.
        if self.weights is not None:
            features=self.contact_history.push(n)
            self.reaction=sum(w*x for w,x in zip(self.weights,features,strict=True))
        elif n>0. and value is not None:
            jac,pos,_=s._point_jacobian(s.q_prev)
            radius=math.hypot(*pos[:2])+math.hypot(s.half_x,s.half_y)
            cap=n*(s.friction_mu*radius+s.torsion_radius)
            self.reaction=self.contact_history.signed_credit(float(jac[:,0]@s.last_table_wrench),float(jac[:,0]@s.last_wrench),cap)
        else:
            self.reaction=self.contact_history.signed_credit(0.,0.,0.)
        if value is None or not s.last_table_allowed:return value
        return subtract_aligned(value,self.reaction)


class Cone(Free):
    calibrated=False
    def reset(self):
        super().reset()
        if getattr(self.ctrl,'MODE',None)!='torque' or self.box_mass<=0:return
        weights=None
        if self.calibrated:
            model=json.loads(MODEL.read_bytes());weights=model['coefficients_Nm_per_N']
            if model['schema']!='week09-r24-load-response-model.v1' or len(weights)!=5 or not all(map(math.isfinite,weights)):
                raise ValueError('invalid normal-only response')
        self.yaw=MemoryYaw(self.box_mass,self._shared,weights)


class Response(Cone):
    calibrated=True
