"""C release/height policy with the same fixed-duration COM blend after an observed prefix."""
import numpy as np

from experiments.r14_stock_handoff.controller import Controller as Height
from experiments.r16_balanced.controller import Controller as Balanced
from experiments.r17_information_prefix.estimate import SCHEMA,decide
from experiments.r17_information_prefix.stream import InformationStream
from sim import config as C


def blend(t, start):
    s = float(np.clip((t-start)/.200,0.,1.))
    return s**3*(10.+s*(-15.+6.*s))


class Controller(Balanced):
    def __init__(self, height, information):
        if information.get('schema')!=SCHEMA:
            raise ValueError('frozen observation model required')
        Height.__init__(self,height,stream_factory=InformationStream)
        self.information_model=information
        self.last_centre=int(information['last_centre'])
        self.sense=(self.last_centre+23)*C.DT
        self.start=(self.last_centre+30)*C.DT

    def reset(self):
        super().reset()
        self.information_result=None
        self.information_offset=np.zeros(3)
        self.information_decided=self.box_mass<=0.
        self.information_active=False
        self.information_weight=0.
        self.information_updates=0

    def compute(self,t,th,thd,tau_s,grip,q_d,qd_d,qdd_d):
        # Keep the registered R14 + balanced reference; replace the early COM
        # policy instead of accidentally applying both corrections.
        command=Height.compute(self,t,th,thd,tau_s,grip,q_d,qd_d,qdd_d)
        if not self.information_decided and round(t/C.DT)>=self.last_centre+23:
            obs=self.registration_stream.information(self.last_centre)
            self.information_result=decide(obs,self.information_model)
            self.information_offset[:2]=.001*np.array(self.information_result['model_action_xy_mm'])
            self.information_active=bool(self.information_result['accept'])
            self.information_decided=True
        if self.information_active and self.information_weight<1. and t+C.DT>self.start:
            w=blend(t+C.DT,self.start)
            if w!=self.information_weight:
                self.nom_box.set_payload(self.box_mass,self._provided_com+w*self.information_offset)
                self.information_weight=w
                self.information_updates+=1
        return command
