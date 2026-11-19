"""Evidence-scoped weak uplift; the guarded full-cycle base is never muted."""
from experiments.r22_guarded.candidate import Detector as Guarded
from experiments.r22_light.witness import LightWitness


class Detector(Guarded):
    def reset(self):
        super().reset()
        self.light_witness=LightWitness(self.box_mass,self.nom,self.nom_box)

    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,base=super().update(t,th,thd,tau_m,tau_s,grip)
        lifted=self.light_witness.feed(t,th,tau_s,grip)
        return residual,bool(base or lifted)
