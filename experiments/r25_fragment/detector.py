"""Preserve every inherited alert; bridge a fragmented R10 normal-model event with signed inverse dynamics."""
from sim import config as C
from experiments.r15_delayed_wrench.observer import LAG, NormalForce
from experiments.r24_composed.detector import Detector as Joint
from experiments.r25_fragment.event import SignedEvent


class BridgeMixin:
    predict = False
    require_stationary = False

    def reset(self):
        super().reset()
        self.event25 = SignedEvent(self.predict, self.require_stationary)
        self.inverse25 = None
        if getattr(self.ctrl, 'MODE', None)=='torque':
            self.inverse25 = NormalForce(self.nom,self.nom_box,dt=C.DT,viscous_spring=C.D,jts_hz=C.JTS_CUTOFF)

    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual, report = super().update(t,th,thd,tau_m,tau_s,grip)
        if self.inverse25 is None:
            return residual,report
        sample=self.inverse25.update(t,th,tau_s,grip)
        n=self.normal_model
        more=self.event25.feed(bool(n and n.detected), None if not n else n.last_error,
                               None if sample is None else sample.force_n,grip,
                               self._shared.release_persistence,LAG,n.threshold if n else float('inf'))
        return residual,bool(report or more)


class Detector(BridgeMixin,Joint):
    """Two signed, independent estimates support the same continuing contact."""


class Predicted(Detector):
    """Extrapolate only an unloading trend through the inverse estimator delay."""
    predict=True


class Stationary(Detector):
    """Stricter admission requiring a flat load or independently strong evidence."""
    require_stationary=True
