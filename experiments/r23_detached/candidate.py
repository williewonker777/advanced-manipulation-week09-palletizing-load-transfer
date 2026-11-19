"""Study-only addition to R22 LIGHT for an unloaded arm after actual detachment."""
import math

from sim import config as C
from experiments.r15_delayed_wrench.observer import NormalForce
from experiments.r22_light.candidate import Detector as Light
from experiments.r22_tangent.state import Evidence
from experiments.r22_tangent.frozen import Detector as Tangent
from experiments.r23_detached.state import OnsetGate


class ReleasedForce:
    def __init__(self, mass, empty, loaded, require_onset):
        self.enabled = math.isfinite(mass) and 0 <= mass <= 8
        self.empty, self.loaded = empty, loaded
        self.require_onset = require_onset
        self.calibrator = self.new_observer() if self.enabled else None
        self.bias = self.prev_grip = self.observer = None
        self.evidence = Evidence()
        self.gate = OnsetGate()
        self.vector = None

    def new_observer(self):
        return NormalForce(self.empty, self.loaded, dt=C.DT, viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)

    def feed(self, t, encoder, jts, grip, incumbent):
        grip = bool(grip)
        detach = self.prev_grip is True and not grip
        self.prev_grip = grip
        self.vector = None
        if not self.enabled:
            return False
        if self.calibrator is not None:
            if t < .5 and not grip:
                self.calibrator.update(t, encoder, jts, False, project=False)
                if self.calibrator.bias is not None:
                    self.bias = self.calibrator.bias.copy()
                    self.calibrator = None
            else:
                self.calibrator = None
            return False
        if grip:
            self.observer = None
            return False
        if detach and self.bias is not None:
            self.observer = self.new_observer()
            self.observer.bias = self.bias.copy()
            self.evidence = Evidence()
            self.gate = OnsetGate()
        if self.observer is None:
            return False
        sample = self.observer.update(t, encoder, jts, False)
        self.vector = None if sample is None else sample.force_n[:2]
        bound = Tangent.bounds_tangent[0]
        above = self.evidence.feed(self.vector, 0, bound)
        return self.gate.feed(self.vector, above, incumbent, bound) if self.require_onset else above


class Free(Light):
    require_onset = False

    def reset(self):
        super().reset()
        self.detached = ReleasedForce(self.box_mass, self.nom, self.nom_box, self.require_onset)

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
        extra = self.detached.feed(t, th, tau_s, grip, incumbent)
        return residual, bool(incumbent or extra)


class Detector(Free):
    require_onset = True
