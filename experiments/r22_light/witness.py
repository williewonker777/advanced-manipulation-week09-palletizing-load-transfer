"""Extra force evidence only while lifting a light object; base detection is unconditional."""
import math

from sim import config as C, trajectory as T
from experiments.r15_delayed_wrench.observer import NormalForce
from experiments.r22_tangent.frozen import Detector as Calibration
from experiments.r22_tangent.state import Evidence, load_class

TARGET_N = 10.
GRAVITY = 9.81


def eligible(mass):
    return math.isfinite(mass) and 0 <= mass * GRAVITY <= TARGET_N


def lift_stop(waypoints, grasp):
    return min(t for t, _ in waypoints if t > grasp)


class LightWitness:
    def __init__(self, mass, nominal, loaded):
        self.mass = mass
        self.enabled = eligible(mass)
        self.nominal, self.loaded = nominal, loaded
        self.stop = lift_stop(T.WAYPOINTS, T.T_GRASP)
        self.state = Evidence()
        self.calibrated = None
        self.observer = self._new() if self.enabled else None
        self.in_lift = False

    def _new(self):
        return NormalForce(self.nominal, self.loaded, dt=C.DT, viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)

    def feed(self, t, th, tau, grip):
        if not self.enabled:
            return False
        if self.calibrated is None:
            # The inherited estimator accepts its static startup calibration before 0.5 s.
            # Failed calibration disables only this optional port, never the base detector.
            if self.observer is not None and t < .5 and not grip:
                self.observer.update(t, th, tau, False, project=False)
                if self.observer.bias is not None:
                    self.calibrated = self.observer.bias.copy()
                    self.observer = None
            return False
        open_window = bool(grip) and T.T_GRASP <= t < self.stop
        continue_event = self.in_lift and bool(grip) and self.state.active
        if not open_window and not continue_event:
            if self.in_lift:
                self.observer = None
                self.state = Evidence()
                self.in_lift = False
            return False
        if not self.in_lift:
            self.observer = self._new()
            self.observer.bias = self.calibrated.copy()
            self.in_lift = True
        sample = self.observer.update(t, th, tau, grip)
        group = load_class(self.mass, grip)
        return self.state.feed(None if sample is None else sample.force_n, group, Calibration.bounds_tangent[group])
