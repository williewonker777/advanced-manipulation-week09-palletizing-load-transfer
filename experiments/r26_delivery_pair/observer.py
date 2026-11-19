"""Public encoder/JTS-only, measured-detachment force reconstruction."""
from sim import config as C
from experiments.r15_delayed_wrench.observer import NormalForce


class ReleasedVector:
    """No sensor input crosses the observed payload transition."""
    def __init__(self, empty, loaded):
        self.empty, self.loaded = empty, loaded
        self.calibration = self.observer = None
        self.calibration = self._new()
        self.bias = None
        self.was_closed = False

    def _new(self):
        return NormalForce(self.empty, self.loaded, dt=C.DT, viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)

    def feed(self,t,th,spring,closed):
        if self.calibration is not None:
            self.calibration.update(t,th,spring,bool(closed),project=False)
            if self.calibration.bias is not None:
                self.bias = self.calibration.bias.copy()
                self.calibration = None
        released = self.was_closed and not closed
        self.was_closed = bool(closed)
        if closed:
            self.observer = None
        if released and self.bias is not None:
            self.observer = self._new()
            self.observer.bias = self.bias.copy()
        if self.observer is not None:
            sample = self.observer.update(t,th,spring,False)
            return None if sample is None else sample.force_n.copy()
        return None
