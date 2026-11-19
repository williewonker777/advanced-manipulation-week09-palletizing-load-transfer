"""Apply the signed continuation only when the original model is observing the carried payload."""
from sim import config as C
from experiments.r15_delayed_wrench.observer import LAG, NormalForce
from experiments.r24_composed.detector import Detector as Joint
from experiments.r25_fragment.event import SignedEvent


class ContinuationMixin:
    def reset(self):
        super().reset()
        self.continuation = SignedEvent(predict=True)
        self.witness = None
        self.static_bias = None
        if getattr(self.ctrl, 'MODE', None) == 'torque' and self.normal_model is not None:
            self.witness = self.new_witness()

    def new_witness(self):
        return NormalForce(self.nom, self.nom_box, dt=C.DT, viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
        if self.static_bias is None:
            if self.witness is not None:
                self.witness.update(t, th, tau_s, grip, project=False)
                if self.witness.bias is not None:
                    self.static_bias = self.witness.bias.copy()
                    self.witness = None
            return residual, report
        n = self.normal_model
        if not n.last_active:
            # last_error is retained even when the old predictor is inactive.
            # A stale value is not an independent observation of present force.
            if self.witness is not None:
                self.witness = None
                self.continuation.reset()
            return residual, report
        if self.witness is None:
            self.witness = self.new_witness()
            self.witness.bias = self.static_bias.copy()
        project = n.detected or self.continuation.observed or self.continuation.bridge
        sample = self.witness.update(t, th, tau_s, grip, project=project)
        if not project:
            if self.continuation.n:
                self.continuation.reset()
            return residual, report
        extra = self.continuation.feed(n.detected, n.last_error, None if sample is None else sample.force_n,
                                       grip, self._shared.release_persistence, LAG, n.threshold)
        return residual, bool(report or extra)


class Detector(ContinuationMixin, Joint):
    """The R24 joint detector plus phase-valid signed horizontal force continuation."""
