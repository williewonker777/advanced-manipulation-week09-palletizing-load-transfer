"""R25 composed OR a completed weak pulse with robust non-support origin."""
from experiments.r25_composed.detector import Detector as R25
from experiments.r25_weak.world_horizontal import WorldHorizontalWitness
from experiments.r26_weak.lifecycle import PulseLifecycle
from experiments.r26_support_certificate.certificate import (
    SupportConeCertificate, lifecycle_eligible)


class CertifiedWeakLifecycleMixin:
    """Composable optional port; incumbent output is unconditional."""

    minimum_load_fraction = .8

    def reset(self):
        super().reset()
        self.r26_cert_force = WorldHorizontalWitness(
            self.nom, self.nom_box, threshold=8.394415779390961,
            support_uncertainty_per_n=.2)
        self.r26_certificate = SupportConeCertificate()
        self.r26_cert_lifecycle = PulseLifecycle()

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
        self.r26_cert_force(t, th, tau_s, grip, self._shared)
        certified = self.r26_certificate.feed(
            self.r26_cert_force.last_raw, self.r26_cert_force.last_score, self._shared)
        physical = bool(grip) and self._shared.load_fraction >= self.minimum_load_fraction
        eligible = lifecycle_eligible(self.r26_cert_lifecycle.phase, physical, certified)
        extra = self.r26_cert_lifecycle.feed(
            self.r26_cert_force.last_vector, self.r26_cert_force.last_score,
            eligible, bool(incumbent))
        return residual, bool(incumbent or extra)


class Detector(CertifiedWeakLifecycleMixin, R25):
    """Standalone frozen development detector."""
