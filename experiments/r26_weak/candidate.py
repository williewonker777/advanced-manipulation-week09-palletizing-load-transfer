"""R25 composed OR an unowned completed weak-pulse lifecycle."""
from experiments.r25_composed.detector import Detector as R25
from experiments.r25_weak.world_horizontal import WorldHorizontalWitness
from experiments.r26_weak.lifecycle import PulseLifecycle


class WeakLifecycleMixin:
    minimum_load_fraction = .8

    def reset(self):
        super().reset()
        self.r26_force = WorldHorizontalWitness(
            self.nom, self.nom_box, threshold=8.394415779390961,
            support_uncertainty_per_n=.2)
        self.r26_lifecycle = PulseLifecycle()

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
        self.r26_force(t, th, tau_s, grip, self._shared)
        eligible = bool(grip) and self._shared.load_fraction >= self.minimum_load_fraction
        extra = self.r26_lifecycle.feed(self.r26_force.last_vector,
                                        self.r26_force.last_score,
                                        eligible, bool(incumbent))
        return residual, bool(incumbent or extra)


class Detector(WeakLifecycleMixin, R25):
    """Standalone frozen development candidate; incumbent is unconditional."""
