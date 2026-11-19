"""R31 held identity plus one normal-calibrated signed vertical carry witness."""
from experiments.r31_held_identity.candidate import Detector as Parent
from experiments.r33_carry_vertical.witness import VerticalCarryWitness


MODEL = {
    "schema": "r33-carry-vertical-model.v1",
    "threshold_N": 2.823020991268468,
    "window_samples": 40,
    "persistence_samples": 2,
    "plan_sha256": "b59d224e9e73e73b96181a429cbf5f0c8fc8b72435cb184e91f6c1efa19ade93",
    "calibration_sha256": "3b90ac81466d367151f0bc23ba0b73c02417dc77ffa0fd8a9361a27a5a1cdaa8",
}


class Detector(Parent):
    """Preserve the parent and OR a source-owned carried vertical notice."""

    def reset(self):
        super().reset()
        self.vertical_carry = VerticalCarryWitness(
            MODEL["threshold_N"], MODEL["window_samples"], MODEL["persistence_samples"])
        self.vertical_carry_extra = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
        projection = self.r27_projection.last_force
        shared = self._shared.last_residual_wrench
        eligible = bool(grip and not self._shared.last_table_allowed)
        extra = self.vertical_carry.extra(
            None if projection is None else float(projection[2]),
            None if shared is None else float(shared[2]),
            eligible,
            bool(incumbent),
        )
        self.vertical_carry_extra = bool(extra)
        return residual, bool(incumbent or extra)
