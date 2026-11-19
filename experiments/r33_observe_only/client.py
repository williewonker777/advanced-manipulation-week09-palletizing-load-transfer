"""Observe unqualified weak-axial notices without connecting them to the R31 alarm."""
from dataclasses import dataclass

from experiments.r33_vertical_composition.candidate import Detector as Research


@dataclass(frozen=True)
class Observation:
    r31_alarm: bool
    pickup_evidence: bool
    carry_evidence: bool
    advisory_started: bool


class ObserveOnly(Research):
    """Research adapter. The value returned to the simulator is the R31 Boolean only."""
    def reset(self):
        super().reset()
        self.observation = Observation(False, False, False, False)
        self._advisory_live = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, _ = super().update(t, th, thd, tau_m, tau_s, grip)
        pickup, carry = bool(self.pick_event), bool(self.carry_event)
        live = pickup or carry
        self.observation = Observation(bool(self.parent_event), pickup, carry,
                                       live and not self._advisory_live)
        self._advisory_live = live
        return residual, self.observation.r31_alarm
