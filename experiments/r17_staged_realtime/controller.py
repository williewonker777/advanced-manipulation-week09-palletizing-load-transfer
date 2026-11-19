"""R17A scheduled reference plus incrementally staged public decisions."""
import numpy as np

from experiments.r14_stock_handoff.controller import Controller as HeightController
from experiments.r14_stock_handoff.controller import RAMP_START, clamp_height_mm
from experiments.r15_early_com.controller import weight
from experiments.r17_realtime.controller import Controller as R17A
from experiments.r17_staged_realtime.staged import Stream
from sim import config as C


class Controller(R17A):
    """Portable staged candidate with the unchanged `(height, early)` interface."""

    def __init__(self, height_calibration, early_calibration):
        super().__init__(height_calibration, early_calibration)
        self._stream_factory = Stream

    def reset(self):
        super().reset()
        if self.box_mass > 0:
            self.registration_stream.configure_staging(self.registration_calibration,
                                                       self.early_calibration)

    def _finish_registration(self, t):
        if self.registration_finished or t < RAMP_START:
            return
        features = self.registration_stream.finish()
        decision = self.registration_stream.finish_height_decision(features)
        if not isinstance(decision, dict) or not {"extra_height_mm", "accept", "half_width_mm"} <= set(decision):
            raise ValueError("invalid staged registration decision")
        values = np.asarray([decision["extra_height_mm"], decision["half_width_mm"]], float)
        if not np.all(np.isfinite(values)) or values[1] < 0.:
            raise ValueError("invalid staged registration values")
        self.registration_result = dict(decision)
        self.registration_accepted = bool(decision["accept"])
        self.registered_height_m = (-.001*clamp_height_mm(values[0])
                                    if self.registration_accepted else 0.)
        self.registration_finished = True; self.registration_finish_t = float(t)

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        # Once both staged decisions are installed (and on the true empty
        # fallback), return to the frozen R17A method with no staging branch.
        if self.box_mass <= 0 or self.registration_finished:
            return super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
        # Exact Early.compute ordering, replacing only its late batch decision.
        command = HeightController.compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
        if not self.early_decided and t >= 2.463:
            row = self.registration_stream.finish_early_decision()
            action = np.asarray(row["model_action_xy_mm"], float)
            if (action.shape != (2,) or np.linalg.norm(action) > 15.000001
                    or (not row["accept"] and np.any(action))):
                raise ValueError("staged early decision")
            self.early_offset[:2] = .001*action; self.early_result = row
            self.early_decided = True; self.early_decided_at_s = float(t)
        if self.early_decided and np.any(self.early_offset):
            value = weight(float(t)+C.DT)
            if value != self._early_weight:
                self.nom_box.set_payload(self.box_mass, self._provided_com+value*self.early_offset)
                self._early_weight = value; self.early_updates += 1
        if self.box_mass > 0 and not self.registration_finished:
            self.registration_stream.advance_height_stage(float(t))
        return command
