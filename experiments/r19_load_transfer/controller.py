"""Support-weight the R17 horizontal information action during load transfer."""
from __future__ import annotations

import numpy as np

from experiments.r17_information_prefix.controller import Controller as Information


def transferred_information_weight(information_weight, hybrid_alpha):
    """Causal fraction of the horizontal COM action still carried by the robot."""
    values = np.asarray([information_weight, hybrid_alpha], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("finite information and handoff weights required")
    if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
        raise ValueError("weights outside [0,1]")
    return float(np.clip(information_weight, 0.0, 1.0)
                 * (1.0 - np.clip(hybrid_alpha, 0.0, 1.0)))


class Controller(Information):
    """R17 with only its extra horizontal COM action handed to the table."""

    def reset(self):
        super().reset()
        self.transfer_information_weight = 0.0
        self.transfer_handoff = 0.0
        self.transfer_reductions = 0
        self.transfer_max_com_delta_m = 0.0
        self.transfer_trace = []

    def _apply_transferred_information(self, t):
        self.transfer_handoff = max(self.transfer_handoff, float(self.hybrid_alpha))
        target = transferred_information_weight(self.information_weight,
                                                self.transfer_handoff)
        action = target * self.information_offset
        # Preserve the operator-provided physical COM and the vertical coordinate.
        com = np.asarray(self._provided_com, dtype=float).copy()
        com[:2] += action[:2]
        before = float(self.transfer_information_weight)
        self.nom_box.set_payload(self.box_mass, com)
        self.transfer_information_weight = target
        if target < before - 1e-15:
            self.transfer_reductions += 1
        self.transfer_max_com_delta_m = max(
            self.transfer_max_com_delta_m,
            float(np.linalg.norm(self.information_weight*self.information_offset-action)),
        )
        if 3.75 <= float(t) <= 3.91:
            self.transfer_trace.append({
                "t": float(t),
                "information_weight": float(self.information_weight),
                "hybrid_alpha": float(self.hybrid_alpha),
                "latched_handoff": float(self.transfer_handoff),
                "effective_weight": target,
                "com": com.tolist(),
            })

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        command = super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
        self._apply_transferred_information(t)
        return command
