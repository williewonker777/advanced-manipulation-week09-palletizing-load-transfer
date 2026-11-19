"""R26 C3 plus normal-only signed lateral loaded-force onset detection."""
from __future__ import annotations

from collections import deque

import numpy as np

from experiments.r15_delayed_wrench.observer import LAG, NormalForce
from experiments.r26_yaw_acquisition.candidate import Detector as R26
from experiments.r27_loaded_observer.lateral_witness import LoadedLateralOnset
from sim import config as C


MODEL_SHA256 = "7e33eff6766c99f374533dd44b1cf5c0128afa771256c7f1e050d529630b1937"
MODEL = {
    "schema": "r27-loaded-lateral-model.v1",
    "proposal": "normal_vertical_coupling_v3",
    "fit_rule": "OLS xy change from [1,z change]; threshold 1.05 * fit max + 0.05 N",
    "grip_stability_samples": 96,
    "onset_lag_samples": 8,
    "sustain_lag_samples": 32,
    "onset_persistence_samples": 3,
    "pending_samples": 25,
    "pulse_samples": 8,
    "rearm_quiet_samples": 25,
    "onset_xy_from_z": [
        [0.0036191655208299978, 0.004439019397068548],
        [-0.27781989114613936, -0.20563339578666828],
    ],
    "sustain_xy_from_z": [
        [0.011690440780641973, 0.015393630159384504],
        [-0.32003120171129956, -0.08215416105297314],
    ],
    "onset_fit_residual_max_N": 6.066470320994524,
    "sustain_fit_residual_max_N": 5.217971677955214,
    "onset_threshold_N": 6.41979383704425,
    "sustain_threshold_N": 5.528870261852975,
    "ownership_release_N": 2.7644351309264876,
    "protocol_sha256": "efcf058b61a1eab589a99e4039f1335db37598920a07b43a555b8b3b21f8d7af",
    "plan_sha256": "9a00dcf6584f51c47a7f51c1c2ac200b0b76b4ad54edd3446149b6a366672fea",
    "fit_npz_sha256": "be2b48cff6f94d3e71351b01525e5eb38ec98a5449627c965d54420e326d1cf9",
}


class Detector(R26):
    """Preserve R26 sources and OR one bounded unsupported-load onset pulse."""

    def reset(self):
        super().reset()
        self.loaded_force_observer = NormalForce(
            self.nom, self.nom_box, dt=C.DT, viscous_spring=C.D,
            jts_hz=C.JTS_CUTOFF)
        self.loaded_force_state = LoadedLateralOnset(MODEL)
        self.loaded_force_eligibility = deque(maxlen=LAG + 1)
        self.loaded_force_grip_run = 0
        self.loaded_force_extra = False
        self.loaded_force_eligible = False
        self.loaded_force_sample = None

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
        self.loaded_force_grip_run = self.loaded_force_grip_run + 1 if grip else 0
        self.loaded_force_eligibility.append((
            int(self.loaded_force_grip_run),
            bool(getattr(self._shared, "last_table_allowed", False))))
        sample = self.loaded_force_observer.update(t, th, tau_s, grip)
        self.loaded_force_sample = (None if sample is None else
                                    np.asarray(sample.force_n, float).copy())
        extra = False
        eligible = False
        if sample is not None and len(self.loaded_force_eligibility) == (
                self.loaded_force_eligibility.maxlen):
            effective_grip_run, effective_table_allowed = self.loaded_force_eligibility[0]
            eligible = bool(
                effective_grip_run >= MODEL["grip_stability_samples"]
                and not effective_table_allowed)
            if eligible:
                effective_tick = int(round(1000.0 * float(sample.effective_time)))
                extra = self.loaded_force_state.feed(
                    effective_tick, sample.force_n, effective_grip_run,
                    bool(incumbent))
            else:
                self.loaded_force_state.reset()
        else:
            self.loaded_force_state.reset()
        self.loaded_force_extra = bool(extra)
        self.loaded_force_eligible = eligible
        return residual, bool(incumbent or extra)
