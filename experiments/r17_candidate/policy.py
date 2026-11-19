"""Dispatch at reset; all supported positive torque episodes run the same candidate."""
import my_detector
import r1_torque
from experiments.r16_payload_contract.controller import attach
from experiments.r17_candidate.domain import check_calibration_ranges, supported_payload

# Cached reference scheduling can have sub-nanoradian non-clock differences.
# Physics tolerances and unchanged trigger events are qualified, not bit identity.
NONCLOCK_EQUIVALENT_TO_D = False


class Controller:
    MODE = 'torque'

    def __init__(self, height=None, early=None):
        self._height, self._early = height, early
        self._active = None

    def reset(self):
        self.loaded_policy = supported_payload(self.box_mass, self.box_com)
        if self.loaded_policy:
            check_calibration_ranges(self._height, self._early)
            from experiments.r17_candidate.timing import Controller as Loaded
            active = Loaded(self._height, self._early)
        else:
            active = r1_torque.Controller()
        attach(self, active)
        active.reset()
        self._active = active
        self.compute = active.compute

    def __getattr__(self, key):
        impl = vars(self).get('_active')
        if impl is None:
            raise AttributeError(key)
        return getattr(impl, key)


class Detector:
    def reset(self):
        in_range = supported_payload(self.box_mass, self.box_com)
        candidate_ctrl = isinstance(self.ctrl, Controller)
        if candidate_ctrl and (self.ctrl._active is None or self.ctrl.loaded_policy != in_range):
            raise ValueError("reset controller and consistent known payload required")
        if candidate_ctrl and in_range:
            from experiments.r17_candidate.observer import Detector as LoadedDetector
            active = LoadedDetector()
        else:
            # Legacy/servo controllers have no R17 observer calibration.
            active = my_detector.Detector()
        attach(self, active)
        active.ctrl = self.ctrl._active if candidate_ctrl else self.ctrl
        active.reset()
        self._active = active
        self.update = active.update

    def __getattr__(self, key):
        impl = vars(self).get('_active')
        if impl is None:
            raise AttributeError(key)
        return getattr(impl, key)
