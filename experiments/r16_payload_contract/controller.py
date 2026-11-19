"""Known-payload dispatch to unchanged implementations, once at episode reset."""
import math
import numpy as np

import my_detector
import r1_torque
from experiments.r14_stock_handoff.detector import Detector as Coupled
from experiments.r15_early_com.controller import Controller as Early
from experiments.r16_balanced.controller import Controller as Balanced


def has_payload(mass, com):
    m = float(mass)
    if not math.isfinite(m) or m < 0.:
        raise ValueError("payload mass must be known, finite and nonnegative")
    center = np.asarray(com, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("known finite three-axis COM required")
    return m > 0.


def attach(owner, impl):
    # This is the documented runner interface, not the estimated plant.
    for name in ('nom', 'nom_box', 'box_mass', 'box_com'):
        setattr(impl, name, getattr(owner, name))


class Controller:
    MODE = 'torque'
    loaded_class = Balanced

    def __init__(self, height_calibration=None, early_calibration=None):
        self._height, self._xy = height_calibration, early_calibration
        self._active = None

    def reset(self):
        loaded = has_payload(self.box_mass, self.box_com)
        impl = self.loaded_class(self._height, self._xy) if loaded else r1_torque.Controller()
        attach(self, impl)
        impl.reset()
        self._active = impl
        # Bind directly, with no per-tick switching or new arithmetic.
        self.compute = impl.compute

    def __getattr__(self, name):
        impl = vars(self).get('_active')
        if impl is None:
            raise AttributeError(name)
        return getattr(impl, name)


class Baseline(Controller):
    """Matched R15 comparator; the historical unwrapped R15 is reported separately."""
    loaded_class = Early


class Detector:
    def reset(self):
        loaded = has_payload(self.box_mass, self.box_com)
        torque = getattr(self.ctrl, 'MODE', None) == 'torque'
        impl = Coupled() if loaded and torque else my_detector.Detector()
        attach(self, impl)
        actual_controller = getattr(self.ctrl, '_active', self.ctrl)
        if actual_controller is None:
            raise ValueError("the controller must be reset before the detector")
        impl.ctrl = actual_controller
        impl.reset()
        self._active = impl
        self.update = impl.update

    def __getattr__(self, name):
        impl = vars(self).get('_active')
        if impl is None:
            raise AttributeError(name)
        return getattr(impl, name)
