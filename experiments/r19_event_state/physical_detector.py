"""R18-fast onset with unloading defined by the frozen physical floors."""
from __future__ import annotations

import numpy as np

from experiments.r18_single_pass.detector import _SinglePass
from experiments.r18_single_pass.recovery import Detector as Recovery
from experiments.r19_event_state.state import EventState


class Detector(Recovery):
    """Drop-in detector; inspect ``_impl.event_state`` after ``reset``."""

    def reset(self):
        super().reset()
        impl = self._impl
        if not isinstance(impl, _SinglePass):
            return
        r18_update = self.update
        impl.event_state = EventState()
        impl.event_active = False
        impl.unload_instantaneous = False

        def observe(*args):
            residual, _ = r18_update(*args)
            supported = bool(impl.prediction_armed and impl.prediction_supported)
            impl.unload_instantaneous = supported and bool(
                np.all(impl.coincidence_physical_ratio <= 1.0)
            )
            impl.event_active = impl.event_state.update(
                impl.recovery_detected,
                impl.unload_instantaneous,
                armed=impl.prediction_armed,
            )
            learned = bool(
                getattr(impl.normal_model, "detected", False)
                or getattr(impl, "r5_extra", False)
                or getattr(impl, "r8_extra", False)
            )
            return residual, bool(impl.cf_detected or learned or impl.event_active)

        self.update = observe
