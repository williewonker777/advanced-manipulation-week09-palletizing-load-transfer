"""Exact R18 onset with an unsigned normalized-impulse event end."""
from experiments.r18_single_pass.detector import _SinglePass
from experiments.r18_single_pass.recovery import Detector as Recovery
from experiments.r19_impulse_end.state import ImpulseEndState


class Detector(Recovery):
    """Drop-in detector; inspect ``_impl.impulse_end_state`` after reset."""
    def reset(self):
        super().reset();impl=self._impl
        if not isinstance(impl,_SinglePass):return
        r18_update=self.update;impl.impulse_end_state=ImpulseEndState();impl.impulse_event_active=False
        def observe(*args):
            residual,_=r18_update(*args)
            impl.impulse_event_active=impl.impulse_end_state.update(
                impl.recovery_detected,impl.coincidence_physical_ratio,
                armed=impl.prediction_armed)
            learned=bool(getattr(impl.normal_model,'detected',False) or getattr(impl,'r5_extra',False) or getattr(impl,'r8_extra',False))
            return residual,bool(impl.cf_detected or learned or impl.impulse_event_active)
        self.update=observe
