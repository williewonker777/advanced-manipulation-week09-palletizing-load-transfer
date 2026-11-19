"""Optional slow-only handoff, downstream of unmodified R19 impulse detection."""
from experiments.r18_single_pass.detector import _SinglePass
from experiments.r19_impulse_end.detector import Detector as Impulse
from experiments.r20_slow_handoff.state import SlowHandoff


class Detector(Impulse):
    def reset(self):
        super().reset()
        impl = self._impl
        if not isinstance(impl, _SinglePass):
            return
        base_update = self.update
        state = SlowHandoff()

        def observe(*args):
            residual, detected = base_update(*args)
            carry = state.update(impl.impulse_event_active, impl.cf_alarm > 0, impl.cf_detected,
                                 impl._shared.release_persistence,
                                 impl.prediction_armed and impl.prediction_supported)
            return residual, bool(detected or carry)

        self.update = observe
