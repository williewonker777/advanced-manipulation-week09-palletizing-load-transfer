"""R19 detector with causal handoff into an already rising incumbent branch."""
from experiments.r18_single_pass.detector import _SinglePass
from experiments.r19_impulse_end.detector import Detector as R19Detector
from experiments.r20_event_bridge.state import HandoffState


class Detector(R19Detector):
    """Experimental drop-in detector; not a submission/default replacement."""

    def reset(self):
        super().reset()
        impl = self._impl
        if not isinstance(impl, _SinglePass):
            return
        r19_update = self.update
        impl.handoff_state = HandoffState()
        impl.handoff_active = False

        def observe(*args):
            residual, detected = r19_update(*args)
            incumbent_evidence = bool(impl.cf_alarm or impl.cf_fast)
            impl.handoff_active = impl.handoff_state.update(
                impl.impulse_event_active,
                incumbent_evidence,
                impl.cf_detected,
                impl.cf_clear,
                impl._shared.release_persistence,
                armed=impl.prediction_armed,
            )
            return residual, bool(detected or impl.handoff_active)

        self.update = observe
