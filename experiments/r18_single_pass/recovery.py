"""Run the same continuously rearming R18 decision on single-pass innovations."""
from experiments.r18_single_pass.detector import Detector as SinglePass, _SinglePass
from experiments.r18_single_pass.reference_recovery import RecoveryState


class Detector(SinglePass):
    def reset(self):
        super().reset()
        impl = self._impl
        if not isinstance(impl, _SinglePass):
            return  # zero or servo: reset has already bound the unchanged R10 detector
        inner = self.update
        impl.recovery = RecoveryState()
        impl.recovery_detected = impl.recovery_instantaneous = False

        def observe(*args):
            residual, _ = inner(*args)
            impl.recovery_instantaneous = bool(impl.coincidence_channels.any())
            impl.recovery_detected = impl.recovery.update(impl.recovery_instantaneous,
                                                          armed=impl.prediction_armed)
            coupled = (impl.cf_detected or getattr(impl.normal_model, 'detected', False)
                       or getattr(impl, 'r5_extra', False) or getattr(impl, 'r8_extra', False))
            return residual, bool(coupled or impl.recovery_detected)

        self.update = observe
