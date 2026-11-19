"""Compose two disjoint physical-state corrections; evaluate the joint behavior rather than adding scores."""
from experiments.r24_loaded.dc import Detector as DriveDetector
from experiments.r24_loaded.memory import MemoryYaw
from experiments.r24_repeat.trial import Detector as PickupDetector, PickupContinuousBalance


class JointBalance(MemoryYaw, PickupContinuousBalance):
    """Both components share one causal momentum history, including at pickup."""
    admitted = False

    def feed(self, th, tau, mass_row, grip):
        if self.admitted:
            # MemoryYaw delegates to PickupContinuousBalance before attributing
            # any post-transport component to observed support.
            return super().feed(th, tau, mass_row, grip)
        value = PickupContinuousBalance.feed(self, th, tau, mass_row, grip)
        self.raw_value, self.reaction = value, 0.
        return value


class Detector(DriveDetector, PickupDetector):
    def reset(self):
        super().reset()
        if self.drive is not None:
            self.yaw = JointBalance(self.box_mass, self._shared, self.yaw.weights)
