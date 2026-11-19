import unittest
from experiments.r22_guarded.candidate import GuardedReleaseBalance
from experiments.r23_detached.free import Detector as Free
from experiments.r24_composed.detector import Detector, JointBalance
from experiments.r24_loaded.dc import Detector as Drive
from experiments.r24_loaded.memory import MemoryYaw
from experiments.r24_repeat.trial import Detector as Pickup, PickupContinuousBalance


class CompositionContract(unittest.TestCase):
    def test_one_report_update_reuses_pickup_then_original(self):
        order=Detector.__mro__;self.assertLess(order.index(Drive),order.index(Pickup))
        self.assertLess(order.index(Pickup),order.index(Free));self.assertEqual(order.count(Free),1)

    def test_loaded_input_wraps_one_pickup_momentum_balance(self):
        order=JointBalance.__mro__;self.assertLess(order.index(MemoryYaw),order.index(PickupContinuousBalance))
        self.assertLess(order.index(PickupContinuousBalance),order.index(GuardedReleaseBalance))
        self.assertEqual(order.count(GuardedReleaseBalance),1)
