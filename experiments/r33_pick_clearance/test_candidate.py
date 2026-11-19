import unittest
import numpy as np
from experiments.r33_pick_clearance.candidate import Route
from experiments.r33_pick_vertical.state import expected

class ClearanceTest(unittest.TestCase):
    def setUp(self):
        self.route=Route()
    def drive(self,heights,vector,parent=()):
        out=[]
        for k,h in enumerate(heights):
            baseline=expected(h,self.route.medians)
            force=np.array(vector,dtype=float);force[2]+=baseline[0]
            out.append(self.route.classify(h,force,vector[2]+baseline[1],k in parent,18.,4))
        return out
    def test_history_is_not_permission_to_report_on_table(self):
        self.assertFalse(any(self.drive([.002]*24,(0,0,-6))))
        self.assertEqual(sum(self.drive([.005]*12,(0,0,-6))),4)
    def test_diagonal_is_not_a_vertical_source(self):
        self.assertFalse(any(self.drive([.016]*40,(8,8,8))))
    def test_high_force_left_to_native(self):
        self.assertFalse(any(self.drive([.025]*40,(0,0,35))))
    def test_unclaimed_bounded_and_native_priority(self):
        self.assertEqual(sum(self.drive([.02]*60,(0,0,10))),4)
        self.route=Route()
        self.assertFalse(any(self.drive([.02]*60,(0,0,10),{7})))
    def test_no_new_source_after_clearance_band(self):
        self.assertFalse(any(self.drive([.048]*40,(0,0,10))))
    def test_held_source_may_finish_then_reset(self):
        self.drive([.023]*18,(0,0,-6))
        self.drive([.024]*40,(0,0,0))
        self.assertEqual(sum(self.drive([.027]*28,(0,0,6))),4)
    def test_missing_sensor_does_not_emit(self):
        self.assertFalse(any(self.drive([.02]*32,(np.nan,0,8))))
