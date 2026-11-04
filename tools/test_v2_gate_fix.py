from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import v2_detector_candidate as L1
import v2_detector_geometry as L3
from tools.production_audit import audit_text


class HeightGateTests(unittest.TestCase):
    def packet(self,z,mass):
        class Packet:
            table_heights=np.array([.1,.2]);height_band=.035;q_prev=np.zeros(6)
            box_mass=mass
            def _point_jacobian(self,q):return None,np.array([0.,0.,z]),None
        return Packet()

    def test_height_gate_independent_of_mass(self):
        for mass in [0.,.25,4.,8.]:
            self.assertFalse(L3.outside_table_height_band(self.packet(.1,mass),True))
            self.assertFalse(L3.outside_table_height_band(self.packet(.23,mass),True))
            self.assertTrue(L3.outside_table_height_band(self.packet(.3,mass),True))
            self.assertFalse(L3.outside_table_height_band(self.packet(.3,mass),False))

    def test_old_eligibility_bug_at_empty_payload(self):
        packet=self.packet(.1,0.)
        old=L1.SharedEstimator._table_allowed(packet,True,.1)
        self.assertFalse(old)  # not old would wrongly enable learning here
        self.assertFalse(L3.outside_table_height_band(packet,True))

    def test_positive_mass_original_protocol_equivalence(self):
        for mass in [.25,4.,8.]:
            for z in [-.1,.08,.12,.23,.3]:
                packet=self.packet(z,mass)
                self.assertEqual(L3.outside_table_height_band(packet,True),
                                 not L1.SharedEstimator._table_allowed(packet,True,z))

    def test_weights_and_thresholds_unchanged(self):
        self.assertEqual(L1.MODEL_PARAMETERS,L3.MODEL_PARAMETERS)
        self.assertEqual(audit_text((ROOT/'v2_detector_geometry.py').read_text())['errors'],[])


if __name__=='__main__':unittest.main()
