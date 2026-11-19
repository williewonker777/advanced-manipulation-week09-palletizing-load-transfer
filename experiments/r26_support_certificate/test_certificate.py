import unittest

import numpy as np

from experiments.r26_support_certificate.certificate import (
    SupportConeCertificate, lifecycle_eligible, signed_cone_lower_bound)


class Packet:
    friction_mu = .1
    last_table_allowed = True
    last_table_wrench = np.array([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])


class SupportCertificateTest(unittest.TestCase):
    def test_full_disk_can_reverse_against_signed_direction(self):
        self.assertEqual(signed_cone_lower_bound([10.0, 0.0], .1, 20.0), 8.0)
        self.assertEqual(signed_cone_lower_bound([-10.0, 0.0], .1, 20.0), 8.0)

    def test_maximum_normal_is_retained_over_observer_lag(self):
        gate = SupportConeCertificate(threshold=5.0, lag=2)
        packet = Packet()
        gate.feed([10.0, 0.0], 10.0, packet)
        packet.last_table_wrench = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        gate.feed([10.0, 0.0], 10.0, packet)
        self.assertEqual(gate.last_maximum_normal, 20.0)
        self.assertEqual(gate.last_lower_bound, 8.0)

    def test_opposite_certified_vectors_are_not_coherent(self):
        gate = SupportConeCertificate(threshold=5.0, lag=0)
        packet = Packet()
        packet.last_table_allowed = False
        self.assertTrue(gate.feed([10.0, 0.0], 10.0, packet))
        self.assertFalse(gate.feed([-10.0, 0.0], 10.0, packet))

    def test_certificate_only_gates_onset_not_trailing_life(self):
        self.assertFalse(lifecycle_eligible("quiet", True, False))
        self.assertTrue(lifecycle_eligible("active", True, False))
        self.assertTrue(lifecycle_eligible("rearm", True, False))
        self.assertFalse(lifecycle_eligible("active", False, True))


if __name__ == "__main__":
    unittest.main()
