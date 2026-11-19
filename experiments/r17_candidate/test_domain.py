import unittest
import numpy as np
from experiments.r17_candidate.domain import CALIBRATED_RANGE_KG, check_calibration_ranges, supported_payload


class DomainTests(unittest.TestCase):
    def test_endpoints_and_full_supported_interval(self):
        lo, hi = CALIBRATED_RANGE_KG
        for value in np.linspace(lo, hi, 127):
            self.assertTrue(supported_payload(value, [0., 0., 0.]))
        for m in (0., .001, 2.5, np.nextafter(lo, 0.), np.nextafter(hi, 99.), 8.15, 100.):
            self.assertFalse(supported_payload(m, [0., 0., 0.]))

    def test_unknown_not_equivalent_to_empty(self):
        for mass, centre in ((-1.,[0.]*3),(np.nan,[0.]*3),(np.inf,[0.]*3),(0.,[1,2]),(0.,[0,np.nan,0])):
            with self.assertRaises(ValueError):
                supported_payload(mass, centre)

    def test_model_cannot_silently_change_range(self):
        good = dict(range=list(CALIBRATED_RANGE_KG))
        check_calibration_ranges(good, {'dynamics': good})
        for bad in ({}, {'range':[3.,7.]}, {'range':[0.,100.]}, {'range':[np.nan,8.]}):
            with self.assertRaises(ValueError):
                check_calibration_ranges(good, {'dynamics': bad})
