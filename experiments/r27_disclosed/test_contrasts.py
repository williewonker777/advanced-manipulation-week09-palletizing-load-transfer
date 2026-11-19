import unittest

import numpy as np

from experiments.r27_disclosed import contrasts


class DirectionTests(unittest.TestCase):
    def test_three_independent_signed_axes_preserve_magnitude_duration(self):
        source=[2.1,72,[3.,-4.,2.]]
        result=contrasts.inputs(source)
        self.assertIs(result['registered'],source)
        self.assertTrue(all(v[:2]==source[:2]for v in result.values()))
        vectors=np.array([result[k][2]for k in ('registered','perpendicular_xy','perpendicular_tilt')])
        np.testing.assert_allclose(vectors@vectors.T,np.eye(3)*29.,atol=1e-10)
        np.testing.assert_array_equal(result['reverse'][2],-vectors[0])

    def test_requires_strict_and_during_genuine_force(self):
        b={'first_nonzero_index':45,'last_nonzero_index':61}
        self.assertFalse(contrasts.acquisition_in_force(dict(strict=False,attributable_rise_indices=[47]),b))
        self.assertFalse(contrasts.acquisition_in_force(dict(strict=True,attributable_rise_indices=[62]),b))
        self.assertTrue(contrasts.acquisition_in_force(dict(strict=True,attributable_rise_indices=[61]),b))
