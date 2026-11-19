import unittest

from experiments.r22_census import check


class CorrectedInputAssessmentTests(unittest.TestCase):
    def test_one_reference_rise_cannot_explain_two(self):
        self.assertEqual(check.unmatched_indices([99,100,110],[100,110]),[100])
        self.assertEqual(check.unmatched_indices([100,102],[99,101]),[])
        self.assertEqual(check.unmatched_indices([101,102],[100]),[102])

    def test_derive_instead_of_assuming_zero_at_nominal_onset(self):
        import numpy as np
        example=None
        for onset in (1.38,1.41,1.44,1.47,1.50,1.9,2.3,2.7,3.1,3.4,3.80,3.83,3.86,3.89,3.92):
            j=int(np.searchsorted(check.TIME,onset))
            if check.TIME[j] > onset:
                example=(onset,j);break
        self.assertIsNotNone(example)
        t,k=example
        _,first,last=check.schedule(t,(10.,0.,0.))
        self.assertEqual(first,k)
        self.assertIn(last-k,(119,120))

    def test_no_force_never_has_an_active_sample(self):
        sha,first,last=check.schedule(None,None)
        self.assertEqual(len(sha),64)
        self.assertIsNone(first);self.assertIsNone(last)
