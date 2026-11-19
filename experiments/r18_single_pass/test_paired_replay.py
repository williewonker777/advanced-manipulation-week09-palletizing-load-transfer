"""Independent public checks of the two-pulse reproduction contract."""
import unittest

import numpy as np

from experiments.r18_paired_impulses.scoring import valid_genuine
from .paired_replay import KINDS, scalar_norm_check, signature


class ReproductionTests(unittest.TestCase):
    def test_digest_is_order_independent_but_keeps_zero_distinct_from_a_miss(self):
        self.assertEqual(signature({'a':[2],'b':None}),signature({'b':None,'a':[2]}))
        self.assertNotEqual(signature({'events':[]}),signature({'events':[0.]}))
        self.assertEqual(len(set(KINDS)),5)

    def test_reference_veto_looks_at_no_second_pulse_twin_not_just_normal(self):
        t=np.arange(6000)*.001
        none={'t':t,'detect':np.zeros(6000,bool)}
        first_only={'t':t,'detect':np.zeros(6000,bool)};first_only['detect'][5430]=True
        both={'t':t,'detect':np.zeros(6000,bool)};both['detect'][[5415,5441,5510]]=True
        legal=valid_genuine(both,[none,first_only],[5415,5441,5510],20,5.415)
        np.testing.assert_array_equal(legal,[5510])

    def test_reference_signature_fails_on_missing_tool_or_changed_detection_time(self):
        arm={'rises_s':{'both':[3.973,5.487]},'gates':{'tool':True,'signed':True}}
        off={**arm,'rises_s':{'both':[3.973,5.488]}}
        self.assertNotEqual(signature(arm),signature(off))
        self.assertNotEqual(signature(arm),signature({**arm,'gates':{'signed':True}}))


class ForceNormChecks(unittest.TestCase):
    def test_scalar_stock_reduction_and_actual_force_corruption(self):
        vectors=np.random.default_rng(321).standard_normal((500,3))
        applied=np.array([np.linalg.norm(row) for row in vectors])
        inputs={'f_ext':applied.copy(),'f_ext_xyz_native':vectors}
        self.assertTrue(scalar_norm_check(inputs)['stock_scalar_bit_exact'])
        inputs['f_ext'][10]=0.
        self.assertFalse(scalar_norm_check(inputs)['stock_scalar_bit_exact'])
