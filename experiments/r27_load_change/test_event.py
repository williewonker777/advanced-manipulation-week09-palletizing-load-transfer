"""Signed window, reset and native-alarm invariants, independent of contact labels."""
import unittest

import numpy as np

from experiments.r27_load_change.event import Changes, Event, WIDTH


class ChangeTests(unittest.TestCase):
    def test_block_means_and_reset(self):
        f=Changes(); out=None
        for i in range(2*WIDTH):
            out=f.feed([i,-i])
        np.testing.assert_allclose(out[0],[19.5,-19.5])
        np.testing.assert_allclose(out[1],[59.5,-59.5])
        out=f.feed([80.,-80.])
        np.testing.assert_allclose(out[0],[20.5,-20.5])
        np.testing.assert_allclose(out[1],[60.5,-60.5])
        self.assertIsNone(f.feed(None));self.assertIsNone(f.feed([80.,-80.]))

    def test_native_is_never_masked_and_untracked_noise_is_quiet(self):
        f=Event([2.]*8); gen=np.random.default_rng(342)
        for native,value in zip(np.arange(500)%31==0,gen.uniform(-.1,.1,(500,2))):
            self.assertEqual(f.feed(value,native),native)
        self.assertTrue(f.feed(None,True))
        self.assertFalse(f.feed(None,False))

    def test_one_excursion_does_not_add_a_return_edge_and_can_yield_to_native(self):
        for inherited in (False,True):
            f=Event([1.]*8);report=[]
            for i in range(420):
                native=inherited and 167<=i<173
                report.append(f.feed([5.,0.] if 120<=i<240 else [0.,0.],native))
                if native:self.assertTrue(report[-1])
            self.assertEqual(sum(v and (not i or not report[i-1]) for i,v in enumerate(report)),1)
            self.assertFalse(report[-1])

    def test_invalid_model_and_nonfinite(self):
        for bad in ([0]*8, [1]*2, [float('nan')]*8):
            with self.assertRaises(ValueError):Event(bad)
        f=Event([1]*8)
        self.assertTrue(f.feed([float('nan'),0.],True))
