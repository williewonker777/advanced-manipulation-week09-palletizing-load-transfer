import unittest
from types import SimpleNamespace

import numpy as np

from experiments.r27_load_change.settled_event import Settled
from experiments.r27_load_change.projection import Projection


class SettledTests(unittest.TestCase):
    def model(self):
        return Settled(dict(joint_bounds_N=[1.]*8,separate_bounds_N=[[2.]*8]*2))

    def trace(self,phantom=False,native=False):
        f=self.model();rows=[]
        for k in range(420):
            value=6. if 120<=k<267 else 0.
            channels=np.array([[value,0.],[0. if phantom else value,0.]])
            inherited=native and 158<=k<189
            rows.append((k,inherited,f.feed(channels,inherited),f.notified))
        return rows

    def test_measurement_absent_or_single_witness_never_alarms(self):
        self.assertIsNone(Projection(2.,(0.,0.,0.)).from_integrated(SimpleNamespace(witness=None)))
        self.assertFalse(any(x[2] for x in self.trace(phantom=True)))

    def test_two_matching_steps_report_once_not_for_the_level(self):
        rows=self.trace();hits=[x[0] for x in rows if x[2]]
        self.assertEqual(len(hits),4);self.assertTrue(120<hits[0]<267)
        self.assertEqual(sum(x[3] for x in rows),1)

    def test_native_acquisition_passes_and_owns_its_plateau(self):
        rows=self.trace(native=True)
        self.assertTrue(all(x[1]==x[2]for x in rows));self.assertFalse(any(x[3]for x in rows))

    def test_clear_or_reset_input_does_not_hold_native(self):
        f=self.model();self.assertTrue(f.feed(None,True));self.assertFalse(f.feed(None,False))
