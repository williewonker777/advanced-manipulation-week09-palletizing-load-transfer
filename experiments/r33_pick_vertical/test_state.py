import json
from pathlib import Path
import unittest
import numpy as np
from .candidate import parameters
from .signal import Sample
from .state import CENTERS,Evidence,expected

class FitTest(unittest.TestCase):
    def test_training_grid_disjoint_and_pins_complete(self):
        from .lift_probe import training
        from tools.d4_robustness import MASSES
        p=parameters()
        self.assertEqual(len(p['traces']),27)
        self.assertEqual(set(p['traces']),{v['case_id'] for v in training()})
        self.assertFalse({v['mass'] for v in training()} & set(MASSES))
        self.assertTrue(all(v['collision'] is None for v in training()))
        self.assertEqual(p['centers'],CENTERS.tolist())
        self.assertGreaterEqual(min(p['bin_trajectory_counts']),26)
        self.assertGreater(p['negative_limit_N'],p['margin_N'])
    def test_disagreement_and_missing_sample(self):
        e=Evidence(np.zeros((len(CENTERS),2)))
        s=Sample(.012,7.,-7.)
        self.assertEqual([e.feed(s) for _ in range(8)],[None]*7+[0.])
        self.assertIsNone(e.feed(None))
        self.assertIsNone(e.feed(Sample(.012,9.,9.)))
    def test_consensus_uses_the_smaller_estimate(self):
        e=Evidence(np.zeros((len(CENTERS),2)))
        for _ in range(8):v=e.feed(Sample(.015,-5.,-11.))
        self.assertEqual(v,-5.)
        self.assertIsNone(e.feed(Sample(.04,-5.,-8.)))
