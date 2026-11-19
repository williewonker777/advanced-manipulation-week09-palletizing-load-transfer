from pathlib import Path
import unittest
from .check import ROOT, level, resolve

class HistoricalProtectionTest(unittest.TestCase):
    def test_working_source_pin(self):
        pin=level.sha(ROOT/'experiments/r33_pick_clearance/candidate.py')
        self.assertEqual(resolve('experiments.r33_pick_clearance.candidate:Detector',pin).__name__,'Detector')
    def test_cannot_register_other_entrants(self):
        for name in ('experiments.r31_held_identity.candidate:Detector','experiments.r33_pick_clearance.plan:Detector',
                     'experiments.r33_pick_clearance.candidate:Probe','experiments.r33_pick_clearance.foo.candidate:Detector'):
            with self.assertRaises(ValueError):resolve(name,'f'*64)
    def test_wrong_content_hash(self):
        with self.assertRaises(ValueError):resolve('experiments.r33_pick_clearance.candidate:Detector','0'*64)
