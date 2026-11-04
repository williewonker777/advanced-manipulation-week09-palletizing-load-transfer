import unittest
from tools.r3_d4_scope import scope


class TestR3D4Scope(unittest.TestCase):
    def test_reference_and_collision_saturation_are_separate(self):
        rows = [dict(kind=kind, saturation_pct=.1, mass=1., table_force_place_max_n=9.81)
                for kind in ('reference', 'collision')]
        result = scope(rows)
        self.assertEqual(result['all']['saturation_episodes'], 2)
        self.assertEqual(result['reference']['saturation_episodes'], 1)
        self.assertEqual(result['reference']['reference_peak_over_mg'], 1)
