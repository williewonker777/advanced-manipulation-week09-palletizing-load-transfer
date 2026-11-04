import unittest
from tools.r3_math import qualify


class TestR3Math(unittest.TestCase):
    def test_rigid_nominal_polarized_action_identity(self):
        result = qualify(seed=1234, samples=3)
        self.assertTrue(result['passed'], result['worst'])
        self.assertEqual(result['episodes_run'], 0)
        self.assertEqual([row['load_fraction'] for row in result['rows']], [0., .5, 1.])
