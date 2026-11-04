import unittest
from tools.r3_holdout_review import compose
from tools.r3_study import baseline_spec


class TestR3HoldoutReview(unittest.TestCase):
    def test_failure_selects_only_frozen_r1_independently(self):
        a = dict(id='A_example', architecture='servo')
        b = dict(id='B_example', architecture='torque')
        result = compose({'chosen_specs': [a, b]},
                         {'servo': {'status': 'FAIL'}, 'torque': {'status': 'PASS'}}, {})
        self.assertEqual(result['architectures']['servo']['effective_selection'], baseline_spec('servo'))
        self.assertEqual(result['architectures']['torque']['effective_selection'], b)
        self.assertTrue(result['no_tuning_permitted'])
