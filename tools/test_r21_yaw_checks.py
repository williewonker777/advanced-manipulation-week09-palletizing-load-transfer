import json
import unittest
from pathlib import Path

from tools.r21_yaw_checks import FILES, GROUP, MODEL_PIN, ROOT, digest, replace_seed


class FrozenSensitivityTests(unittest.TestCase):
    def test_plan_targets_both_known_limits_not_hidden_success(self):
        plan = json.loads((ROOT / GROUP / 'noise_plan.json').read_bytes())
        self.assertEqual(plan['masses'], [0, 8])
        self.assertEqual(plan['time'], [1.42, 3.86])
        self.assertEqual(len(set(plan['seeds'])), 4)
        self.assertEqual(plan['force'], [0, 10, 0])
        self.assertIn('not hidden mechanics', plan['classification'])

    def test_source_list_is_relative_and_frozen_model_matches(self):
        self.assertEqual(len(FILES), len(set(FILES)))
        for name in FILES:
            self.assertFalse(Path(name).is_absolute())
            self.assertNotIn('..', Path(name).parts)
        self.assertEqual(digest(ROOT / GROUP / 'normal_bins.json'), MODEL_PIN)

    def test_seed_replacement_requires_exactly_one_default(self):
        self.assertEqual(replace_seed('JTS_SEED = 7  # value\n', 29),
                         'JTS_SEED = 29  # value\n')
        for raw in ('', 'JTS_SEED = 29  # value\n', 'JTS_SEED = 7  # a\nJTS_SEED = 7  # b\n'):
            with self.assertRaises(ValueError):
                replace_seed(raw, 41)
