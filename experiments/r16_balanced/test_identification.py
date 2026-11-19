import json
from pathlib import Path
import unittest

import numpy as np

from experiments.r16_balanced.identify import features, refit_from_table, solve


class FitTests(unittest.TestCase):
    def test_recreates_frozen_response_from_score_table_alone(self):
        root = Path(__file__).parent
        doc = json.loads((root/'coefficients.json').read_text())
        plan = json.loads((root/'plan.json').read_text())
        self.assertEqual([r['id'] for r in doc['training']], plan['train_ids'])
        match = refit_from_table(doc['training'])
        np.testing.assert_allclose(match, [doc['relief'],doc['phase'],doc['min_margin_weight']],
                                   rtol=0, atol=1e-10)
        self.assertGreater(match[2], 0.)

    def test_force_windows_do_not_hide_either_side_of_detachment(self):
        time = np.arange(3600,4100)/1000
        force = np.ones(500); force[100]=1.08; force[290]=.6; force[300]=1.12
        peak_pre, peak_post, neg_min, neg_mean = features(force,time)
        self.assertEqual((peak_pre,peak_post,neg_min), (1.08,1.12,-.6))
        self.assertAlmostEqual(neg_mean,-.98)
        with self.assertRaises(ValueError):
            features(force[:-1],time[:-1])

    def test_infeasible_positive_slack_remains_negative(self):
        result = solve([[0.,0.],[1.,0.]], [-.1,1.])
        self.assertAlmostEqual(result[2], -.1)

    def test_duplicate_rows_and_nonfinite_scores_fail_closed(self):
        with self.assertRaises(ValueError):
            refit_from_table([{'id':'a'},{'id':'a'}])
        with self.assertRaises(ValueError):
            solve([[np.nan,0.]],[1.])


if __name__ == '__main__':
    unittest.main()
