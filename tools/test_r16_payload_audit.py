"""Public contract tests for the portable R16 stock payload audit."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.r16_robust_prefix import robust
from tools import r16_payload_audit as audit


class PayloadAuditTest(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((audit.ROOT/audit.SOURCE_PATHS['plan']).read_text())
        self.calibration = json.loads((audit.ROOT/audit.SOURCE_PATHS['calibration']).read_text())
        self.reference = json.loads((audit.ROOT/audit.SOURCE_PATHS['reference']).read_text())

    def test_frozen_inputs_and_provenance(self):
        audit.validate_inputs(self.plan, self.calibration, self.reference)
        bad = json.loads(json.dumps(self.plan)); bad['fresh_groups'][0]['mass'] = -1
        with self.assertRaises(ValueError):
            audit.validate_inputs(bad, self.calibration, self.reference)
        bad = json.loads(json.dumps(self.calibration)); bad['portable_provenance']['truth_used_in_fit'] = True
        with self.assertRaises(ValueError):
            audit.validate_inputs(self.plan, bad, self.reference)
        self.assertIn('cannot independently regenerate', self.calibration['portable_provenance']['stock_regeneration'])

    def test_none_and_contact_enabled_zero_remain_distinct(self):
        case = self.plan['fresh_groups'][0]; x, y, z = case['shift']
        self.assertIsNone(audit.offset(case, 'none'))
        self.assertEqual(audit.offset(case, 'explicit_zero'), (0., 0., 0.))
        self.assertEqual(audit.offset(case, 'xy'), (x, y, 0.))
        self.assertEqual(audit.offset(case, 'up'), (x, y, z))
        self.assertEqual(audit.offset(case, 'down'), (x, y, -z))

    def _row(self, signature, jacobian=True):
        count = len(robust.USED_CENTRES)
        jac = np.tile(np.eye(6, 3)[None], (count, 1, 1)) if jacobian else np.zeros((count, 6, 3))
        return {'mass': np.array(5.), 'signature': np.asarray(signature, float),
                'y': np.zeros((count, 6)), 'jac': jac, 'stiffness': np.zeros((count, 6)),
                'prior_com_mm': np.zeros(3), 'completed_at_s': np.array(3.447)}

    def _small_calibration(self):
        count = len(robust.USED_CENTRES)
        return {'schema': 'r16-robust-prefix-calibration.v1',
                'normalization': {'mass_center': 5., 'signature_center': [0., 0., 0., 0.],
                    'signature_scale': [1., 1., 1., 1.], 'mass_range': [3., 8.],
                    'signature_radius_max': 2.},
                'dynamics': {'bias': np.zeros((6, count, 6)).tolist(), 'sigma': np.ones(6).tolist()},
                'baseline_xy': np.zeros((6, 2)).tolist(), 'normal_guard_mm': .2}

    def test_out_of_support_and_unidentifiable_force_zero_action(self):
        outside = robust.decide(self._row([9., 9., 9., 9.]), self._small_calibration())
        self.assertFalse(outside['signature_supported']); self.assertFalse(outside['accept'])
        self.assertEqual(outside['model_action_xy_mm'], [0.0, 0.0])
        rank_bad = robust.decide(self._row([0., 0., 0., 0.], jacobian=False), self._small_calibration())
        self.assertFalse(rank_bad['identifiable']); self.assertFalse(rank_bad['accept'])
        self.assertEqual(rank_bad['model_action_xy_mm'], [0.0, 0.0])

    def test_metrics_fail_wrong_confident_sign(self):
        rows = []
        for _ in range(8):
            rows.append({'true_xy_mm': [0., 0.], 'point_error_mm': [.1, 0.],
                         'action_error_mm': [0., 0.], 'accept': False, 'covered': True,
                         'model_action_xy_mm': [0., 0.]})
        for vector in ([3., 4.], [-3., 4.], [4., -3.])*4:
            rows.append({'true_xy_mm': list(vector), 'point_error_mm': [.1, -.1],
                         'action_error_mm': [.1, -.1], 'accept': True, 'covered': True,
                         'model_action_xy_mm': list(vector)})
        self.assertTrue(audit.measures(rows)['qualified'])
        rows[-1]['model_action_xy_mm'] = [-4., 3.]
        self.assertFalse(audit.measures(rows)['qualified'])

    def test_source_has_no_private_fixture_raw_or_absolute_contract(self):
        source = Path(audit.__file__).read_text()
        self.assertNotIn('r5_fixture', source)
        self.assertNotIn('savez', source)
        # Reject a checkout or user-directory literal on this or any host;
        # the distribution scanner independently checks generic path layouts.
        self.assertNotIn(str(Path.home().resolve()) + os.sep, source)
        self.assertNotIn(str(audit.ROOT.resolve()) + os.sep, source)
        self.assertIn('predictions_sealed_before_truth', source)
        self.assertIn('score_capsule', source)

    def test_snapshot_remains_bound_after_source_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for key, relative in audit.SOURCE_PATHS.items():
                path = root/relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((key+'-original').encode())
            with patch.object(audit, 'ROOT', root):
                blobs, pins = audit.source_inputs()
            target = root/audit.SOURCE_PATHS['robust']; target.write_bytes(b'mutated')
            self.assertEqual(blobs['robust'], b'robust-original')
            self.assertEqual(pins['robust'], audit.sha(b'robust-original'))
            self.assertNotEqual(pins['robust'], audit.sha(target.read_bytes()))

    def test_help_from_unrelated_cwd_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, '-B', str(Path(audit.__file__).resolve()), '--help'],
                                    cwd=folder, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0, result.stderr); self.assertIn('--kit', result.stdout)
            existing = Path(folder)/'existing'; existing.mkdir()
            argv = ['r16_payload_audit.py', '--kit', folder, '--out', str(existing)]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                audit.main()


if __name__ == '__main__':
    unittest.main()
