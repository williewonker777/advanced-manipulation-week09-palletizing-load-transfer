import os
import unittest
from pathlib import Path
from unittest.mock import patch
from tools.r2_study import specs, candidate_classes, validate_ridge, require_private, ROOT


class StudyTests(unittest.TestCase):
    def test_candidate_grid(self):
        values = specs()
        self.assertEqual(len(values), 31)
        self.assertEqual(len({s['id'] for s in values}), 31)
        self.assertEqual({s['gamma'] for s in values if s.get('law') == 'NRIC'}, {1., 2., 4.})

    def test_observer_change_requires_new_model(self):
        spec = next(s for s in specs() if s['id'] == 'C5_prne_plus_ridge')
        with self.assertRaises(ValueError):
            candidate_classes('servo', spec)
        with self.assertRaises(ValueError):
            validate_ridge({'protocol_sha256': 'wrong'})

    def test_external_private_directory_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                require_private('/tmp/private_evidence')
        with patch.dict(os.environ, {'R2_PRIVATE_EVIDENCE_ROOT': '/tmp/private_evidence'}):
            self.assertEqual(require_private('/tmp/private_evidence/a'), Path('/tmp/private_evidence/a'))
            with self.assertRaises(ValueError):
                require_private(ROOT/'publish/repo')
            with self.assertRaises(ValueError):
                require_private('/tmp/wrong')


if __name__ == '__main__':
    unittest.main()
