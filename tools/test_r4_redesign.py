import copy
import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools import r4_redesign as X, r4_study as S


class _Body:
    id = 1


class _Model:
    def body(self, name):
        if name != 'link_6':
            raise KeyError(name)
        return _Body()


class _Data:
    xpos = np.array([[9., 9., 9.], [1., 2., .5]])
    xmat = np.array([[1., 0., 0., 0., 1., 0., 0., 0., 1.],
                     [1., 0., 0., 0., 1., 0., 0., 0., 1.]])


class _Packet:
    class nom:
        m = _Model()
        d = _Data()
    bottom_offset = np.array([0., 0., .1])
    table_heights = np.array([.1, .2])
    height_band = .035


class RedesignTests(unittest.TestCase):
    def test_gate_is_pure_read_idempotent_and_grip_gated(self):
        packet = _Packet()
        before_xpos = packet.nom.d.xpos.copy()
        before_xmat = packet.nom.d.xmat.copy()
        self.assertFalse(X.pure_read_gate(packet, False))
        self.assertTrue(X.pure_read_gate(packet, True))
        self.assertTrue(X.pure_read_gate(packet, True))
        np.testing.assert_array_equal(packet.nom.d.xpos, before_xpos)
        np.testing.assert_array_equal(packet.nom.d.xmat, before_xmat)

    def test_rebuild_changes_only_the_registered_gate_line(self):
        item = X._original_sources()['W40_ACTIVE.py']
        source = (X.GENERATED / 'W40_ACTIVE.py').read_text(encoding='utf-8')
        # The test works both before prepare and against the exact archived copy
        # after prepare.
        if X.OLD_GATE not in source:
            source = (X.ARCHIVE / 'W40_ACTIVE.py').read_text(encoding='utf-8')
        revised = X.revised_source(source, item['parameters'])
        self.assertEqual(revised.count(X.NEW_GATE), 1)
        self.assertEqual(revised.replace(X.NEW_GATE, X.OLD_GATE, 1), source)
        self.assertTrue(X.literal_audit(revised)['passed'])

    def test_registered_design_is_explicit_not_sampled(self):
        design = X._registered()['revalidation']['cases']
        self.assertEqual(len(design), 38)
        self.assertEqual([row['case_id'] for row in design], list(range(38)))

    def test_fresh_episode_has_integrity_and_bundle_prechecks(self):
        source = inspect.getsource(X._online)
        loop = source.index("for original in baseline['results']:")
        run = source.index('run_episode(', loop)
        self.assertLess(source.index('S.check_integrity(ROOT)', loop), run)
        self.assertLess(source.index('S.runtime_bundle() != bundle', loop), run)
        validate = inspect.getsource(X.validate)
        self.assertLess(validate.index('S.check_integrity(ROOT)'),
                        validate.index('bundle = S.runtime_bundle()'))

    def test_resolver_signatures_choose_only_revised_w40(self):
        with patch.object(S, 'redesign_active', return_value=True):
            self.assertEqual(S.equivalence_path().name, 'R4_REDESIGN_EQUIVALENCE.json')
            self.assertEqual(S.validation_document_path('W40_ACTIVE', 'servo').name,
                             'R4_REVALIDATION_W40_ACTIVE_servo.json')
            self.assertEqual(S.validation_baseline_path('W40_ACTIVE', 'torque').name,
                             'R4_BASE_revalidation_torque.json')
            self.assertEqual(S.validation_document_path('CUSUM_ACTIVE_K05', 'servo').name,
                             'R4_VALIDATION_CUSUM_ACTIVE_K05_servo.json')
            self.assertEqual(S.redesign_decision_path().name, 'R4_REDESIGN_DECISION_V2.json')
            self.assertEqual(S.validation_review_path().name, 'R4_REVALIDATION_REVIEW.json')

    def test_literal_store_free_requires_all_three_new_proofs(self):
        spec = next(item for item in S.protocol()['candidate_grid']
                    if item['id'] == 'W40_ACTIVE')
        actual_read = S.read
        def proof(passed):
            return lambda path: ({'passed': passed} if Path(path).name.startswith('R4_REDESIGN_')
                                 else actual_read(path))
        with patch.object(S, 'redesign_active', return_value=True), \
             patch.object(Path, 'is_file', return_value=True), \
             patch.object(S, 'read', side_effect=proof(True)):
            self.assertTrue(S.literal_store_free(spec))
            self.assertFalse(S.literal_store_free({'id': 'CUSUM_ACTIVE_K05'}))
        with patch.object(S, 'redesign_active', return_value=True), \
             patch.object(Path, 'is_file', return_value=True), \
             patch.object(S, 'read', side_effect=proof(False)):
            self.assertFalse(S.literal_store_free(spec))


if __name__ == '__main__':
    unittest.main()
