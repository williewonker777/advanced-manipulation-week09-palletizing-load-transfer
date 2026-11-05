import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import r1_detector
from tools import r4_study as S, r4_review as G


class StudyTests(unittest.TestCase):
    def test_registered_balanced_case_pairs_and_disjoint_normal_masses(self):
        p = S.protocol()
        masses = []
        for stage in ('fit', 'calibration', 'audit'):
            cond = p['calibration']['normal'][stage]['conditions']
            self.assertEqual(len(cond), 16 if stage == 'audit' else 24)
            unique = set(c['mass'] for c in cond)
            self.assertFalse(unique.intersection(masses))
            self.assertFalse(unique.intersection([0., 4., 8.]))
            masses.extend(unique)
        for stage in ('validation', 'development'):
            rows = S.base_cases(stage, 'servo')
            self.assertEqual(len(rows), 38)
            collisions = [r for r in rows if r['kind'] == 'collision']
            for a, b in zip(collisions[::2], collisions[1::2]):
                self.assertEqual(a['mass'], b['mass'])
                self.assertEqual(a['com'], b['com'])
                self.assertEqual(a['collision'][0], b['collision'][0])
                np.testing.assert_array_equal(a['collision'][1], -np.asarray(b['collision'][1]))

    def test_exact_scalar_prediction_both_modes(self):
        times = np.arange(6000) * .001
        m, c = 3.4512, [.0123, -.0227, .0351]
        for mode in ('servo', 'torque'):
            actual = S.prediction(times, m, c, mode)
            p = r1_detector.MODEL_PARAMETERS[mode]
            mean, scale, w = np.asarray(p['mean']), np.asarray(p['scale']), np.asarray(p['weights'])
            expected = np.vstack([np.clip((r1_detector.normal_features(float(t), m, c) - mean) / scale, -8., 8.) @ w for t in times])
            np.testing.assert_array_equal(actual, expected)

    def test_normal_firewall_rejects_wrong_stage_count_source_and_condition(self):
        rows = S.base_cases('fit', 'servo')
        doc = dict(schema='R4_BASE_STREAMS.v1', stage='fit', architecture='servo',
                   protocol_sha256=S.sha(S.P), registration=S.REGISTRATION,
                   source_sha256=S.science_hashes(), results=rows)
        S.validate_normal_document(doc, 'fit', 'servo')
        for field, wrong in [('stage', 'validation'), ('source_sha256', {}), ('results', rows[:-1])]:
            bad = copy.deepcopy(doc); bad[field] = wrong
            with self.assertRaises(RuntimeError): S.validate_normal_document(bad, 'fit', 'servo')
        bad = copy.deepcopy(doc); bad['results'][0]['mass'] += .01
        with self.assertRaises(RuntimeError): S.validate_normal_document(bad, 'fit', 'servo')

    def test_precollision_gate_prevents_running_any_case(self):
        with patch.object(S, 'check_integrity'), patch.object(S, 'require_equivalence', side_effect=RuntimeError('sealed')), patch.object(S.R, 'run_case') as run:
            with self.assertRaisesRegex(RuntimeError, 'sealed'): S.collect('validation', Path('/unused'))
            run.assert_not_called()

    def test_normal_stage_cannot_accept_explicit_collision_or_harness(self):
        with patch.object(S, 'check_integrity'), patch.object(S.R, 'run_case') as run:
            with self.assertRaisesRegex(RuntimeError, 'normal stages'):
                S.collect('fit', Path('/unused'), explicit=[{'collision': [2., [10., 0., 0.]]}])
            run.assert_not_called()

    def test_stale_equivalence_certificate_is_rejected(self):
        with patch.object(S, 'runtime_bundle', return_value={'new': 'hash'}), patch.object(S, 'read', return_value={'passed': True, 'runtime_bundle_sha256': {'old': 'hash'}}):
            with self.assertRaisesRegex(RuntimeError, 'stale'): S.require_equivalence()

    def test_raw_boundary_rejects_non_private_path(self):
        with self.assertRaises(ValueError): S.raw({'private_raw_basename': 'x'}, S.ROOT)


class GateTests(unittest.TestCase):
    def rows(self):
        physical = {key: 0. for key in G.PHYSICAL_METRICS}
        physical.update(diverged=False, g4={})
        return dict(results=[dict(physical, case_id=i, mass=2., com=[0., 0., .03],
             kind='collision' if i else 'reference', collision=[2., [10., 0., 0.]] if i else None,
             pick_offset=None, phase='transport', false_alarms=0,
             collision_outcome='detected' if i else None,
             tool_contact_delay_ms=None) for i in range(2)])

    def test_or_masking_is_not_counted_as_preserved_detection(self):
        base = self.rows(); candidate = copy.deepcopy(base)
        candidate['results'][1]['collision_outcome'] = 'masked'
        result = G.review(base, candidate, 0)
        self.assertFalse(result['passed']); self.assertEqual(result['lost_R1'], 1)

    def test_casewise_false_alarm_increase_cannot_hide_in_total(self):
        base = self.rows(); base['results'][0]['false_alarms'] = 2
        candidate = copy.deepcopy(base)
        candidate['results'][0]['false_alarms'] = 0
        candidate['results'][1]['false_alarms'] = 1
        result = G.review(base, candidate, 0)
        self.assertFalse(result['passed'])

    def test_insufficient_gain_fails_even_if_all_other_values_identical(self):
        base = self.rows()
        self.assertTrue(G.review(base, copy.deepcopy(base), 0)['passed'])
        self.assertFalse(G.review(base, copy.deepcopy(base), 2)['passed'])


if __name__ == '__main__': unittest.main()
