import copy
import unittest

from tools.r20_handoff_check import check_stages
from tools.r20_horizontal_check import calibrated_bound


def stage(name):
    rows = []
    for i in range(18 if name == 'exposed' else 6):
        r = {'id': str(i), 'episodes': 2 if name == 'normal' else 8}
        if name == 'normal':
            r.update(passed=True, candidates={'slow': {'gates': {'no_added_false': True}}})
        else:
            certified = name == 'exposed' and i < 3
            checks = {'force_and_prefix_valid': not certified, 'retention': True}
            cert_checks = {key: True for key in (
                'authorized_legacy_batched_norm_only',
                'assessment_sha256_matches_initial_r19',
                'initial_r19_and_slow_assessment_bit_equal',
                'all_four_twins_6000_ticks',
                't_native_second_committed_scalar_exact',
                'stock_scalar_exact_in_actual_assessment',
                'legacy_batched_norm_mismatch_reproduced')}
            direct = ({'status': 'direct_stock_rerun', 'episodes': 4, 'passed': True,
                       'assessment_sha256': 'assessment', 'checks': cert_checks,
                       'authorization_evidence': {
                           'initial_force_gate_failures': {
                               'r19': ['signed_profiles_exact'],
                               'slow': ['signed_profiles_exact']},
                           'initial_signed_and_scalar_exact': {'r19': True, 'slow': True},
                           'initial_stock_scalar_all_twins_exact': {
                               'r19': True, 'slow': True},
                           'initial_r19_and_slow_records_equal': True,
                           'initial_r19_to_slow_physical_equal': True},
                       'twins': {twin: {
                           'samples': 6000,
                           'exact': {key: True for key in
                                     ('t', 'native', 'extra', 'combined', 'scalar')},
                           'all_arrays_exact': True,
                           'actual_array_sha256': {key: '0' * 64 for key in
                                                   ('t', 'native', 'extra', 'combined', 'scalar')},
                           'batched_norm_mismatch_ticks': [1] if twin == 'first' else [],
                           'max_batched_norm_difference_N': 1e-15,
                           'stock_scalar_bit_exact': True,
                       } for twin in ('none', 'first', 'second', 'both')}} if certified else
                      {'status': 'not_required', 'episodes': 0, 'passed': None})
            failed = ['signed_profiles_exact'] if certified else []
            r.update(checks=checks, retained=all(checks.values()),
                     direct_force_certificate=direct,
                     supplemental_continuation_retained=True,
                     strict={'slow': not certified, 'r19': not certified},
                     one_per_slot={'slow': True, 'r19': True},
                     failed_original_gates={'slow': failed, 'r19': failed},
                     assessment_sha256={'slow': 'assessment', 'r19': 'assessment'})
        rows.append(r)
    result = {'passed': all(r.get('retained', r.get('passed')) and
                            (name != 'fresh' or r.get('strict', {}).get('slow', True))
                            for r in rows), 'rows': rows}
    if name == 'exposed':
        result.update(certified_continuation_passed=True, certificate_episodes=12,
                      strict_counts={'r19': 15, 'slow': 15})
    return result


class CertTest(unittest.TestCase):
    def setUp(self):
        self.doc = {'schema': 'r20-slow-handoff-qualification.v2',
                    'eligible_for_submission': False,
                    'independent_detection_gain_claimed': False,
                    'base_episodes': 204, 'certificate_episodes': 12, 'episodes': 216,
                    'initial_without_direct_certificate': {
                        'public_gate_halted_after': 'exposed', 'passed': False,
                        'failing_ids': ['0', '1', '2']},
                    'stages': {s: stage(s) for s in ('exposed', 'normal', 'fresh')}}

    def test_closed_continuation_and_real_denominators(self):
        check_stages(self.doc)
        for bad in ('drop', 'repeat', 'reopen', 'fake_episodes', 'reordered', 'promote'):
            x = copy.deepcopy(self.doc)
            if bad == 'drop':
                x['stages']['normal']['rows'].pop()
            if bad == 'repeat':
                x['stages']['fresh']['rows'][1]['id'] = '0'
            if bad == 'reopen':
                r = x['stages']['normal']['rows'][0]
                r['candidates']['slow']['gates']['no_added_false'] = r['passed'] = False
                x['stages']['normal']['passed'] = False
            if bad == 'fake_episodes':
                x['episodes'] += 1
            if bad == 'reordered':
                x['stages'] = dict(reversed(list(x['stages'].items())))
            if bad == 'promote':
                x['eligible_for_submission'] = True
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_stages(x)

    def test_reject_ignoring_unpassed_original_gate_and_accept_honest_failure(self):
        r = self.doc['stages']['fresh']['rows'][0]
        r['failed_original_gates']['slow'] = ['both_second_genuine']
        with self.assertRaises(ValueError):
            check_stages(self.doc)
        r['strict']['slow'] = self.doc['stages']['fresh']['passed'] = False
        check_stages(self.doc)

    def test_direct_certificate_cannot_change_original_failure(self):
        row = self.doc['stages']['exposed']['rows'][0]
        row['retained'] = True
        with self.assertRaises(ValueError):
            check_stages(self.doc)

    def test_direct_certificate_requires_every_exact_trace_field(self):
        row = self.doc['stages']['exposed']['rows'][0]
        del row['direct_force_certificate']['twins']['both']['exact']['scalar']
        with self.assertRaises(ValueError):
            check_stages(self.doc)

    def test_non_roundoff_failure_halts_without_certificate(self):
        x = copy.deepcopy(self.doc)
        row = x['stages']['exposed']['rows'][0]
        row['failed_original_gates']['r19'].append('force_free_prefixes')
        row['failed_original_gates']['slow'].append('force_free_prefixes')
        row['direct_force_certificate'] = {
            'status': 'not_run_non_roundoff_force_failure', 'episodes': 0, 'passed': False}
        row['supplemental_continuation_retained'] = False
        x['stages']['exposed']['certified_continuation_passed'] = False
        x['stages']['exposed']['certificate_episodes'] = 8
        x['stages'] = {'exposed': x['stages']['exposed']}
        x['base_episodes'], x['certificate_episodes'], x['episodes'] = 144, 8, 152
        check_stages(x)

    def test_calibration_uses_only_absolute_axis_maxima_from_each_unique_normal(self):
        rows = [{'mass': m, 'com': (c, 0, 0), 'max': {'early': {'residual_xy': [c, 1]}}}
                for m in (0, 1, 2, 3) for c in range(5)]
        self.assertEqual(calibrated_bound({'rows': rows}), 4 * 1.02)
        for replacement in (rows[:-1], rows[:-1] + rows[:1]):
            with self.assertRaises(ValueError):
                calibrated_bound({'rows': replacement})
        rows[0]['max']['early']['residual_xy'][0] = float('nan')
        with self.assertRaises(ValueError):
            calibrated_bound({'rows': rows})


if __name__ == '__main__':
    unittest.main()
