"""Displayed course outcomes and true defaults cannot be silently replaced by positive scores."""
from copy import deepcopy
import json
import unittest

from tools import r20_official_compare as matrix


class MatrixTest(unittest.TestCase):
    def setUp(self):
        self.raw = {m: (matrix.ROOT / f'results/evaluation_baseline_{m}.txt').read_text()
                    for m in matrix.MODES}
        self.stock = {m: matrix.table(text) for m, text in self.raw.items()}
        self.release = json.loads((matrix.ROOT / 'results/r10/official_regression.json').read_bytes())
        self.r10 = {m: self.release[m]['control_and_collisions'] for m in matrix.MODES}

    def runs(self):
        return {arm: {m: {'cells': deepcopy(src[m])} for m in matrix.MODES}
                for arm, src in (('stock', self.stock), ('r10', self.r10), ('combined', self.r10))}

    def test_original_report_distinguishes_absent_and_masked(self):
        self.assertEqual(matrix.summarize(self.stock['servo'])['impact_outcomes'],
                         dict(detected=12, missed=4, indistinguishable=4, already_active=4))
        self.assertEqual(matrix.summarize(self.stock['torque'])['no_impact_false_rises'], 15)
        self.assertEqual(matrix.summarize(self.stock['torque'])['worst_false_count_in_an_impact_run'], 12)
        self.assertEqual(matrix.summarize(self.r10['torque'])['impact_outcomes']['detected'], 18)
        self.assertEqual(matrix.summarize(self.r10['servo'])['no_impact_false_rises'], 0)

    def test_parser_rejects_duplicate_partial_corrupt_or_inconsistent_rows(self):
        lines = self.raw['torque'].splitlines()
        exemplar = next(line for line in lines if '1.42s |' in line)
        for invalid in ('\n'.join(lines + [exemplar]), '\n'.join(l for l in lines if l != exemplar),
                        '\n'.join(lines).replace('     경보 중 |', '     이상 |', 1),
                        '\n'.join(lines).replace(exemplar, exemplar.replace('40 N', '10 N'))):
            with self.subTest(case=invalid[-15:]), self.assertRaises(ValueError):
                matrix.table(invalid)

    def test_changes_include_a_lost_detection_not_only_faster_ones(self):
        old = self.r10['torque']
        new = deepcopy(old)
        new['collisions'][1]['cells'][:2] = ['놓침', '40']
        self.assertEqual([x['candidate'] for x in matrix.compare(old, new)], ['놓침', '40'])
        self.assertEqual([x['force_N'] for x in matrix.compare(old, new)], [10, 15])
        new['collisions'].reverse()
        with self.assertRaises(ValueError):
            matrix.compare(old, new)

    def test_scope_and_stock_provenance_are_hard_checks(self):
        matrix.verify_runs(self.runs(), self.release)
        for a, mode, idx in (('combined', 'servo', 3), ('combined', 'torque', 0), ('stock', 'torque', 3),
                             ('r10', 'torque', 3)):
            rows = self.runs()
            rows[a][mode]['cells']['collisions'][idx]['cells'][0] = '1'
            with self.subTest(a=a, mode=mode), self.assertRaises(ValueError):
                matrix.verify_runs(rows, self.release)
        rows = self.runs()
        rows['combined']['torque']['cells']['collisions'][3]['cells'][0] = '1'
        matrix.verify_runs(rows, self.release)  # A real in-domain difference must survive the check.

    def test_unmodified_defaults_and_explicit_candidate_modules(self):
        self.assertEqual(matrix.modules('stock', 'servo'), ('servo_controller', 'collision_detector'))
        self.assertEqual(matrix.modules('r10', 'torque'), ('my_torque', 'my_detector'))
        self.assertEqual(matrix.modules('combined', 'servo'), ('my_servo', matrix.POLICY))
        self.assertEqual(matrix.modules('combined', 'torque'), (matrix.POLICY, matrix.POLICY))
        for arm, mode in [('r10', 'other'), ('unknown', 'torque')]:
            with self.assertRaises(ValueError):
                matrix.modules(arm, mode)


if __name__ == '__main__':
    unittest.main()
