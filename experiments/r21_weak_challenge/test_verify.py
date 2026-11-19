import copy
import unittest

from experiments.r21_weak_challenge import verify


class ResultInvariantTests(unittest.TestCase):
    def setUp(self):
        self.plan = {kind: [{'id': f'{kind}{i}', 'role': 'test'} for i in range(n)]
                     for kind, n in (('forced', 15), ('negative_only', 3))}
        self.pick, self.yaw = (self.fixture(c) for c in ('r21_pick', 'r21_yaw'))

    def fixture(self, arm):
        arms = ['r10', 'r20', arm]
        result = {'episodes': 99, 'passed': True, 'arms': arms}
        for kind, size, gates in (('forced', 6, verify.FORCED_CHECKS),
                                  ('negative_only', 3, verify.NEGATIVE_CHECKS)):
            result[kind] = [
                dict(case=c, id=c['id'], role=c['role'], episodes=size, passed=True,
                     arms={k: {} for k in arms}, comparisons={
                         k: {'passed': True, 'checks': dict.fromkeys(gates, True)} for k in arms[1:]})
                for c in self.plan[kind]]
        return result

    def check(self):
        return tuple(verify.paired_rows(self.plan, self.pick, self.yaw))

    def test_two_exact_cohorts(self):
        left, right = self.check()
        self.assertEqual(list(left), list(right))
        self.assertEqual(len(left), 15)

    def test_equal_wrong_or_only_one_wrong_count_is_rejected(self):
        for p, y in ((98, 99), (99, 98), (98, 98)):
            with self.subTest(p=p, y=y), self.assertRaisesRegex(ValueError, 'incomplete'):
                self.pick['episodes'], self.yaw['episodes'] = p, y
                self.check()

    def test_duplicate_not_collapsed(self):
        self.pick['forced'].append(copy.deepcopy(self.pick['forced'][0]))
        with self.assertRaisesRegex(ValueError, 'cohort'):
            self.check()

    def test_missing_gate_not_implicit_pass(self):
        del self.yaw['negative_only'][0]['comparisons']['r21_yaw']['checks']['physical_equal']
        with self.assertRaisesRegex(ValueError, 'gate'):
            self.check()

    def test_changed_negative_comparator_not_hidden(self):
        self.yaw['negative_only'][0]['arms']['r10']['false_alarm'] = True
        with self.assertRaisesRegex(ValueError, 'comparators'):
            self.check()
