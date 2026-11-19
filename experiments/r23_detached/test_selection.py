import unittest

from experiments.r23_detached.selection import pick


class PolicyTests(unittest.TestCase):
    def test_dominance_only_after_guards(self):
        scores={'onset':{1,2},'free':{1,2,3}}
        self.assertIsNone(pick([],scores))
        self.assertEqual(pick(['onset'],scores),'onset')
        self.assertEqual(pick(['onset','free'],scores),'free')

    def test_no_sum_across_incomparable_panels(self):
        self.assertEqual(pick(['onset','free'],{'onset':{1},'free':{2,3}}),'onset')
        self.assertEqual(pick(['onset','free'],{'onset':{1},'free':{1}}),'onset')
