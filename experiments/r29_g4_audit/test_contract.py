import unittest
from experiments.r29_g4_audit.contract import ALARM, PHYSICAL, no_force_edges, physical_view


def rec(ticks=(),false=()):
    return dict(rises=list(ticks),rise_times_s=[x/1000 for x in ticks],false_times_s=[x/1000 for x in false],
                physical_sha256='a',g4={k:0 for k in PHYSICAL|ALARM},ending_table_mean_N=1.,ending_table_max_N=2.)


class ContractTest(unittest.TestCase):
    def test_remove_only_certified_false(self):
        v=no_force_edges(rec([100,200,300],[200]),rec([100,300]))
        self.assertTrue(v['passed']);self.assertEqual(v['removed_certified_false'],[200])
        self.assertFalse(no_force_edges(rec([100,200]),rec([100]))['passed'])

    def test_single_new_edge_cannot_cover_two_old_edges(self):
        self.assertFalse(no_force_edges(rec([100,102]),rec([101]))['passed'])
        self.assertFalse(no_force_edges(rec([100]),rec([100,102]))['passed'])

    def test_new_false_and_reclassification_fail(self):
        self.assertFalse(no_force_edges(rec([100]),rec([100,102],[102]))['passed'])
        self.assertFalse(no_force_edges(rec([100]),rec([100],[100]))['passed'])

    def test_unknown_false_annotation_fails_closed(self):
        for r in (rec([100],[104]),rec([100],[100,100]),rec([100,100])):
            with self.assertRaises(ValueError): no_force_edges(rec(),r)

    def test_controller_and_observer_fields_have_separate_ownership(self):
        a=rec();b=rec();b['g4']['false_alarm_times']=[1]
        self.assertEqual(physical_view(a),physical_view(b))
        for k in PHYSICAL:
            b=rec();b['g4'][k]='mutated'
            self.assertNotEqual(physical_view(a),physical_view(b))
        b=rec();b['g4']['future_field']=0
        with self.assertRaises(ValueError):physical_view(b)
