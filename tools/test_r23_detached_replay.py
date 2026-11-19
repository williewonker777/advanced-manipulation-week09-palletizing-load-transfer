import unittest
from pathlib import PurePosixPath

from tools import r23_detached_replay as replay


class DistributionSourceTests(unittest.TestCase):
    def test_all_current_studies_exclude_licensed_sources(self):
        for study in (None, 'pairs', 'timing', 'traces', 'tool'):
            with self.subTest(study=study):
                blobs,pins=replay.sources(study)
                self.assertEqual(set(blobs),set(pins))
                self.assertGreater(len(blobs),len(replay.OWN))
                for name in blobs:
                    self.assertFalse(PurePosixPath(name).is_absolute())
                    self.assertNotIn('..',PurePosixPath(name).parts)
                    self.assertFalse(name.startswith(('sim/','model/')))
                    self.assertNotEqual(name,'evaluate.py')
                self.assertIn('experiments/r23_detached/candidate.py',blobs)
                self.assertIn('experiments/r15_delayed_wrench/observer.py',blobs)

    def test_choices_separate_exposed_and_new_evaluation(self):
        self.assertNotIn('independent',replay.MODES)
        self.assertNotIn('holdout',replay.MODES)
        self.assertEqual(replay.MODES['previous'],('exposed','--panel','full'))

    def test_tool_witness_is_only_in_separate_pregrasp_study(self):
        dependency = 'experiments/r23_tool_contact/candidate.py'
        self.assertIn(dependency, replay.sources('tool')[0])
        self.assertNotIn(dependency, replay.sources('target')[0])
        self.assertEqual(replay.MODES['tool'], ('contact_case',))
