import unittest
from pathlib import Path

from tools import r22_census, r22_studies


class StudyInputsTests(unittest.TestCase):
    def test_scoped_runtime_and_census_dependencies_exist(self):
        names=r22_census.dependencies()
        self.assertEqual(len(names),len(set(names)))
        for n in names:
            self.assertFalse(Path(n).is_absolute())
            self.assertNotIn('..',Path(n).parts)
            self.assertTrue((r22_studies.ROOT/n).is_file(),n)

    def test_no_stock_code_bundled_as_authored_runtime(self):
        for study in ('official','pairs','timing'):
            b,pins=r22_studies.sources(study)
            self.assertEqual(set(b),set(pins))
            self.assertTrue(set(r22_studies.RUNTIME)<=set(b))
            self.assertFalse(any(n.startswith(('sim/','model/')) or n=='evaluate.py' for n in b))


class CensusAccountingTests(unittest.TestCase):
    def result(self):
        ids = (39, 84, 167, 174, 330, 339, 349)
        return {
            'arm': 'core', 'cohort': 'four', 'cases': 7, 'collisions': 3,
            'reference_cache_sha256': None, 'simulated_episodes': 21,
            'rows': [{'id': i, 'kind': 'reference' if i < 189 else 'collision',
                      'input_certified': True, 'protection': i != 330} for i in ids],
            'input_failures': [], 'protection_failures': [330],
        }

    def test_complete_negative_conclusion_is_publishable(self):
        r22_census.validate_accounting(self.result(), 'core', 'four')

    def test_empty_misidentified_or_uncounted_cohort_is_not_publishable(self):
        for field, value in (('rows', []), ('rows', [self.result()['rows'][0]] * 7),
                             ('cohort', 'six'), ('arm', 'light'), ('cases', 0),
                             ('collisions', 0), ('reference_cache_sha256', 'cached'),
                             ('simulated_episodes', 7), ('protection_failures', [])):
            with self.subTest(field=field):
                result = self.result()
                result[field] = value
                with self.assertRaises(ValueError):
                    r22_census.validate_accounting(result, 'core', 'four')
