"""Presentation tests do not execute a controller or alter scientific inputs."""
import unittest
from tools import r8_report as R


class R8ReportTests(unittest.TestCase):
    def test_sources_and_negative_decision(self):
        c=R.context(R.ROOT);texts=R.render(c)
        self.assertFalse(c['R8_ADOPTION']['adopted'])
        self.assertTrue(c['R8_HOLDOUT_REVIEW']['passed'])
        self.assertIn('생산 import 실패 1건',texts[1])
        self.assertIn('775시험',texts[0])

    def test_old_readme_content_preserved(self):
        c=R.context(R.ROOT);readme=R.render(c)[0]
        for line in c['old_readme'].splitlines()[1:]:
            self.assertIn(line,readme)

    def test_official_source_precision(self):
        c=R.context(R.ROOT);text=R.official(c)
        for rec in c['R8_OFFICIAL_REGRESSION']['records'].values():
            cells=rec['control_and_collisions']['control']['8.0']
            self.assertIn(cells[0]+'/'+cells[1],text)
        self.assertNotIn('1.170',text)


if __name__=='__main__':unittest.main()
