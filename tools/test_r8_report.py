"""Presentation tests do not execute a controller or alter scientific inputs."""
import unittest
import hashlib
import json
from copy import deepcopy
from tools import r8_report as R


class R8ReportTests(unittest.TestCase):
    def test_sources_and_negative_decision(self):
        c=R.context(R.ROOT);texts=R.render(c)
        self.assertFalse(c['R8_ADOPTION']['adopted'])
        self.assertTrue(c['R8_HOLDOUT_REVIEW']['passed'])
        r=c['R8_ADOPTION']['restoration']
        self.assertTrue(r['effective_adopted'])
        self.assertEqual('C1',r['effective_final_selection'])
        self.assertEqual(R.sha(R.ROOT/'my_detector.py'),r['effective_detector_sha256'])
        self.assertIn('생산 import 실패 1건',texts[1])
        self.assertIn('778시험',texts[0])
        self.assertIn('결과를 본 뒤',texts[0])
        for mutate in ('adopted','holdout_servo_gain'):
            changed=deepcopy(c);changed['R8_ADOPTION'][mutate]=99
            with self.assertRaises(AssertionError):R.validate_restoration(changed)
        changed=deepcopy(c);changed['R8_ADOPTION']['restoration']['new_official_batches']=True
        with self.assertRaises(AssertionError):R.validate_restoration(changed)

    def test_old_readme_content_preserved(self):
        c=R.context(R.ROOT);readme=R.render(c)[0]
        updated=('**최종 제출:', '| 약한 집기·놓기 충돌 놓침 |', '| 최대 1 ms 초과 |', '| 계산 비용 증가 |', '| 충돌 감지(전체 D4·운반) |', '| 약한 집기·놓기 충돌 |', '| 계산 시간(1 ms) |', '| 숨은 조건 일반화 |')
        for line in c['old_readme'].splitlines()[1:]:
            if not line.startswith(updated):self.assertIn(line,readme)
        for token in ('공식 18→23/24','1 ms 초과 0→14주기','감사 최대 7.99 ms','G4','24 충족','5항목','약한 수직 충돌','토크 감지'):
            self.assertIn(token,readme)
        archive=R.read_json(R.ROOT/(R.D+'R8_ARCHIVE.json'))
        self.assertEqual(archive['original_readme'],c['old_readme'])

    def test_official_source_precision(self):
        c=R.context(R.ROOT);text=R.official(c)
        for rec in c['R8_OFFICIAL_REGRESSION']['records'].values():
            cells=rec['control_and_collisions']['control']['8.0']
            self.assertIn(cells[0]+'/'+cells[1],text)
        self.assertNotIn('1.170',text)


if __name__=='__main__':unittest.main()
