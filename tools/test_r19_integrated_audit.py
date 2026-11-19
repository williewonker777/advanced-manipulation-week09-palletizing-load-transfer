import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import r19_integrated_audit as audit


def kit_shape(parent):
    kit=parent/'kit';kit.mkdir()
    for d in ('sim','model'):(kit/d).mkdir()
    (kit/'evaluate.py').touch()
    return kit


class ProtectedAuditTest(unittest.TestCase):
    def test_rejects_reusing_output_before_building_overlay(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);kit=kit_shape(t);h=t/'cal.json';h.write_text('{}');out=t/'out.json';out.write_text('retained')
            with mock.patch.object(audit.assembly,'main') as assemble, self.assertRaises(SystemExit):
                audit.main(['--kit',str(kit),'--height-calibration',str(h),'--out',str(out)])
            self.assertEqual(out.read_text(),'retained');assemble.assert_not_called()

    def test_uses_exact_existing_staged_scorer_and_guarded_policy(self):
        target=audit.ROOT/'experiments/r19_combined_gate'
        plan=(target/'plan.json').read_bytes()
        self.assertEqual(json.loads(plan)['required_pins']['r18_integrated_study_sha256'],
                         hashlib.sha256((audit.ROOT/'experiments/r18_combined_gate/study.py').read_bytes()).hexdigest())
        source=(target/'study.py').read_text()
        self.assertIn('from experiments.r18_combined_gate import study as frozen',source)
        self.assertIn('frozen.main()',source)

    def test_no_output_in_course_tree(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);kit=kit_shape(t);h=t/'cal.json';h.write_text('{}')
            with mock.patch.object(audit.assembly,'main') as assemble, self.assertRaises(SystemExit):
                audit.main(['--kit',str(kit),'--height-calibration',str(h),'--out',str(kit/'out')])
            assemble.assert_not_called()
