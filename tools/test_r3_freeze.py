import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import r3_freeze as F
from tools import r3_extended as E, r3_study as R


class TestR3Freeze(unittest.TestCase):
    def _fixture(self, root, eligible=False):
        records = {
            'PROTOCOL_R3.json': {},
            'R3_BOUND_STABILITY_REVIEW.json': {'protocol_gate_check': {'status': 'PASS'}},
            'R3_CHALLENGERS.json': {'architectures': {
                mode: dict(selected=name, validation_eligible=eligible,
                           irreversibly_ineligible_for_adoption=not eligible)
                for mode, name in [('servo', 'A_handoff_long'), ('torque', 'B_mass')]}},
            'R3_REDESIGN_DECISION.json': {},
            'R3_PRNE_MATH.json': {'passed': True},
            'R3_PERTURB_PARITY.json': {'results': [{'passed': True}]},
            'R3_PERTURBATIONS.json': {}, 'R3_SIGNAL_ANALYSIS.json': {},
            'R3_D4_SCOPE_PARITY.json': {}, 'R3_REPLAY_TIMING.json': {},
            'R3_PREFREEZE_INTEGRITY.json': {'passed': True},
        }
        for name in ('R3_DEVELOPMENT_REVIEW.json', 'R3_D4_REVIEW.json', 'R3_PERTURB_REVIEW.json'):
            records[name] = {'architectures': {mode: {'review': {'status': 'PASS'}}
                                                for mode in ('servo', 'torque')}}
        for name, value in records.items():
            (root / name).write_text(json.dumps(value))

    def test_green_diagnostic_stages_never_revive_rejected_sources(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self._fixture(root)
            with (patch.object(F, 'D', root), patch.object(F, 'DECISION', root / 'decision.json'),
                  patch.object(F, 'SELECTION', root / 'selection.json'),
                  patch.object(F, 'FREEZE', root / 'freeze.json')):
                result = F.prepare()
            self.assertEqual([item['id'] for item in result['chosen_specs']], ['R1', 'R1'])
            self.assertTrue(result['no_runner_up_or_diagnostic_revival'])

    def test_freeze_includes_all_required_and_transitive_runtime_sources(self):
        selected = [R.baseline_spec(mode) for mode in ('servo', 'torque')]
        names = set(F.source_paths())
        self.assertTrue(E._required_freeze_paths(selected) <= names)
        self.assertIn('tools/production_audit.py', names)
        self.assertIn('tools/sweep.py', names)

    def test_negative_round_freeze_refuses_a_new_eligibility_claim(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self._fixture(root, eligible=True)
            with (patch.object(F, 'D', root), patch.object(F, 'DECISION', root / 'decision.json'),
                  patch.object(F, 'SELECTION', root / 'selection.json'),
                  patch.object(F, 'FREEZE', root / 'freeze.json')):
                with self.assertRaisesRegex(RuntimeError, 'irreversible rejection'):
                    F.prepare()
                self.assertFalse((root / 'decision.json').exists())
