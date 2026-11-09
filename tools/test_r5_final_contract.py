"""Lock the append-only final contract veto and exact precommitted fallback."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'results/v2_protocol'


def read(name):
    return json.loads((D / name).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FinalContractTests(unittest.TestCase):
    def test_final_submission_is_the_exact_prefrozen_r1_fallback(self):
        final = read('R5_FINAL_ADOPTION.json')
        freeze = read('R5_FREEZE.json')
        self.assertEqual({'servo': 'R1', 'torque': 'R1'}, final['effective_variants'])
        self.assertEqual('R1', final['final_assembly'])
        self.assertTrue(final['fallback_only_from_prefrozen_assemblies'])
        self.assertFalse(final['runner_up_promoted'])
        for stem in ('servo', 'torque', 'detector'):
            name = 'my_' + stem + '.py'
            self.assertEqual((ROOT / name).read_bytes(),
                             (ROOT / ('r1_' + stem + '.py')).read_bytes())
            self.assertEqual(final['final_files'][name], sha(ROOT / name))
        record = freeze['assemblies']['R1']
        self.assertEqual(record['sha256'], sha(ROOT / record['path']))
        self.assertEqual((ROOT / record['path']).read_bytes(),
                         (ROOT / 'my_detector.py').read_bytes())

    def test_historical_positive_decision_is_not_rewritten(self):
        final = read('R5_FINAL_ADOPTION.json')
        previous = read('R5_ADOPTION.json')
        self.assertEqual({'servo': 'SCALE_STATE', 'torque': 'R1'},
                         previous['effective_variants'])
        self.assertEqual(sha(D / 'R5_ADOPTION.json'), final['supersedes']['sha256'])
        self.assertTrue(final['original_adoption_record_not_edited'])
        self.assertTrue(final['positive_sealed_measurements_retained'])
        holdout = read('R5_HOLDOUT_REVIEW.json')
        self.assertEqual('PASS', holdout['entries'][0]['status'])
        self.assertEqual('PASS', read('R5_OFFICIAL_REGRESSION.json')['status'])
        self.assertEqual(0, final['new_holdout_or_official_batches'])

    def test_existing_full_suite_failure_is_recorded_as_a_veto(self):
        contract = read('R5_FINAL_CONTRACT_REVIEW.json')
        final = read('R5_FINAL_ADOPTION.json')
        self.assertEqual('FAIL', contract['status'])
        self.assertEqual(526, contract['tests_run'])
        self.assertEqual(8, contract['errors'])
        self.assertEqual(8, len(contract['error_test_names']))
        self.assertEqual(sha(D / 'R5_FINAL_CONTRACT_REVIEW.json'),
                         final['contract_review']['sha256'])
        self.assertIn('FINAL_SUBMISSION_COMPATIBILITY_FAILED', final['reasons']['servo'])
        for key in ('scientific_source_changes', 'model_or_threshold_changes',
                    'protocol_or_gate_changes', 'new_holdout_or_official_batches'):
            self.assertEqual(0, contract[key])
        self.assertFalse(contract['numeric_gates_relaxed'])

    def test_rejected_assembly_and_its_defect_remain_frozen_evidence(self):
        contract = read('R5_FINAL_CONTRACT_REVIEW.json')
        freeze = read('R5_FREEZE.json')
        record = freeze['assemblies']['A_STATE']
        source = ROOT / record['path']
        self.assertEqual(record['sha256'], sha(source))
        self.assertEqual(record['sha256'], contract['tested_assembly']['sha256'])
        declarations = [node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name)
                                and target.id == '_R5_ENABLED_MODES'
                                for target in node.targets)]
        self.assertEqual(1, len(declarations))
        modes = ast.literal_eval(declarations[0].value)
        self.assertEqual('servo', modes)
        with self.assertRaises(TypeError):
            _ = None not in modes

    def test_only_submission_alias_differs_from_the_sealed_science_manifest(self):
        freeze = read('R5_FREEZE.json')
        self.assertEqual(119, len(freeze['source_files']))
        self.assertNotEqual(freeze['source_files']['my_detector.py'],
                            sha(ROOT / 'my_detector.py'))
        self.assertEqual(freeze['assemblies']['R1']['sha256'],
                         sha(ROOT / 'my_detector.py'))
        self.assertFalse(read('R5_FINAL_ADOPTION.json')['candidate_repair_or_new_configuration_after_seal'])


if __name__ == '__main__':
    unittest.main()
