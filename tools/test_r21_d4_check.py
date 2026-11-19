import json
from pathlib import Path
import unittest

from tools import r21_d4_check as check


class D4CheckTests(unittest.TestCase):
    def test_dependency_manifest_is_relative_and_complete(self):
        self.assertEqual(len(check.DEPENDENCIES), len(set(check.DEPENDENCIES)))
        self.assertTrue(all(not Path(name).is_absolute() and ".." not in Path(name).parts
                            for name in check.DEPENDENCIES))
        for required in (check.PLAN, check.RUNNER, check.AUDIT_PLAN, check.AUDIT,
                         check.EXPECTED, "tools/check_integrity.py",
                         "tools/d4_robustness.py", "tools/sweep.py"):
            self.assertIn(required, check.DEPENDENCIES)
        blobs, pins = check.snapshot()
        self.assertEqual(set(blobs), set(check.DEPENDENCIES))
        self.assertTrue(all(len(value) == 64 for value in pins.values()))

    def test_commands_are_no_bytecode_and_worker_bounded(self):
        commands = check.commands("python", 3)
        self.assertEqual(len(commands), 2)
        for command, _ in commands:
            self.assertEqual(command[1:3], ["-B", "-m"])
            self.assertEqual(command[command.index("--workers") + 1], "3")
        self.assertEqual(commands[0][0][3], "experiments.r21_full_protection.runner_v2")
        self.assertEqual(commands[1][0][3], "experiments.r21_full_protection.force_audit")

    def test_first_difference_is_exact_and_path_specific(self):
        self.assertIsNone(check.first_difference({"a": [1, 2]}, {"a": [1, 2]}))
        self.assertEqual(check.first_difference({"a": [1, 2]}, {"a": [1, 3]}),
                         "science.a[1]: 2 != 3")
        self.assertEqual(check.first_difference({"a": 1}, {"b": 1}),
                         "science: keys differ")

    def test_frozen_summary_declares_exact_regressions_and_exposure(self):
        summary = json.loads((check.ROOT / check.EXPECTED).read_text())
        self.assertEqual(summary["verdict"], "FAIL")
        self.assertIn("exposed", summary["classification"])
        failures = summary["confirmed_policy_comparison_failures"]
        self.assertEqual(failures["case_ids"], [174, 330, 339, 349])
        self.assertEqual(failures["new_no_force_false_case_ids"], [174])
        self.assertEqual(failures["new_duplicate_case_ids"], [330, 339, 349])
        audit = summary["separate_exact_stock_input_audit"]
        self.assertTrue(audit["all_signed_input_exact"])
        self.assertTrue(audit["all_stock_scalar_exact_same_per_tick_order"])


if __name__ == "__main__":
    unittest.main()
