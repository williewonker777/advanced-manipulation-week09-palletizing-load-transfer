import unittest

from experiments.r21_release_gate import audit


class AuditSummaryTest(unittest.TestCase):
    def test_collision_count_ignores_misses_and_metadata(self):
        report = {"collisions": [
            {"cells": ["10", "놓침", "경보 중", "33", "1회", "10 N"]},
            {"cells": ["구분 불가", "4", "놓침", "2", "0회", "15 N"]},
        ]}
        self.assertEqual(audit.collision_count(report), 4)

    def test_cells_excludes_execution_metadata(self):
        report = {"episodes": 29, "control": {"0.0": []}, "collisions": []}
        self.assertEqual(audit.cells(report), {
            "control": {"0.0": []}, "collisions": []})

    def test_control_comparison_ignores_only_wall_clock_compute_column(self):
        report = {"control": {"0.0": ["2.09", "4.05", "0.0", "9.0", "4.8",
                                         "999", "0", "0", "0", "정상"]}}
        self.assertEqual(audit.control_without_compute(report), {
            "0.0": ["2.09", "4.05", "0.0", "9.0", "4.8", "0", "0", "0", "정상"]})

    def test_frozen_R10_sources_are_bound(self):
        self.assertEqual(audit.expected_source("torque"),
                         "8f66f8f03cde65fcc6d8ef2f11643691cd8fe2e96dad6eaeaa17a0c7aa3db371")
        self.assertEqual(audit.expected_source("servo"),
                         "40706968fb72ed8d7a864e44e828787ae8db0365c2927e5166e4a5b2d83bba8b")


if __name__ == "__main__":
    unittest.main()
