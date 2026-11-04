import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools import v2_study


class V2StudyCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = v2_study.load_protocol()

    def test_stage_case_counts(self):
        self.assertEqual(len(v2_study.build_control_cases(self.protocol, "v1", "servo")), 4)
        self.assertEqual(len(v2_study.build_development_cases(self.protocol, "v1", "servo")), 34)
        self.assertEqual(len(v2_study.build_d4_cases("v1", "servo")), 385)
        self.assertEqual(len(v2_study.build_timing_cases("v1", "servo")), 3)

    def test_development_direction_balance(self):
        cases = v2_study.build_development_cases(self.protocol, "v1", "servo")
        collisions = [case for case in cases if case["kind"] == "collision"]
        counts = {
            direction: sum(case["direction"] == direction for case in collisions)
            for direction in self.protocol["development"]["directions"]
        }
        self.assertEqual(set(counts.values()), {4})
        references = [case for case in cases if case["kind"] == "reference"]
        self.assertEqual(len(references), 2)

    def test_holdout_requires_matching_freeze_and_is_balanced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "PROTOCOL.json"
            freeze_path = root / "FREEZE.json"
            protocol_path.write_bytes(v2_study.PROTOCOL_PATH.read_bytes())
            with self.assertRaisesRegex(RuntimeError, "required freeze is missing"):
                v2_study.build_holdout_cases(
                    self.protocol, "v2", "servo",
                    protocol_path=protocol_path, freeze_path=freeze_path,
                )
            freeze_path.write_text(json.dumps({"protocol_sha256": "wrong"}))
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                v2_study.build_holdout_cases(
                    self.protocol, "v2", "servo",
                    protocol_path=protocol_path, freeze_path=freeze_path,
                )
            digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
            freeze_path.write_text(json.dumps({"protocol_sha256": digest}))
            cases = v2_study.build_holdout_cases(
                self.protocol, "v2", "servo",
                protocol_path=protocol_path, freeze_path=freeze_path,
            )
        self.assertEqual(len(cases), 38)
        collisions = [case for case in cases if case["kind"] == "collision"]
        references = [case for case in cases if case["kind"] == "reference"]
        offsets = [case for case in cases if case["kind"] == "offset"]
        self.assertEqual((len(references), len(collisions)), (4, 30))
        self.assertEqual(len(offsets), 4)
        self.assertTrue(all(case["pick_offset"] is None for case in references + collisions))
        self.assertEqual(
            {case["pick_offset"] for case in offsets},
            {tuple(offset) for offset in self.protocol["holdout"]["offset_m"]},
        )
        for field, expected in (("phase", 10), ("direction", 3), ("force_n", 6)):
            values = {case[field] for case in collisions}
            counts = {value: sum(case[field] == value for case in collisions) for value in values}
            self.assertEqual(set(counts.values()), {expected})

    def test_g4_window_metrics(self):
        times = np.arange(0.0, 6.0, 0.001)
        table = np.zeros_like(times)
        final = (times >= 3.88) & (times < 3.9)
        table[final] = np.arange(1, final.sum() + 1, dtype=float)
        table[np.argmin(np.abs(times - 4.0))] = 50.0
        metrics = v2_study.g4_window_metrics({"t": times, "f_table": table}, 2.0)
        self.assertAlmostEqual(metrics["table_mean_n"], 10.5)
        self.assertEqual(metrics["table_min_n"], 1.0)
        self.assertEqual(metrics["place_peak_n"], 50.0)
        self.assertAlmostEqual(metrics["table_mean_over_mg"], 10.5 / 19.62)
        zero = v2_study.g4_window_metrics({"t": times, "f_table": table}, 0.0)
        self.assertIsNone(zero["table_mean_over_mg"])
        self.assertIsNone(zero["place_peak_over_mg"])
        unavailable = v2_study.g4_window_metrics(
            {"t": times[:1000], "f_table": table[:1000]}, 2.0
        )
        self.assertFalse(unavailable["available"])
        self.assertIsNone(unavailable["table_mean_n"])

    def test_evidence_result_is_relative_identifier(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence_id = v2_study._save_evidence(
                {"t": np.asarray([0.0]), "qpos": np.ones((1, 2))},
                Path(directory), "case-0001",
            )
            self.assertEqual(evidence_id, "case-0001.npz")
            self.assertFalse(Path(evidence_id).is_absolute())
            with np.load(Path(directory) / evidence_id) as evidence:
                self.assertEqual(set(evidence.files), {"t"})


if __name__ == "__main__":
    unittest.main()


class FreezeSourceTests(unittest.TestCase):
    def test_frozen_artifact_mismatch_is_rejected(self):
        from tools.v2_study import verify_freeze_sources
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'weights.json').write_text('fixed')
            import hashlib
            expected = hashlib.sha256(b'fixed').hexdigest()
            verify_freeze_sources({'source_files': {'weights.json': expected}}, root)
            (root/'weights.json').write_text('changed')
            with self.assertRaises(RuntimeError):
                verify_freeze_sources({'source_files': {'weights.json': expected}}, root)
            with self.assertRaises(RuntimeError):
                verify_freeze_sources({}, root)


class TimingHistogramTests(unittest.TestCase):
    def test_all_samples_retained_in_distribution(self):
        d = v2_study._distribution([100000, 200000, 1500000, 2000000000])
        self.assertEqual(d['samples'], 4)
        self.assertEqual(sum(d['histogram_counts']) + d['histogram_underflow'] + d['histogram_overflow'], 4)
        self.assertEqual(d['histogram_overflow'], 1)
        self.assertEqual(d['max_us'], 2000000.)
