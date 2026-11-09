"""Byte-contract tests for the immutable R7 assembly copy."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import r7_build as B


class R7BuildTests(unittest.TestCase):
    def test_inputs_and_outputs_are_the_pinned_immutable_bytes(self):
        inherited = B.verify_inputs()
        self.assertEqual(8, len(inherited["models"]))
        outputs = {name: path.read_bytes() for name, path in B.SOURCES.items()}
        record = B._record(outputs)
        self.assertEqual("R7_BUILD.v1", record["schema"])
        self.assertEqual(0, record["assembly_logic_changes"])
        self.assertEqual(0, record["model_weight_or_threshold_updates"])
        self.assertTrue(all(row["byte_equal_to_source"] for row in record["generated"]))
        self.assertEqual(B.EXPECTED_HASHES,
            {row["name"]: row["sha256"] for row in record["generated"]})

    def test_build_writes_exact_copies_and_strict_record_in_a_sandbox(self):
        with tempfile.TemporaryDirectory(dir=B.ROOT / "results") as name:
            root = Path(name)
            outputs = {key: root / (key + ".py") for key in B.SOURCES}
            record_path = root / "R7_BUILD.json"
            with mock.patch.object(B, "OUTPUTS", outputs), \
                    mock.patch.object(B, "BUILD_RECORD", record_path):
                record = B.build()
                self.assertEqual(record, B.build(check=True))
            for key, path in outputs.items():
                self.assertEqual(B.SOURCES[key].read_bytes(), path.read_bytes())
            self.assertIn('"schema": "R7_BUILD.v1"', record_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
