import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from .parity import compare_traces, inspect_submission


class SubmissionContract(unittest.TestCase):
    def make_zip(self, root, names=("my_servo.py", "my_torque.py", "my_detector.py")):
        target = root / "submission.zip"
        with zipfile.ZipFile(target, "w") as archive:
            for name in names:
                archive.writestr(name, b"value = 1\n")
        return target

    def test_accepts_exact_three_root_python_members(self):
        with tempfile.TemporaryDirectory() as folder:
            target = self.make_zip(Path(folder))
            report = inspect_submission(target)
            self.assertEqual(report["members"], ["my_detector.py", "my_servo.py", "my_torque.py"])

    def test_rejects_archive_with_an_extra_member(self):
        with tempfile.TemporaryDirectory() as folder:
            target = self.make_zip(Path(folder), ("my_servo.py", "my_torque.py", "my_detector.py", "notes.txt"))
            with self.assertRaisesRegex(ValueError, "exact three-file submission"):
                inspect_submission(target)

    def test_rejects_archive_member_with_path_or_symlink_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = self.make_zip(root)
            unsafe = root / "unsafe.zip"
            with zipfile.ZipFile(unsafe, "w") as archive:
                for name in ("my_servo.py", "my_torque.py"):
                    archive.writestr(name, b"value=1\n")
                info = zipfile.ZipInfo("folder/my_detector.py")
                archive.writestr(info, b"value=1\n")
            with self.assertRaisesRegex(ValueError, "exact three-file submission"):
                inspect_submission(unsafe)

    def test_reports_member_hashes_over_extracted_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            target = self.make_zip(Path(folder))
            report = inspect_submission(target)
            expected = hashlib.sha256(b"value = 1\n").hexdigest()
            self.assertEqual(set(report["sha256"].values()), {expected})


class TraceParity(unittest.TestCase):
    def test_accepts_bit_identical_detector_and_physical_arrays(self):
        values = {name: np.arange(12).reshape(6, 2) for name in ("q", "tau", "r")}
        values["detect"] = np.array([False, True, True, False, False, True])
        report = compare_traces(values, {key: value.copy() for key, value in values.items()})
        self.assertTrue(report["bit_identical"])

    def test_identifies_exact_array_that_differs(self):
        expected = {"r": np.array([[0.0], [1.0]]), "detect": np.array([False, True])}
        actual = {key: value.copy() for key, value in expected.items()}
        actual["r"][1, 0] = np.nextafter(1.0, 2.0)
        report = compare_traces(expected, actual)
        self.assertEqual(report["different"], ["r"])

    def test_rejects_missing_trace_key(self):
        with self.assertRaisesRegex(ValueError, "trace keys differ"):
            compare_traces({"r": np.zeros(1)}, {"detect": np.zeros(1)})


if __name__ == "__main__":
    unittest.main()
