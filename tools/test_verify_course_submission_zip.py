import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.verify_course_submission_zip import EXPECTED_NAMES, ROOT, inspect_zip, normalized


class ZipContract(unittest.TestCase):
    def make(self, names=None, detector=b"import numpy as np\n"):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "in.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name in names or EXPECTED_NAMES:
                source = (ROOT / name).read_bytes() if name in {"my_servo.py", "my_torque.py"} else detector
                archive.writestr(name, source)
        return path

    def test_exact_three_top_level_python_files(self):
        path = self.make()
        files, digest = inspect_zip(path, None)
        self.assertEqual(set(files), EXPECTED_NAMES)
        self.assertEqual(len(digest), 64)

    def test_rejects_extra_and_dependency(self):
        with self.assertRaisesRegex(ValueError, "exactly three"):
            inspect_zip(self.make(EXPECTED_NAMES | {"README.md"}), None)
        with self.assertRaisesRegex(ValueError, "non-submission"):
            inspect_zip(self.make(detector=b"from experiments.foo import X\n"), None)

    def test_official_parser_excludes_host_timing(self):
        text = """
  8kg | 1.17 | 3.91 | 0.0 | 9.7 / 14.6 | 79 N | 999 | 0 / 0 / 0 회 | 정상
  8kg 1.42s | 66 | 55 | 36 | 24 | 0회 | 10 N
"""
        cells = normalized(text)
        self.assertEqual(cells["control"]["8.0"],
                         ["1.17","3.91","0.0","9.7","14.6","79","0","0","0","정상"])
        self.assertEqual(cells["collisions"][0]["cells"], ["66","55","36","24","0회","10 N"])


if __name__ == "__main__":
    unittest.main()
