"""Integrity checker regression tests use temporary trees, never the real kit."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from tools.check_integrity import check_integrity


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("tools", "sim", "model"):
            (self.root / name).mkdir()
        self.data = {"sim/runner.py": b"immutable\n", "model/mesh.stl": b"mesh"}
        for name, content in self.data.items():
            (self.root / name).write_bytes(content)
        self.manifest = self.root / "tools/kit_manifest.sha256"
        self.manifest.write_text("".join(f"{hashlib.sha256(v).hexdigest()}  {k}\n"
                                         for k, v in self.data.items()))

    def test_pristine(self):
        self.assertEqual(check_integrity(self.root)["files"], 2)

    def test_modified(self):
        (self.root / "sim/runner.py").write_text("modified")
        with self.assertRaisesRegex(ValueError, "changed="):
            check_integrity(self.root)

    def test_missing(self):
        (self.root / "model/mesh.stl").unlink()
        with self.assertRaisesRegex(ValueError, "missing="):
            check_integrity(self.root)

    def test_extra_bytecode(self):
        (self.root / "sim/cache.pyc").touch()
        with self.assertRaisesRegex(ValueError, "extra="):
            check_integrity(self.root)

    def test_symlink(self):
        (self.root / "sim/alias").symlink_to("runner.py")
        with self.assertRaisesRegex(ValueError, "symlink"):
            check_integrity(self.root)

    def test_unsafe_manifest(self):
        self.manifest.write_text(f"{'0' * 64}  ../secret\n")
        with self.assertRaisesRegex(ValueError, "Invalid"):
            check_integrity(self.root)

    def test_empty_manifest(self):
        self.manifest.write_text("")
        with self.assertRaisesRegex(ValueError, "Empty"):
            check_integrity(self.root)


if __name__ == "__main__":
    unittest.main()
