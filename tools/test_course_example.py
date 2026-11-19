import base64
import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from tools import course_example as subject


class CourseExampleTest(unittest.TestCase):
    def test_rejects_output_inside_kit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kit = root / "kit"; kit.mkdir()
            guide = root / "guide.html"; guide.write_text("guide")
            with self.assertRaisesRegex(subject.CourseError, "external"):
                subject.safe_paths(kit, guide, kit / "output")

    def test_unrecognized_guide_is_explicitly_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            guide = Path(tmp) / "guide.html"; guide.write_text("revision")
            with self.assertRaisesRegex(subject.CourseError, "unrecognized_revision"):
                subject.verify_guide(guide)

    def test_extracts_only_expected_png_from_signed_guide(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = b"synthetic-png"
            guide = Path(tmp) / "guide.html"
            guide.write_bytes(b"data:image/png;base64," + base64.b64encode(png))
            with mock.patch.object(subject, "GUIDE_SHA", subject.digest(guide)), \
                 mock.patch.object(subject, "REFERENCE_PNG_SHA", hashlib.sha256(png).hexdigest()):
                self.assertEqual(subject.verify_guide(guide), (subject.digest(guide), 1))

    def test_wrong_stock_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); kit = root / "kit"; (kit / "sim").mkdir(parents=True)
            (kit / "model").mkdir()
            manifest = root / "manifest"; manifest.write_text("")
            with mock.patch.object(subject, "MANIFEST_SHA", subject.digest(manifest)):
                with self.assertRaisesRegex(subject.CourseError, "stock course source mismatch"):
                    subject.verify_kit(kit, manifest)

    def test_extra_protected_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); kit = root / "kit"; (kit / "sim").mkdir(parents=True)
            (kit / "model").mkdir(); original = kit / "sim/original.py"
            original.write_text("original")
            manifest = root / "manifest"
            manifest.write_text(subject.digest(original) + "  sim/original.py\n")
            with mock.patch.object(subject, "MANIFEST_SHA", subject.digest(manifest)), \
                 mock.patch.object(subject, "SOURCES", {}):
                self.assertEqual(len(subject.verify_kit(kit, manifest)), 1)
                (kit / "sim/extra.py").write_text("extra")
                with self.assertRaisesRegex(subject.CourseError, "exactly match"):
                    subject.verify_kit(kit, manifest)


    def test_render_difference_does_not_diagnose_the_renderer_or_reject_original_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            out = base / "output"

            def different_plot(*args, **kwargs):
                (kwargs["cwd"] / "result.png").write_bytes(b"different plot bytes")
                return mock.Mock(stdout="stock run")

            with (mock.patch.object(subject, "safe_paths", return_value=(base, base / "guide", out)),
                  mock.patch.object(subject, "verify_kit", return_value={}),
                  mock.patch.object(subject, "verify_guide", return_value=(subject.GUIDE_SHA, 12)),
                  mock.patch.object(subject.subprocess, "run", side_effect=different_plot)):
                record = subject.run(base, base / "guide", out)
            self.assertTrue(record["source_authenticity"]["unchanged_after_run"])
            self.assertFalse(record["render_identity"]["byte_identical_to_guide"])
            self.assertEqual(record["render_identity"]["status"], "not_byte_identical")


if __name__ == "__main__":
    unittest.main()
