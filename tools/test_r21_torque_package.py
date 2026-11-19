import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from experiments.r21_release_gate import build
from tools import r21_torque_package


def digest(data):
    return hashlib.sha256(data).hexdigest()


class TorquePackageTest(unittest.TestCase):
    def audit_runner(self, kit, output):
        with tempfile.TemporaryDirectory() as tmp:
            detector = Path(tmp) / "detector.py"
            receipt = build.build(detector, variant="yaw")
        output.write_text(json.dumps({
            "schema": "r21-yaw-release-audit.v1",
            "variant": "yaw",
            "status": "installable_research_option",
            "default_submission_promoted": False,
            "build": receipt,
            "kit": {"status": "ok", "manifest_sha256": "kit-pin"},
            "evaluator_sha256": "evaluator-pin",
            "controllers": {"torque": build.sha(build.ROOT.joinpath("my_torque.py").read_bytes())},
            "recognized_collision_cells": {"torque": 24, "servo": 23},
            "gates": {
                "standalone_default_filename": True,
                "frozen_yaw_torque_table_exact": True,
                "servo_R10_fallback_exact": True,
                "protected_kit_unchanged": True,
            },
        }, sort_keys=True))

    def test_archive_contains_only_two_python_sources_readme_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); kit = base / "kit"; kit.mkdir(); out = base / "yaw.zip"
            result = r21_torque_package.create_package(
                kit, out, audit_runner=self.audit_runner)
            with ZipFile(out) as archive:
                self.assertEqual(archive.namelist(), [
                    "torque_controller.py", "collision_detector.py",
                    "README.md", "receipt.json"])
                receipt = json.loads(archive.read("receipt.json"))
                self.assertNotIn(str(base), archive.read("README.md").decode())
                self.assertNotIn(str(base), json.dumps(receipt))
                self.assertEqual(receipt["audit_sha256"], result["audit_sha256"])
                self.assertEqual(receipt["full_d4"]["verdict"], "FAIL")
                self.assertEqual(receipt["full_d4"]["policy_regression_case_ids"], [174, 330, 339, 349])

    def test_same_inputs_produce_byte_identical_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); kit = base / "kit"; kit.mkdir()
            first, second = base / "one.zip", base / "two.zip"
            r21_torque_package.create_package(kit, first, audit_runner=self.audit_runner)
            r21_torque_package.create_package(kit, second, audit_runner=self.audit_runner)
            self.assertEqual(digest(first.read_bytes()), digest(second.read_bytes()))

    def test_existing_or_symlink_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); kit = base / "kit"; kit.mkdir()
            existing = base / "existing.zip"; existing.write_bytes(b"x")
            with self.assertRaises(ValueError):
                r21_torque_package.create_package(kit, existing, audit_runner=self.audit_runner)
            link = base / "link.zip"; link.symlink_to(base / "missing.zip")
            with self.assertRaises(ValueError):
                r21_torque_package.create_package(kit, link, audit_runner=self.audit_runner)

    def test_file_raced_into_place_during_audit_is_not_deleted_or_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); kit = base / "kit"; kit.mkdir(); out = base / "raced.zip"

            def racing_audit(kit_path, audit_path):
                self.audit_runner(kit_path, audit_path)
                out.write_bytes(b"unrelated-sentinel")

            with self.assertRaises(FileExistsError):
                r21_torque_package.create_package(kit, out, audit_runner=racing_audit)
            self.assertEqual(out.read_bytes(), b"unrelated-sentinel")

    def test_readme_states_known_limitations_and_non_hardware_scope(self):
        text = r21_torque_package._readme().decode()
        self.assertIn("case 174", text)
        self.assertIn("330, 339, 349", text)
        self.assertIn("40/129 ms", text)
        self.assertIn("6--6.5 N", text)
        self.assertIn("not hardware use", text)

    def test_output_under_repository_or_kit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            kit = Path(tmp) / "kit"; kit.mkdir()
            with self.assertRaises(ValueError):
                r21_torque_package.create_package(
                    kit, build.ROOT / "forbidden.zip", audit_runner=self.audit_runner)
            with self.assertRaises(ValueError):
                r21_torque_package.create_package(
                    kit, kit / "forbidden.zip", audit_runner=self.audit_runner)


if __name__ == "__main__":
    unittest.main()
