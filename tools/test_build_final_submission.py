"""Packaging invariants of the final hand-in (no simulator needed)."""
import ast
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from tools import build_combined_submission as base
from tools import build_final_submission as final


class FinalHandInContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files, cls.receipt = final.materialize()
        cls.detector = cls.files['my_detector.py'].decode()

    def test_controllers_are_unchanged_r1(self):
        for name in ('my_servo.py', 'my_torque.py'):
            self.assertEqual(final.digest(self.files[name]), base.RELEASE[name])

    def test_torque_linkage_is_the_verified_r10_r31_body(self):
        codex = base.materialize()[0]['my_detector.py'].decode()
        self.assertEqual(final.digest(codex.encode()), final.CODEX_COMBINED_DETECTOR)
        body = codex[len(final.OLD_HEAD):-len(base.DISPATCH)]
        self.assertEqual(self.detector.count(body), 1)
        self.assertTrue(self.detector.startswith(final.NEW_HEAD + body))
        self.assertEqual(self.detector.count((base.ROOT / 'my_detector.py').read_text()), 1)

    def test_servo_closure_and_dispatch(self):
        self.assertEqual(set(self.receipt['servo_sources']), set(final.SERVO_PINS))
        self.assertEqual(len(self.receipt['servo_modules']), 3)
        for token in ("_submission_torque_class(), 'r31-torque'",
                      "_submission_servo_class(), 'r32-servo'",
                      "_submission_r10.Detector, 'r10'"):
            self.assertEqual(self.detector.count(token), 1)
        self.assertFalse(self.receipt['new_detection_rules'])

    def test_no_research_imports_or_dynamic_loading(self):
        allow = {'mujoco', 'numpy', 'sim', 'collections', 'dataclasses', 'math', 'copy', 'json',
                 'types', 'base64', 'io', 'functools'}
        for node in ast.walk(ast.parse(self.detector)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split('.')[0], allow)
            if isinstance(node, ast.ImportFrom):
                self.assertIn(node.module.split('.')[0], allow)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                self.assertNotIn(node.id, {'open', 'Path', 'exec', 'eval', '__file__', '__import__'})

    def test_committed_submission_matches_reproducible_build(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            ra, rb = final.build(a), final.build(b)
            self.assertEqual(ra, rb)
            za = a / final.ZIP_NAME
            self.assertEqual(za.read_bytes(), (b / final.ZIP_NAME).read_bytes())
            self.assertEqual(ra['zip_sha256'], hashlib.sha256(za.read_bytes()).hexdigest())
            with ZipFile(za) as z:
                self.assertEqual(sorted(z.namelist()), ['my_detector.py', 'my_servo.py', 'my_torque.py'])
                self.assertIsNone(z.testzip())
                for n in z.namelist():
                    self.assertEqual(z.read(n), (base.ROOT / 'submission' / n).read_bytes())
            with self.assertRaises(ValueError):
                final.build(a)


if __name__ == '__main__':
    unittest.main()
