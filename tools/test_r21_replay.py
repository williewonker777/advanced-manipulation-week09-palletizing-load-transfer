import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import r21_replay as replay


class ReplayTests(unittest.TestCase):
    def test_manifest_authentication_and_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            kit = Path(folder)
            (kit / "sim").mkdir(); (kit / "model").mkdir()
            (kit / "sim/a.py").write_bytes(b"a")
            (kit / "evaluate.py").write_bytes(b"eval")
            raw = (hashlib.sha256(b"a").hexdigest() + "  sim/a.py\n").encode()
            with mock.patch.object(replay, "MANIFEST_SHA256", replay.digest(raw)), \
                    mock.patch.object(replay, "EVALUATOR_SHA256", replay.digest(b"eval")):
                self.assertEqual(replay.authenticate_kit(kit, raw)["sim/a.py"], hashlib.sha256(b"a").hexdigest())
                (kit / "sim/extra.py").write_bytes(b"extra")
                with self.assertRaisesRegex(ValueError, "unexpected"):
                    replay.authenticate_kit(kit, raw)
                (kit / "sim/extra.py").unlink()
                (kit / "sim/a.py").write_bytes(b"b")
                with self.assertRaisesRegex(ValueError, "authentication failed"):
                    replay.authenticate_kit(kit, raw)

    def test_output_must_be_fresh_external(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); kit = base / "kit"; kit.mkdir()
            (kit / "sim").mkdir(); (kit / "model").mkdir(); (kit / "evaluate.py").touch()
            out = base / "out"
            self.assertEqual(replay.validate_paths(kit, out)[1], out.resolve())
            out.mkdir()
            with self.assertRaisesRegex(ValueError, "fresh"):
                replay.validate_paths(kit, out)

    def test_surface_proof_requires_four_rows_and_arms(self):
        arm = {"strict_exact_force": True, "force_exact": True, "retained_vs": {}, "score_sha256": "a" * 64}
        variant = {"arms": {name: dict(arm) for name in ("r10", "r20", "pick", "yaw")}}
        doc = {"schema": "r21-surface-pair-assay.v1", "episodes": 128,
               "rows": [{"id": str(i), "material_effect_observed": {"q": True},
                          "variants": {"stock": variant, "changed": variant}} for i in range(4)]}
        proof = replay.surface_proof(json.loads(json.dumps(doc)))
        self.assertEqual(len(proof["rows"]), 4)
        self.assertTrue(proof["four_twins"])
        doc["rows"].pop()
        with self.assertRaisesRegex(ValueError, "cohort"):
            replay.surface_proof(doc)

    def test_weak_commands_use_both_drivers_and_verify(self):
        commands = replay.weak_commands("python")
        modules = [command[command.index("-m") + 1] for command, _ in commands]
        self.assertEqual(modules, ["experiments.r21_weak_challenge.evaluator",
                                  "experiments.r21_weak_challenge.yaw_driver",
                                  "experiments.r21_weak_challenge.verify"])

    def test_history_snapshot_includes_model_and_both_plans(self):
        blobs, _ = replay.snapshot("history")
        self.assertIn("experiments/r21_yaw_continuity/model.json", blobs)
        self.assertIn("experiments/r21_yaw_continuity/plan.json", blobs)
        self.assertIn("experiments/r21_surface_pairs/plan.json", blobs)
        self.assertNotIn("experiments/r21_weak_challenge/plan.json", blobs)

    def test_publish_never_replaces_existing_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); result = base / "result"; result.mkdir()
            (result / "proof.json").write_text("{}")
            out = base / "out"; out.mkdir()
            marker = out / "user"; marker.write_text("keep")
            with self.assertRaises(FileExistsError):
                replay.publish(result, out)
            self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
