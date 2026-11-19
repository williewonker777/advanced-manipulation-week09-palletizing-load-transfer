from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.r26_transfer_portable import evaluator
from tools import r26_landing_replay as replay


class R26PortableReplayTests(unittest.TestCase):
    def test_catalog_separates_original_from_research(self):
        catalog = replay.describe()
        self.assertEqual(catalog["original"]["entrypoint"], ["run.py", "--box", "8"])
        self.assertIn("Only the original", catalog["separation"])
        self.assertEqual(catalog["research_arms"]["registered-c1"]["status"],
                         "frozen_registered_candidate")
        for arm in catalog["research_arms"].values():
            self.assertNotEqual(arm.get("status"), "authentic_original_course")

    def test_scenarios_are_named_and_fixed(self):
        self.assertEqual(set(evaluator.SCENARIOS), set(replay.SCENARIOS))
        kind, case, collision = evaluator.scenario("signed-z-negative")
        self.assertEqual(kind, "signed_late_force")
        self.assertEqual(collision, [4.18, [0.0, 0.0, -16.0]])
        self.assertEqual(case["mass"], 5.8)

    def test_unfrozen_c1_fails_before_simulation(self):
        original = evaluator.registrations
        def unfrozen():
            data = original()
            data["arms"]["registered-c1"] = {
                "controller": "experiments.r26_delivery_pair.controller:Controller",
                "detector": None, "status": "not_frozen"}
            return data
        with patch.object(evaluator, "registrations", side_effect=unfrozen), \
             patch.object(evaluator.mujoco, "__version__", "3.17.0"):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                evaluator.evaluate("normal-sh5", ("registered-c1",))

    def test_runtime_rejects_wrong_mujoco(self):
        with tempfile.TemporaryDirectory() as folder:
            kit = Path(folder)
            (kit / "requirements.txt").write_text("mujoco==3.17.0\n")
            with patch("mujoco.__version__", "3.16.0"):
                with self.assertRaisesRegex(ValueError, "3.17.0"):
                    replay.runtime_gate(kit)

    def test_closure_is_kit_free_and_complete(self):
        document = replay.closure()
        self.assertEqual(set(document["files"]),
                         set(document["overlay_files"]) | set(document["public_files"]))
        self.assertIn("experiments/r26_delivery_pair/candidate.py", document["overlay_files"])
        self.assertIn("tools/r26_landing_replay.py", document["public_files"])
        self.assertFalse(any(name.startswith(("sim/", "model/"))
                             for name in document["files"]))

    def test_original_delegates_to_hash_pinned_course_runner(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = {"source_authenticity": {"status": "authentic_original_course"}}
            with patch.object(replay.course_example, "run", return_value=expected) as run:
                code = replay.main(["original", "--kit", str(root), "--guide", str(root / "g"),
                                    "--out", str(root / "o"), "--box", "8"])
            self.assertEqual(code, 0)
            run.assert_called_once_with(root, root / "g", root / "o")

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                replay.replay(root, root, "normal-sh5", ("r24",))


if __name__ == "__main__":
    unittest.main()
