import importlib.util
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch

from tools import r26_yaw_acquisition_replay as replay


HAS_COURSE_KIT = importlib.util.find_spec("sim") is not None
if HAS_COURSE_KIT:
    from experiments.r26_yaw_onset_portable import target


@unittest.skipUnless(HAS_COURSE_KIT, "external course kit required")
class TargetTests(unittest.TestCase):
    def test_registry_and_candidate_are_sealed(self):
        result = target.sealed.registration()
        self.assertEqual(result["case"]["id"], "s25_long_transfer_signed")
        self.assertEqual(target.digest(
            target.ROOT / "experiments/r26_yaw_acquisition/candidate.py"),
            target.CANDIDATE_SHA256)

    def test_tick_intervals_use_recorded_ticks(self):
        trace = [
            {"tick": tick, "active": value}
            for tick, value in enumerate((False, True, True, False, True), start=10)
        ]
        self.assertEqual(target.tick_intervals(trace, "active"), [[11, 12], [14, 14]])


class ReplayTests(unittest.TestCase):
    def test_no_default_and_fixed_catalog(self):
        catalog = replay.describe()
        self.assertFalse(catalog["default_run"])
        self.assertEqual(catalog["case"], "s25_long_transfer_signed")
        self.assertEqual(catalog["episodes"], 12)

    def test_extension_is_relative_pinned_and_kit_free(self):
        data = replay.extension()
        self.assertEqual(set(data["files"]),
                         set(data["overlay_files"]) | set(data["public_files"]))
        self.assertIn("experiments/r26_yaw_acquisition/candidate.py",
                      data["overlay_files"])
        for name in data["files"]:
            path = PurePosixPath(name)
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)
            self.assertFalse(name.startswith(("sim/", "model/")))
            self.assertNotEqual(name, "evaluate.py")

    def test_existing_output_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kit, out = root / "kit", root / "result"
            kit.mkdir()
            out.write_text("sentinel")
            with (patch.object(replay, "ROOT", root),
                  patch.object(replay.sealed.shared, "validate_paths",
                               side_effect=ValueError("fresh output required"))):
                with self.assertRaises(SystemExit):
                    replay.main(["replay", "--kit", str(kit), "--out", str(out)])
            self.assertEqual(out.read_text(), "sentinel")

    def test_result_contract_rejects_independent_claim(self):
        result = {
            "schema": "r26-known-signed-yaw-acquisition-diagnostic.v1",
            "classification": replay.CLASSIFICATION,
            "independent": True,
            "unopened_r26_conditions_used": False,
            "default_promoted": False,
            "original_course_physics": True,
            "episodes": 12,
            "candidate_sha256":
                "212f4e79367cb636c0143a2d1f8aa7ba5584aaaca5ecde6b412d01a4985b6784",
            "rows": [{"variant": name, "passed": True} for name in
                     ("stock", "actuator_shift", "contact_shift")],
            "passed": True,
        }
        with self.assertRaises(ValueError):
            replay.validate_result(result)


if __name__ == "__main__":
    unittest.main()
