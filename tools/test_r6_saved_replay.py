"""Synthetic pure-helper tests for the R6 saved-stream replay harness."""

import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools import r6_saved_replay as R

ROOT = Path(__file__).resolve().parents[1]


def synthetic_signals(samples=3):
    arrays = {}
    for name in R.SIGNALS:
        if name in {"t", "grip", "active", "r1_detect"}:
            arrays[name] = np.zeros(samples, dtype=float if name == "t" else bool)
        else:
            arrays[name] = np.zeros((samples, 6), dtype=float)
    return arrays


class SavedReplayHelperTests(unittest.TestCase):
    def test_exact_comparison_uses_dtype_shape_and_raw_bytes(self):
        bits = np.array([0x7ff8000000000001, 0x8000000000000000], dtype=np.uint64)
        left, right = bits.view(np.float64), bits.copy().view(np.float64)
        self.assertTrue(R.exact_array_comparison(left, right)["exact"])
        changed_nan = np.array(
            [0x7ff8000000000002, 0x8000000000000000], dtype=np.uint64).view(np.float64)
        self.assertFalse(R.exact_array_comparison(left, changed_nan)["exact"])
        self.assertFalse(R.exact_array_comparison(left, left.astype(np.float32))["exact"])
        self.assertFalse(R.exact_array_comparison(left, left.reshape(2, 1))["exact"])
        self.assertFalse(R.exact_array_comparison(np.array([-0.0]), np.array([0.0]))["exact"])

    def test_signal_schema_is_exact_and_shape_bound(self):
        signals = synthetic_signals(3)
        self.assertTrue(R.validate_signal_arrays(signals, samples=3))
        missing = dict(signals)
        missing.pop("active")
        with self.assertRaises(ValueError):
            R.validate_signal_arrays(missing, samples=3)
        wrong = dict(signals)
        wrong["th"] = np.zeros((3, 5))
        with self.assertRaises(ValueError):
            R.validate_signal_arrays(wrong, samples=3)

    def test_compare_replay_gates_actual_and_recorded_arrays(self):
        signals = synthetic_signals(3)
        trace = {
            "residual": signals["r_joint"].copy(), "detect": np.zeros(3, bool),
            "base_detect": signals["r1_detect"].copy(), "own": np.zeros(3, bool),
            "score": np.zeros(3), "ema": np.zeros(3), "monitor_present": np.ones(3, bool),
            "packet_r_joint": signals["r_joint"].copy(),
            "packet_wrench": signals["wrench"].copy(), "active": signals["active"].copy(),
            "controller_output": np.zeros((3, 4, 6)), "applied_command": np.zeros((3, 6)),
        }
        monitor = {name: trace[name].copy() for name in ("detect", "own", "score", "ema")}
        row = {"mode": "servo", "source_variant": "SCALE_STATE"}
        comparisons = R.compare_replay(row, signals, monitor, trace, trace)
        self.assertTrue(all(item["exact"] for item in comparisons.values()))
        changed = {name: value.copy() for name, value in trace.items()}
        changed["score"][1] = -0.0
        comparisons = R.compare_replay(row, signals, monitor, trace, changed)
        self.assertFalse(comparisons["old_vs_new:score"]["exact"])

    def test_create_only_and_hash_barrier_helpers(self):
        with tempfile.TemporaryDirectory() as folder_name:
            folder = Path(folder_name)
            raw, sidecar = folder / "raw.bin", folder / "sidecar.json"
            raw.write_bytes(b"raw")
            sidecar.write_text("{}")
            row = {"raw": {"absolute_path": str(raw), "sha256": R.sha256_path(raw)},
                   "monitor": None,
                   "sidecar": {"absolute_path": str(sidecar), "sha256": R.sha256_path(sidecar)}}
            R._verify_all_paths_before_npz_open([row])
            raw.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                R._verify_all_paths_before_npz_open([row])
            target = folder / "created.json"
            R.write_create(target, {"ok": True})
            with self.assertRaises(FileExistsError):
                R.write_create(target, {"ok": True})

    def test_aggregate_requires_exact_inventory(self):
        index = {"streams": [{"stream_id": "a"}, {"stream_id": "b"}]}
        results = {
            "a": {"stage": "D4", "mode": "servo", "samples": 3, "passed": True},
            "b": {"stage": "D4", "mode": "servo", "samples": 3, "passed": True},
        }
        expected = [{"stage": "D4", "mode": "servo", "streams": 2,
                     "samples": 6, "passed_streams": 2,
                     "all_bit_comparisons_passed": True}]
        self.assertEqual(expected, R.aggregate_results(index, results))
        with self.assertRaises(RuntimeError):
            R.aggregate_results(index, {"a": results["a"]})

    def test_harness_has_no_physics_rng_or_monkeypatch_entrypoints(self):
        path = ROOT / "tools/r6_saved_replay.py"
        text = path.read_text(encoding="utf-8")
        ast.parse(text, filename=str(path))
        self.assertNotIn("Plant(", text)
        self.assertNotIn("mj_step", text)
        self.assertNotIn("default_rng", text)
        self.assertNotIn("monkeypatch", text.lower())
        self.assertNotIn("unittest.mock", text)


if __name__ == "__main__":
    unittest.main()
