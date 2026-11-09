"""Simulation-free tests for the portable R5 normal-only learning tool."""

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sim import trajectory as T
from tools import r5_normal_learning as L


ROOT = Path(__file__).resolve().parents[1]


def episode(role="fit", mode="servo"):
    t = np.arange(6000, dtype=float) * 0.001
    refs = [T.reference(float(value)) for value in t]
    q = np.vstack([value[0] for value in refs])
    qd = np.vstack([value[1] for value in refs])
    qdd = np.vstack([value[2] for value in refs])
    grip = (t >= T.T_GRASP) & (t < T.T_RELEASE)
    active = grip & (t > 1.5) & (t < 3.8)
    value = {
        "t": t, "th": q.copy(), "thd": qd.copy(),
        "tau_m": np.zeros((6000, 6)), "tau_s": np.zeros((6000, 6)),
        "r_joint": np.zeros((6000, 6)), "wrench": np.zeros((6000, 6)),
        "grip": grip, "active": active, "r1_detect": np.zeros(6000, dtype=bool),
        "ref_q": q, "ref_qd": qd, "ref_qdd": qdd, "mass": 3.25,
        "com": np.array([0.01, -0.015, 0.03]), "mode": mode,
        "role": role, "normal_only": True,
    }
    prediction = L._r1_prediction(value)
    phase = L._phase_index(t)
    offsets = np.array([[0., 0.], [.4, -.2], [-.3, .25], [.1, .2], [0., 0.]])
    ripple = np.column_stack((.2 * np.sin(5 * t), .15 * np.cos(4 * t)))
    value["wrench"][:, :2] = prediction + offsets[phase] + ripple
    return value


def write_signal(path, ep):
    np.savez_compressed(path, **{key: ep[key] for key in L.SIGNAL_KEYS})
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AlgorithmTests(unittest.TestCase):
    def test_registered_grid_and_numeric_constants_are_fixed(self):
        specs = L.specifications()
        self.assertEqual(8, len(specs))
        self.assertEqual([64, 128, 64, 128],
                         [spec["bands"] for spec in specs if "bands" in spec])
        self.assertEqual([377003, 377004, 377005, 377006],
                         [spec["parameter_seed"] for spec in specs if "parameter_seed" in spec])
        self.assertEqual([20, 50], [spec["lag_ms"] for spec in specs if "lag_ms" in spec])
        self.assertEqual((.01, .1, 4, 8, .25),
                         (L.RIDGE, L.EMA_ALPHA, L.ON_COUNT, L.CLEAR_COUNT, L.SCALE_FLOOR))
        self.assertEqual([2, 2, 2, 2, 3, 3, 2, 2],
                         [spec["target_dims"] for spec in specs])

    def test_one_synthetic_fit_and_calibration_obey_sealed_formula(self):
        fit_ep = episode("fit")
        cal_ep = episode("calibration")
        model = L.fit("scale_active_phase_mad", [fit_ep])
        self.assertEqual("r5.normal-model.v1", model["format"])
        phase = L._phase_index(fit_ep["t"])
        residual = fit_ep["wrench"][:, :2] - L._r1_prediction(fit_ep)
        for index in range(5):
            selected = fit_ep["active"] & (phase == index)
            self.assertEqual(int(np.sum(selected)), model["phase_fit_counts"][index])
            if not np.any(selected):
                np.testing.assert_array_equal(np.zeros(2), model["location"][index])
                self.assertEqual(1.0, model["phase_scale"][index])
                continue
            location = np.median(residual[selected], axis=0)
            np.testing.assert_allclose(location, model["location"][index], rtol=0., atol=1e-14)
            norm = np.linalg.norm(residual[selected] - location, axis=1)
            expected = max(float(np.median(np.abs(norm - np.median(norm))) * 1.4826), .25)
            self.assertAlmostEqual(expected, model["phase_scale"][index], places=14)
        result = L.trace(model, cal_ep)
        peak = L._four_run_peak(result["ema"], result["active"])
        threshold = L.calibrate(model, [cal_ep])
        self.assertEqual(1.05 * peak + 1e-12, threshold)
        self.assertFalse(np.any(L.alarms(result, threshold)))
        result["r1_detect"][200] = True
        self.assertTrue(L.alarms(result, threshold)[200])
        self.assertFalse(np.any(L._own_detection(result, threshold)))

    def test_role_and_field_firewalls(self):
        audit = episode("audit")
        with self.assertRaises(ValueError):
            L.fit("scale_active_phase_mad", [audit])
        bad = episode("fit")
        bad["seed"] = 3
        with self.assertRaises(ValueError):
            L.fit("scale_active_phase_mad", [bad])
        bad = episode("calibration")
        bad["normal_only"] = False
        model = L.fit("scale_active_phase_mad", [episode("fit")])
        with self.assertRaises(ValueError):
            L.calibrate(model, [bad])

    def test_algorithm_source_is_portable_and_neutral(self):
        source = inspect.getsource(L)
        disallowed = ("/ho" + "me/", "WORK" + "_ROOT", "TA" + "SK-", "Cl" + "aude",
                      "Co" + "dex", "G" + "PT", "O" + "MX")
        for text in disallowed:
            self.assertNotIn(text, source)
        self.assertIn("normal_features", source)
        self.assertIn("MODEL_PARAMETERS", source)


class LoaderTests(unittest.TestCase):
    def test_loader_strips_row_metadata_and_never_reads_physical_file(self):
        design = json.loads(L.DESIGN_PATH.read_text())
        case = deepcopy(design["roles"]["fit"][0])
        ep = episode("fit")
        ep["mass"] = case["mass"]
        ep["com"] = np.asarray(case["com"])
        with tempfile.TemporaryDirectory() as folder_name:
            folder = Path(folder_name)
            signal = folder / "000.signals.npz"
            digest = write_signal(signal, ep)
            row = {
                "case": case, "role": "fit", "mode": "servo", "normal_only": True,
                "official": False, "signal_file": signal.name, "signal_sha256": digest,
                "physical_file": "must-not-be-opened.physical.npz",
                "physical_sha256": "unused", "source_hashes": {"seed": "metadata-only"},
                "status": {"diverged": False, "samples": 6000},
            }
            loaded = L._load_episode(row, "fit", "servo", folder.resolve())
            self.assertEqual(L._REQUIRED, set(loaded))
            self.assertFalse(any(key in loaded for key in ("seed", "case", "physical_file")))

    def test_index_rejects_wrong_plan_before_opening_signals(self):
        _, design, plan_sha, design_sha = L._public_contract()
        with tempfile.TemporaryDirectory() as folder_name:
            root = Path(folder_name)
            folder = root / "fit/servo"
            folder.mkdir(parents=True)
            (folder / "index.json").write_text(json.dumps({
                "schema": "R5_S2_DATA_INDEX.v1", "role": "fit", "mode": "servo",
                "plan_sha256": "0" * 64, "design_sha256": design_sha, "rows": [],
            }))
            with self.assertRaisesRegex(ValueError, "provenance"):
                L._load_index(root, "fit", "servo", design, plan_sha, design_sha)

    def test_output_must_be_empty_and_outside_logs_or_registered_models(self):
        with tempfile.TemporaryDirectory() as folder_name:
            root = Path(folder_name)
            dataset = root / "dataset"
            dataset.mkdir()
            occupied = root / "occupied"
            occupied.mkdir()
            (occupied / "keep").write_text("x")
            with self.assertRaises(ValueError):
                L._prepare_output(dataset, occupied)
            with self.assertRaises(ValueError):
                L._prepare_output(dataset, root / "logs/new")
            empty = root / "empty"
            empty.mkdir()
            _, selected = L._prepare_output(dataset, empty)
            self.assertEqual(empty.resolve(), selected)
        with self.assertRaises(ValueError):
            L._prepare_output(Path("/tmp/input"), ROOT / "models/r5/new")

    def test_signal_allowlist_has_no_oracle_or_provenance_fields(self):
        self.assertEqual(13, len(L.SIGNAL_KEYS))
        for name in ("seed", "label", "physical", "contact", "table_force", "tool_force"):
            self.assertNotIn(name, L.SIGNAL_KEYS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
