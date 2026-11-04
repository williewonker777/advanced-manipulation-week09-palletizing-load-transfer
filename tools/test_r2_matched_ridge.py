#!/usr/bin/env python3
"""Pure tests for the matched R1-observer ridge attribution ablation."""
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.r2_matched_ridge import make_detector
from tools.r2_matched_ridge import OUTPUT_KEYS, _load_permitted, fit, replay_one
from sim.nominal import Nominal
from sim import trajectory as T
import r1_servo


def _parameters(threshold):
    return {"mean": np.zeros(77), "scale": np.ones(77),
            "weights": np.zeros((77, 2)), "threshold": threshold}


def test_factory_reuses_r1_shared_core_and_replaces_only_ridge():
    controller = r1_servo.Controller()
    controller.nom, controller.nom_box = Nominal(), Nominal()
    controller.box_mass, controller.box_com = 2.0, (0.01, 0.0, 0.02)
    controller.nom_box.set_payload(controller.box_mass, controller.box_com)
    controller.reset()
    detector = make_detector({"servo": _parameters(4.25), "torque": _parameters(5.25)})()
    detector.ctrl = controller
    detector.nom, detector.nom_box = controller.nom, controller.nom_box
    detector.box_mass, detector.box_com = controller.box_mass, controller.box_com
    detector.reset()
    assert detector._shared is controller.shared_estimator
    assert detector.normal_model.threshold == 4.25


def test_replay_reads_permitted_and_writes_only_allowed_arrays():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); source = root / "source.npz"; output = root / "allowed.npz"
        times = np.arange(5) * 0.001
        q = np.array([T.reference(float(t))[0] for t in times])
        np.savez_compressed(source, obs_t=times, encoder=q,
                            encoder_velocity=np.zeros_like(q), jts=np.zeros_like(q),
                            grip=np.zeros(5, bool), forbidden_oracle=np.arange(5))
        loaded = _load_permitted(source)
        assert "forbidden_oracle" not in loaded
        replay_one("servo", {"mass": 0.75, "com": [0.01, 0.0, 0.02]}, source, output)
        with np.load(output, allow_pickle=False) as archive:
            assert set(archive.files) == set(OUTPUT_KEYS)
            assert archive["x"].shape == (5, 77)
            assert archive["y"].shape == (5, 2)


def test_fit_exact_contract_on_synthetic_allowed_data():
    import json
    import hashlib
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); records = []
        rng = np.random.default_rng(4)
        for mode in ("servo", "torque"):
            for split in ("train", "validation"):
                path = root / f"{mode}_{split}_0000.npz"
                x = rng.normal(size=(30, 77)); x[:, 0] = 1.0
                y = x[:, :2] * 0.2; active = np.ones(30, bool)
                np.savez_compressed(path, t=np.arange(30) * .001, x=x, y=y, active=active)
                records.append({"file": path.name, "mode": mode, "split": split,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest = {"protocol_sha256": "a" * 64, "seed": 369601,
                    "target_definition": "synthetic permitted target", "records": records}
        (root / "manifest.json").write_text(json.dumps(manifest))
        model = fit(root, root / "model.json")
        assert set(model["models"]) == {"servo", "torque"}
        assert all(value["threshold"] >= 3.0 for value in model["models"].values())
        assert model["candidate_episodes"] == 0 and not model["collision_results_used"]


def main():
    tests = [test_factory_reuses_r1_shared_core_and_replaces_only_ridge,
             test_replay_reads_permitted_and_writes_only_allowed_arrays,
             test_fit_exact_contract_on_synthetic_allowed_data]
    for test in tests:
        test()
    print(f"{len(tests)} matched-ridge pure tests passed")


if __name__ == "__main__":
    main()
