#!/usr/bin/env python3
"""Replay permitted normal observations through R1, then fit matched ridge.

Collection reads only the five documented runtime arrays from each source
archive and writes physically separate ``t,x,y,active`` archives.  Fitting is
an offline NumPy-only subcommand suitable for ``.venv-train``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PERMITTED_KEYS = ("obs_t", "encoder", "encoder_velocity", "jts", "grip")
OUTPUT_KEYS = ("t", "x", "y", "active")
RIDGE = 0.02


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_permitted(path):
    """Materialize only permitted members; never index other archive keys."""
    with np.load(path, allow_pickle=False) as archive:
        missing = set(PERMITTED_KEYS) - set(archive.files)
        if missing:
            raise ValueError(f"missing permitted inputs in {Path(path).name}: {sorted(missing)}")
        return {key: np.asarray(archive[key]) for key in PERMITTED_KEYS}


def _controller_core(mode, condition):
    from sim.nominal import Nominal

    controller = importlib.import_module("r1_" + mode).Controller()
    controller.nom = Nominal()
    controller.nom_box = Nominal()
    controller.nom_box.set_payload(condition["mass"], condition["com"])
    controller.box_mass = float(condition["mass"])
    controller.box_com = tuple(condition["com"])
    controller.reset()
    return controller.shared_estimator


def replay_one(mode, condition, source, destination):
    """Replay one archive without reading any posthoc/oracle member."""
    from r1_detector import outside_table_height_band
    from tools.v2_learned_core import normal_features

    arrays = _load_permitted(source)
    sizes = {len(value) for value in arrays.values()}
    if len(sizes) != 1:
        raise ValueError("permitted source arrays have inconsistent lengths")
    core = _controller_core(mode, condition)
    features, targets, active = [], [], []
    rows = zip(*(arrays[key] for key in PERMITTED_KEYS), strict=True)
    for t, encoder, velocity, jts, grip in rows:
        grip = bool(grip)
        core.update(float(t), encoder, velocity, jts, grip)
        features.append(normal_features(float(t), condition["mass"], condition["com"]))
        targets.append(core.last_wrench[:2].copy())
        active.append(outside_table_height_band(core, grip))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, t=arrays["obs_t"], x=np.asarray(features),
                        y=np.asarray(targets), active=np.asarray(active, dtype=bool))
    with np.load(destination, allow_pickle=False) as written:
        if set(written.files) != set(OUTPUT_KEYS):
            raise RuntimeError("allowed-only archive contract violated")


def _target_difference(old_path, new_path):
    with np.load(old_path, allow_pickle=False) as old, np.load(new_path, allow_pickle=False) as new:
        if set(old.files) != set(OUTPUT_KEYS) or set(new.files) != set(OUTPUT_KEYS):
            raise ValueError("comparison inputs must be allowed-only archives")
        if not np.array_equal(old["t"], new["t"]):
            raise ValueError("pRNE and R1 replay timestamps differ")
        delta = np.linalg.norm(old["y"] - new["y"], axis=1)
        active = np.asarray(new["active"], bool)
        selected = delta[active]
        return {"samples": int(len(delta)), "active_samples": int(active.sum()),
                "active_rms_N": float(np.sqrt(np.mean(selected ** 2))) if len(selected) else 0.0,
                "active_p99_N": float(np.percentile(selected, 99)) if len(selected) else 0.0,
                "maximum_N": float(delta.max(initial=0.0))}


def collect(protocol_path, source_dir, prne_allowed_dir, output_dir):
    protocol_path, source_dir = Path(protocol_path), Path(source_dir)
    prne_allowed_dir, output_dir = Path(prne_allowed_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refuse to overwrite non-empty matched-data directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(protocol_path.read_text())
    source_manifest = json.loads((prne_allowed_dir / "manifest.json").read_text())
    expected = {(mode, split, index): condition
                for mode in ("servo", "torque")
                for split in ("train", "validation")
                for index, condition in enumerate(protocol[split])}
    records, comparisons = [], []
    for row in source_manifest["records"]:
        name = row["file"]
        parts = name.removesuffix(".npz").split("_")
        mode, index = row["mode"], int(parts[-1])
        key = (mode, row["split"], index)
        if key not in expected or row["condition"] != expected[key]:
            raise ValueError(f"condition mismatch for {name}")
        source = source_dir / name
        destination = output_dir / name.replace("C5_prne_only", "C5_ridge_only_matched")
        replay_one(mode, row["condition"], source, destination)
        comparison = _target_difference(prne_allowed_dir / name, destination)
        comparisons.append(dict(file=destination.name, mode=mode, split=row["split"], **comparison))
        records.append({"file": destination.name, "mode": mode, "split": row["split"],
                        "condition": row["condition"],
                        "source": "R1 observer replay from permitted encoder/JTS inputs only",
                        "source_raw_sha256": sha256(source), "sha256": sha256(destination),
                        "active_samples": comparison["active_samples"]})
    if set(expected) != {(r["mode"], r["split"], int(r["file"].split("_")[-1][:-4]))
                        for r in records}:
        raise ValueError("source manifest does not cover the registered conditions exactly")
    manifest = {
        "protocol_sha256": sha256(protocol_path), "seed": protocol["seeds"]["training"],
        "target_definition": "R1 observer last_wrench[:2] replayed from permitted observed inputs only",
        "features_definition": "normal_features77 with registered known mass/COM",
        "permitted_source_keys": list(PERMITTED_KEYS), "output_keys": list(OUTPUT_KEYS),
        "records": records, "prne_target_difference": comparisons,
        "source_manifest_sha256": sha256(prne_allowed_dir / "manifest.json"),
        "normal_source_sha256": {
            name: sha256(ROOT / name) for name in
            ("r1_servo.py", "r1_torque.py", "r1_detector.py",
             "tools/v2_learned_core.py", "tools/r2_matched_ridge.py")},
        "official_conditions_used": False, "holdout_opened": False,
        "candidate_episodes": 0, "collision_results_used": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def standardized(x, mean, scale):
    return np.clip((x - mean) / scale, -8.0, 8.0)


def validation_score(x, y, active, mean, scale, weights):
    errors = np.linalg.norm(y - standardized(x, mean, scale) @ weights, axis=1)
    ema, scores = 0.0, []
    for value, gate in zip(errors, active, strict=True):
        ema = ema + 0.1 * (value - ema) if gate else 0.0
        scores.append(ema)
    return np.asarray(scores)


def fit(data_dir, output, seed=369601):
    data_dir, output = Path(data_dir), Path(output)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if seed != manifest["seed"]:
        raise ValueError("fit seed must equal registered training seed")
    result = {"protocol_sha256": manifest["protocol_sha256"], "seed": seed,
              "method": "train-only normalized ridge .02; disjoint no-collision validation threshold",
              "training_target": manifest["target_definition"],
              "data_sha256": sha256(manifest_path), "models": {}, "validation": {},
              "dependencies": {"numpy": np.__version__}, "official_conditions_used": False,
              "holdout_opened": False, "candidate_episodes": 0,
              "collision_results_used": False,
              "normal_source_sha256": {
                  name: sha256(ROOT / name) for name in
                  ("r1_servo.py", "r1_torque.py", "r1_detector.py",
                   "tools/v2_learned_core.py", "tools/r2_matched_ridge.py")}}
    for mode in ("servo", "torque"):
        rows = [row for row in manifest["records"] if row["mode"] == mode]
        matrices, targets = [], []
        for row in rows:
            path = data_dir / row["file"]
            if sha256(path) != row["sha256"]:
                raise ValueError("data hash mismatch: " + row["file"])
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != set(OUTPUT_KEYS):
                    raise ValueError("fit input is not allowed-only")
                if row["split"] == "train":
                    mask = np.asarray(archive["active"], bool)
                    matrices.append(archive["x"][mask]); targets.append(archive["y"][mask])
        x, y = np.concatenate(matrices), np.concatenate(targets)
        mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
        mean[0], scale[0] = 0.0, 1.0
        z = standardized(x, mean, scale)
        penalty = RIDGE * np.eye(z.shape[1]); penalty[0, 0] = 0.0
        weights = np.linalg.solve(z.T @ z + penalty, z.T @ y)
        episodes = []
        for row in rows:
            if row["split"] != "validation":
                continue
            with np.load(data_dir / row["file"], allow_pickle=False) as archive:
                scores = validation_score(archive["x"], archive["y"], archive["active"],
                                          mean, scale, weights)
            episodes.append({"file": row["file"], "maximum_score": float(scores.max())})
        maximum = max(row["maximum_score"] for row in episodes)
        threshold = max(3.0, 1.2 * maximum + 0.5)
        result["models"][mode] = {"mean": mean.tolist(), "scale": scale.tolist(),
                                  "weights": weights.tolist(), "threshold": threshold}
        result["validation"][mode] = {"episodes": episodes, "threshold": threshold,
                                      "train_samples": len(x),
                                      "train_rmse_N": float(np.sqrt(np.mean((z @ weights - y) ** 2)))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("collect")
    replay.add_argument("--protocol", type=Path, required=True)
    replay.add_argument("--source-dir", type=Path, required=True)
    replay.add_argument("--prne-allowed-dir", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    trainer = sub.add_parser("fit")
    trainer.add_argument("--data-dir", type=Path, required=True)
    trainer.add_argument("--out", type=Path, required=True)
    trainer.add_argument("--seed", type=int, default=369601)
    args = parser.parse_args()
    if args.command == "collect":
        result = collect(args.protocol, args.source_dir, args.prne_allowed_dir, args.output_dir)
        print(json.dumps({"records": len(result["records"]), "candidate_episodes": 0}))
    else:
        result = fit(args.data_dir, args.out, args.seed)
        print(json.dumps(result["validation"], indent=2))


if __name__ == "__main__":
    main()
