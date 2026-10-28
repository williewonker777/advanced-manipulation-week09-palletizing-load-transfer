#!/usr/bin/env python3
"""Collision-blind calibration of symmetric main and grip-off fast channels."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.dont_write_bytecode = True

from check_integrity import check_integrity

check_integrity(ROOT)

import numpy as np

from experiments import d3_detector_fast
import servo_controller
import torque_controller
from sim.nominal import Nominal
from sim.runner import run_episode


PROTOCOL_PATH = ROOT / "results" / "D3_FAST_PROTOCOL.json"
OUTPUT_PATH = ROOT / "results" / "D3_FAST_CALIBRATION.json"
EXPECTED_PROTOCOL_SHA256 = "fcbee8133ebd6f353cbf89520bf981077c001ed010ab1350af5835726cf3a0c5"
EXPECTED_CONTROLLER_SHA256 = {
    "servo_controller.py": "50b5d2b05a45d540615a4af4373780d86872c60c8049c8af75dd6a4c4836e416",
    "torque_controller.py": "a9d13fcddf89d9ab9bb92346bafe6d2d18b76cd5d605710cbdf3d9dddb8619d9",
}
COM_GRID = list(
    itertools.product((-0.08, 0.0, 0.08), (-0.05, 0.0, 0.05), (-0.04, 0.0, 0.04))
)
CONTROLLERS = {
    "servo_controller": servo_controller.Controller,
    "torque_controller": torque_controller.Controller,
}


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_provenance():
    protocol_sha = file_sha256(PROTOCOL_PATH)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(f"protocol hash changed: {protocol_sha}")
    for name, expected in EXPECTED_CONTROLLER_SHA256.items():
        actual = file_sha256(ROOT / name)
        if actual != expected:
            raise ValueError(f"controller hash changed: {name} {actual}")
    return protocol_sha


def cases():
    train_rng = np.random.default_rng(9363)
    order = train_rng.permutation(len(COM_GRID))
    masses = np.r_[0.0, 8.0, train_rng.uniform(0.0, 8.0, 10)]
    train = [(float(m), tuple(COM_GRID[int(i)])) for m, i in zip(masses, order[:12], strict=True)]

    hold_rng = np.random.default_rng(10363)
    remaining = np.asarray(order[12:], int)
    hold_masses = hold_rng.uniform(0.0, 8.0, 6)
    hold_indices = hold_rng.permutation(remaining)[:6]
    holdout = [
        (float(m), tuple(COM_GRID[int(i)]))
        for m, i in zip(hold_masses, hold_indices, strict=True)
    ]
    return train, holdout


def feature_groups(detector):
    residual = detector.last_residual_wrench
    raw = detector.last_wrench
    innovation = detector.last_fast_innovation
    main = np.array(
        [
            max(abs(residual[0]), abs(residual[1])),
            max(-raw[2], 0.0),
            max(abs(residual[3]), abs(residual[4])),
            abs(residual[5]),
            0.0 if detector.last_table_allowed else max(raw[2], 0.0),
        ],
        dtype=float,
    )
    fast = np.array(
        [
            max(abs(innovation[0]), abs(innovation[1])),
            max(-innovation[2], 0.0),
            max(abs(innovation[3]), abs(innovation[4])),
            abs(innovation[5]),
            max(innovation[2], 0.0),
        ],
        dtype=float,
    )
    if detector.last_grip:
        fast[:] = 0.0
    return main, fast


class ColdRecordingDetector(d3_detector_fast.Detector):
    def reset(self):
        super().reset()
        self.main_groups = []
        self.fast_groups = []

    def _directional_alarm(self, residual, raw_wrench, table_allowed):
        del residual, raw_wrench, table_allowed
        return False, 0.0

    def _fast_directional_alarm(self, innovation):
        del innovation
        return False, 0.0

    def update(self, *args):
        result = super().update(*args)
        main, fast = feature_groups(self)
        self.main_groups.append(main)
        self.fast_groups.append(fast)
        return result


class ValidationDetector(d3_detector_fast.Detector):
    def reset(self):
        super().reset()
        self.main_groups = []
        self.fast_groups = []

    def update(self, *args):
        result = super().update(*args)
        main, fast = feature_groups(self)
        self.main_groups.append(main)
        self.fast_groups.append(fast)
        return result


def window_peak_min(values, persistence):
    values = np.asarray(values, float)
    if len(values) < persistence:
        return 0.0
    windows = np.lib.stride_tricks.sliding_window_view(values, persistence)
    return float(np.max(np.min(windows, axis=-1)))


def collect(case_list, detector_type, split, evidence):
    available_kib = int(
        next(
            line.split()[1]
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
    )
    if available_kib * 1024 < 6 * 1024**3:
        raise RuntimeError("No-collision calibration requires at least 6 GiB MemAvailable")
    rows = []
    traces = {}
    for controller_name, controller_type in CONTROLLERS.items():
        for index, (mass, com) in enumerate(case_list):
            detector = detector_type()
            result, _ = run_episode(
                controller_type(), box_mass=mass, box_com=com, detector=detector, collision=None
            )
            main = np.asarray(detector.main_groups)
            fast = np.asarray(detector.fast_groups)
            key = f"{split}_{controller_name}_{index:02d}"
            traces[key] = {"main": main, "fast": fast}
            rows.append(
                {
                    "key": key,
                    "split": split,
                    "controller": controller_name,
                    "mass_kg": mass,
                    "com_m": list(com),
                    "false_alarms": int(result["false_alarms"]),
                    "main_group_peaks": main.max(axis=0).tolist(),
                    "fast_group_peaks": fast.max(axis=0).tolist(),
                }
            )
    evidence.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key, channel_traces in traces.items():
        arrays[f"{key}_main"] = channel_traces["main"]
        arrays[f"{key}_fast"] = channel_traces["fast"]
    np.savez_compressed(evidence / f"fast_{split}_features.npz", **arrays)
    return rows, traces


def select_channel(traces, channel, persistence_candidates):
    all_values = np.concatenate([trace[channel] for trace in traces.values()], axis=0)
    median = np.median(all_values, axis=0)
    mad = np.median(np.abs(all_values - median), axis=0)
    margins = np.array([0.5, 0.5, 0.05, 0.05, 0.5])
    noise_scale = np.maximum(1.4826 * mad, margins)
    candidates = []
    for persistence in persistence_candidates:
        base = np.zeros(5)
        for trace in traces.values():
            values = trace[channel]
            for group in range(5):
                base[group] = max(base[group], window_peak_min(values[:, group], persistence))
        thresholds = 1.25 * base + margins
        q = 0.0
        for trace in traces.values():
            score = np.max(trace[channel] / thresholds, axis=1)
            q = max(q, window_peak_min(score, persistence))
        guard_scale = max(1.0, 1.25 * q)
        thresholds *= guard_scale
        cost = float(np.mean(np.log1p(thresholds / noise_scale)) + persistence / 20.0)
        candidates.append(
            {
                "persistence_samples": int(persistence),
                "persistent_base": base.tolist(),
                "unit_margins": margins.tolist(),
                "robust_noise_scale": noise_scale.tolist(),
                "feature_switch_q": q,
                "feature_switch_scale": guard_scale,
                "threshold_groups": thresholds.tolist(),
                "cost": cost,
            }
        )
    chosen = min(candidates, key=lambda item: (item["cost"], item["persistence_samples"]))
    return candidates, chosen


def installed_parameters():
    detector = d3_detector_fast.Detector()
    detector.nom = Nominal()
    detector.nom_box = Nominal()
    detector.box_mass = 1.0
    detector.box_com = (0.0, 0.0, 0.0)
    detector.reset()
    if detector.force_threshold[0] != detector.force_threshold[1]:
        raise ValueError("main T must enforce Fx == Fy")
    if detector.moment_threshold[0] != detector.moment_threshold[1]:
        raise ValueError("main T must enforce Mx == My")
    if detector.fast_force_threshold[0] != detector.fast_force_threshold[1]:
        raise ValueError("fast T must enforce Fx == Fy")
    if detector.fast_moment_threshold[0] != detector.fast_moment_threshold[1]:
        raise ValueError("fast T must enforce Mx == My")
    return {
        "main": {
            "threshold_groups": [
                float(detector.force_threshold[0]),
                float(detector.force_threshold[2]),
                float(detector.moment_threshold[0]),
                float(detector.moment_threshold[2]),
                float(detector.out_of_height_fz_threshold),
            ],
            "persistence_samples": int(detector.persistence),
        },
        "fast": {
            "threshold_groups": [
                float(detector.fast_force_threshold[0]),
                float(detector.fast_force_threshold[2]),
                float(detector.fast_moment_threshold[0]),
                float(detector.fast_moment_threshold[2]),
                float(detector.fast_positive_fz_threshold),
            ],
            "persistence_samples": int(detector.fast_persistence),
        },
    }


def verify_installed(output):
    installed = installed_parameters()
    for channel in ("main", "fast"):
        selected = output["selection"][channel]["selected"]
        actual = installed[channel]
        if actual["persistence_samples"] != selected["persistence_samples"] or not np.allclose(
            actual["threshold_groups"], selected["threshold_groups"], rtol=0.0, atol=1e-10
        ):
            raise ValueError(f"installed {channel} parameters do not equal selection")
    return installed


def score_margin(traces, channel, thresholds, persistence):
    q = 0.0
    for trace in traces.values():
        score = np.max(trace[channel] / thresholds, axis=1)
        q = max(q, window_peak_min(score, persistence))
    return {"max_persistent_score": q, "dimensionless_score_margin": 1.0 - q}


def phase_collect(evidence):
    protocol_sha = verify_provenance()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    train, holdout = cases()
    rows, traces = collect(train, ColdRecordingDetector, "training_cold", evidence)
    main_candidates, main_selected = select_channel(
        traces, "main", protocol["selection"]["main_persistence_candidates"]
    )
    fast_candidates, fast_selected = select_channel(
        traces, "fast", protocol["selection"]["fast_persistence_candidates"]
    )
    output = {
        "protocol_sha256": protocol_sha,
        "collision_cases_used": 0,
        "holdout_used_for_selection": False,
        "candidate_source_sha256": file_sha256(ROOT / "experiments" / "d3_detector_fast.py"),
        "training_cases": [{"mass_kg": m, "com_m": list(c)} for m, c in train],
        "holdout_cases_reserved": [{"mass_kg": m, "com_m": list(c)} for m, c in holdout],
        "cold_training_rows": rows,
        "selection": {
            "main": {"candidates": main_candidates, "selected": main_selected},
            "fast": {"candidates": fast_candidates, "selected": fast_selected},
            "joint_cost": main_selected["cost"] + fast_selected["cost"],
        },
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


def phase_validate_training(evidence):
    protocol_sha = verify_provenance()
    output = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    if output["protocol_sha256"] != protocol_sha:
        raise ValueError("calibration record protocol hash mismatch")
    installed = verify_installed(output)
    train, _ = cases()
    rows, traces = collect(train, ValidationDetector, "training_replay", evidence)
    source_sha = file_sha256(ROOT / "experiments" / "d3_detector_fast.py")
    output["installed"] = installed
    output["training_replay"] = {
        "source_sha256": source_sha,
        "rows": rows,
        "false_alarms": sum(row["false_alarms"] for row in rows),
        "main": score_margin(
            traces,
            "main",
            np.asarray(installed["main"]["threshold_groups"]),
            installed["main"]["persistence_samples"],
        ),
        "fast": score_margin(
            traces,
            "fast",
            np.asarray(installed["fast"]["threshold_groups"]),
            installed["fast"]["persistence_samples"],
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


def phase_validate_holdout(evidence):
    protocol_sha = verify_provenance()
    output = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    if output["protocol_sha256"] != protocol_sha:
        raise ValueError("calibration record protocol hash mismatch")
    if output.get("training_replay", {}).get("false_alarms") != 0:
        raise ValueError("training replay has false alarms; holdout must remain untouched")
    installed = verify_installed(output)
    source_sha = file_sha256(ROOT / "experiments" / "d3_detector_fast.py")
    if output["training_replay"].get("source_sha256") != source_sha:
        raise ValueError("training and holdout source hashes differ")
    _, holdout = cases()
    rows, traces = collect(holdout, ValidationDetector, "holdout", evidence)
    output["holdout_validation"] = {
        "source_sha256": source_sha,
        "rows": rows,
        "false_alarms": sum(row["false_alarms"] for row in rows),
        "main": score_margin(
            traces,
            "main",
            np.asarray(installed["main"]["threshold_groups"]),
            installed["main"]["persistence_samples"],
        ),
        "fast": score_margin(
            traces,
            "fast",
            np.asarray(installed["fast"]["threshold_groups"]),
            installed["fast"]["persistence_samples"],
        ),
        "parameters_changed_from_holdout": False,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("collect", "validate-training", "validate-holdout"), required=True
    )
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "collect":
        phase_collect(args.evidence_dir)
    elif args.phase == "validate-training":
        phase_validate_training(args.evidence_dir)
    else:
        phase_validate_holdout(args.evidence_dir)


if __name__ == "__main__":
    main()
