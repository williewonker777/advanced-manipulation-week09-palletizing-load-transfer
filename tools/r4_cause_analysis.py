"""Offline R1 D4 miss analysis using only cached, permitted detector signals."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = ROOT / "results" / "v2_protocol"
sys.path.insert(0, str(ROOT))

import r1_detector  # noqa: E402


SCORE_NAMES = (
    "xy_raw_active",
    "xy_ema01_active",
    "xy_mean40_active",
    "joint_l2",
    "fz_positive",
    "fz_abs",
    "fz_hp40",
)
RAW_KEYS = (
    "obs_t",
    "estimated_wrench",
    "joint_residual",
    "active",
    "detect",
)


def rolling_mean(values: np.ndarray, width: int = 40) -> np.ndarray:
    """Causal trailing mean, including the current sample."""
    values = np.asarray(values, dtype=float)
    csum = np.r_[0.0, np.cumsum(values, dtype=float)]
    end = np.arange(values.size) + 1
    start = np.maximum(end - width, 0)
    return (csum[end] - csum[start]) / np.minimum(end, width)


def active_rolling_mean(values: np.ndarray, active: np.ndarray,
                        width: int = 40) -> np.ndarray:
    """Trailing mean over the current contiguous active segment."""
    values = np.asarray(values, dtype=float)
    active = np.asarray(active, dtype=bool)
    result = np.zeros(values.size, dtype=float)
    buffer = np.empty(width, dtype=float)
    count = 0
    total = 0.0
    cursor = 0
    for index, (value, enabled) in enumerate(zip(values, active)):
        if not enabled:
            count = 0
            total = 0.0
            cursor = 0
            continue
        if count < width:
            buffer[count] = value
            count += 1
            total += value
        else:
            total += value - buffer[cursor]
            buffer[cursor] = value
            cursor = (cursor + 1) % width
        result[index] = total / count
    return result


def active_ema(values: np.ndarray, active: np.ndarray,
               alpha: float = 0.1) -> np.ndarray:
    """EMA that resets to zero whenever the learned gate is inactive."""
    result = np.zeros(len(values), dtype=float)
    state = 0.0
    for index, (value, enabled) in enumerate(zip(values, active)):
        if enabled:
            state += alpha * (float(value) - state)
            result[index] = state
        else:
            state = 0.0
    return result


def consecutive_crossing(values: np.ndarray, boundary: np.ndarray | float,
                         length: int = 4) -> np.ndarray:
    """True at samples completing a strict consecutive crossing run."""
    over = np.asarray(values) > boundary
    if length == 1:
        return over
    count = 0
    result = np.zeros(over.size, dtype=bool)
    for index, value in enumerate(over):
        count = count + 1 if value else 0
        result[index] = count >= length
    return result


def collision_is_missed(row: dict) -> bool:
    """Cross-check the public label against the evaluator fields."""
    evaluated = row.get("detect_delay_ms") is None and not bool(
        row.get("alarm_before_collision", False)
    )
    if "collision_outcome" in row:
        if (row["collision_outcome"] == "missed") != evaluated:
            raise AssertionError("collision outcome conflicts with evaluator fields")
    return evaluated


def empirical_roc(normal_episode_scores: np.ndarray,
                  collision_episode_scores: np.ndarray) -> dict:
    """Episode-level empirical ROC under a strict ``score > threshold`` rule."""
    normal = np.asarray(normal_episode_scores, dtype=float)
    collision = np.asarray(collision_episode_scores, dtype=float)
    if normal.ndim != 1 or collision.ndim != 1 or not normal.size or not collision.size:
        raise ValueError("ROC inputs must be non-empty one-dimensional arrays")
    values = np.unique(np.r_[normal, collision])[::-1]
    thresholds = np.r_[np.inf, np.nextafter(values, -np.inf)]
    false_positive_rate = np.asarray(
        [np.mean(normal > threshold) for threshold in thresholds], dtype=float
    )
    true_positive_rate = np.asarray(
        [np.mean(collision > threshold) for threshold in thresholds], dtype=float
    )
    auc = float(np.trapezoid(true_positive_rate, false_positive_rate))
    return {
        "normal_episodes": int(normal.size),
        "collision_episodes": int(collision.size),
        "threshold": [None if np.isinf(x) else float(x) for x in thresholds],
        "false_positive_rate": false_positive_rate.tolist(),
        "true_positive_rate": true_positive_rate.tolist(),
        "auc": auc,
    }


def score_signals(raw: dict[str, np.ndarray],
                  predicted_xy: np.ndarray) -> dict[str, np.ndarray]:
    """Build the seven predeclared diagnostic score families."""
    active = np.asarray(raw["active"], dtype=bool)
    wrench = np.asarray(raw["estimated_wrench"], dtype=float)
    xy = np.linalg.norm(wrench[:, :2] - predicted_xy, axis=1)
    fz = wrench[:, 2]
    return {
        "xy_raw_active": np.where(active, xy, 0.0),
        "xy_ema01_active": active_ema(xy, active),
        "xy_mean40_active": active_rolling_mean(xy, active),
        "joint_l2": np.linalg.norm(raw["joint_residual"], axis=1),
        "fz_positive": np.maximum(fz, 0.0),
        "fz_abs": np.abs(fz),
        "fz_hp40": np.abs(fz - rolling_mean(fz)),
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_raw(row: dict, raw_dir: Path,
              manifest: list[str]) -> dict[str, np.ndarray]:
    path = raw_dir / row["private_raw_basename"]
    digest = _hash(path)
    if digest != row["private_raw_sha256"]:
        raise RuntimeError(f"raw trace hash mismatch: {path.name}")
    manifest.append(f"{path.name}:{digest}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in RAW_KEYS}


def _prediction(t: np.ndarray, mass: float, com: list[float], mode: str,
                cache: dict[tuple, np.ndarray]) -> np.ndarray:
    key = (mode, float(mass), tuple(float(x) for x in com))
    if key not in cache:
        parameters = r1_detector.MODEL_PARAMETERS[mode]
        mean = np.asarray(parameters["mean"], dtype=float)
        scale = np.asarray(parameters["scale"], dtype=float)
        weights = np.asarray(parameters["weights"], dtype=float)
        features = np.vstack([r1_detector.normal_features(x, mass, com) for x in t])
        cache[key] = np.clip((features - mean) / scale, -8.0, 8.0) @ weights
    return cache[key]


def _counts(rows: list[dict], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def _window(t: np.ndarray, start: float, duration: float) -> np.ndarray:
    return (t >= start) & (t < start + duration)


def _first_true_delay_ms(flag: np.ndarray, t: np.ndarray, start: float,
                         duration: float) -> float | None:
    indices = np.flatnonzero(flag & _window(t, start, duration))
    return None if not indices.size else float((t[indices[0]] - start) * 1000.0)


def _analyze_mode(mode: str, public_path: Path, raw_dir: Path) -> dict:
    public = json.loads(public_path.read_text())
    rows = public["results"]
    groups = {kind: [row for row in rows if row["kind"] == kind]
              for kind in ("reference", "collision", "offset")}
    expected = {"reference": 189, "collision": 180, "offset": 16}
    actual = {kind: len(group) for kind, group in groups.items()}
    if actual != expected:
        raise AssertionError(f"unexpected D4 composition for {mode}: {actual}")

    misses = [row for row in groups["collision"] if collision_is_missed(row)]
    if len(misses) != 69:
        raise AssertionError(f"unexpected miss count for {mode}: {len(misses)}")

    manifest: list[str] = []
    prediction_cache: dict[tuple, np.ndarray] = {}
    time_grid = None
    maxima = {name: -np.inf for name in SCORE_NAMES}
    envelopes: dict[str, np.ndarray] = {}
    reference_episode_maxima = {name: [] for name in SCORE_NAMES}
    collision_episode_maxima = {name: [] for name in SCORE_NAMES}

    # The reference-only pass defines diagnostics before collision inspection.
    for row in groups["reference"]:
        raw = _read_raw(row, raw_dir, manifest)
        t = np.asarray(raw["obs_t"], dtype=float)
        if time_grid is None:
            time_grid = t.copy()
            if not np.allclose(np.diff(time_grid), 0.001, rtol=0.0, atol=1e-12):
                raise AssertionError("cached observations are not on the 1 ms grid")
        elif not np.array_equal(t, time_grid):
            raise AssertionError("reference time grids differ")
        scores = score_signals(
            raw, _prediction(t, row["mass"], row["com"], mode, prediction_cache)
        )
        for name, values in scores.items():
            maxima[name] = max(maxima[name], float(np.max(values)))
            reference_episode_maxima[name].append(float(np.max(values)))
            if name not in envelopes:
                envelopes[name] = values.copy()
            else:
                np.maximum(envelopes[name], values, out=envelopes[name])

    thresholds = {name: float(np.nextafter(value, np.inf))
                  for name, value in maxima.items()}
    family = {name: {"all_collision_any": 0, "miss_extra_any": 0,
                     "all_collision_four": 0, "miss_extra_four": 0}
              for name in SCORE_NAMES}
    union_global_any: set[int] = set()
    union_global_four: set[int] = set()
    union_time_any: set[int] = set()
    union_time_four: set[int] = set()
    all_below_global: set[int] = set()
    mean40_extra: list[dict] = []
    raw_delay_mismatches: list[int] = []
    learned_onset_violations = 0
    pulse_gate_off = 0
    place_eval_gate_off = 0
    pick_first_active_ms: list[float] = []
    pick_gate = Counter()

    for row in groups["collision"]:
        raw = _read_raw(row, raw_dir, manifest)
        t = np.asarray(raw["obs_t"], dtype=float)
        if not np.array_equal(t, time_grid):
            raise AssertionError("collision time grid differs from references")
        start = float(row["collision"][0])
        evaluation = _window(t, start, 0.32)
        pulse = _window(t, start, 0.12)
        active = np.asarray(raw["active"], dtype=bool)
        detected = np.asarray(raw["detect"], dtype=bool)
        scores = score_signals(
            raw, _prediction(t, row["mass"], row["com"], mode, prediction_cache)
        )
        missed = collision_is_missed(row)
        case_id = int(row["case_id"])

        rises = detected & ~np.r_[False, detected[:-1]]
        replay_delay = _first_true_delay_ms(rises, t, start, 0.32)
        public_delay = row["detect_delay_ms"]
        equal_delay = ((replay_delay is None and public_delay is None) or
                       (replay_delay is not None and public_delay is not None and
                        abs(replay_delay - float(public_delay)) <= 1e-6))
        if not equal_delay:
            raw_delay_mismatches.append(case_id)

        if not np.any(active[pulse]):
            pulse_gate_off += int(missed)
        if row["phase"] == "place" and not np.any(active[evaluation]):
            place_eval_gate_off += int(missed)
        if row["phase"] == "pick" and missed:
            active_count = int(np.count_nonzero(active[pulse]))
            pick_gate["off"] += active_count == 0
            pick_gate["full"] += active_count == int(np.count_nonzero(pulse))
            pick_gate["partial"] += 0 < active_count < int(np.count_nonzero(pulse))
            delay = _first_true_delay_ms(active, t, start, 0.32)
            if delay is not None:
                pick_first_active_ms.append(delay)

        model_threshold = float(r1_detector.MODEL_PARAMETERS[mode]["threshold"])
        learned_onset = consecutive_crossing(
            scores["xy_ema01_active"], model_threshold, 4
        )
        learned_onset_violations += int(np.count_nonzero(learned_onset & ~detected))

        crossed_any = set()
        crossed_four = set()
        time_any = set()
        time_four = set()
        for name, values in scores.items():
            collision_episode_maxima[name].append(float(np.max(values[evaluation])))
            one = bool(np.any((values > thresholds[name]) & evaluation))
            four = bool(np.any(consecutive_crossing(values, thresholds[name], 4)
                               & evaluation))
            tone = bool(np.any((values > envelopes[name]) & evaluation))
            tfour = bool(np.any(consecutive_crossing(values, envelopes[name], 4)
                                & evaluation))
            family[name]["all_collision_any"] += int(one)
            family[name]["all_collision_four"] += int(four)
            if one:
                crossed_any.add(name)
            if four:
                crossed_four.add(name)
            if tone:
                time_any.add(name)
            if tfour:
                time_four.add(name)
            if missed:
                family[name]["miss_extra_any"] += int(one)
                family[name]["miss_extra_four"] += int(four)

        if missed:
            if crossed_any:
                union_global_any.add(case_id)
            if crossed_four:
                union_global_four.add(case_id)
            if time_any:
                union_time_any.add(case_id)
            if time_four:
                union_time_four.add(case_id)
            if not crossed_any:
                all_below_global.add(case_id)
            if "xy_mean40_active" in crossed_any:
                flags = (scores["xy_mean40_active"] >
                         thresholds["xy_mean40_active"]) & evaluation
                first = np.flatnonzero(flags)
                first_ms = None if not first.size else float(
                    (t[first[0]] - start) * 1000.0
                )
                mean40_extra.append({"phase": row["phase"],
                                     "force_n": row["force_n"],
                                     "direction": row["direction"],
                                     "first_cross_ms": first_ms})

    # Offsets are included in the provenance check even though no score uses them.
    for row in groups["offset"]:
        raw = _read_raw(row, raw_dir, manifest)
        if not np.array_equal(raw["obs_t"], time_grid):
            raise AssertionError("offset time grid differs from references")

    if len(manifest) != 385 or len(set(manifest)) != 385:
        raise AssertionError("raw provenance is incomplete or duplicated")
    manifest_sha = hashlib.sha256("\n".join(sorted(manifest)).encode()).hexdigest()
    quantiles = ([] if not pick_first_active_ms else
                 [float(x) for x in np.quantile(pick_first_active_ms,
                                                [0, .25, .5, .75, 1])])
    roc = {}
    for name in SCORE_NAMES:
        curve = empirical_roc(reference_episode_maxima[name],
                              collision_episode_maxima[name])
        curve["observed_zero_fpr_point"] = {
            "threshold": thresholds[name],
            "false_positive_rate": 0.0,
            "true_positive_rate": family[name]["all_collision_any"] / 180.0,
            "collision_episodes_above": family[name]["all_collision_any"],
        }
        roc[name] = curve
    return {
        "architecture": mode,
        "case_counts": actual,
        "miss_count": len(misses),
        "miss_strata": {
            "mass_kg": _counts(misses, "mass"),
            "phase": _counts(misses, "phase"),
            "force_n": _counts(misses, "force_n"),
            "direction": _counts(misses, "direction"),
            "vertical_pair_phase": _counts(
                [row for row in misses if row["direction"] in ("+z", "-z")],
                "phase",
            ),
        },
        "learned_gate": {
            "misses_gate_off_entire_physical_pulse": pulse_gate_off,
            "place_misses_gate_off_entire_evaluator_window": place_eval_gate_off,
            "pick_miss_pulse_gate": dict(pick_gate),
            "pick_first_active_delay_ms_quantiles": quantiles,
            "pick_first_active_at_or_after_120ms": sum(
                value >= 120.0 for value in pick_first_active_ms
            ),
        },
        "diagnostic_global_zero_reference_false_alarm_thresholds": thresholds,
        "episode_empirical_roc": roc,
        "global_headroom": family,
        "miss_union": {
            "global_any": len(union_global_any),
            "global_four_consecutive": len(union_global_four),
            "aligned_time_envelope_any": len(union_time_any),
            "aligned_time_envelope_four_consecutive": len(union_time_four),
            "all_families_below_global_reference_maximum": len(all_below_global),
        },
        "xy_mean40_global_extra_miss_strata": {
            "count": len(mean40_extra),
            "four_consecutive_count": family["xy_mean40_active"]["miss_extra_four"],
            "phase": _counts(mean40_extra, "phase"),
            "force_n": _counts(mean40_extra, "force_n"),
            "direction": _counts(mean40_extra, "direction"),
            "first_cross_after_physical_pulse": sum(
                row["first_cross_ms"] is not None and row["first_cross_ms"] >= 120.0
                for row in mean40_extra
            ),
        },
        "replay_checks": {
            "public_raw_detect_rise_delay_mismatch_count": len(raw_delay_mismatches),
            "public_raw_detect_rise_delay_mismatch_case_ids": raw_delay_mismatches,
            "learned_ema_four_sample_onset_not_implying_raw_detect":
                learned_onset_violations,
            "note": "EMA 4-sample onset implication only; the 8-sample clear latch waveform was not replayed.",
        },
        "raw_provenance": {
            "verified_files": len(manifest),
            "manifest_sha256": manifest_sha,
            "public_index_sha256": _hash(public_path),
        },
    }


def analyze(raw_dir: Path, public_dir: Path = PUBLIC_DIR) -> dict:
    modes = {}
    for mode in ("servo", "torque"):
        path = public_dir / f"R3_D4_R1_{mode}.json"
        modes[mode] = _analyze_mode(mode, path, raw_dir)
    verified = sum(value["raw_provenance"]["verified_files"]
                   for value in modes.values())
    if verified != 770:
        raise AssertionError(f"expected 770 verified raw files, got {verified}")
    return {
        "schema": "R4_CAUSE_ANALYSIS.v1",
        "scope": "cached R1 D4 development evidence only",
        "official_or_holdout_used": False,
        "score_families": list(SCORE_NAMES),
        "windows": {"physical_pulse_s": 0.12,
                    "evaluator_half_open_s": 0.32,
                    "trailing_mean_samples": 40,
                    "sample_period_s": 0.001},
        "assumptions_and_limits": [
            "Global thresholds are nextafter(reference maximum) diagnostics, not operational thresholds.",
            "The aligned per-millisecond reference envelope reuses the same data and is an optimistic descriptive upper bound.",
            "No result supplies a global separability or safety guarantee.",
            "The learned replay checks EMA plus four-sample onset implication only, not the eight-sample clear latch waveform.",
            "All cached cases and outliers are retained; no official or holdout result is read.",
            "Only permitted cached signals and scores derived from them are used.",
        ],
        "analysis_sources_sha256": {
            "tools/r4_cause_analysis.py": _hash(ROOT / "tools" / "r4_cause_analysis.py"),
            "r1_detector.py": _hash(ROOT / "r1_detector.py"),
            "R3_D4_R1_servo.json": modes["servo"]["raw_provenance"]["public_index_sha256"],
            "R3_D4_R1_torque.json": modes["torque"]["raw_provenance"]["public_index_sha256"],
        },
        "raw_provenance": {
            "verified_files": verified,
            "combined_manifest_sha256": hashlib.sha256(
                "\n".join(modes[mode]["raw_provenance"]["manifest_sha256"]
                          for mode in ("servo", "torque")).encode()
            ).hexdigest(),
        },
        "modes": modes,
    }


def write_roc_plot(result: dict, path: Path) -> None:
    """Render the episode-level diagnostic ROC without exposing raw arrays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "xy_raw_active": "XY raw (active)",
        "xy_ema01_active": "XY EMA 0.1 (active)",
        "xy_mean40_active": "XY mean 40 ms (active)",
        "joint_l2": "Joint residual L2",
        "fz_positive": "positive Fz",
        "fz_abs": "absolute Fz",
        "fz_hp40": "Fz high-pass 40 ms",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharex=True, sharey=True)
    for axis, mode in zip(axes, ("servo", "torque")):
        for name in SCORE_NAMES:
            curve = result["modes"][mode]["episode_empirical_roc"][name]
            axis.plot(curve["false_positive_rate"], curve["true_positive_rate"],
                      linewidth=1.5, label=f"{labels[name]} (AUC {curve['auc']:.3f})")
        axis.plot([0, 1], [0, 1], color="0.65", linestyle="--", linewidth=1)
        axis.set_title("Servo R1" if mode == "servo" else "Torque R1")
        axis.set_xlabel("Observed false-positive rate (normal episodes)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("True-positive rate (collision episodes)")
    axes[1].legend(fontsize=7, loc="lower right")
    figure.suptitle("Cached R1 D4 episode-level empirical ROC (same-data diagnostic)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="directory containing cached R1 D4 NPZ traces")
    parser.add_argument("--out", type=Path, required=True,
                        help="JSON output path")
    parser.add_argument("--plot", type=Path,
                        default=ROOT / "results" / "figures" / "r4_diagnostic_roc.png",
                        help="episode-level diagnostic ROC figure path")
    args = parser.parse_args()
    result = analyze(args.raw_dir)
    write_roc_plot(result, args.plot)
    result["analysis_sources_sha256"]["results/figures/r4_diagnostic_roc.png"] = _hash(
        args.plot
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for mode, values in result["modes"].items():
        print(mode, json.dumps({
            "misses": values["miss_count"],
            "mean40_extra": values["xy_mean40_global_extra_miss_strata"]["count"],
            "union": values["miss_union"],
            "raw_verified": values["raw_provenance"]["verified_files"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
