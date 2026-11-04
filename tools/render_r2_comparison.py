#!/usr/bin/env python3
"""Render reproducible four-way baseline/v1/R1/R2 comparison media.

The two 3.7 kg examples are nonofficial illustrations, not model-selection
evidence.  Episodes are collected serially through the public runner.  Replay,
normalised table support and residual-derived wrench are offline displays only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import trajectory as T  # noqa: E402
from sim.nominal import Nominal  # noqa: E402
from sim.runner import run_episode  # noqa: E402
from tools.check_integrity import check_integrity  # noqa: E402
from tools.render_comparison import (  # noqa: E402
    COM,
    FONT_BOLD,
    FONT_REGULAR,
    HEIGHT,
    VIEW_HEIGHT,
    WIDTH,
    encode_media,
    frame_times,
    gl_identity,
    media_record,
    phase,
    require_software_environment,
    residual_to_wrench,
    run_summary,
    timewarp_segments,
)
from tools.sweep import classify_collision  # noqa: E402
from tools.v2_resources import wait_for_capacity  # noqa: E402


VARIANT_ORDER = ("baseline", "v1", "r1", "r2")
COLORS = {"baseline": "#697386", "v1": "#009e73", "r1": "#0072b2", "r2": "#cc79a7"}
LABELS = {"baseline": "기준", "v1": "v1", "r1": "R1", "r2": "R2"}
SCENARIOS = (
    {
        "stem": "07_r2_servo_comparison",
        "title": "서보 A · 3.7 kg 비공식 예시 · 운반 +y 15 N",
        "mode": "servo", "mass": 3.7, "collision": (3.0, (0.0, 15.0, 0.0)),
    },
    {
        "stem": "08_r2_torque_comparison",
        "title": "토크 B · 3.7 kg 비공식 예시 · 운반 -x 15 N",
        "mode": "torque", "mass": 3.7, "collision": (3.0, (-15.0, 0.0, 0.0)),
    },
)


def combinations(mode: str) -> dict[str, tuple[str, str]]:
    """Return the frozen public modules used in the four displayed columns."""
    return {
        "baseline": (f"{mode}_controller", "collision_detector"),
        "v1": (f"v1_{mode}", "v1_detector"),
        "r1": (f"r1_{mode}", "r1_detector"),
        "r2": (f"my_{mode}", "my_detector"),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    files = {
        "servo_controller.py", "torque_controller.py", "collision_detector.py",
        "v1_servo.py", "v1_torque.py", "v1_detector.py",
        "r1_servo.py", "r1_torque.py", "r1_detector.py",
        "my_servo.py", "my_torque.py", "my_detector.py",
        "sim/runner.py", "tools/render_r2_comparison.py",
    }
    return {name: sha256(ROOT / name) for name in sorted(files)}


def identical_source_aliases(hashes: dict[str, str]) -> dict[str, str]:
    """Declare only byte-identical R2 sources as aliases of frozen R1."""
    pairs = (("my_servo.py", "r1_servo.py"), ("my_torque.py", "r1_torque.py"),
             ("my_detector.py", "r1_detector.py"))
    return {r2: r1 for r2, r1 in pairs if hashes.get(r2) == hashes.get(r1)}


def collect_variant(controller_name: str, detector_name: str, mass: float, collision):
    host = wait_for_capacity()
    controller = importlib.import_module(controller_name).Controller()
    detector = importlib.import_module(detector_name).Detector()
    result, log = run_episode(controller, box_mass=mass, box_com=COM,
                              detector=detector, collision=collision)
    log["wrench_offline"] = residual_to_wrench(log["r"], log["q"])
    nominal = Nominal()
    actual = np.asarray([nominal.tcp_position(q) for q in log["q"]])
    desired = np.asarray([nominal.tcp_position(q) for q in log["q_d"]])
    log["tcp_error_xyz_mm"] = 1e3 * (actual - desired)
    log["support_normalized"] = np.asarray(log["f_table"], float) / (mass * 9.81)
    return result, log, host


class FourWayReplayView:
    """Identical-camera replay view for one offline comparison column."""

    def __init__(self, log: dict[str, Any], width: int):
        self.model = log["plant"].m
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=VIEW_HEIGHT, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.25, 0.1, 0.3]
        self.camera.distance = 2.8
        self.camera.azimuth = 150
        self.camera.elevation = -20
        self.log = log
        self.nominal = Nominal()

    def render(self, index: int, force_vector: np.ndarray | None) -> np.ndarray:
        self.data.qpos[:] = self.log["qpos"][index]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        desired = self.nominal.tcp_position(self.log["q_d"][index])
        geom = self.renderer.scene.geoms[self.renderer.scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.array([0.015] * 3), desired, np.eye(3).ravel(),
                           np.array([0.0, 0.9, 0.9, 0.85]))
        self.renderer.scene.ngeom += 1
        if force_vector is not None and np.linalg.norm(force_vector) > 0:
            start = self.data.xpos[self.model.body("link_6").id].copy()
            end = start + np.asarray(force_vector, float) * (0.20 / 15.0)
            arrow = self.renderer.scene.geoms[self.renderer.scene.ngeom]
            mujoco.mjv_initGeom(arrow, mujoco.mjtGeom.mjGEOM_ARROW, np.ones(3),
                               np.zeros(3), np.eye(3).ravel(),
                               np.array([0.95, 0.05, 0.05, 1.0]))
            mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, start, end)
            arrow.rgba[:] = [0.95, 0.05, 0.05, 1.0]
            self.renderer.scene.ngeom += 1
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def display_labels(mode: str, hashes: dict[str, str]) -> dict[str, str]:
    labels = dict(LABELS)
    controller_equal = hashes[f"my_{mode}.py"] == hashes[f"r1_{mode}.py"]
    detector_equal = hashes["my_detector.py"] == hashes["r1_detector.py"]
    if controller_equal and detector_equal:
        labels["r2"] = "R2=R1 (동일 바이트)"
    return labels


def plot_panel(logs: dict[str, dict[str, Any]], cursor: float,
               labels: dict[str, str], force_label: str) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.3), dpi=100)
    fig.patch.set_facecolor("#f6f7f9")
    t = logs["baseline"]["t"]
    for name in VARIANT_ORDER:
        axes[0].plot(t, logs[name]["e_mm"], color=COLORS[name], lw=1.0, label=labels[name])
        axes[2].plot(t, logs[name]["support_normalized"], color=COLORS[name], lw=1.0,
                     label=labels[name])
    axes[1].plot(t, logs["baseline"]["f_ext"], color="#d62728", lw=1.25,
                 label=force_label)
    for band, name in enumerate(VARIANT_ORDER):
        on = logs[name]["detect"].astype(float)
        axes[3].fill_between(t, band, band + 0.82 * on, where=on > 0,
                             color=COLORS[name], alpha=.8, step="post")
    for ax in axes:
        ax.axvline(cursor, color="#d62728", lw=1.25)
        ax.set_xlim(0, T.T_END)
        ax.grid(alpha=.22)
        ax.tick_params(labelsize=7)
    axes[0].set_title("TCP 오차 [mm]", fontsize=9)
    axes[1].set_title(f"인가 외력 크기 [N] ({force_label})", fontsize=9)
    axes[2].set_title("테이블 지지력 / (m g) · G4", fontsize=9)
    axes[3].set_title("경보 4개 밴드", fontsize=9)
    axes[3].set_ylim(-.1, 4.0)
    axes[3].set_yticks([.4, 1.4, 2.4, 3.4], [labels[x] for x in VARIANT_ORDER])
    for ax in axes[:3]:
        ax.legend(loc="upper right", fontsize=5.5, framealpha=.75)
    fig.tight_layout(pad=.75)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def outcome_text(result: dict[str, Any]) -> str:
    if result.get("alarm_before_collision"):
        return "충돌 전 경보"
    delay = result.get("detect_delay_ms")
    return "놓침" if delay is None else f"감지 {delay:.0f} ms"


def compose_frame(images: list[np.ndarray], panel: np.ndarray, cursor: float,
                  logs: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]],
                  index: int, title: str, labels: dict[str, str], force_label: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (WIDTH, HEIGHT), "white")
    x = 0
    for image in images:
        canvas.paste(Image.fromarray(image), (x, 0))
        x += image.shape[1]
    canvas.paste(Image.fromarray(panel), (0, VIEW_HEIGHT))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, WIDTH, 50), fill=(9, 20, 35, 225))
    draw.text((16, 7), title, font=_font(19, True), fill="white")
    draw.text((WIDTH - 245, 10), f"t={cursor:5.3f}s · {phase(cursor)}",
              font=_font(16, True), fill="white")
    widths = [WIDTH // 4, WIDTH // 4, WIDTH // 4, WIDTH - 3 * (WIDTH // 4)]
    x = 0
    for width, name in zip(widths, VARIANT_ORDER):
        alarm = bool(logs[name]["detect"][index])
        color = (235, 65, 65, 240) if alarm else (20, 145, 95, 235)
        draw.rounded_rectangle((x + 6, 57, x + width - 6, 137), radius=7,
                               fill=(8, 18, 30, 205))
        draw.text((x + 13, 62), f"{labels[name]} · {'경보' if alarm else '정상'}",
                  font=_font(14, True), fill=color)
        draw.text((x + 13, 86), f"TCP {logs[name]['e_mm'][index]:.2f} mm",
                  font=_font(12), fill="white")
        draw.text((x + 13, 107), outcome_text(results[name]), font=_font(12, True),
                  fill=(255, 210, 80, 255))
        x += width
    draw.text((12, VIEW_HEIGHT - 22),
              f"청록 구: 목표 TCP · 빨간 화살표: {force_label} (15 N = 0.20 m) · 아래: 기록된 테이블 지지력/(m g)",
              font=_font(11, True), fill=(0, 255, 245, 255))
    return np.asarray(canvas)


def save_raw(raw: Path, stem: str, logs: dict[str, dict[str, Any]]) -> None:
    signals: dict[str, np.ndarray] = {}
    for name, log in logs.items():
        for key in ("t", "qpos", "e_mm", "f_table", "r", "detect", "f_ext",
                    "support_normalized"):
            signals[f"{name}_{key}"] = log[key]
        signals[f"{name}_estimated_wrench_offline_true_q"] = log["wrench_offline"]
        signals[f"{name}_tcp_error_xyz_mm"] = log["tcp_error_xyz_mm"]
    np.savez_compressed(raw / f"{stem}.npz", **signals)


def render_scenario(scenario: dict[str, Any], runs: dict[str, Any], output: Path,
                    raw: Path, ffmpeg: Path, hashes: dict[str, str]):
    results = {name: runs[name][0] for name in VARIANT_ORDER}
    logs = {name: runs[name][1] for name in VARIANT_ORDER}
    save_raw(raw, scenario["stem"], logs)
    frame_dir = raw / f"{scenario['stem']}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    widths = [WIDTH // 4, WIDTH // 4, WIDTH // 4, WIDTH - 3 * (WIDTH // 4)]
    views = {name: FourWayReplayView(logs[name], width)
             for name, width in zip(VARIANT_ORDER, widths)}
    collision = scenario["collision"]
    force_label = "+y 15 N" if collision[1][1] > 0 else "-x 15 N"
    labels = display_labels(scenario["mode"], hashes)
    times = frame_times(collision[0])
    try:
        for number, cursor in enumerate(times):
            index = min(int(round(cursor / 1e-3)), len(logs["baseline"]["t"]) - 1)
            force = None
            if logs["baseline"]["f_ext"][index] > 0:
                vector = np.asarray(collision[1], float)
                force = vector * (logs["baseline"]["f_ext"][index] / np.linalg.norm(vector))
            images = [views[name].render(index, force) for name in VARIANT_ORDER]
            panel = plot_panel(logs, float(cursor), labels, force_label)
            frame = compose_frame(images, panel, float(cursor), logs, results, index,
                                  scenario["title"], labels, force_label)
            from PIL import Image
            Image.fromarray(frame).save(frame_dir / f"{number:05d}.png", optimize=True)
    finally:
        for view in views.values():
            view.close()
    mp4, gif = output / f"{scenario['stem']}.mp4", output / f"{scenario['stem']}.gif"
    encode_media(ffmpeg, frame_dir, mp4, gif)
    return results, logs, times, [media_record(mp4, ffmpeg), media_record(gif, ffmpeg)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/media")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_software_environment()
    output, raw = args.output_dir.resolve(), args.raw_dir.resolve()
    if output != (ROOT / "artifacts/media").resolve():
        raise RuntimeError("public output must be artifacts/media")
    if raw == ROOT or ROOT in raw.parents:
        raise RuntimeError("raw/intermediate data must be outside the workcopy")
    ffmpeg = args.ffmpeg if args.ffmpeg.exists() else Path(shutil.which(str(args.ffmpeg)) or "")
    if not ffmpeg.exists():
        raise RuntimeError("ffmpeg not found")
    output.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    owned = [output / f"{scenario['stem']}{suffix}" for scenario in SCENARIOS
             for suffix in (".mp4", ".gif")]
    owned.append(output / "r2_manifest.json")
    for path in owned:
        if path.exists():
            path.unlink()

    integrity_before = check_integrity(root=ROOT)
    hashes = source_hashes()
    all_runs: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        mode = scenario["mode"]
        all_runs[mode] = {}
        for name, (controller, detector) in combinations(mode).items():
            reference = collect_variant(controller, detector, scenario["mass"], None)
            collision = collect_variant(controller, detector, scenario["mass"], scenario["collision"])
            all_runs[mode][name] = {"reference": reference, "collision": collision}

    probe_log = all_runs["servo"]["baseline"]["collision"][1]
    probe_view = FourWayReplayView(probe_log, WIDTH // 4)
    probe_view.render(0, None)
    backend = gl_identity()
    probe_view.close()

    records = []
    for scenario in SCENARIOS:
        mode_runs = all_runs[scenario["mode"]]
        collision_runs = {name: mode_runs[name]["collision"] for name in VARIANT_ORDER}
        results, logs, times, artifacts = render_scenario(
            scenario, collision_runs, output, raw, ffmpeg, hashes)
        variants, classifications, references = {}, {}, {}
        for name in VARIANT_ORDER:
            controller, detector = combinations(scenario["mode"])[name]
            ref_result, ref_log, ref_host = mode_runs[name]["reference"]
            _, _, collision_host = mode_runs[name]["collision"]
            variants[name] = {"controller": controller, "detector": detector}
            classifications[name] = classify_collision(
                results[name], ref_log["detect"], scenario["collision"][0])
            references[name] = {
                "controller": controller, "detector": detector,
                "box_mass_kg": scenario["mass"], "box_com_m": list(COM),
                "false_alarm_rises": int(ref_result["false_alarms"]),
                "resource_guard_before_reference": ref_host,
                "resource_guard_before_collision": collision_host,
            }
        records.append({
            "name": scenario["stem"],
            "condition": {"mode": scenario["mode"], "box_mass_kg": scenario["mass"],
                          "box_com_m": list(COM),
                          "collision": {"time_s": scenario["collision"][0],
                                        "force_n": list(scenario["collision"][1])}},
            "variants": variants,
            "display_labels": display_labels(scenario["mode"], hashes),
            "plot_signals": {"g4_support_normalized": "recorded f_table / (box_mass_kg * 9.81)",
                             "offline_wrench": "nominal Jacobian true-q display only"},
            "timewarp": timewarp_segments(scenario["collision"][0]),
            "source_simulation_frames": {name: len(logs[name]["t"]) for name in VARIANT_ORDER},
            "rendered_frame_count": len(times), "software_gl": backend,
            "episode_summary": {name: run_summary(results[name]) for name in VARIANT_ORDER},
            "evaluate_classification": classifications,
            "classification_method": "tools.sweep.classify_collision with matched no-collision reference",
            "reference_no_collision": references, "artifacts": artifacts,
        })

    integrity_after = check_integrity(root=ROOT)
    total_bytes = sum(item["bytes"] for row in records for item in row["artifacts"])
    if total_bytes >= 30 * 1024**2:
        raise AssertionError(f"R2 media total exceeds 30 MiB: {total_bytes}")
    version = subprocess.check_output([str(ffmpeg), "-version"], text=True).splitlines()[0]
    manifest = {
        "schema_version": 3,
        "generation_command": "env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=glfw LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe __GLX_VENDOR_LIBRARY_NAME=mesa .venv/bin/python tools/render_r2_comparison.py --output-dir artifacts/media --raw-dir \"$RAW_DIR\" --ffmpeg ffmpeg",
        "simulation_source": "sixteen fresh sequential public run_episode episodes: matched no-collision and collision for four displayed variants in two nonofficial 3.7 kg examples",
        "scope_note": "Illustrative multidirectional transport examples only; not official evaluation, holdout evidence, or a model-selection input.",
        "rendering_note": "qpos replay in separate MjData with an identical camera/time cursor for four columns",
        "software_gl": backend, "ffmpeg": version, "source_hashes": hashes,
        "identical_source_aliases": identical_source_aliases(hashes),
        "integrity_before": integrity_before, "integrity_after": integrity_after,
        "total_media_bytes": total_bytes, "scenarios": records,
    }
    manifest_path = output / "r2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({"manifest": manifest_path.relative_to(ROOT).as_posix(),
                      "total_media_bytes": total_bytes, "backend": backend,
                      "scenarios": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
