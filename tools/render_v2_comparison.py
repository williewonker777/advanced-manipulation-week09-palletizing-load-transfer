#!/usr/bin/env python3
"""Render reproducible baseline/v1/v2 comparison media.

All simulation samples come from the public ``sim.runner.run_episode`` API and
are collected sequentially after the conservative host-capacity guard.  MuJoCo
state and detector residuals are used only after each episode for visualization;
they are never supplied to a controller or detector.
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
    FPS,
    GIF_FPS,
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


COLLISION = (3.0, (0.0, 15.0, 0.0))
VARIANT_ORDER = ("baseline", "v1", "v2")
COLORS = {"baseline": "#697386", "v1": "#009e73", "v2": "#0072b2"}
LABELS = {"baseline": "기준", "v1": "v1", "v2": "v2"}
SCENARIOS = (
    {
        "stem": "05_v2_servo_transport_collision",
        "title": "서보 A · 3.7 kg (비정수 질량) · 운반 +y 15 N 충돌 · 기준/v1/v2",
        "mode": "servo",
        "mass": 3.7,
    },
    {
        "stem": "06_v2_torque_transport_collision",
        "title": "토크 B · 3.7 kg (비정수 질량) · 운반 +y 15 N 충돌 · 기준/v1/v2",
        "mode": "torque",
        "mass": 3.7,
    },
)


def combinations(mode: str) -> dict[str, tuple[str, str]]:
    """Return the frozen public modules used by each displayed column."""
    return {
        "baseline": (f"{mode}_controller", "collision_detector"),
        "v1": (f"v1_{mode}", "v1_detector"),
        "v2": (f"my_{mode}", "my_detector"),
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
        "my_servo.py", "my_torque.py", "my_detector.py",
        "sim/runner.py", "tools/render_v2_comparison.py",
    }
    return {name: sha256(ROOT / name) for name in sorted(files)}


def collect_variant(controller_name: str, detector_name: str, mass: float, collision):
    """Run one guarded episode and attach analysis-only plotting signals."""
    host = wait_for_capacity()
    controller = importlib.import_module(controller_name).Controller()
    detector = importlib.import_module(detector_name).Detector()
    result, log = run_episode(
        controller, box_mass=mass, box_com=COM, detector=detector,
        collision=collision,
    )
    log["wrench_offline"] = residual_to_wrench(log["r"], log["q"])
    nominal = Nominal()
    actual = np.asarray([nominal.tcp_position(q) for q in log["q"]])
    desired = np.asarray([nominal.tcp_position(q) for q in log["q_d"]])
    log["tcp_error_xyz_mm"] = 1e3 * (actual - desired)
    return result, log, host


class ThreeWayReplayView:
    """Identical-camera replay view for one of the three offline columns."""

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
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.015, 0.015, 0.015]), desired, np.eye(3).ravel(),
            np.array([0.0, 0.9, 0.9, 0.85]),
        )
        self.renderer.scene.ngeom += 1
        if force_vector is not None and np.linalg.norm(force_vector) > 0:
            start = self.data.xpos[self.model.body("link_6").id].copy()
            end = start + np.asarray(force_vector, float) * (0.20 / 15.0)
            arrow = self.renderer.scene.geoms[self.renderer.scene.ngeom]
            mujoco.mjv_initGeom(
                arrow, mujoco.mjtGeom.mjGEOM_ARROW, np.ones(3), np.zeros(3),
                np.eye(3).ravel(), np.array([0.95, 0.05, 0.05, 1.0]),
            )
            mujoco.mjv_connector(
                arrow, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, start, end,
            )
            arrow.rgba[:] = [0.95, 0.05, 0.05, 1.0]
            self.renderer.scene.ngeom += 1
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def plot_panel(logs: dict[str, dict[str, Any]], cursor: float) -> np.ndarray:
    """Draw readable error/force/alarm plots for the three variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.3), dpi=100)
    fig.patch.set_facecolor("#f6f7f9")
    t = logs["baseline"]["t"]
    for name in VARIANT_ORDER:
        axes[0].plot(t, logs[name]["e_mm"], color=COLORS[name], lw=1.05,
                     label=LABELS[name])
        axes[2].plot(t, logs[name]["wrench_offline"][:, 1], color=COLORS[name],
                     lw=1.05, label=LABELS[name])
    axes[1].plot(t, logs["baseline"]["f_ext"], color="#d62728", lw=1.25,
                 label="인가 +y 힘")
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
    axes[1].set_title("실제 인가 외력 +y [N]", fontsize=9)
    axes[2].set_title("잔차→추정 Fy [N] (오프라인)", fontsize=9)
    axes[3].set_title("경보 3개 밴드", fontsize=9)
    axes[3].set_ylim(-.1, 3.0)
    axes[3].set_yticks([.4, 1.4, 2.4], [LABELS[x] for x in VARIANT_ORDER])
    for ax in axes[:3]:
        ax.legend(loc="upper right", fontsize=6, framealpha=.75)
    index = min(int(round(cursor / 1e-3)), len(t) - 1)
    values = " / ".join(
        f"{LABELS[name]} {np.linalg.norm(logs[name]['tcp_error_xyz_mm'][index]):.1f}"
        for name in VARIANT_ORDER
    )
    fig.suptitle(f"현재 TCP |오차| [mm]: {values}", fontsize=9, y=.995)
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
                  index: int, title: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (WIDTH, HEIGHT), "white")
    x = 0
    for image in images:
        canvas.paste(Image.fromarray(image), (x, 0))
        x += image.shape[1]
    canvas.paste(Image.fromarray(panel), (0, VIEW_HEIGHT))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, WIDTH, 50), fill=(9, 20, 35, 225))
    draw.text((16, 7), title, font=_font(20, True), fill="white")
    draw.text((WIDTH - 245, 10), f"t={cursor:5.3f}s · {phase(cursor)}",
              font=_font(16, True), fill="white")
    widths = [WIDTH // 3, WIDTH // 3, WIDTH - 2 * (WIDTH // 3)]
    x = 0
    for width, name in zip(widths, VARIANT_ORDER):
        alarm = bool(logs[name]["detect"][index])
        color = (235, 65, 65, 240) if alarm else (20, 145, 95, 235)
        draw.rounded_rectangle((x + 8, 57, x + width - 8, 137), radius=7,
                               fill=(8, 18, 30, 205))
        draw.text((x + 18, 62), f"{LABELS[name]} · {'경보' if alarm else '정상'}",
                  font=_font(17, True), fill=color)
        draw.text((x + 18, 87), f"TCP {logs[name]['e_mm'][index]:.2f} mm",
                  font=_font(13), fill="white")
        draw.text((x + 18, 108), outcome_text(results[name]),
                  font=_font(13, True), fill=(255, 210, 80, 255))
        x += width
    draw.text((12, VIEW_HEIGHT - 22),
              "청록 구: 목표 TCP (반지름 15 mm) · 빨간 화살표: 인가 +y 외력 (15 N = 0.20 m)",
              font=_font(12, True), fill=(0, 255, 245, 255))
    return np.asarray(canvas)


def save_raw(raw: Path, stem: str, logs: dict[str, dict[str, Any]]) -> None:
    signals: dict[str, np.ndarray] = {}
    for name, log in logs.items():
        for key in ("t", "qpos", "e_mm", "f_table", "r", "detect", "f_ext"):
            signals[f"{name}_{key}"] = log[key]
        signals[f"{name}_estimated_wrench_offline_true_q"] = log["wrench_offline"]
        signals[f"{name}_tcp_error_xyz_mm"] = log["tcp_error_xyz_mm"]
    np.savez_compressed(raw / f"{stem}.npz", **signals)


def render_scenario(scenario: dict[str, Any], runs: dict[str, Any], output: Path,
                    raw: Path, ffmpeg: Path, backend: dict[str, str]):
    results = {name: runs[name][0] for name in VARIANT_ORDER}
    logs = {name: runs[name][1] for name in VARIANT_ORDER}
    save_raw(raw, scenario["stem"], logs)
    frame_dir = raw / f"{scenario['stem']}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    widths = [WIDTH // 3, WIDTH // 3, WIDTH - 2 * (WIDTH // 3)]
    views = {name: ThreeWayReplayView(logs[name], width)
             for name, width in zip(VARIANT_ORDER, widths)}
    times = frame_times(COLLISION[0])
    try:
        for number, cursor in enumerate(times):
            index = min(int(round(cursor / 1e-3)), len(logs["baseline"]["t"]) - 1)
            force = None
            if logs["baseline"]["f_ext"][index] > 0:
                force = np.asarray(COLLISION[1]) * (
                    logs["baseline"]["f_ext"][index] / np.linalg.norm(COLLISION[1])
                )
            images = [views[name].render(index, force) for name in VARIANT_ORDER]
            panel = plot_panel(logs, float(cursor))
            frame = compose_frame(images, panel, float(cursor), logs, results, index,
                                  scenario["title"])
            from PIL import Image
            Image.fromarray(frame).save(frame_dir / f"{number:05d}.png", optimize=True)
    finally:
        for view in views.values():
            view.close()
    mp4 = output / f"{scenario['stem']}.mp4"
    gif = output / f"{scenario['stem']}.gif"
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
    owned.append(output / "v2_manifest.json")
    for path in owned:
        if path.exists():
            path.unlink()

    integrity_before = check_integrity(root=ROOT)
    all_runs: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        mode = scenario["mode"]
        all_runs[mode] = {}
        for name, (controller, detector) in combinations(mode).items():
            reference = collect_variant(controller, detector, scenario["mass"], None)
            collision = collect_variant(controller, detector, scenario["mass"], COLLISION)
            all_runs[mode][name] = {"reference": reference, "collision": collision}

    probe_log = all_runs["servo"]["baseline"]["collision"][1]
    probe = ThreeWayReplayView(probe_log, WIDTH // 3)
    probe.render(0, None)
    backend = gl_identity()
    probe.close()

    records = []
    for scenario in SCENARIOS:
        mode_runs = all_runs[scenario["mode"]]
        collision_runs = {name: mode_runs[name]["collision"] for name in VARIANT_ORDER}
        results, logs, times, artifacts = render_scenario(
            scenario, collision_runs, output, raw, ffmpeg, backend,
        )
        variants = {}
        classifications = {}
        references = {}
        for name in VARIANT_ORDER:
            controller, detector = combinations(scenario["mode"])[name]
            ref_result, ref_log, ref_host = mode_runs[name]["reference"]
            _, _, collision_host = mode_runs[name]["collision"]
            variants[name] = {"controller": controller, "detector": detector}
            classifications[name] = classify_collision(
                results[name], ref_log["detect"], COLLISION[0],
            )
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
                          "collision": {"time_s": COLLISION[0],
                                        "force_n": list(COLLISION[1])}},
            "variants": variants,
            "timewarp": timewarp_segments(COLLISION[0]),
            "source_simulation_frames": {name: len(logs[name]["t"])
                                         for name in VARIANT_ORDER},
            "rendered_frame_count": len(times),
            "software_gl": backend,
            "episode_summary": {name: run_summary(results[name]) for name in VARIANT_ORDER},
            "evaluate_classification": classifications,
            "classification_method": "tools.sweep.classify_collision; alarm-before, miss, matching no-collision reference active within +/-20 ms at detected-rise index, then detected",
            "reference_no_collision": references,
            "artifacts": artifacts,
        })

    integrity_after = check_integrity(root=ROOT)
    total_bytes = sum(item["bytes"] for row in records for item in row["artifacts"])
    if total_bytes >= 30 * 1024**2:
        raise AssertionError(f"v2 media total exceeds 30 MiB: {total_bytes}")
    version = subprocess.check_output([str(ffmpeg), "-version"], text=True).splitlines()[0]
    manifest = {
        "schema_version": 2,
        "generation_command": "env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=glfw LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe __GLX_VENDOR_LIBRARY_NAME=mesa .venv/bin/python tools/render_v2_comparison.py --output-dir artifacts/media --raw-dir \"$RAW_DIR\" --ffmpeg ffmpeg",
        "simulation_source": "twelve fresh sequential unmodified sim.runner.run_episode episodes at 3.7 kg: matching no-collision reference plus collision for each displayed variant",
        "scope_note": "Unseen noninteger 3.7 kg comparison. Final L3 uses the preregistered geometric height gate for every mass, not the rejected exploratory mass-envelope fallback. Official canonical conditions are separately reported as compatibility checks.",
        "rendering_note": "qpos replay in separate MjData with identical camera and time cursor for all columns",
        "force_plot_note": "joint residual r converted only offline with nominal Jacobian and recorded q; estimated Fy is not sensor force or ground truth and is never a controller input",
        "software_gl": backend,
        "ffmpeg": version,
        "source_hashes": source_hashes(),
        "integrity_before": integrity_before,
        "integrity_after": integrity_after,
        "total_media_bytes": total_bytes,
        "scenarios": records,
    }
    manifest_path = output / "v2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({"manifest": manifest_path.relative_to(ROOT).as_posix(),
                      "total_media_bytes": total_bytes, "backend": backend,
                      "scenarios": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
