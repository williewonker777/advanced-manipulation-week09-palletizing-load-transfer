#!/usr/bin/env python3
"""Render the course original and the final hand-in side by side.

One official evaluator condition: 8 kg box, +y 10 N horizontal collision at
1.42 s (right after pick).  Three columns run through the public runner:
the course-provided servo controller + collision detector (the original demo
setup), and the files in ``submission/`` for servo A and torque B.  Each
column is classified exactly like ``evaluate.py`` (a matching no-collision run
decides "구분 불가").  Replay and plots are offline displays; the evaluator
outputs in ``results/final/official/`` remain the score record.

Run inside a kit overlay: a repository copy plus the licensed ``sim/``,
``model/`` and the course ``servo_controller.py`` / ``collision_detector.py``::

    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=glfw LIBGL_ALWAYS_SOFTWARE=1 \\
      GALLIUM_DRIVER=llvmpipe __GLX_VENDOR_LIBRARY_NAME=mesa \\
      python tools/render_original_vs_final.py --output-dir artifacts/media --raw-dir "$RAW_DIR" --ffmpeg ffmpeg
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
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
from tools.render_comparison import (  # noqa: E402
    FPS, GIF_FPS, _font, frame_times, gl_identity, media_record, phase,
    require_software_environment, sha256,
)

STEM = "11_original_vs_final"
MASS = 8.0
COLLISION = (1.42, (0.0, 10.0, 0.0))  # evaluate.py uses (time, [0, F, 0])
TITLE = "원본 vs 최종 · 8 kg · 집기 직후 +y 10 N 충돌 (공식 평가 조건)"
COLUMNS = (
    ("original", "원본 · 수업 제공 서보", "#697386"),
    ("servo", "서보 A 최종", "#0072b2"),
    ("torque", "토크 B 최종", "#d55e00"),
)
# Official record (results/final/official): stock servo 8 kg·1.42 s 10 N = "구분 불가".
EXPECTED = {"original": "구분 불가", "servo": "66 ms", "torque": "43 ms"}
WIDTH, VIEW_HEIGHT, PLOT_HEIGHT = 1280, 380, 720
HEIGHT = VIEW_HEIGHT + PLOT_HEIGHT
VIEW_WIDTHS = (426, 427, 427)
GIF_WIDTH = 960


def load_submission(name: str):
    path = ROOT / "submission" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"final_submission_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def factories():
    detector = load_submission("my_detector")
    stock_ctrl = importlib.import_module("servo_controller")
    stock_det = importlib.import_module("collision_detector")
    return {
        "original": (stock_ctrl.Controller, stock_det.Detector),
        "servo": (load_submission("my_servo").Controller, detector.Detector),
        "torque": (load_submission("my_torque").Controller, detector.Detector),
    }


def classify(result: dict[str, Any], reference_detect: np.ndarray) -> str:
    """Same cell rule as evaluate.py for one collision run."""
    if result["alarm_before_collision"]:
        return "경보 중"
    if result["detect_delay_ms"] is None:
        return "놓침"
    k = int(round((COLLISION[0] + result["detect_delay_ms"] / 1e3) / 1e-3))
    if np.asarray(reference_detect)[max(0, k - 20):k + 21].any():
        return "구분 불가"
    return f"{result['detect_delay_ms']:.0f} ms"


def collect(make) -> tuple[dict[str, Any], dict[str, Any], str]:
    ctrl_cls, det_cls = make
    _, reference = run_episode(ctrl_cls(), box_mass=MASS, detector=det_cls())
    result, log = run_episode(ctrl_cls(), box_mass=MASS, detector=det_cls(), collision=COLLISION)
    log["support_normalized"] = np.asarray(log["f_table"], float) / (MASS * 9.81)
    return result, log, classify(result, reference["detect"])


class View:
    def __init__(self, log: dict[str, Any], width: int):
        self.model = log["plant"].m
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=VIEW_HEIGHT, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.25, 0.1, 0.50]
        self.camera.distance = 2.5
        self.camera.azimuth = 150
        self.camera.elevation = -20
        self.log = log
        self.nominal = Nominal()

    def render(self, index: int, force: np.ndarray | None) -> np.ndarray:
        self.data.qpos[:] = self.log["qpos"][index]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        geom = self.renderer.scene.geoms[self.renderer.scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.015] * 3),
                            self.nominal.tcp_position(self.log["q_d"][index]), np.eye(3).ravel(),
                            np.array([0.0, 0.9, 0.9, 0.85]))
        self.renderer.scene.ngeom += 1
        if force is not None:
            start = self.data.xpos[self.model.body("link_6").id].copy()
            arrow = self.renderer.scene.geoms[self.renderer.scene.ngeom]
            mujoco.mjv_initGeom(arrow, mujoco.mjtGeom.mjGEOM_ARROW, np.ones(3), np.zeros(3),
                                np.eye(3).ravel(), np.array([0.95, 0.05, 0.05, 1.0]))
            mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, 0.014, start,
                                 start + force * (0.20 / 15.0))
            arrow.rgba[:] = [0.95, 0.05, 0.05, 1.0]
            self.renderer.scene.ngeom += 1
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def plot_panel(logs: dict[str, dict[str, Any]], cursor: float) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(4, 1, figsize=(WIDTH / 100, PLOT_HEIGHT / 100), dpi=100, sharex=True)
    fig.patch.set_facecolor("#f6f7f9")
    t = logs["servo"]["t"]
    for mode, label, color in COLUMNS:
        axes[0].plot(t, logs[mode]["e_mm"], color=color, lw=1.5, label=label)
        axes[2].plot(t, logs[mode]["support_normalized"], color=color, lw=1.5, label=label)
    axes[1].plot(t, logs["servo"]["f_ext"], color="#d62728", lw=1.8, label="인가 외력 +y 10 N")
    for band, (mode, label, color) in enumerate(COLUMNS):
        on = logs[mode]["detect"].astype(float)
        axes[3].fill_between(t, band, band + 0.8 * on, where=on > 0, color=color, alpha=.9,
                             step="post")
    titles = ("TCP 오차 [mm]", "인가 외력 [N]", "작업대 지지력 / 박스 무게 (1.0 = 박스 무게)",
              "충돌 경보")
    for ax, title in zip(axes, titles):
        ax.axvline(cursor, color="#d62728", lw=1.6)
        ax.axvspan(COLLISION[0], COLLISION[0] + 0.12, color="#e41a1c", alpha=.10)
        ax.set_xlim(0, T.T_END)
        ax.grid(alpha=.25)
        ax.tick_params(labelsize=10)
        ax.set_title(title, fontsize=13, loc="left", pad=3, fontweight="bold")
    axes[2].axhline(1.0, color="#333", ls=":", lw=1.1)
    axes[3].set_ylim(-.15, 3.0)
    axes[3].set_yticks([.4, 1.4, 2.4], [c[1] for c in COLUMNS], fontsize=10)
    axes[3].set_xlabel("시간 [s]", fontsize=11)
    for ax in axes[:3]:
        ax.legend(loc="upper right", fontsize=10, ncol=3, framealpha=.85)
    fig.tight_layout(pad=.6, h_pad=.4)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def compose(images: list[np.ndarray], panel: np.ndarray, cursor: float, index: int,
            logs: dict[str, dict[str, Any]], results: dict[str, Any],
            cells: dict[str, str]) -> np.ndarray:
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "white")
    x = 0
    for image in images:
        canvas.paste(Image.fromarray(image), (x, 0))
        x += image.shape[1]
    canvas.paste(Image.fromarray(panel), (0, VIEW_HEIGHT))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, WIDTH, 46), fill=(9, 20, 35, 225))
    draw.text((16, 7), TITLE, font=_font(21, True), fill="white")
    draw.text((WIDTH - 250, 10), f"t = {cursor:5.3f} s · {phase(cursor)}", font=_font(17, True),
              fill="white")
    x = 0
    for width, (mode, label, _) in zip(VIEW_WIDTHS, COLUMNS):
        alarm = bool(logs[mode]["detect"][index])
        state = (235, 65, 65, 240) if alarm else (20, 165, 105, 235)
        draw.rounded_rectangle((x + 6, 52, x + width - 6, 152), radius=8, fill=(8, 18, 30, 210))
        draw.text((x + 14, 56), f"{label} · {'경보' if alarm else '정상'}", font=_font(17, True),
                  fill=state)
        draw.text((x + 14, 81), f"TCP 오차 {logs[mode]['e_mm'][index]:.2f} mm", font=_font(14),
                  fill="white")
        cell = cells[mode]
        verdict = f"충돌 감지 {cell}" if cell.endswith("ms") else cell
        draw.text((x + 14, 103), f"결과: {verdict}", font=_font(14, True), fill=(255, 210, 80, 255))
        draw.text((x + 14, 125), f"이 실행의 오감지 {int(results[mode]['false_alarms'])}회",
                  font=_font(13), fill=(230, 230, 230, 255))
        x += width
    draw.text((10, VIEW_HEIGHT - 22),
              "청록 구: 목표 TCP(반지름 15 mm) · 빨간 화살표: 인가 외력 +y (15 N = 0.20 m)",
              font=_font(13, True), fill=(0, 255, 245, 255))
    return np.asarray(canvas)


def encode(ffmpeg: Path, frames: Path, mp4: Path, gif: Path) -> None:
    subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(FPS),
                    "-i", str(frames / "%05d.png"), "-map_metadata", "-1", "-c:v", "libx264",
                    "-preset", "slow", "-crf", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(mp4)], check=True)
    palette = frames / "palette.png"
    vf = f"fps={GIF_FPS},scale={GIF_WIDTH}:-2:flags=lanczos"
    subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4),
                    "-vf", vf + ",palettegen=max_colors=96:stats_mode=diff", str(palette)], check=True)
    subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4),
                    "-i", str(palette), "-lavfi",
                    vf + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                    "-map_metadata", "-1", str(gif)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/media")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    args = parser.parse_args()
    require_software_environment()
    ffmpeg = args.ffmpeg if args.ffmpeg.exists() else Path(shutil.which(str(args.ffmpeg)) or "")
    if not ffmpeg.exists():
        raise RuntimeError("ffmpeg not found")
    output, raw = args.output_dir.resolve(), args.raw_dir.resolve()
    if raw.is_relative_to(ROOT):
        raise RuntimeError("raw frames must stay outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    frames = raw / f"{STEM}_frames"
    frames.mkdir(parents=True, exist_ok=False)

    makers = factories()
    runs = {mode: collect(makers[mode]) for mode, _, _ in COLUMNS}
    results = {mode: runs[mode][0] for mode in runs}
    logs = {mode: runs[mode][1] for mode in runs}
    cells = {mode: runs[mode][2] for mode in runs}
    if cells != EXPECTED or results["servo"]["false_alarms"] or results["torque"]["false_alarms"]:
        raise RuntimeError(f"differs from the official record: {cells} {results}")

    views = [View(logs[mode], width) for width, (mode, _, _) in zip(VIEW_WIDTHS, COLUMNS)]
    gl = gl_identity()
    times = frame_times(COLLISION[0])
    direction = np.asarray(COLLISION[1], float) / np.linalg.norm(COLLISION[1])
    try:
        from PIL import Image
        for number, cursor in enumerate(times):
            index = min(int(round(cursor / 1e-3)), len(logs["servo"]["t"]) - 1)
            images = []
            for view, (mode, _, _) in zip(views, COLUMNS):
                magnitude = float(logs[mode]["f_ext"][index])
                images.append(view.render(index, direction * magnitude if magnitude > 0 else None))
            frame = compose(images, plot_panel(logs, float(cursor)), float(cursor), index, logs,
                            results, cells)
            Image.fromarray(frame).save(frames / f"{number:05d}.png", optimize=True)
    finally:
        for view in views:
            view.close()

    mp4, gif = output / f"{STEM}.mp4", output / f"{STEM}.gif"
    encode(ffmpeg, frames, mp4, gif)
    version = subprocess.check_output([str(ffmpeg), "-version"], text=True).splitlines()[0]
    manifest = {
        "schema_version": 2,
        "generation_command": ("env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=glfw "
                               "LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe "
                               "__GLX_VENDOR_LIBRARY_NAME=mesa python tools/render_original_vs_final.py "
                               "--output-dir artifacts/media --raw-dir \"$RAW_DIR\" --ffmpeg ffmpeg"),
        "simulation_source": ("six fresh unmodified sim.runner.run_episode episodes at 8 kg: a no-collision "
                              "reference and the official 1.42 s +y 10 N collision for the course servo "
                              "controller + detector and for the submission/ servo A and torque B"),
        "scope_note": ("Illustration of one official evaluator condition; cells use the evaluate.py rule. "
                       "The evaluator outputs in results/final/official are the score record."),
        "rendering_note": ("qpos replay in separate MjData; desired TCP marker radius 15 mm; force arrow "
                           "0.20 m per 15 N; slow motion around pick, collision and release; one graph "
                           "per row"),
        "software_gl": gl, "ffmpeg": version,
        "kit_manifest_sha256": sha256(ROOT / "tools/kit_manifest.sha256"),
        "source_hashes": {f"submission/{name}": sha256(ROOT / "submission" / name)
                          for name in ("my_servo.py", "my_torque.py", "my_detector.py")},
        "episodes": {mode: {"cell": cells[mode],
                            "detect_delay_ms": results[mode].get("detect_delay_ms"),
                            "false_alarms": results[mode].get("false_alarms"),
                            "alarm_before_collision": bool(results[mode].get("alarm_before_collision"))}
                     for mode in logs},
        "frames": int(len(times)), "fps": FPS, "gif_fps": GIF_FPS,
        "media": [media_record(mp4, ffmpeg), media_record(gif, ffmpeg)],
    }
    manifest["total_media_bytes"] = sum(item["bytes"] for item in manifest["media"])
    (output / "final_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"episodes": manifest["episodes"], "bytes": manifest["total_media_bytes"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
