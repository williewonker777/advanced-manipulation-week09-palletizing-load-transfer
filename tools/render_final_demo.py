#!/usr/bin/env python3
"""Render the final hand-in demo: servo A and torque B side by side.

One official evaluator condition: 8 kg box, +y 10 N horizontal collision at
1.42 s (right after pick).  Both columns run the files in ``submission/``
(R1 controllers + final detector) through the public runner.  Replay and
plots are offline displays only; the evaluator tables in
``results/final/official/`` remain the score record.  Run inside a kit overlay
(repository copy + licensed ``sim/`` and ``model/``) with software GL::

    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=glfw LIBGL_ALWAYS_SOFTWARE=1 \\
      GALLIUM_DRIVER=llvmpipe __GLX_VENDOR_LIBRARY_NAME=mesa \\
      python tools/render_final_demo.py --output-dir artifacts/media --raw-dir "$RAW_DIR" --ffmpeg ffmpeg
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import trajectory as T  # noqa: E402
from sim.runner import run_episode  # noqa: E402
from tools.render_comparison import (  # noqa: E402
    FPS, GIF_FPS, HEIGHT, VIEW_HEIGHT, WIDTH, ReplayView, _font, frame_times, gl_identity,
    media_record, phase, require_software_environment, sha256,
)

STEM = "11_final_servo_torque"
MASS = 8.0
COLLISION = (1.42, (0.0, 10.0, 0.0))  # evaluate.py: (time, [0, F, 0])
EXPECTED_DELAY_MS = {"servo": 66.0, "torque": 43.0}  # results/final/official 8 kg · 1.42 s · 10 N
TITLE = "최종 제출본 · 8 kg · 집기 직후 +y 10 N 충돌 (공식 평가 조건)"
COLUMNS = (("servo", "서보 A 최종", "#0072b2"), ("torque", "토크 B 최종", "#d55e00"))
GIF_SIZE = (960, 540)


def load_submission(name: str):
    path = ROOT / "submission" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"final_submission_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect(detector_module, mode: str):
    controller = load_submission(f"my_{mode}").Controller()
    detector = detector_module.Detector()
    result, log = run_episode(controller, box_mass=MASS, detector=detector, collision=COLLISION)
    log["support_normalized"] = np.asarray(log["f_table"], float) / (MASS * 9.81)
    log["selected_method"] = getattr(detector, "selected_method", "?")
    return result, log


def outcome(result: dict[str, Any]) -> str:
    if result.get("alarm_before_collision"):
        return "충돌 전 경보"
    delay = result.get("detect_delay_ms")
    return "놓침" if delay is None else f"충돌 감지 {delay:.0f} ms"


def plot_panel(logs: dict[str, dict[str, Any]], cursor: float) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.3), dpi=100)
    fig.patch.set_facecolor("#f6f7f9")
    t = logs["servo"]["t"]
    for mode, label, color in COLUMNS:
        axes[0].plot(t, logs[mode]["e_mm"], color=color, lw=1.1, label=label)
        axes[2].plot(t, logs[mode]["support_normalized"], color=color, lw=1.1, label=label)
    axes[1].plot(t, logs["servo"]["f_ext"], color="#d62728", lw=1.3, label="+y 10 N")
    for band, (mode, label, color) in enumerate(COLUMNS):
        on = logs[mode]["detect"].astype(float)
        axes[3].fill_between(t, band, band + 0.82 * on, where=on > 0, color=color, alpha=.85,
                             step="post")
    for ax in axes:
        ax.axvline(cursor, color="#d62728", lw=1.25)
        ax.axvspan(COLLISION[0], COLLISION[0] + 0.12, color="#e41a1c", alpha=.08)
        ax.set_xlim(0, T.T_END)
        ax.grid(alpha=.22)
        ax.tick_params(labelsize=7)
    axes[0].set_title("TCP 오차 [mm]", fontsize=9)
    axes[1].set_title("인가 외력 [N]", fontsize=9)
    axes[2].set_title("작업대 지지력 / 박스 무게", fontsize=9)
    axes[2].axhline(1.0, color="#555", ls=":", lw=.9)
    axes[3].set_title("충돌 경보", fontsize=9)
    axes[3].set_ylim(-.1, 2.0)
    axes[3].set_yticks([.4, 1.4], [c[1] for c in COLUMNS])
    for ax in axes[:3]:
        ax.legend(loc="upper right", fontsize=6.5, framealpha=.75)
    fig.tight_layout(pad=.75)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def compose_frame(images: dict[str, np.ndarray], panel: np.ndarray, cursor: float, index: int,
                  logs: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]]) -> np.ndarray:
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "white")
    canvas.paste(Image.fromarray(images["servo"]), (0, 0))
    canvas.paste(Image.fromarray(images["torque"]), (WIDTH // 2, 0))
    canvas.paste(Image.fromarray(panel), (0, VIEW_HEIGHT))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, WIDTH, 52), fill=(9, 20, 35, 220))
    draw.text((18, 9), TITLE, font=_font(21, True), fill="white")
    draw.text((WIDTH - 245, 11), f"t = {cursor:5.3f} s · {phase(cursor)}", font=_font(17, True),
              fill="white")
    for x, (mode, label, _) in zip((12, WIDTH // 2 + 12), COLUMNS):
        log = logs[mode]
        alarm = bool(log["detect"][index])
        color = (235, 65, 65, 240) if alarm else (20, 165, 105, 235)
        draw.rounded_rectangle((x, 60, x + 300, 148), radius=8, fill=(8, 18, 30, 205))
        draw.text((x + 10, 64), f"{label} · {'경보' if alarm else '정상'}", font=_font(18, True),
                  fill=color)
        draw.text((x + 10, 90), f"TCP 오차 {log['e_mm'][index]:.2f} mm", font=_font(14), fill="white")
        draw.text((x + 10, 113), "결과: " + outcome(results[mode]), font=_font(14, True),
                  fill=(255, 210, 80, 255))
    draw.text((12, VIEW_HEIGHT - 23),
              "청록 구: 목표 TCP(반지름 15 mm) · 빨간 화살표: 인가 외력 +y (15 N = 0.20 m)",
              font=_font(13, True), fill=(0, 255, 245, 255))
    return np.asarray(canvas)


def encode(ffmpeg: Path, frames: Path, mp4: Path, gif: Path) -> None:
    common = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
              "-framerate", str(FPS), "-i", str(frames / "%05d.png")]
    subprocess.run(common + ["-map_metadata", "-1", "-c:v", "libx264", "-preset", "slow",
                             "-crf", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                             str(mp4)], check=True)
    palette = frames / "palette.png"
    vf = f"fps={GIF_FPS},scale={GIF_SIZE[0]}:{GIF_SIZE[1]}:flags=lanczos"
    subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4),
                    "-vf", vf + ",palettegen=max_colors=96:stats_mode=diff", str(palette)],
                   check=True)
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

    detector_module = load_submission("my_detector")
    runs = {mode: collect(detector_module, mode) for mode, _, _ in COLUMNS}
    results = {mode: runs[mode][0] for mode in runs}
    logs = {mode: runs[mode][1] for mode in runs}
    for mode, expected in EXPECTED_DELAY_MS.items():
        got = results[mode].get("detect_delay_ms")
        if got is None or abs(float(got) - expected) > 0.5 or results[mode].get("false_alarms"):
            raise RuntimeError(f"{mode} differs from the official record: {results[mode]}")

    views = {mode: ReplayView(logs[mode]) for mode in logs}
    gl = gl_identity()
    times = frame_times(COLLISION[0])
    direction = np.asarray(COLLISION[1], float) / np.linalg.norm(COLLISION[1])
    try:
        from PIL import Image
        for number, cursor in enumerate(times):
            index = min(int(round(cursor / 1e-3)), len(logs["servo"]["t"]) - 1)
            images = {}
            for mode in logs:
                magnitude = float(logs[mode]["f_ext"][index])
                images[mode] = views[mode].render(index, direction * magnitude if magnitude > 0 else None)
            frame = compose_frame(images, plot_panel(logs, float(cursor)), float(cursor), index,
                                  logs, results)
            Image.fromarray(frame).save(frames / f"{number:05d}.png", optimize=True)
    finally:
        for view in views.values():
            view.close()

    mp4, gif = output / f"{STEM}.mp4", output / f"{STEM}.gif"
    encode(ffmpeg, frames, mp4, gif)
    version = subprocess.check_output([str(ffmpeg), "-version"], text=True).splitlines()[0]
    manifest = {
        "schema_version": 1,
        "generation_command": ("env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=glfw "
                               "LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe "
                               "__GLX_VENDOR_LIBRARY_NAME=mesa python tools/render_final_demo.py "
                               "--output-dir artifacts/media --raw-dir \"$RAW_DIR\" --ffmpeg ffmpeg"),
        "simulation_source": ("two fresh unmodified sim.runner.run_episode episodes (servo A, torque B) "
                              "with the submission/ files at the official 8 kg, 1.42 s, +y 10 N condition"),
        "scope_note": ("Illustration of one official evaluator condition with the final hand-in; "
                       "the evaluator outputs in results/final/official are the score record."),
        "rendering_note": ("qpos replay in separate MjData; desired TCP marker radius 15 mm; "
                           "force arrow scale 0.20 m per 15 N; slow motion around pick, collision "
                           "and release"),
        "software_gl": gl, "ffmpeg": version,
        "source_hashes": {f"submission/{name}": sha256(ROOT / "submission" / name)
                          for name in ("my_servo.py", "my_torque.py", "my_detector.py")},
        "episodes": {mode: {"selected_method": logs[mode]["selected_method"],
                            "detect_delay_ms": results[mode].get("detect_delay_ms"),
                            "false_alarms": results[mode].get("false_alarms"),
                            "alarm_before_collision": bool(results[mode].get("alarm_before_collision"))}
                     for mode in logs},
        "frames": int(len(times)), "fps": FPS, "gif_fps": GIF_FPS, "gif_size": list(GIF_SIZE),
        "media": [media_record(mp4, ffmpeg), media_record(gif, ffmpeg)],
    }
    manifest["total_media_bytes"] = sum(item["bytes"] for item in manifest["media"])
    (output / "final_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"mp4": mp4.name, "gif": gif.name, "episodes": manifest["episodes"],
                      "bytes": manifest["total_media_bytes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
