#!/usr/bin/env python3
"""Render reproducible baseline-versus-proposed comparison media.

The simulation data come from the unmodified ``sim.runner.run_episode`` API.
Rendering replays the returned qpos history in a separate MjData object, so it
cannot affect controller or detector behavior.  Estimated forces are an
offline Jacobian conversion of the detector's public joint residual log.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import re

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_integrity import check_integrity  # noqa: E402
from tools.sweep import classify_collision  # noqa: E402
from sim import config as C, trajectory as T  # noqa: E402
from sim.nominal import Nominal  # noqa: E402
from sim.runner import run_episode  # noqa: E402


WIDTH, HEIGHT, VIEW_HEIGHT = 1280, 720, 390
FPS, GIF_FPS = 15, 8
MASS, COM = 8.0, (0.02, -0.01, 0.03)
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

SCENARIOS = (
    {
        "stem": "01_servo_no_collision",
        "title": "서보 A · 8 kg · 충돌 없음",
        "baseline_ctrl": "servo_controller", "ours_ctrl": "my_servo",
        "collision": None,
    },
    {
        "stem": "02_torque_no_collision",
        "title": "토크 B · 8 kg · 충돌 없음",
        "baseline_ctrl": "torque_controller", "ours_ctrl": "my_torque",
        "collision": None,
    },
    {
        "stem": "03_servo_place_collision",
        "title": "서보 A · 8 kg · 놓기 +y 15 N 충돌",
        "baseline_ctrl": "servo_controller", "ours_ctrl": "my_servo",
        "collision": (3.86, (0.0, 15.0, 0.0)),
    },
    {
        "stem": "04_servo_transport_collision",
        "title": "서보 A · 8 kg · 운반 +y 15 N 충돌",
        "baseline_ctrl": "servo_controller", "ours_ctrl": "my_servo",
        "collision": (3.0, (0.0, 15.0, 0.0)),
    },
)


def available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found")


def require_software_environment() -> None:
    required = {
        "MUJOCO_GL": "glfw", "LIBGL_ALWAYS_SOFTWARE": "1",
        "GALLIUM_DRIVER": "llvmpipe", "__GLX_VENDOR_LIBRARY_NAME": "mesa",
    }
    wrong = {key: os.environ.get(key) for key, value in required.items()
             if os.environ.get(key) != value}
    if wrong:
        raise RuntimeError(f"software GL environment is not exact: {wrong}")
    if available_memory_bytes() < 6 * 1024**3:
        raise RuntimeError("less than 6 GiB MemAvailable")


def gl_identity() -> dict[str, str]:
    from OpenGL.GL import GL_RENDERER, GL_VENDOR, GL_VERSION, glGetString

    values = {
        "vendor": (glGetString(GL_VENDOR) or b"").decode("ascii", "replace"),
        "renderer": (glGetString(GL_RENDERER) or b"").decode("ascii", "replace"),
        "version": (glGetString(GL_VERSION) or b"").decode("ascii", "replace"),
    }
    text = f"{values['vendor']} {values['renderer']}".lower()
    if "mesa" not in text or not ("llvmpipe" in text or "softpipe" in text):
        raise RuntimeError(f"software renderer not proven: {values}")
    return values


def timewarp_segments(collision_time: float | None) -> list[dict[str, float]]:
    slow = [(1.25, 1.70), (3.65, 4.20)]
    if collision_time is not None and not any(a <= collision_time <= b for a, b in slow):
        slow.append((collision_time - 0.15, collision_time + 0.30))
    points = sorted({0.0, T.T_END, *(x for span in slow for x in span)})
    segments = []
    for start, end in zip(points, points[1:]):
        if end <= 0 or start >= T.T_END:
            continue
        start, end = max(0.0, start), min(T.T_END, end)
        midpoint = (start + end) / 2
        speed = 0.35 if any(a <= midpoint < b for a, b in slow) else 1.30
        segments.append({"simulation_start_s": start, "simulation_end_s": end,
                         "playback_speed": speed})
    return segments


def frame_times(collision_time: float | None, fps: int = FPS) -> np.ndarray:
    segments = timewarp_segments(collision_time)
    out: list[float] = []
    for segment in segments:
        duration = (segment["simulation_end_s"] - segment["simulation_start_s"]) / segment["playback_speed"]
        count = max(1, int(round(duration * fps)))
        out.extend(np.linspace(segment["simulation_start_s"], segment["simulation_end_s"], count, endpoint=False))
    out.append(float(T.T_END - 1e-3))
    return np.asarray(out)


def phase(t: float) -> str:
    if t < T.T_GRASP:
        return "접근"
    if t < 2.0:
        return "집기·상승"
    if t < 3.65:
        return "운반"
    if t < T.T_RELEASE:
        return "놓기 접근"
    return "하중 이양·복귀"


def run_variant(controller_name: str, detector_name: str, collision):
    controller = importlib.import_module(controller_name).Controller()
    detector = importlib.import_module(detector_name).Detector()
    result, log = run_episode(
        controller, box_mass=MASS, box_com=COM, detector=detector,
        collision=collision,
    )
    wrench = residual_to_wrench(log["r"], log["q"])
    log["wrench_offline"] = wrench
    nom = Nominal()
    actual_tcp = np.array([nom.tcp_position(q) for q in log["q"]])
    desired_tcp = np.array([nom.tcp_position(q) for q in log["q_d"]])
    log["tcp_error_xyz_mm"] = 1e3 * (actual_tcp - desired_tcp)
    return result, log


def residual_to_wrench(residual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Analysis-only true-q conversion; never exposed to controller/detector."""
    nom = Nominal()
    wrench = np.zeros_like(residual)
    body = nom.m.body("link_6").id
    bottom_offset = np.array([0.0, 0.0, C.BOX_POS[2] + 0.5 * C.BOX_SIZE[2]])
    for k, (rk, qk) in enumerate(zip(residual, q)):
        nom.d.qpos[:] = qk
        mujoco.mj_kinematics(nom.m, nom.d)
        mujoco.mj_comPos(nom.m, nom.d)
        rotation = nom.d.xmat[body].reshape(3, 3)
        point = nom.d.xpos[body] + rotation @ bottom_offset
        jacp, jacr = np.zeros((3, 6)), np.zeros((3, 6))
        mujoco.mj_jac(nom.m, nom.d, jacp, jacr, point, body)
        jacobian = np.vstack((jacp, jacr))
        try:
            wrench[k] = np.linalg.solve(jacobian.T, rk)
        except np.linalg.LinAlgError:
            wrench[k] = np.linalg.lstsq(jacobian.T, rk, rcond=1e-5)[0]
    return wrench


def run_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "tcp_rms_mm": round(float(result["rms_mm"]), 2),
        "tcp_max_mm": round(float(result["max_mm"]), 2),
        "saturation_pct": round(float(result["saturation_pct"]), 2),
        "torque_jitter": round(float(result["torque_jitter"]), 3),
        "false_alarm_rises_in_this_episode": int(result["false_alarms"]),
        "false_alarm_times_s": [round(float(x), 3) for x in result["false_alarm_times"]],
        "detect_delay_ms_raw": None if result.get("detect_delay_ms") is None else round(float(result["detect_delay_ms"])),
        "alarm_before_collision": bool(result.get("alarm_before_collision", False)),
        "diverged": bool(result["diverged"]),
    }


def validate_with_run_py(controller: str, detector: str, collision, expected: dict[str, Any],
                         raw: Path, tag: str) -> dict[str, Any]:
    """Execute the user-facing runner and compare its printed episode metrics."""
    plot_path = raw / "run_py" / f"{tag}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "run.py", "--box", "8", "--ctrl", controller,
               "--detector", detector, "--plot", str(plot_path)]
    portable = [".venv/bin/python", "run.py", "--box", "8", "--ctrl", controller,
                "--detector", detector, "--plot", f"$RAW_DIR/run_py/{tag}.png"]
    if collision is not None:
        command += ["--collision", str(collision[0]), "--col-force", "15"]
        portable += ["--collision", str(collision[0]), "--col-force", "15"]
    output = subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT)
    (raw / "run_py" / f"{tag}.txt").write_text(output, encoding="utf-8")
    match = re.search(r"RMS ([0-9.]+) mm, 최대 ([0-9.]+) mm \| 토크 포화 ([0-9.]+)% \| 토크 떨림 ([0-9.]+)", output)
    false = re.search(r"오감지 (\d+)회", output)
    if not match or not false:
        raise AssertionError(f"could not parse run.py output for {tag}")
    parsed = {
        "tcp_rms_mm": float(match.group(1)), "tcp_max_mm": float(match.group(2)),
        "saturation_pct": float(match.group(3)), "torque_jitter": float(match.group(4)),
        "false_alarm_rises_in_this_episode": int(false.group(1)),
        "alarm_before_collision": "충돌 전부터 경보 중" in output,
    }
    delay = re.search(r"충돌 감지 지연 ([0-9.]+) ms", output)
    parsed["detect_delay_ms_raw"] = None if delay is None else float(delay.group(1))
    comparisons = {
        "tcp_rms_mm": float(f"{expected['rms_mm']:.2f}"),
        "tcp_max_mm": float(f"{expected['max_mm']:.2f}"),
        "saturation_pct": float(f"{expected['saturation_pct']:.1f}"),
        "torque_jitter": float(f"{expected['torque_jitter']:.2f}"),
        "false_alarm_rises_in_this_episode": expected["false_alarms"],
        "alarm_before_collision": expected.get("alarm_before_collision", False),
        "detect_delay_ms_raw": None if expected.get("detect_delay_ms") is None else float(f"{expected['detect_delay_ms']:.0f}"),
    }
    if parsed != comparisons:
        raise AssertionError(f"run.py mismatch for {tag}: parsed={parsed}, direct={comparisons}")
    return {"command": " ".join(portable), "parsed": parsed, "matches_direct_run_episode": True}


def verify_expected(stem: str, baseline: dict[str, Any], ours: dict[str, Any]) -> None:
    expected = {
        "01_servo_no_collision": ((6.14, 14.00, 7), (1.17, 3.91, 0)),
        "02_torque_no_collision": ((7.04, 13.18, 9), (3.23, 8.44, 0)),
    }
    if stem in expected:
        for label, result, values in (("baseline", baseline, expected[stem][0]), ("ours", ours, expected[stem][1])):
            actual = (round(result["rms_mm"], 2), round(result["max_mm"], 2), result["false_alarms"])
            if actual != values:
                raise AssertionError(f"{stem} {label}: expected {values}, got {actual}")


class ReplayView:
    def __init__(self, log: dict[str, Any]):
        import mujoco

        self.mujoco = mujoco
        self.model = log["plant"].m
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=VIEW_HEIGHT, width=WIDTH // 2)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.25, 0.1, 0.3]
        self.camera.distance = 2.8
        self.camera.azimuth = 150
        self.camera.elevation = -20
        self.log = log
        self.nom = Nominal()

    def render(self, index: int, force_vector: np.ndarray | None = None) -> np.ndarray:
        m = self.mujoco
        self.data.qpos[:] = self.log["qpos"][index]
        m.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        self.last_force_screen_direction = None
        # Actual-scale 15 mm desired-TCP marker; visualization only.
        desired = self.nom.tcp_position(self.log["q_d"][index])
        geom = self.renderer.scene.geoms[self.renderer.scene.ngeom]
        m.mjv_initGeom(geom, m.mjtGeom.mjGEOM_SPHERE, np.array([0.015, 0.015, 0.015]),
                      desired, np.eye(3).ravel(), np.array([0.0, 0.9, 0.9, 0.85]))
        self.renderer.scene.ngeom += 1
        if force_vector is not None and np.linalg.norm(force_vector) > 0:
            start = self.data.xpos[self.model.body("link_6").id].copy()
            # Explicit visualization scale: a 15 N vector is drawn 0.20 m long.
            end = start + np.asarray(force_vector, float) * (0.20 / 15.0)
            arrow = self.renderer.scene.geoms[self.renderer.scene.ngeom]
            m.mjv_initGeom(arrow, m.mjtGeom.mjGEOM_ARROW, np.ones(3), np.zeros(3),
                          np.eye(3).ravel(), np.array([0.95, 0.05, 0.05, 1.0]))
            m.mjv_connector(arrow, m.mjtGeom.mjGEOM_ARROW, 0.012, start, end)
            arrow.rgba[:] = [0.95, 0.05, 0.05, 1.0]
            self.renderer.scene.ngeom += 1
            head = np.zeros(3); forward = np.zeros(3); up = np.zeros(3); right = np.zeros(3)
            m.mjv_cameraFrame(head, forward, up, right, self.data, self.camera)
            projected = np.array([np.dot(force_vector, right), -np.dot(force_vector, up)])
            if np.linalg.norm(projected) > 1e-9:
                self.last_force_screen_direction = projected / np.linalg.norm(projected)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def plot_panel(base_log: dict[str, Any], ours_log: dict[str, Any], cursor: float,
               collision) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 3.3), dpi=100)
    fig.patch.set_facecolor("#f6f7f9")
    t = base_log["t"]
    specs = [
        (axes[0, 0], base_log["e_mm"], ours_log["e_mm"], "TCP 오차 [mm]"),
        (axes[0, 1], base_log["tau"][:, 1], ours_log["tau"][:, 1], "J2 모터 토크 [N·m]"),
        (axes[0, 2], base_log["f_table"], ours_log["f_table"], "작업대 수직력 [N]"),
        (axes[1, 0], base_log["wrench_offline"][:, 1], ours_log["wrench_offline"][:, 1], "추정 외력 Fy [N] (오프라인)"),
        (axes[1, 1], base_log["detect"].astype(float), ours_log["detect"].astype(float), "경보 상태"),
    ]
    for ax, y0, y1, title in specs:
        ax.plot(t, y0, color="#697386", lw=1.0, label="기준")
        ax.plot(t, y1, color="#0072b2", lw=1.15, label="우리")
        ax.axvline(cursor, color="#d62728", lw=1.4)
        ax.set_xlim(0, T.T_END)
        ax.set_title(title, fontsize=9, pad=2)
        ax.grid(alpha=.22)
        ax.tick_params(labelsize=7)
    axes[0, 2].axhline(1.1 * MASS * 9.81, color="#d55e00", ls="--", lw=1, label="1.1 mg")
    if collision:
        tc, force = collision
        axes[1, 0].axvspan(tc, tc + 0.12, color="#e41a1c", alpha=.12)
        direction = float(np.sign(force[1]))
        axes[1, 0].plot(t, direction * base_log["f_ext"], color="#e41a1c", ls=":", lw=1,
                        label="실제 인가 +y 힘")
    axes[1, 1].set_ylim(-0.1, 1.25)
    axes[1, 1].set_yticks([0, 1])
    for ax in axes.flat[:5]:
        ax.legend(loc="upper right", fontsize=6, ncol=2, framealpha=.75)
    inset = axes[1, 2]
    index = min(int(round(cursor / 1e-3)), len(t) - 1)
    b_err, o_err = base_log["tcp_error_xyz_mm"], ours_log["tcp_error_xyz_mm"]
    inset.plot(b_err[:, 0], b_err[:, 1], color="#697386", lw=.7, alpha=.45)
    inset.plot(o_err[:, 0], o_err[:, 1], color="#0072b2", lw=.7, alpha=.45)
    inset.scatter([0], [0], marker="+", s=70, color="black", label="목표")
    inset.scatter([b_err[index, 0]], [b_err[index, 1]], s=26, color="#697386", label="기준 현재")
    inset.scatter([o_err[index, 0]], [o_err[index, 1]], s=26, color="#0072b2", label="우리 현재")
    inset.set_xlim(-15, 15); inset.set_ylim(-15, 15); inset.set_aspect("equal", adjustable="box")
    inset.set_xlabel("x 오차 [mm]", fontsize=7); inset.set_ylabel("y 오차 [mm]", fontsize=7)
    inset.set_title("TCP 오차 확대 (실제 mm 축)\n"
                    f"z 기준/우리 {b_err[index,2]:+.1f}/{o_err[index,2]:+.1f} mm · "
                    f"|e| {np.linalg.norm(b_err[index]):.1f}/{np.linalg.norm(o_err[index]):.1f}",
                    fontsize=8, pad=2)
    inset.grid(alpha=.22); inset.tick_params(labelsize=6)
    inset.legend(loc="upper right", fontsize=5, framealpha=.75)
    fig.tight_layout(pad=.7)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def compose_frame(left: np.ndarray, right: np.ndarray, panel: np.ndarray,
                  t: float, base_log: dict[str, Any], ours_log: dict[str, Any], index: int,
                  collision, title: str, force_screen_direction: np.ndarray | None) -> np.ndarray:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (WIDTH, HEIGHT), "white")
    canvas.paste(Image.fromarray(left), (0, 0))
    canvas.paste(Image.fromarray(right), (WIDTH // 2, 0))
    canvas.paste(Image.fromarray(panel), (0, VIEW_HEIGHT))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, WIDTH, 52), fill=(9, 20, 35, 220))
    draw.text((18, 8), title, font=_font(22, True), fill="white")
    draw.text((WIDTH - 260, 10), f"t = {t:5.3f} s · {phase(t)}", font=_font(18, True), fill="white")
    for x, label, log in ((12, "기준", base_log), (WIDTH // 2 + 12, "우리", ours_log)):
        alarm = bool(log["detect"][index])
        color = (235, 65, 65, 235) if alarm else (20, 145, 95, 230)
        draw.rounded_rectangle((x, 60, x + 330, 157), radius=8, fill=(8, 18, 30, 205))
        draw.text((x + 10, 65), f"{label} · {'경보' if alarm else '정상'}", font=_font(18, True), fill=color)
        draw.text((x + 10, 88), f"TCP 오차 {log['e_mm'][index]:.2f} mm", font=_font(14), fill="white")
        draw.text((x + 10, 108), f"이 에피소드 오감지 시작 {int(log.get('_episode_false_alarms', 0))}회",
                  font=_font(13), fill=(235, 235, 235, 255))
        draw.text((x + 10, 128), str(log.get("_episode_outcome", "에피소드 요약: 충돌 없음")),
                  font=_font(13, True), fill=(255, 210, 80, 255))
    draw.text((12, VIEW_HEIGHT - 23),
              "청록 구 = 목표 TCP(반지름 15 mm) · 우하단 = 실제 mm 축 오차 확대",
              font=_font(13, True), fill=(0, 255, 245, 255))
    if collision and base_log["f_ext"][index] > 0:
        draw.text((WIDTH // 2 - 125, 60),
                  f"인가 외력 +y {base_log['f_ext'][index]:.1f} N · 화살표 15 N = 0.20 m",
                  font=_font(15, True), fill=(255, 80, 80, 255), anchor="ma")
        if force_screen_direction is not None:
            magnitude = float(base_log["f_ext"][index])
            length = 85.0 * magnitude / 15.0
            direction = force_screen_direction
            normal = np.array([-direction[1], direction[0]])
            for origin_x in (530, 1170):
                start = np.array([origin_x, 205.0])
                end = start + length * direction
                draw.line((tuple(start), tuple(end)), fill=(255, 45, 45, 255), width=7)
                back = end - 18 * direction
                draw.polygon((tuple(end), tuple(back + 9 * normal), tuple(back - 9 * normal)),
                             fill=(255, 45, 45, 255))
                draw.text((origin_x - 110, 225), "world +y 화면 투영(표시용)",
                          font=_font(12, True), fill=(255, 90, 90, 255))
                draw.text((origin_x - 110, 242), "2D 배율: 15 N = 85 px",
                          font=_font(12, True), fill=(255, 90, 90, 255))
    return np.asarray(canvas)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe(ffmpeg: Path, path: Path) -> dict[str, Any]:
    probe = ffmpeg.with_name("ffprobe")
    output = subprocess.check_output([
        str(probe), "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames:format=duration",
        "-of", "json", str(path),
    ], text=True)
    data = json.loads(output)
    stream = data["streams"][0]
    average_rate = stream["avg_frame_rate"]
    render_rate = stream["r_frame_rate"]
    duration = float(data["format"]["duration"])
    frames = int(stream["nb_read_frames"])
    fps = frames / duration
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "avg_frame_rate": average_rate, "r_frame_rate": render_rate,
            "fps": fps, "frame_count": frames, "duration_s": duration}


def media_record(path: Path, ffmpeg: Path) -> dict[str, Any]:
    record = {"file": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
              "sha256": sha256(path)}
    record.update(ffprobe(ffmpeg, path))
    return record


def encode_media(ffmpeg: Path, frames: Path, mp4: Path, gif: Path) -> None:
    common = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
              "-framerate", str(FPS), "-i", str(frames / "%05d.png")]
    subprocess.run(common + ["-map_metadata", "-1", "-c:v", "libx264", "-preset", "slow",
                             "-crf", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                             str(mp4)], check=True)
    palette = frames / "palette.png"
    vf = f"fps={GIF_FPS},scale=640:360:flags=lanczos"
    subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4),
                    "-vf", vf + ",palettegen=max_colors=96:stats_mode=diff", str(palette)], check=True)
    subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4),
                    "-i", str(palette), "-lavfi", vf + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                    "-map_metadata", "-1", str(gif)], check=True)


def render_scenario(scenario: dict[str, Any], baseline, ours, output: Path, raw: Path,
                    ffmpeg: Path, backend: dict[str, str], classification: dict[str, Any] | None,
                    run_py_check: dict[str, Any], reference_info: dict[str, Any]):
    stem = scenario["stem"]
    base_result, base_log = baseline
    our_result, our_log = ours
    base_log["_episode_false_alarms"] = int(base_result["false_alarms"])
    our_log["_episode_false_alarms"] = int(our_result["false_alarms"])
    if scenario["collision"] is None:
        base_log["_episode_outcome"] = our_log["_episode_outcome"] = "에피소드 요약: 충돌 없음"
    else:
        base_log["_episode_outcome"] = ("에피소드 요약: 충돌 전 경보 · 판정 불가"
                                        if base_result["alarm_before_collision"] else
                                        f"에피소드 요약: 감지 {base_result['detect_delay_ms']:.0f} ms")
        our_log["_episode_outcome"] = ("에피소드 요약: 충돌 전 경보 · 판정 불가"
                                       if our_result["alarm_before_collision"] else
                                       f"에피소드 요약: 감지 {our_result['detect_delay_ms']:.0f} ms")
    verify_expected(stem, base_result, our_result)
    raw_signals = {}
    for label, log in (("baseline", base_log), ("ours", our_log)):
        for key in ("t", "qpos", "e_mm", "tau", "f_table", "r", "detect", "f_ext"):
            raw_signals[f"{label}_{key}"] = log[key]
        raw_signals[f"{label}_wrench_offline_true_q"] = log["wrench_offline"]
        raw_signals[f"{label}_tcp_error_xyz_mm"] = log["tcp_error_xyz_mm"]
    np.savez_compressed(raw / f"{stem}.npz", **raw_signals)
    frame_dir = raw / f"{stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    left, right = ReplayView(base_log), ReplayView(our_log)
    times = frame_times(None if scenario["collision"] is None else scenario["collision"][0])
    for number, t in enumerate(times):
        index = min(int(round(t / 1e-3)), len(base_log["t"]) - 1)
        panel = plot_panel(base_log, our_log, float(t), scenario["collision"])
        force_vector = None
        if scenario["collision"] is not None and base_log["f_ext"][index] > 0:
            commanded = np.asarray(scenario["collision"][1], float)
            force_vector = commanded * (base_log["f_ext"][index] / np.linalg.norm(commanded))
        left_image = left.render(index, force_vector)
        right_image = right.render(index, force_vector)
        image = compose_frame(left_image, right_image, panel, float(t), base_log, our_log, index,
                              scenario["collision"], scenario["title"], left.last_force_screen_direction)
        from PIL import Image
        Image.fromarray(image).save(frame_dir / f"{number:05d}.png", optimize=True)
    left.close(); right.close()
    mp4, gif = output / f"{stem}.mp4", output / f"{stem}.gif"
    encode_media(ffmpeg, frame_dir, mp4, gif)
    return {
        "name": stem,
        "condition": {"box_mass_kg": MASS, "box_com_m": list(COM),
                      "collision": None if scenario["collision"] is None else
                      {"time_s": scenario["collision"][0], "force_n": list(scenario["collision"][1])}},
        "combination": {"baseline": f"{scenario['baseline_ctrl']} + collision_detector",
                        "ours": f"{scenario['ours_ctrl']} + my_detector"},
        "timewarp": timewarp_segments(None if scenario["collision"] is None else scenario["collision"][0]),
        "source_simulation_frames": len(base_log["t"]), "rendered_frame_count": len(times),
        "software_gl": backend,
        "run_py_equivalent": {"baseline": run_summary(base_result), "ours": run_summary(our_result)},
        "run_py_validation": run_py_check,
        "evaluate_classification": classification,
        "classification_method": "tools.sweep.classify_collision; alarm-before, miss, matching-reference active within +/-20 ms at detected-rise index, then detected",
        "reference_no_collision": reference_info,
        "artifacts": [media_record(mp4, ffmpeg), media_record(gif, ffmpeg)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    owned.append(output / "media_manifest.json")
    for old in owned:
        if old.exists():
            old.unlink()
    integrity_before = check_integrity(root=ROOT)

    runs: dict[tuple[str, str, float | None], Any] = {}
    for scenario in SCENARIOS:
        collision_time = None if scenario["collision"] is None else scenario["collision"][0]
        for ctrl, det in ((scenario["baseline_ctrl"], "collision_detector"),
                          (scenario["ours_ctrl"], "my_detector")):
            key = (ctrl, det, collision_time)
            runs[key] = run_variant(ctrl, det, scenario["collision"])

    # Renderer creation establishes the GL context. Fail closed unless GL itself
    # reports Mesa llvmpipe/softpipe, not merely an environment flag.
    probe = ReplayView(runs[("servo_controller", "collision_detector", None)][1])
    probe.render(0)
    backend = gl_identity()
    probe.close()

    run_py_checks = {}
    for scenario in SCENARIOS:
        ct = None if scenario["collision"] is None else scenario["collision"][0]
        for label, ctrl, det in (("baseline", scenario["baseline_ctrl"], "collision_detector"),
                                 ("ours", scenario["ours_ctrl"], "my_detector")):
            direct = runs[(ctrl, det, ct)][0]
            tag = f"{scenario['stem']}_{label}"
            run_py_checks[(scenario["stem"], label)] = validate_with_run_py(
                ctrl, det, scenario["collision"], direct, raw, tag)

    records = []
    for scenario in SCENARIOS:
        ct = None if scenario["collision"] is None else scenario["collision"][0]
        baseline = runs[(scenario["baseline_ctrl"], "collision_detector", ct)]
        ours = runs[(scenario["ours_ctrl"], "my_detector", ct)]
        classification = None
        if ct is not None:
            base_ref_result, base_ref_log = runs[(scenario["baseline_ctrl"], "collision_detector", None)]
            our_ref_result, our_ref_log = runs[(scenario["ours_ctrl"], "my_detector", None)]
            reference_base = base_ref_log["detect"]
            reference_ours = our_ref_log["detect"]
            classification = {
                "baseline": classify_collision(baseline[0], reference_base, ct),
                "ours": classify_collision(ours[0], reference_ours, ct),
            }
            expected = (("alarm_before_collision", "detected", 69) if ct == 3.86
                        else ("detected", "detected", 83))
            got = (classification["baseline"]["collision_outcome"],
                   classification["ours"]["collision_outcome"],
                   round(classification["ours"]["detect_delay_ms"] or -1))
            if got != expected:
                raise AssertionError(f"{scenario['stem']} classification expected {expected}, got {got}")
            if ct == 3.0 and round(classification["baseline"]["detect_delay_ms"] or -1) != 68:
                raise AssertionError("transport baseline expected 68 ms")
            exact = ((True, None, 6, 69, 0) if ct == 3.86 else
                     (False, 68, 8, 83, 0))
            actual = (bool(baseline[0]["alarm_before_collision"]),
                      None if baseline[0]["detect_delay_ms"] is None else round(baseline[0]["detect_delay_ms"]),
                      int(baseline[0]["false_alarms"]), round(ours[0]["detect_delay_ms"] or -1),
                      int(ours[0]["false_alarms"]))
            if actual != exact:
                raise AssertionError(f"{scenario['stem']} expected exact {exact}, got {actual}")
            reference_info = {
                "baseline": {"controller": scenario["baseline_ctrl"], "detector": "collision_detector",
                             "box_mass_kg": MASS, "box_com_m": list(COM),
                             "false_alarm_rises": int(base_ref_result["false_alarms"])},
                "ours": {"controller": scenario["ours_ctrl"], "detector": "my_detector",
                         "box_mass_kg": MASS, "box_com_m": list(COM),
                         "false_alarm_rises": int(our_ref_result["false_alarms"])},
                "matching_rule": "same controller, detector, mass and COM; no collision",
            }
        else:
            reference_info = {
                "baseline": {"controller": scenario["baseline_ctrl"], "detector": "collision_detector",
                             "box_mass_kg": MASS, "box_com_m": list(COM),
                             "false_alarm_rises": int(baseline[0]["false_alarms"])},
                "ours": {"controller": scenario["ours_ctrl"], "detector": "my_detector",
                         "box_mass_kg": MASS, "box_com_m": list(COM),
                         "false_alarm_rises": int(ours[0]["false_alarms"])},
                "matching_rule": "this no-collision episode is the matching reference",
            }
        run_py_check = {label: run_py_checks[(scenario["stem"], label)]
                        for label in ("baseline", "ours")}
        records.append(render_scenario(scenario, baseline, ours, output, raw, ffmpeg,
                                       backend, classification, run_py_check, reference_info))

    integrity_after = check_integrity(root=ROOT)
    total_bytes = sum(a["bytes"] for record in records for a in record["artifacts"])
    if total_bytes >= 30 * 1024**2:
        raise AssertionError(f"media total exceeds 30 MiB: {total_bytes}")
    for record in records:
        for artifact in record["artifacts"]:
            limit = 5 if artifact["file"].endswith(".mp4") else 6
            if artifact["bytes"] >= limit * 1024**2:
                raise AssertionError(f"{artifact['file']} exceeds {limit} MiB")
    version = subprocess.check_output([str(ffmpeg), "-version"], text=True).splitlines()[0].split()[:3]
    manifest = {
        "schema_version": 1,
        "generation_command": "env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=glfw LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe __GLX_VENDOR_LIBRARY_NAME=mesa .venv/bin/python tools/render_comparison.py --output-dir artifacts/media --raw-dir \"$RAW_DIR\" --ffmpeg ffmpeg",
        "simulation_source": "eight fresh unmodified sim.runner.run_episode media episodes plus eight independent run.py verification episodes",
        "rendering_note": "qpos replay in separate MjData; physical geometry is actual scale; desired TCP marker radius is 15 mm; anchored 3D force arrow scale is 0.20 m per 15 N; separate depth-independent 2D direction legend is display-only, projects world +y through the camera, and is normalized to 85 px at 15 N",
        "force_plot_note": "detector joint residual converted offline with analysis-only true link q to bottom-centre wrench; never fed to control or detection",
        "ffmpeg": " ".join(version),
        "integrity_before": integrity_before,
        "integrity_after": integrity_after,
        "total_media_bytes": total_bytes,
        "scenarios": records,
    }
    manifest_path = output / "media_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": manifest_path.relative_to(ROOT).as_posix(),
                      "total_media_bytes": total_bytes, "backend": backend,
                      "scenarios": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
