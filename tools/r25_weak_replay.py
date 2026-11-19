"""Reproduce the *rejected* R25 weak-onset options, including the failures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import r25_replay as base


CANDIDATES = {"immediate": "experiments.r25_weak.normal_force_detector:Detector",
              "deferred": "experiments.r25_weak.normal_force_detector_v2:Detector"}
LOCK = "experiments/r25_weak/nearby_replay.json"


def inputs():
    if any((ROOT / name).exists() for name in ("sim", "model", "evaluate.py")):
        raise ValueError("use the kit-free public source snapshot")
    blobs, pins = base.source_blobs(base.closure())
    lock = json.loads((ROOT / LOCK).read_bytes())
    for name, h in lock["weak_source_pins"].items():
        if not base.relative_name(name) or not name.startswith("experiments/r25_weak/"):
            raise ValueError("invalid weak study member")
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("regular weak study member required")
        data = path.read_bytes()
        if h != base.digest(data):
            raise ValueError("weak study source changed: " + name)
        blobs[name], pins[name] = data, h
    return blobs, pins, lock["expected_raw_sha256"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--candidate", required=True, choices=tuple(CANDIDATES))
    p.add_argument("--workers", type=int, choices=(1, 2), default=1)
    q = p.parse_args()
    try:
        kit, out = base.shared.validate_paths(q.kit, q.out)
        blobs, pins, expected = inputs()
        manifest = (ROOT / base.shared.KIT_MANIFEST).read_bytes()
        base.shared.authenticate_kit(kit, manifest)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", MUJOCO_GL="disable")
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory(prefix="r25-weak-", dir=out.parent) as folder:
            temporary = Path(folder)
            base.shared.install_overlay(kit, temporary, manifest, blobs, pins)
            base.shared.authenticate_kit(temporary, manifest)
            completed = subprocess.run(
                [sys.executable, "-B", "-m", "experiments.r25_weak.evaluate_world_heldback",
                 "--registered", "experiments/r25_weak/PLAN.json",
                 "--r24-plan", "experiments/r24_holdout/plan.json",
                 "--candidate", CANDIDATES[q.candidate], "--workers", str(q.workers),
                 "--out", "raw.json"], cwd=temporary, env=env, check=False)
            if completed.returncode:
                raise RuntimeError("weak study process failed")
            result = (temporary / "raw.json").read_bytes()
            if base.digest(result) != expected[q.candidate]:
                raise ValueError("research result does not match the original failures")
            base.shared.authenticate_kit(kit, manifest)
            base.shared.authenticate_kit(temporary, manifest)
            for name, h in pins.items():
                if base.digest(ROOT / name) != h:
                    raise ValueError("authored source changed during replay: " + name)
            with out.open("xb") as stream:
                stream.write(result)
        print("Reproduced 32 development rows / 128 episodes. Candidate remains rejected.")
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
