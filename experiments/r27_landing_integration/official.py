"""Run authentic stock evaluate.py for frozen C1 and the R27 landing pilot."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from tools.r20_official_compare import summarize, table


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EVALUATOR = "ee7effc27ac0006c7d2f392523e3e1e4a45626d1113116cba40e00978bd2786f"
FROZEN = {
    "plan.json": "4d37047e107c9881f09ba90ecca55fc263ad0bc5338485852b8c10c4fed79c8b",
    "model.json": "447660558c5ba67ea65e53be568e493bdd5f42532b558f440884b3fb610b5939",
    "candidate.py": "ffde18b42b8ccd8250eae184147f7e498003356cfdb1a7cd46ccc5edfce0fb95",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(item):
    name, detector, folder, env = item
    target = folder / (name + ".txt")
    completed = subprocess.run(
        [sys.executable, "-B", "evaluate.py", "--ctrl",
         "experiments.r26_delivery_pair.controller", "--detector", detector,
         "--out", str(target)], cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode or sum(line.strip().startswith("실행 중: 박스")
                                   for line in completed.stdout.splitlines()) != 29:
        raise RuntimeError(completed.stdout + "\n" + completed.stderr)
    cells = table(target.read_text())
    return name, {"detector": detector, "episodes": 29, "cells": cells,
                  "counts": summarize(cells),
                  "stdout_sha256": hashlib.sha256(
                      completed.stdout.encode()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    import mujoco
    if mujoco.__version__ != "3.17.0" or sha(ROOT / "evaluate.py") != EVALUATOR:
        raise ValueError("authentic MuJoCo 3.17/evaluator required")
    if any(sha(HERE / name) != expected for name, expected in FROZEN.items()):
        raise ValueError("frozen candidate/model/plan changed")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MUJOCO_GL="disable")
    env.pop("PYTHONPATH", None)
    arms = {
        "c1": "experiments.r26_delivery_pair.candidate",
        "r27": "experiments.r27_landing_integration.candidate",
    }
    args.out.mkdir()
    with tempfile.TemporaryDirectory(dir=args.out) as temp:
        jobs = [(name, detector, Path(temp), env) for name, detector in arms.items()]
        with ThreadPoolExecutor(2) as pool:
            records = dict(pool.map(score, jobs))
    doc = {
        "schema": "r27-landing-authentic-official-comparison.v1",
        "classification": "Post-disclosure authentic official development sweep; not blind.",
        "runtime": {"mujoco": mujoco.__version__,
                    "evaluator_sha256": EVALUATOR},
        "sources": dict(FROZEN), "episodes": 58, "records": records,
        "same_control_cells": (records["c1"]["cells"]["control"] ==
                               records["r27"]["cells"]["control"]),
    }
    (args.out / "result.json").write_text(
        json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": doc["episodes"],
                      "same_control_cells": doc["same_control_cells"],
                      "counts": {name: record["counts"]
                                 for name, record in records.items()}}, indent=2))


if __name__ == "__main__":
    main()
