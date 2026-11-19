"""Bit-exact parity of a three-file hand-in against R10-servo/R31-torque.

The submission arm runs from an authenticated course-kit copy with isolated
Python startup.  Consequently an undeclared import from this research tree
cannot make a broken hand-in appear valid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any
import zipfile

import numpy as np

import my_detector
import my_servo
import my_torque
from experiments.r21_full_protection import runner_v2 as raw
from experiments.r24_loaded.d4 import load_cases
from experiments.r31_held_identity.candidate import Detector as R31
from tools import r21_replay


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = frozenset(("my_servo.py", "my_torque.py", "my_detector.py"))
TRACE_KEYS = raw.PHYSICAL + ("r", "detect")
SCOPES = {
    # A clean trajectory and one ordinary collision per controller.
    "smoke": {"servo": (170, 189), "torque": (43, 266)},
    # Disclosed normal, weak/multiaxis/strong, pre/post release and offset rows.
    "representative": {
        "servo": (0, 170, 189, 266, 310, 377),
        "torque": (43, 170, 190, 197, 287, 310, 313, 314, 318, 377),
    },
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def inspect_submission(path: Path) -> dict[str, Any]:
    """Authenticate the exact user-facing assignment envelope."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular submission ZIP required")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if set(names) != EXPECTED or len(names) != len(EXPECTED):
            raise ValueError("exact three-file submission required")
        blobs = {}
        for item in infos:
            mode = item.external_attr >> 16
            if (Path(item.filename).name != item.filename or item.is_dir()
                    or (mode & 0o170000) == 0o120000):
                raise ValueError("regular root-level Python members required")
            blob = archive.read(item)
            compile(blob, item.filename, "exec")
            blobs[item.filename] = blob
        if archive.testzip() is not None:
            raise ValueError("corrupt submission ZIP")
    return {
        "zip_sha256": sha_bytes(path.read_bytes()),
        "members": sorted(blobs),
        "sha256": {name: sha_bytes(blob) for name, blob in sorted(blobs.items())},
        "blobs": blobs,
    }


def compare_traces(expected: dict[str, np.ndarray], actual: dict[str, np.ndarray]) -> dict[str, Any]:
    if set(expected) != set(actual):
        raise ValueError("trace keys differ")
    different = []
    hashes = {}
    for key in sorted(expected):
        one, two = np.asarray(expected[key]), np.asarray(actual[key])
        same = one.dtype == two.dtype and one.shape == two.shape and np.array_equal(one, two)
        if not same:
            different.append(key)
        hashes[key] = {
            "expected": raw.array_sha(one),
            "submission": raw.array_sha(two),
        }
    return {"bit_identical": not different, "different": different, "sha256": hashes}


class TracePlant(raw.stock_runner.Plant):
    latest = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signed = []
        type(self).latest = self

    def push(self, force):
        self.signed.append(np.asarray(force, float).copy())
        return super().push(force)


def reference_trace(mode: str, case: dict[str, Any]) -> dict[str, np.ndarray]:
    controller = my_servo.Controller() if mode == "servo" else my_torque.Controller()
    detector = my_detector.Detector() if mode == "servo" else R31()
    original = raw.stock_runner.Plant
    TracePlant.latest = None
    raw.stock_runner.Plant = TracePlant
    try:
        _, log = raw.stock_runner.run_episode(
            controller, box_mass=case["mass"], box_com=tuple(case["com"]),
            detector=detector, collision=case.get("collision"),
            pick_offset=case.get("pick_offset"))
    finally:
        raw.stock_runner.Plant = original
    if TracePlant.latest is None:
        raise RuntimeError("reference Plant was not instrumented")
    return {**{key: np.asarray(log[key]) for key in TRACE_KEYS},
            "signed_push": np.asarray(TracePlant.latest.signed)}


WORKER = r'''import json,sys
from pathlib import Path
import numpy as np
from sim import runner
import my_detector
import my_servo
import my_torque

case=json.loads(Path(sys.argv[1]).read_text())
mode=sys.argv[2]
out=Path(sys.argv[3])
class TracePlant(runner.Plant):
    latest=None
    def __init__(self,*a,**k):
        super().__init__(*a,**k);self.signed=[];type(self).latest=self
    def push(self,force):
        self.signed.append(np.asarray(force,float).copy());return super().push(force)
old=runner.Plant;runner.Plant=TracePlant
try:
    ctrl=my_servo.Controller() if mode=='servo' else my_torque.Controller()
    metrics,log=runner.run_episode(ctrl,box_mass=case['mass'],box_com=tuple(case['com']),
        detector=my_detector.Detector(),collision=case.get('collision'),pick_offset=case.get('pick_offset'))
finally:runner.Plant=old
if TracePlant.latest is None:raise RuntimeError('submission Plant was not instrumented')
keys=('t','q','q_d','tau','e_mm','f_table','f_tool','qpos','f_ext','r','detect')
np.savez(out,**{k:np.asarray(log[k]) for k in keys},signed_push=np.asarray(TracePlant.latest.signed))
'''


def kit_digest(kit: Path) -> dict[str, str]:
    manifest = (ROOT / r21_replay.KIT_MANIFEST).read_bytes()
    return r21_replay.authenticate_kit(kit, manifest)


def submission_trace(blobs: dict[str, bytes], kit: Path, mode: str,
                     case: dict[str, Any], temporary_root: Path) -> dict[str, np.ndarray]:
    with TemporaryDirectory(prefix="assignment-parity-", dir=temporary_root) as folder:
        work = Path(folder)
        for tree in ("sim", "model"):
            shutil.copytree(kit / tree, work / tree, symlinks=False)
        shutil.copyfile(kit / "evaluate.py", work / "evaluate.py")
        for name, blob in blobs.items():
            (work / name).write_bytes(blob)
        (work / "worker.py").write_text(WORKER)
        case_file, output = work / "case.json", work / "trace.npz"
        case_file.write_text(json.dumps(case, sort_keys=True, allow_nan=False))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   MUJOCO_GL="disable")
        env.pop("PYTHONPATH", None)
        isolated = ("import runpy,sys;sys.path.insert(0,'.');"
                    "sys.argv=['worker.py',*sys.argv[1:]];"
                    "runpy.run_path('worker.py',run_name='__main__')")
        result = subprocess.run(
            [sys.executable, "-I", "-c", isolated, str(case_file), mode, str(output)],
            cwd=work, env=env, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("isolated submission failed: " + (result.stderr or result.stdout)[-1200:])
        with np.load(output) as archive:
            return {key: archive[key].copy() for key in archive.files}


def run(package: Path, kit: Path, scope: str, out: Path, temporary_root: Path) -> dict[str, Any]:
    submission = inspect_submission(package)
    kit = kit.resolve()
    before = kit_digest(kit)
    cases = load_cases()
    rows = []
    for mode, ids in SCOPES[scope].items():
        for case_id in ids:
            case = cases[case_id]
            expected = reference_trace(mode, case)
            actual = submission_trace(submission["blobs"], kit, mode, case, temporary_root)
            comparison = compare_traces(expected, actual)
            rows.append({"mode": mode, "case_id": case_id, "kind": case["kind"],
                         "phase": case["phase"], "direction": case["direction"],
                         "force_n": case["force_n"], **comparison})
    after = kit_digest(kit)
    if before != after:
        raise ValueError("course kit changed")
    result = {
        "schema": "assignment-r10-servo-r31-torque-parity.v1",
        "scope": scope,
        "submission": {key: value for key, value in submission.items() if key != "blobs"},
        "reference": {"servo": "R10", "torque": "R31 held-identity"},
        "actual_simulation_episodes": 2 * len(rows),
        "package_episodes": len(rows),
        "reference_episodes": len(rows),
        "all_bit_identical": all(row["bit_identical"] for row in rows),
        "different_rows": [row["case_id"] for row in rows if not row["bit_identical"]],
        "kit_unchanged": True,
        "kit_files": before,
        "rows": rows,
        "limits": "Disclosed deterministic simulator rows; not independent or hardware validation.",
    }
    if out.exists() or out.is_symlink():
        raise ValueError("fresh output required")
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--scope", choices=SCOPES, default="smoke")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tmp", required=True, type=Path)
    args = parser.parse_args()
    if not args.tmp.is_dir() or any(args.out.is_relative_to(args.kit / part) for part in ("sim", "model")):
        parser.error("existing temporary root and external result required")
    result = run(args.package, args.kit, args.scope, args.out, args.tmp)
    print(result["actual_simulation_episodes"], result["all_bit_identical"], result["different_rows"])


if __name__ == "__main__":
    main()
