"""Final integrity-gated official matrix, separate from historical ablations."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.check_integrity import check_integrity
from tools.compare_baseline import official_cells

COMBINATIONS = {
    "servo_base_base": ("servo_controller", "collision_detector"),
    "servo_base_final": ("servo_controller", "my_detector"),
    "servo_final_base": ("my_servo", "collision_detector"),
    "servo_final_final": ("my_servo", "my_detector"),
    "torque_base_base": ("torque_controller", "collision_detector"),
    "torque_base_final": ("torque_controller", "my_detector"),
    "torque_final_base": ("my_torque", "collision_detector"),
    "torque_final_final": ("my_torque", "my_detector"),
}


def hashes():
    return {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
            for n in ("my_servo.py", "my_torque.py", "my_detector.py")}


def memory_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError('MemAvailable unavailable')


def run_one(item, evidence):
    key, (controller, detector) = item
    check_integrity(ROOT)
    command = [sys.executable, 'tools/evaluate_checked.py', '--ctrl', controller,
               '--detector', detector, '--out', f'results/evaluation_final_{key}.txt']
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    with (evidence/f'{key}.log').open('w') as out:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=out,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'Official evaluator failed: {key}')
    return key


def collect():
    data = {}
    for key, (ctrl, det) in COMBINATIONS.items():
        control, collision = official_cells((ROOT/f'results/evaluation_final_{key}.txt').read_text())
        data[key] = {"controller": ctrl, "detector": det,
                     "control": {str(m): dict(rms_mm=float(c[0]), max_mm=float(c[1]),
                         saturation_pct=float(c[2]), pick_nm=float(c[3]), place_nm=float(c[4]),
                         table_n=float(c[5]), false_alarms=[int(x) for x in c[6:9]], status=c[9])
                                 for m, c in control.items()},
                     "collisions": [{"mass": m, "time": t, "10N": c[0], "15N": c[1],
                                      "40N": c[2], "100N": c[3], "false_max": c[4],
                                      "minimum": c[5]} for (m, t), c in collision.items()]}
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--evidence-dir', type=Path)
    args = parser.parse_args()
    if not args.run:
        parser.error('--run is required: collect-only cannot authenticate stale evaluation text')
    frozen = hashes()
    memory = memory_gib()
    start = time.perf_counter()
    if args.run:
        if memory < 6:
            raise SystemExit('RAM below 6 GiB')
        if args.evidence_dir is None:
            parser.error('--run requires --evidence-dir')
        args.evidence_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda item: run_one(item, args.evidence_dir), COMBINATIONS.items()))
        if hashes() != frozen:
            raise RuntimeError('production sources changed during official evaluation')
    data = {"official": True, "source": "evaluate_checked.py -> unchanged evaluate.py",
            "source_sha256": frozen, "mem_available_gib_before": memory,
            "elapsed_seconds": time.perf_counter()-start,
            "evaluation_sha256": {key: hashlib.sha256((ROOT/f'results/evaluation_final_{key}.txt').read_bytes()).hexdigest()
                                  for key in COMBINATIONS}, "combinations": collect()}
    (ROOT/'results/FINAL_OFFICIAL.json').write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    print('Final official combinations:', len(data['combinations']))


if __name__ == '__main__':
    main()
