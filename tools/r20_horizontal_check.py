"""Recheck early horizontal force detection using a local course kit, not research control."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.r13_grasp_audit import digest, install_snapshot, validate_paths
from tools.r15_early_check import has_symlink_component
from tools.r20_official_compare import compare, run_pair
from tools.validate_submission import prepare_overlay

GROUP = 'experiments/r20_horizontal/'
PIN = {'candidate_detector.py': '1770c36e002404a660c9cfc1ef45ddbb6cb1f53a3ec973c82b760e60881e7025',
       'FRESH_PLAN.json': '6fcfb117da8ca8d106a02bcec0c336ac7d316440e9e0da278223d6954927fa6d'}


def calibrated_bound(normal):
    rows = normal['rows']
    if (len(rows) != 20 or len({(r['mass'], tuple(r['com'])) for r in rows}) != 20 or
            any(len(r['max']['early']['residual_xy']) != 2 for r in rows)):
        raise ValueError('incomplete normal-only grid')
    values = [v for row in rows for v in row['max']['early']['residual_xy']]
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('invalid horizontal residual')
    return 1.02 * max(values)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    try:
        kit, out = validate_paths(a.kit, a.out)
        if has_symlink_component(a.out) or out.is_relative_to(ROOT):
            raise ValueError('output required outside both source and kit')
        release = json.loads((ROOT / 'results/r10/official_regression.json').read_bytes())['torque']
        with tempfile.TemporaryDirectory(prefix='horizontal-', dir=out.parent) as tmp:
            base = Path(tmp)
            c, d = prepare_overlay(kit, base, 'my', 'torque')
            if (c, d) != (release['source_sha256'], release['detector_sha256']):
                raise ValueError('changed R10 reference pair')
            expected = {'my_torque.py': c, 'my_detector.py': d}
            for name in ('__init__.py', 'calibrate_normal.py', 'trace_detector.py', 'fresh.py', *PIN):
                expected[GROUP + name] = PIN.get(name) or digest(ROOT / GROUP / name)
            for rel, pin in expected.items():
                f, target = ROOT / rel, base / rel
                if f.is_symlink() or not f.is_file():
                    raise ValueError('regular authored source required')
                target.parent.mkdir(parents=True, exist_ok=True)
                install_snapshot(f.read_bytes(), pin, target)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='disable')
            env.pop('PYTHONPATH', None)

            def run_module(name, *flags):
                subprocess.run([sys.executable, '-B', '-m', 'experiments.r20_horizontal.' + name, *flags],
                               cwd=base, env=env, check=True)

            run_module('calibrate_normal', '--out', 'normal.json')
            bound = calibrated_bound(json.loads((base / 'normal.json').read_bytes()))
            if not math.isclose(bound, 18.3753578739, rel_tol=0, abs_tol=5e-10):
                raise ValueError('frozen normal-only bound does not reproduce')
            baseline = run_pair(base, 'r10', 'torque', env)
            new = run_pair(base, 'horizontal', 'torque', env)
            if baseline['cells'] != release['control_and_collisions']:
                raise ValueError('official R10 did not reproduce')
            run_module('fresh', '--plan', GROUP + 'FRESH_PLAN.json', '--out', 'fresh.json')
            fresh = json.loads((base / 'fresh.json').read_bytes())
            if (fresh['plan_sha256'] != PIN['FRESH_PLAN.json'] or
                    fresh['candidate_sha256'] != PIN['candidate_detector.py'] or
                    len(fresh['cases']) != 6 or fresh['passed'] is not all(fresh['gates'].values())):
                raise ValueError('incomplete independent evidence')
            if any(digest(ROOT / rel) != v or digest(base / rel) != v for rel, v in expected.items()):
                raise ValueError('source changed during evaluation')
            evidence = {'schema': 'r20-horizontal-check.v1', 'original_demo': False,
                        'submission_changed': False, 'episodes': {'normal_calibration': 20, 'official': 58,
                        'post_freeze': 24}, 'evaluator_sha256': digest(base / 'evaluate.py'),
                        'source_sha256': expected, 'calibrated_threshold_N': bound, 'r10': baseline,
                        'candidate': new, 'official_changes': compare(baseline['cells'], new['cells']),
                        'fresh': fresh}
        with out.open('x', encoding='utf-8') as f:
            json.dump(evidence, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
    except (KeyError, ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        p.error(str(exc))
    print('Official recognized collisions:', baseline['impact_outcomes']['detected'], '->',
          new['impact_outcomes']['detected'], '/ 24; fresh retention:', fresh['passed'])


if __name__ == '__main__':
    main()
