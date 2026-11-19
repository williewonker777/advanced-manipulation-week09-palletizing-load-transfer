"""Regenerate the 16-normal yaw envelopes, then compare seed-only copies of the course physics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.r18_seed_repeatability.kit import replace_seed
from tools.r13_grasp_audit import digest, install_snapshot
from tools.r21_replay import authenticate_kit, validate_paths
from tools.validate_submission import prepare_overlay

GROUP = 'experiments/r21_yaw_impulse/'
MOD = 'experiments.r21_yaw_impulse.'
FILES = ('my_torque.py', 'my_detector.py', 'experiments/r20_horizontal/candidate_detector.py',
         'experiments/r18_paired_impulses/runner.py', 'experiments/r21_weak_challenge/evaluator.py', *(
             GROUP + x for x in ('state.py', 'detector.py', 'frozen.py', 'coarse.py', 'calibrate.py',
                                 'coarse_calibration.py', 'plan.json', 'normal_bins.json', 'noise.py', 'noise_plan.json')))
MODEL_PIN = '5f22f640eb8824f764ffeec7ecbef7ba23df02a874547b93b8ddc72925e25c15'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args(argv)
    kit, out = validate_paths(a.kit, a.out)
    manifest = (ROOT/'tools/kit_manifest.sha256').read_bytes()
    original = authenticate_kit(kit, manifest)
    if any((ROOT/n).is_symlink() for n in FILES):
        raise ValueError('regular authored files required')
    pins = {n: digest(ROOT/n) for n in FILES}
    if pins[GROUP+'normal_bins.json'] != MODEL_PIN:
        raise ValueError('changed frozen model')
    plan = json.loads((ROOT/GROUP/'noise_plan.json').read_bytes())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MUJOCO_GL='disable')
    env.pop('PYTHONPATH', None)
    with tempfile.TemporaryDirectory(prefix='yaw-check-', dir=out.parent) as tmp:
        root = Path(tmp)

        def overlay(label):
            base = root/label
            base.mkdir()
            src, det = prepare_overlay(kit, base, 'my', 'torque')
            if (src, det) != (pins['my_torque.py'], pins['my_detector.py']):
                raise ValueError('R10 copy differs')
            for rel, pin in pins.items():
                target = base/rel
                target.parent.mkdir(parents=True, exist_ok=True)
                install_snapshot((ROOT/rel).read_bytes(), pin, target)
            return base

        def module(base, name, *options):
            subprocess.run([sys.executable, '-B', '-m', MOD+name, *options], cwd=base, env=env, check=True)

        base = overlay('original')
        module(base, 'calibrate', '--jobs', '2', '--out', 'global.json')
        module(base, 'coarse_calibration', '--jobs', '2', '--out', 'model.json')
        bounds = json.loads((base/'global.json').read_bytes())['bounds_Nm']
        if (digest(base/'model.json') != MODEL_PIN or bounds != [3.23779270273884, 5.111942835466052]):
            raise ValueError('original-kit normal calibration did not reproduce')

        def adjusted(seed):
            fixture = overlay('seed-'+str(seed))
            cfg = fixture/'sim/config.py'
            cfg.write_text(replace_seed(cfg.read_text(), seed))
            module(fixture, 'noise', '--seed', str(seed), '--out', 'data.json')
            return {'seed': seed, 'fixture_config_sha256': digest(cfg), **json.loads((fixture/'data.json').read_bytes())}

        with ThreadPoolExecutor(max_workers=2) as pool:
            seeds = list(pool.map(adjusted, plan['seeds']))
        if original != authenticate_kit(kit, manifest) or any(pins[k] != digest(ROOT/k) for k in pins):
            raise ValueError('inputs changed')
        rows = [r for s in seeds for r in s['rows']]
        result = {'schema': 'r21-yaw-normal-and-seeds.v1', 'global_normal_bounds_Nm': bounds,
                  'normal_episodes': 32, 'normal_bins_sha256': MODEL_PIN, 'authored_pins': pins,
                  'fixture': 'Only JTS_SEED differs inside each alternative physics copy; exposed geometry.',
                  'noise_episodes': sum(s['episodes'] for s in seeds), 'seeds': seeds,
                  'strict_of_16': {a: sum(r['scores'][a]['strict_true_positive'] for r in rows)
                                   for a in ('r20', 'yaw')}}
        with out.open('x') as fp:
            json.dump(result, fp, indent=2, allow_nan=False)
            fp.write('\n')
        print(result['strict_of_16'])


if __name__ == '__main__':
    main()
