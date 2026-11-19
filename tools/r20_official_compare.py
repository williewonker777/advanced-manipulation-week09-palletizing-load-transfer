"""Compare course defaults, R10 and the staged research pair on the original official table."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.r19_combined_gate import prepare_overlay as assembly
from tools.compare_baseline import official_cells
from tools.r13_grasp_audit import digest, install_snapshot
from tools.r15_early_check import validate_paths
from tools.r19_impulse_audit import PIN_ALGORITHM

STOCK = {
    'servo_controller.py': '50b5d2b05a45d540615a4af4373780d86872c60c8049c8af75dd6a4c4836e416',
    'torque_controller.py': 'a9d13fcddf89d9ab9bb92346bafe6d2d18b76cd5d605710cbdf3d9dddb8619d9',
    'collision_detector.py': 'fe4b1101ba41ffa46e04500f7153a0c569125b20083d8cd3a9fe6d8d69bda884',
}
POLICY = 'experiments.r20_official.policy'
MODES = ('servo', 'torque')
ARMS = ('stock', 'r10', 'combined')
FORCES = (10, 15, 40, 100)
TIMES = (1.42, 3.0, 3.86)
STATUS = {'놓침': 'missed', '경보 중': 'already_active', '구분 불가': 'indistinguishable'}


def modules(arm, mode):
    if mode not in MODES:
        raise ValueError('unknown controller mode')
    if arm == 'stock':
        return mode + '_controller', 'collision_detector'
    if arm == 'r10':
        return 'my_' + mode, 'my_detector'
    if arm == 'horizontal' and mode == 'torque':
        return 'my_torque', 'experiments.r20_horizontal.candidate_detector'
    if arm == 'combined':
        return (POLICY if mode == 'torque' else 'my_servo'), POLICY
    raise ValueError('unknown comparison arm or mode')


def table(report):
    controls, impacts = official_cells(report)
    keys = [(m, t) for m in (0., 8.) for t in TIMES]
    if (len(re.findall(r'^\s*\d+(?:\.\d+)?kg\s', report, re.M)) != 9
            or sorted(controls) != [0., 4., 8.] or sorted(impacts) != keys):
        raise ValueError('expected complete official grid')
    for cells in controls.values():
        if (len(cells) != 10 or cells[-1] not in ('정상', '발산') or
                any(not re.fullmatch(r'\d+(?:\.\d+)?', x) for x in cells[:6]) or
                any(not x.isdecimal() for x in cells[6:9])):
            raise ValueError('invalid official control row')
    for cells in impacts.values():
        if (len(cells) != 6 or any(not x.isdecimal() and x not in STATUS for x in cells[:4])
                or not re.fullmatch(r'\d+회', cells[4])):
            raise ValueError('invalid official collision row')
        minimum = next((f'{f} N' for f, v in zip(FORCES, cells) if v.isdecimal()), '—')
        if cells[5] != minimum:
            raise ValueError('inconsistent minimum detected force')
    return {'control': {str(m): controls[m] for m in sorted(controls)},
            'collisions': [{'mass': m, 'time': t, 'cells': impacts[m, t]} for m, t in keys]}


def summarize(cells):
    counts = dict.fromkeys(('detected', *STATUS.values()), 0)
    for row in cells['collisions']:
        for value in row['cells'][:4]:
            counts[STATUS.get(value, 'detected')] += 1
    return {'impact_outcomes': counts, 'no_impact_false_rises': sum(
        sum(map(int, row[6:9])) for row in cells['control'].values()),
        # evaluate.py reports a maximum, not a sum, over the four forces in each row.
        'worst_false_count_in_an_impact_run': max(int(r['cells'][4][:-1]) for r in cells['collisions'])}


def compare(reference, candidate):
    changes = []
    for old, new in zip(reference['collisions'], candidate['collisions'], strict=True):
        if (old['mass'], old['time']) != (new['mass'], new['time']):
            raise ValueError('mismatched grid')
        for force, a, b in zip(FORCES, old['cells'][:4], new['cells'][:4], strict=True):
            if a != b:
                changes.append({'mass_kg': old['mass'], 'time_s': old['time'], 'force_N': force,
                                'reference': a, 'candidate': b})
    return changes


def run_pair(base, arm, mode, env):
    ctrl, det = modules(arm, mode)
    dest = base / (arm + '-' + mode + '.txt')
    process = subprocess.run([sys.executable, '-B', 'evaluate.py', '--ctrl', ctrl,
                              '--detector', det, '--out', dest.name], cwd=base, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode:
        raise RuntimeError(f'{arm}/{mode}: {process.stderr.strip()}')
    episodes = sum(line.strip().startswith('실행 중: 박스') for line in process.stdout.splitlines())
    if episodes != 29:
        raise ValueError(f'incomplete {arm}/{mode}: {episodes}/29')
    cells = table(dest.read_text(encoding='utf-8'))
    return {'modules': [ctrl, det], 'episodes': episodes, 'cells': cells, **summarize(cells)}


def verify_runs(runs, release):
    for mode in MODES:
        if runs['stock'][mode]['cells'] != table((ROOT / f'results/evaluation_baseline_{mode}.txt').read_text()):
            raise ValueError('original stock control did not reproduce: ' + mode)
        if runs['r10'][mode]['cells'] != release[mode]['control_and_collisions']:
            raise ValueError('protected R10 control did not reproduce: ' + mode)
    candidate, reference = runs['combined'], runs['r10']
    if (candidate['servo']['cells'] != reference['servo']['cells'] or
            candidate['torque']['cells']['control']['0.0'] != reference['torque']['cells']['control']['0.0'] or
            candidate['torque']['cells']['collisions'][:3] != reference['torque']['cells']['collisions'][:3]):
        raise ValueError('out-of-domain/servo fallback changed a displayed non-clock cell')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', type=Path, required=True)
    parser.add_argument('--height-calibration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='new external JSON file')
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    try:
        kit, height, out = validate_paths(args.kit, args.height_calibration, args.out)
        release = json.loads((ROOT / 'results/r10/official_regression.json').read_bytes())
        with tempfile.TemporaryDirectory(prefix='course-matrix-', dir=out.parent) as tmp:
            base, receipt = Path(tmp) / 'overlay', Path(tmp) / 'receipt.json'
            assembly.main(['--kit', str(kit), '--height', str(height), '--early',
                           str(ROOT / 'results/r15/early_stock_calibration.json'),
                           '--overlay', str(base), '--auth', str(receipt)])
            auth = json.loads(receipt.read_bytes())
            pins = auth['source_pins']
            # Restore authenticated stock defaults before running the original evaluator.
            for rel, expected in STOCK.items():
                source = kit / rel
                if source.is_symlink() or not source.is_file() or digest(source) != expected:
                    raise ValueError('original stock module required: ' + rel)
                install_snapshot(source.read_bytes(), expected, base / rel)
            for rel in ('my_servo.py', 'my_torque.py', 'experiments/r19_impulse_end/__init__.py',
                        'experiments/r19_impulse_end/state.py', 'experiments/r19_impulse_end/detector.py',
                        'experiments/r20_official/__init__.py', 'experiments/r20_official/policy.py'):
                source = ROOT / rel
                if source.is_symlink() or not source.is_file():
                    raise ValueError('regular source required: ' + rel)
                pins[rel] = digest(source)
                target = base / rel
                target.parent.mkdir(exist_ok=True, parents=True)
                install_snapshot(source.read_bytes(), pins[rel], target)
            for rel, expected in PIN_ALGORITHM.items():
                if rel.startswith('experiments/r19_impulse_end/') and pins[rel] != expected:
                    raise ValueError('impulse source does not match frozen R19')
            for mode in MODES:
                if (pins[f'my_{mode}.py'] != release[mode]['source_sha256'] or
                        pins['my_detector.py'] != release[mode]['detector_sha256']):
                    raise ValueError('R10 submission differs')
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='disable')
            env.pop('PYTHONPATH', None)
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                pending = {(arm, mode): pool.submit(run_pair, base, arm, mode, env)
                           for arm in ARMS for mode in MODES}
                runs = {arm: {mode: pending[arm, mode].result() for mode in MODES} for arm in ARMS}
            verify_runs(runs, release)
            if (any(digest(ROOT / rel) != pin or digest(base / rel) != pin for rel, pin in pins.items())
                    or any(digest(base / rel) != pin or digest(kit / rel) != pin for rel, pin in STOCK.items())
                    or digest(base / 'evaluate.py') != auth['evaluator_sha256']):
                raise ValueError('source changed during evaluation')
        doc = {'schema': 'r20-original-official-matrix.v1', 'original_demo': False,
               'previously_exposed_grid': True, 'promoted': False, 'timing_included': False,
               'evaluator_sha256': auth['evaluator_sha256'], 'kit_manifest_sha256': auth['kit_manifest_sha256'],
               'stock_module_sha256': STOCK, 'research_source_sha256': pins,
               'height_source_sha256': auth['height_source_sha256'], 'early_source_sha256': auth['early_source_sha256'],
               'runs': runs, 'r10_to_combined_collision_changes': {m: compare(runs['r10'][m]['cells'],
                    runs['combined'][m]['cells']) for m in MODES}}
        with out.open('x', encoding='utf-8') as fp:
            json.dump(doc, fp, ensure_ascii=False, indent=2, allow_nan=False)
            fp.write('\n')
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({a: {m: runs[a][m]['impact_outcomes'] for m in MODES} for a in ARMS}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
