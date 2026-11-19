"""Run the two self-contained submissions with a local kit, never in the original."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CURRENT_RELEASE = 'r10'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from tools.check_integrity import check_integrity
from tools.compare_baseline import official_cells


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_overlay(kit, overlay, prefix, mode, *, archived_r9=False):
    """Copy only from the local kit; verify copied bytes before any import."""
    for tree in ('sim', 'model'):
        if (kit / tree).is_symlink():
            raise ValueError(f'unexpected symlink: {tree}')
        shutil.copytree(kit / tree, overlay / tree, symlinks=True)
    (overlay / 'tools').mkdir()
    shutil.copyfile(ROOT / 'tools/kit_manifest.sha256', overlay / 'tools/kit_manifest.sha256')
    pins = json.loads((ROOT / 'results/v2_protocol/PROTOCOL_R7.json').read_text())['runtime_pins']
    if digest(overlay / 'tools/kit_manifest.sha256') != pins['tools/kit_manifest.sha256']:
        raise ValueError('original kit manifest differs')
    check_integrity(overlay)
    evaluator = overlay / 'evaluate.py'
    shutil.copyfile(kit / 'evaluate.py', evaluator)
    if digest(evaluator) != pins['evaluate.py']:
        raise ValueError('evaluate.py does not match the recorded kit')
    controller = overlay / f'{mode}_controller.py'
    detector = overlay / 'collision_detector.py'
    ctrl_source = ROOT / (f'r1_{mode}.py' if archived_r9 else f'{prefix}_{mode}.py')
    det_source = ROOT / ('experiments/r9_final/FAST2.py' if archived_r9 else f'{prefix}_detector.py')
    shutil.copyfile(ctrl_source, controller)
    shutil.copyfile(det_source, detector)
    return digest(controller), digest(detector)


def check_release(records, expected):
    if set(records) != {'servo', 'torque'} or set(expected) != set(records):
        raise ValueError('both modes are required')
    for mode, record in records.items():
        # Deliberately exclude host-dependent timing from the table comparison.
        for field in ('source_sha256', 'detector_sha256', 'kit_integrity',
                      'self_contained_default_names', 'episodes', 'control_and_collisions'):
            if record[field] != expected[mode][field]:
                raise ValueError(f'release differs: {mode}.{field}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit', type=Path, help='licensed icpbl_sim directory (default: this execution tree)')
    ap.add_argument('--prefix', default='my')
    ap.add_argument('--out', type=Path, required=True, help='new or empty output directory')
    ap.add_argument('--check-release', action='store_true',
                    help='require the selected release sources and all recorded non-timing cells')
    ap.add_argument('--archived-r9', action='store_true',
                    help='run the archived R1+FAST2 sources and compare to R9, not the live R10 release')
    args = ap.parse_args()
    kit = (args.kit or ROOT).resolve()
    out = args.out.resolve()
    if (not args.prefix.isidentifier() or (args.archived_r9 and args.prefix != 'my')
            or any(out.is_relative_to(kit / tree) for tree in ('sim', 'model'))
            or (args.kit and out.is_relative_to(kit))):
        ap.error('choose a module prefix and an output directory outside the input kit')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        ap.error('refusing to overwrite existing output')
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    records = {}
    for mode in ('servo', 'torque'):
        with tempfile.TemporaryDirectory(prefix='submission-', dir=out) as folder:
            overlay = Path(folder)
            controller, detector = prepare_overlay(kit, overlay, args.prefix, mode,
                                                  archived_r9=args.archived_r9)
            command = [sys.executable, 'evaluate.py', '--ctrl', mode + '_controller',
                       '--detector', 'collision_detector', '--out', 'evaluation.txt']
            result = subprocess.run(command, cwd=overlay, env=env, capture_output=True, text=True)
            (out / f'{mode}.log').write_text(result.stdout + result.stderr)
            result.check_returncode()
            report = (overlay / 'evaluation.txt').read_text()
            (out / f'{mode}.txt').write_text(report)
            control, collisions = official_cells(report)
            episodes = sum(line.strip().startswith('실행 중: 박스') for line in result.stdout.splitlines())
            if episodes != 29 or len(control) != 3 or len(collisions) != 6:
                raise ValueError(f'incomplete official output: {mode}')
            records[mode] = {
                'kit_integrity': check_integrity(overlay), 'source_sha256': controller,
                'detector_sha256': detector, 'episodes': episodes, 'self_contained_default_names': True,
                'control_and_collisions': {
                    'control': {str(m): cells for m, cells in control.items()},
                    'collisions': [dict(mass=m, time=t, cells=cells)
                                   for (m, t), cells in collisions.items()]}}
    (out / 'summary.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    release = 'r9' if args.archived_r9 else CURRENT_RELEASE
    if args.check_release:
        check_release(records, json.loads((ROOT / 'results' / release / 'official_regression.json').read_text()))
    print('Completed 29 episodes per mode.' + (f' {release.upper()} release match.' if args.check_release else ''))


if __name__ == '__main__':
    main()
