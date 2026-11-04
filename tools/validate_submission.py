"""Exercise self-contained entries under the kit's default module names, offline."""
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
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from tools.check_integrity import check_integrity
from tools.compare_baseline import official_cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', default='my')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    check_integrity(ROOT)
    args.out.mkdir(parents=True, exist_ok=True)
    records = {}
    for mode, default in [('servo', 'servo_controller'), ('torque', 'torque_controller')]:
        with tempfile.TemporaryDirectory(prefix='submission-', dir=ROOT/'publish') as name:
            overlay = Path(name)
            for tree in ('sim', 'model'):
                shutil.copytree(ROOT/tree, overlay/tree)
            shutil.copyfile(ROOT/'evaluate.py', overlay/'evaluate.py')
            (overlay/'tools').mkdir()
            shutil.copyfile(ROOT/'tools/kit_manifest.sha256', overlay/'tools/kit_manifest.sha256')
            for source, target in [(f'{args.prefix}_{mode}.py', f'{default}.py'),
                                   (f'{args.prefix}_detector.py', 'collision_detector.py')]:
                shutil.copyfile(ROOT/source, overlay/target)
            env = os.environ.copy()
            env.pop('PYTHONPATH', None)
            env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            command = [sys.executable, 'evaluate.py', '--ctrl', default, '--out', 'evaluation.txt']
            r = subprocess.run(command, cwd=overlay, env=env, capture_output=True, text=True)
            (args.out/f'{mode}.log').write_text(r.stdout+r.stderr)
            r.check_returncode()
            content = (overlay/'evaluation.txt').read_text()
            (args.out/f'{mode}.txt').write_text(content)
            records[mode] = {'kit_integrity': check_integrity(overlay),
                             'source_sha256': hashlib.sha256((ROOT/f'{args.prefix}_{mode}.py').read_bytes()).hexdigest(),
                             'self_contained_default_names': True,
                             'control_and_collisions': {'control': official_cells(content)[0], 'collisions': [dict(mass=k[0], time=k[1], cells=v) for k, v in official_cells(content)[1].items()]}}
    (args.out/'summary.json').write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')
    print('Default filename evaluations passed:', list(records))


if __name__ == '__main__':
    main()
