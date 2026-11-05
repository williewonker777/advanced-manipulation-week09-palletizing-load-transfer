"""Add the R4 overlay to the preserved public baseline, with no network I/O."""
from pathlib import Path
import argparse
import hashlib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import build_public as B

BASELINE = '1531409ad06e43bee40c79065e523bc8bb084101'
DEST = ROOT / 'publish/repo'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checksum(root):
    root = Path(root)
    names = sorted(p.relative_to(root).as_posix() for p in root.rglob('*')
                   if p.is_file() and '.git' not in p.relative_to(root).parts
                   and p.relative_to(root).as_posix() != 'artifacts/SHA256SUMS')
    target = root / 'artifacts/SHA256SUMS'
    target.write_text(''.join(f'{sha((root / n).read_bytes())}  {n}\n' for n in names))
    return dict(files=len(names) + 1,
                bytes=sum((root / n).stat().st_size for n in names) + target.stat().st_size)


def curate(destination=DEST):
    destination = Path(destination)
    tracked = subprocess.check_output(['git', 'ls-tree', '-rz', '--name-only', BASELINE],
                                      cwd=destination).decode().split('\0')
    selected = {}
    for name in filter(None, tracked):
        if name == 'artifacts/SHA256SUMS': continue
        # Use the exact parent object, not a possibly edited staging worktree.
        selected[name] = subprocess.check_output(['git', 'show', BASELINE + ':' + name], cwd=destination)
    additions = {'README.md', 'my_servo.py', 'my_torque.py', 'my_detector.py',
                 'docs/R1_README.md', 'docs/R2_README.md',
                 'docs/R3_README.md', 'results/presentation_R3.pdf',
                 'results/presentation.pdf', 'report/PRESENTATION_R4.md'}
    for directory, patterns in (
        ('tools', ('r4_*.py', 'test_r4_*.py')),
        ('experiments', ('r4_*.py',)), ('experiments/r4_generated', ('*.py',)),
        ('experiments/r4_original', ('*.py',)),
        ('docs', ('R4_*.md',)),
        ('results/v2_protocol', ('R4*.json', 'PROTOCOL_R4*.json')),
        ('results/figures', ('r4_*.png',)),
    ):
        for pattern in patterns:
            additions.update(p.relative_to(ROOT).as_posix() for p in (ROOT / directory).glob(pattern))
    official = ROOT / 'results/v2_protocol/R4_OFFICIAL_DEFAULT'
    if official.is_dir():
        additions.update(p.relative_to(ROOT).as_posix() for p in official.iterdir() if p.suffix in ('.json', '.txt'))
    media = ROOT / 'artifacts/media'
    additions.update(p.relative_to(ROOT).as_posix() for p in media.glob('*r4*') if p.suffix in ('.json', '.gif', '.mp4'))
    for name in sorted(additions):
        path = ROOT / name
        if path.is_symlink() or not path.is_file(): raise ValueError('unsafe or missing overlay file: ' + name)
        data = path.read_bytes()
        if path.suffix == '.json': data = B.portable_json_bytes(data)
        selected[name] = data
    manifest = 'results/v2_protocol/R4_REPORT_MANIFEST.json'
    if manifest in selected:
        selected[manifest] = B.rebind_public_report_manifest(selected[manifest], selected)
    existing = {p.relative_to(destination).as_posix() for p in destination.rglob('*')
                if p.is_file() and '.git' not in p.relative_to(destination).parts}
    unexpected = existing - set(selected) - {'artifacts/SHA256SUMS'}
    if unexpected: raise ValueError('unexpected public overlay files: ' + repr(sorted(unexpected)))
    for name, data in selected.items():
        path = destination / name
        if path.is_symlink(): raise ValueError('unsafe destination symlink')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return checksum(destination)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checksums-only', action='store_true')
    ap.add_argument('--root', type=Path, default=DEST)
    args = ap.parse_args()
    print(checksum(args.root) if args.checksums_only else curate(args.root))


if __name__ == '__main__': main()
