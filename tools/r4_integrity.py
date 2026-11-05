"""Read-only source/environment/archive integrity against a local inventory."""
from pathlib import Path
from importlib import metadata
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_integrity import check_integrity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(root, hashes):
    return {name: (Path(root) / name).is_file() and sha(Path(root) / name) == digest
            for name, digest in hashes.items()}


def verify(inventory, original, root=ROOT):
    root, original = Path(root), Path(original)
    baseline = json.loads(Path(inventory).read_text())
    read = lambda name: json.loads((root / 'results/v2_protocol' / name).read_text())
    groups = {}
    for label, hashes, location in (
        ('R3', read('R3_FREEZE.json')['source_files'], root),
        ('R2', read('R2_FREEZE.json')['source_files'], root),
        ('original', baseline['original_inventory'], original),
    ):
        values = check(location, hashes)
        groups[label] = dict(files=len(values), unchanged=all(values.values()),
                             failed=[n for n, ok in values.items() if not ok])
    exact_original = {p.relative_to(original).as_posix() for p in original.rglob('*') if p.is_file()} == set(baseline['original_inventory'])
    preserved = {n: h for n, h in baseline['tracked_sha256'].items()
                 if (n.startswith(('docs/R1_', 'docs/R2_', 'docs/R3_', 'report/PRESENTATION_R2', 'report/PRESENTATION_R3'))
                     or n in ('results/presentation_R1.pdf', 'results/presentation_R2.pdf'))}
    archives = dict(check(root, preserved))
    archives.update({n: sha(root / n) == digest for n, digest in read('R4_BASELINE.json')['archives'].items()})
    packages = {d.metadata['Name']: d.version for d in metadata.distributions()}
    aliases = {mode: sha(root / ('my_' + mode + '.py')) == sha(root / ('r1_' + mode + '.py'))
               for mode in ('servo', 'torque', 'detector')}
    record = dict(schema='R4_INTEGRITY.v1', groups=groups, original_inventory_exact=exact_original,
                  protected=check_integrity(root), official_packages_count=len(packages),
                  official_packages_unchanged=packages == baseline['official_packages'],
                  archives=archives, final_R1_aliases=aliases)
    # A selected detector may be the sole mutable submission alias. Old R3
    # bytes are preserved in r1_detector and the baseline Git object. This
    # narrow exception never applies to another path or a rejected round.
    allowed = []
    selection = root / 'results/v2_protocol/R4_PREFREEZE_DECISION.json'
    if selection.exists() and any(v != 'R1' for v in json.loads(selection.read_text())['chosen'].values()):
        allowed = ['my_detector.py']
    record['declared_alias_exception'] = allowed
    record['passed'] = (all(set(g['failed']) <= set(allowed) for g in groups.values())
                        and exact_original and all(archives.values())
                        and record['official_packages_unchanged'] and aliases['servo'] and aliases['torque'])
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inventory', type=Path, required=True)
    ap.add_argument('--original', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists(): raise RuntimeError('retain every integrity checkpoint')
    value = verify(args.inventory, args.original)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0 if value['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
