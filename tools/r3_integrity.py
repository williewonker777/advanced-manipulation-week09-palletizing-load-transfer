"""Hash-only local integrity evidence; no installs, evaluation, or network."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
from importlib import metadata
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_integrity import check_integrity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_hashes(root, expected):
    return {name: (Path(root) / name).is_file() and sha(Path(root) / name) == digest
            for name, digest in expected.items()}


def verify(baseline_inventory, original_inventory, original_root):
    baseline = json.loads(Path(baseline_inventory).read_text())
    original = json.loads(Path(original_inventory).read_text())['original_files']
    r2 = json.loads((ROOT / 'results/v2_protocol/R2_FREEZE.json').read_text())['source_files']
    original_checks = check_hashes(original_root, original)
    actual_original = {p.relative_to(original_root).as_posix()
                       for p in original_root.rglob('*') if p.is_file()}
    old_source_checks = check_hashes(ROOT, r2)
    packages = {item.metadata['Name']: item.version for item in metadata.distributions()}
    retained = ('docs/R1_README.md', 'docs/R2_README.md', 'docs/R2_RESULTS.md',
                'docs/R2_METHODS.md', 'report/PRESENTATION_R2.md',
                'results/presentation_R1.pdf')
    # R2 archive was created from the comparison-start current PDF.
    archive = sha(ROOT / 'results/presentation_R2.pdf') == baseline['tracked_sha256'][
        'results/presentation.pdf']
    readme_archive = sha(ROOT / 'docs/R2_README.md') == baseline['tracked_sha256']['README.md']
    retained_checks = {name: sha(ROOT / name) == baseline['tracked_sha256'][name]
                       for name in retained if name in baseline['tracked_sha256']}
    result = dict(schema='R3_INTEGRITY.v1', checked_utc=datetime.now(timezone.utc).isoformat(),
                  protected=check_integrity(ROOT),
                  original_files=len(original_checks), original_unchanged=all(original_checks.values()),
                  original_inventory_exact=actual_original == set(original),
                  original_failed=[name for name, ok in original_checks.items() if not ok],
                  R2_frozen_files=len(old_source_checks), R2_frozen_unchanged=all(old_source_checks.values()),
                  R2_failed=[name for name, ok in old_source_checks.items() if not ok],
                  official_packages_count=len(packages),
                  official_packages_unchanged=packages == baseline['official_packages'],
                  retained_reports=retained_checks, R2_archive_pdf_unchanged=archive,
                  R2_archive_readme_unchanged=readme_archive,
                  final_R1_alias={mode: sha(ROOT / ('my_' + mode + '.py')) ==
                                  sha(ROOT / ('r1_' + mode + '.py'))
                                  for mode in ('servo', 'torque', 'detector')})
    result['passed'] = (result['original_unchanged'] and result['original_inventory_exact']
                        and result['R2_frozen_unchanged'] and result['official_packages_unchanged']
                        and all(retained_checks.values()) and archive and readme_archive
                        and all(result['final_R1_alias'].values()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-inventory', type=Path, required=True)
    parser.add_argument('--original-inventory', type=Path, required=True)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('retain every integrity checkpoint; choose a new output')
    result = verify(args.baseline_inventory, args.original_inventory, args.original_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
