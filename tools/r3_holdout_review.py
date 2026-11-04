"""Pure post-holdout review and immutable effective-source selection."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import r3_extended as E
from tools import r3_official_regression as O
from tools import r3_select as S
from tools import r3_study as R

D = ROOT / 'results/v2_protocol'


def compose(freeze, reviews, hashes):
    if set(reviews) != {'servo', 'torque'} or any(
            row.get('status') not in ('PASS', 'FAIL') for row in reviews.values()):
        raise ValueError('both independently reviewed modes are required')
    chosen = {spec['architecture']: spec for spec in freeze['chosen_specs']}
    return dict(schema='R3_HOLDOUT_REVIEW.v1',
                recorded_utc=datetime.now(timezone.utc).isoformat(), **hashes,
                no_tuning_permitted=True, no_runner_up_or_diagnostic_revival=True,
                architectures={mode: dict(status=review['status'], review=review,
                    effective_selection=chosen[mode] if review['status'] == 'PASS'
                    else R.baseline_spec(mode)) for mode, review in reviews.items()})


def build():
    if (D / 'R3_OFFICIAL_OPENED.json').exists():
        raise RuntimeError('never amend holdout review after official exposure')
    freeze = json.loads(E.FREEZE.read_text())
    E.verify_freeze(freeze['chosen_specs'])
    hashes = dict(freeze_sha256=R.sha(E.FREEZE), holdout_sha256=R.sha(E.HOLDOUT_RESULT),
                  protocol_sha256=R.sha(R.PROTOCOL), opened_marker_sha256=R.sha(E.HOLDOUT_MARKER))
    opened = json.loads(E.HOLDOUT_MARKER.read_text())
    if (opened.get('batch_attempt') != 1 or opened.get('chosen_specs') != freeze['chosen_specs']
            or opened.get('local_commit') != freeze['local_commit']
            or opened.get('protocol_sha256') != hashes['protocol_sha256']):
        raise RuntimeError('opened marker and frozen source differ')
    holdout = json.loads(E.HOLDOUT_RESULT.read_text())
    completed = O._holdout_complete(holdout, freeze, hashes['freeze_sha256'],
                                    hashes['protocol_sha256'], hashes['opened_marker_sha256'])
    return compose(freeze, completed['reviews'], hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=D / 'R3_HOLDOUT_REVIEW.json')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('holdout review is one-shot; do not overwrite')
    result = build()
    S._atomic_write(args.output, result)
    print({mode: row['status'] for mode, row in result['architectures'].items()})


if __name__ == '__main__':
    main()
