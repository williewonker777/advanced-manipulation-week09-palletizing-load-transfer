"""Bit-exact parity of the final hand-in against the R32 (servo) and R31 (torque) modules.

Reuses ``experiments/assignment_combined_qa/parity.py``: the reference runs in-process
from this repository's research modules, the hand-in runs in an isolated copy of the
licensed kit (``python -I``).  Physical arrays, residual ``r``, ``detect`` and the
signed force pushes must be bit-identical.  Usage::

    # inside a kit overlay (repository copy + licensed sim/, model/, evaluate.py)
    python -B tools/parity_final_submission.py --package <final zip> --kit <icpbl_sim> \
        --out <new result.json> --tmp <existing temporary directory>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def load():
    """Import the reference modules.  Like the other research reproduction tools this
    needs a kit overlay: a copy of this repository with the licensed kit's ``sim/``,
    ``model/`` and ``evaluate.py`` added (they are not distributed here)."""
    if not (ROOT / 'sim').is_dir() or not (ROOT / 'model').is_dir():
        raise SystemExit('run inside a kit overlay: copy this repository and add the licensed '
                         "kit's sim/, model/ and evaluate.py")
    from experiments.assignment_combined_qa import parity
    from experiments.r32_servo_ownership.candidate import Detector as R32
    # Reference servo detector is R32 (the base tool used R10); torque stays R31.
    parity.my_detector = SimpleNamespace(Detector=R32)
    return parity


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--kit', type=Path, required=True)
    ap.add_argument('--scope', choices=('smoke', 'representative'), default='representative')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tmp', type=Path, required=True)
    a = ap.parse_args()
    p = load()
    if a.out.exists():
        raise SystemExit('fresh output required')
    sub = p.inspect_submission(a.package.resolve())
    kit = a.kit.resolve()
    before = p.kit_digest(kit)
    cases = p.load_cases()
    rows = []
    for mode, ids in p.SCOPES[a.scope].items():
        for cid in ids:
            case = cases[cid]
            cmp = p.compare_traces(p.reference_trace(mode, case),
                                   p.submission_trace(sub['blobs'], kit, mode, case, a.tmp.resolve()))
            rows.append({'mode': mode, 'case_id': cid, 'kind': case['kind'], 'phase': case['phase'],
                         'direction': case['direction'], 'force_n': case['force_n'], **cmp})
            print(mode, cid, case['kind'], 'bit_identical' if cmp['bit_identical'] else 'DIFF ' + str(cmp['different']), flush=True)
    if p.kit_digest(kit) != before:
        raise SystemExit('course kit changed')
    result = {'schema': 'week09-final-r32-servo-r31-torque-parity.v1', 'scope': a.scope,
              'submission': {k: v for k, v in sub.items() if k != 'blobs'},
              'reference': {'servo': 'R32 servo ownership (experiments.r32_servo_ownership.candidate)',
                            'torque': 'R31 held identity (experiments.r31_held_identity.candidate)'},
              'episodes_each_side': len(rows),
              'all_bit_identical': all(r['bit_identical'] for r in rows),
              'different_rows': [r['case_id'] for r in rows if not r['bit_identical']],
              'kit_unchanged': True, 'rows': rows,
              'limits': 'Disclosed deterministic simulator rows; not independent or hardware validation.'}
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print('ALL_BIT_IDENTICAL' if result['all_bit_identical'] else 'MISMATCH', result['different_rows'])


if __name__ == '__main__':
    main()
