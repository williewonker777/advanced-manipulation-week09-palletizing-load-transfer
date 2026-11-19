"""Verify branch/residual/alarm bytes on the entire archived R8 holdout.

Inputs are licensed/private R8 traces, not a new holdout. No hidden trace
fields reach the detector/controller. Timing is deliberately not measured.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.check_integrity import check_integrity
from tools.r9_timing import module


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(root, mode, variant, index, case, design_sha):
    path = root / 'holdout' / mode / variant / f'{index:03d}.json'
    rec = json.loads(path.read_text())
    raw = path.with_suffix('.npz')
    if (rec['case'] != case or rec['status'] != 'success' or rec['variant'] != variant
            or rec['mode'] != mode or rec['plan_sha256'] != design_sha
            or rec['raw_file'] != raw.name or sha(raw) != rec['raw_sha256']):
        raise ValueError(f'unbound archived trace: {index}/{variant}')
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', default='servo', choices=('servo', 'torque'))
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--candidate', type=Path, default=ROOT / 'my_detector.py')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    check_integrity(ROOT)
    controller = module(ROOT / f'r1_{args.mode}.py').Controller
    from sim.drive import Drive
    from sim.nominal import Nominal
    from sim.trajectory import reference

    cls = module(args.candidate).Detector
    review = json.loads((ROOT / 'results/v2_protocol/R8_HOLDOUT_REVIEW.json').read_text())
    design = args.archive / 'HOLDOUT_DESIGN.json'
    design_sha = sha(design)
    cases = json.loads(design.read_text())['cases']
    if design_sha != review['design_sha256'] or len(cases) != 114:
        raise ValueError('expected complete historical R8 holdout design')
    rows = []
    for i, case in enumerate(cases):
        reference_path = record(args.archive, args.mode, 'C1', i, case, design_sha)
        base_path = record(args.archive, args.mode, 'fallback', i, case, design_sha)
        with np.load(reference_path, allow_pickle=False) as z:
            t, th, thd, tm, ts, grip, expected_r, expected_hits = (
                z[k] for k in ('t', 'th', 'thd', 'tau_m', 'tau_s', 'grip', 'physical_r', 'detect'))
        with np.load(base_path, allow_pickle=False) as z:
            expected_base = z['detect']
        ctrl, det = controller(), cls()
        det.ctrl = ctrl
        if args.mode == "servo":
            ctrl.drive = Drive()
        for obj in (ctrl, det):
            obj.nom, obj.nom_box = Nominal(), Nominal()
            obj.box_mass, obj.box_com = case['mass'], tuple(case['com'])
            obj.nom_box.set_payload(obj.box_mass, obj.box_com)
            obj.reset()
        if len(t) != 6000:
            raise ValueError('truncated episode')
        hits = np.empty(6000, dtype=bool)
        base = np.empty(6000, dtype=bool)
        residual = np.empty_like(expected_r)
        for k, time in enumerate(t):
            residual[k], hits[k] = det.update(time, th[k], thd[k], tm[k], ts[k], grip[k])
            base[k] = det.r8_base_detect
            ctrl.compute(time, th[k], thd[k], ts[k], grip[k], *reference(time))
        ok = all(x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()
                 for x, y in ((residual, expected_r), (hits, expected_hits), (base, expected_base)))
        rows.append(dict(case=case['source_case_id'], identical=ok,
                         extra_cycles=int(np.count_nonzero(hits & ~base))))
        if (i + 1) % 20 == 0:
            print(f'{i + 1}/{len(cases)} replayed', flush=True)
    ok = all(row['identical'] for row in rows)
    result = dict(mode=args.mode, identical=ok, candidate_sha256=sha(args.candidate), design_sha256=design_sha,
                  streams=len(cases), cycles=6000 * len(cases), rows=rows)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print('identical:', ok, flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
