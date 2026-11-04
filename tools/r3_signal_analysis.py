"""Descriptive development residual attenuation, never a threshold search."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import r3_select as S

D = ROOT / 'results/v2_protocol'


def innovations(collision, reference, start_s):
    mask = ((collision['obs_t'] >= start_s)
            & (collision['obs_t'] < start_s + .12))
    if not mask.any() or not np.array_equal(collision['obs_t'], reference['obs_t']):
        raise ValueError('signal traces require identical time arrays and force-on samples')
    residual = collision['joint_residual'] - reference['joint_residual']
    wrench = collision['estimated_wrench'] - reference['estimated_wrench']
    return dict(
        joint_residual_innovation_peak_nm=float(np.linalg.norm(residual[mask], axis=1).max()),
        horizontal_wrench_innovation_peak_n=float(np.linalg.norm(wrench[mask, :2], axis=1).max()),
        linear_wrench_innovation_peak_n=float(np.linalg.norm(wrench[mask, :3], axis=1).max()),
        active_force_on_samples=int(np.count_nonzero(collision['active'][mask])),
        force_on_samples=int(mask.sum()),
    )


def read_raw(row, private_dir):
    path = private_dir / row['private_raw_basename']
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['private_raw_sha256']:
        raise RuntimeError('private trace hash mismatch')
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in ('obs_t', 'joint_residual', 'estimated_wrench', 'active')}


def analyze(private_dir):
    selected = json.loads((D / 'R3_DIAGNOSTIC_SELECTION.json').read_text())['chosen_specs']
    groups = []
    artifacts = {}
    for spec in selected:
        mode = spec['architecture']
        paths = [D / f"R3_DEVELOPMENT_{variant}_{mode}.json" for variant in ('R1', spec['id'])]
        base, candidate = [json.loads(path.read_text()) for path in paths]
        artifacts.update({path.relative_to(ROOT).as_posix():
                          hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
        refs = [{(r['mass'], tuple(r['com'])): r for r in data['results']
                 if r['kind'] == 'reference'} for data in (base, candidate)]
        rows = []
        for left, right in S.pair_results(base, candidate):
            if right['kind'] != 'collision':
                continue
            key = (right['mass'], tuple(right['com']))
            values = [innovations(read_raw(row, private_dir),
                                  read_raw(reference[key], private_dir),
                                  float(row['collision'][0]))
                      for row, reference in zip((left, right), refs)]
            ratios = {name: (values[1][name] / values[0][name]
                             if values[0][name] > 0 else None)
                      for name in values[0] if name.endswith(('_nm', '_n'))}
            rows.append(dict(case_id=right['case_id'], mass=right['mass'],
                             phase=right['phase'], direction=right['direction'],
                             force_n=right['force_n'], R1=values[0], candidate=values[1],
                             peak_ratio_vs_R1=ratios,
                             R1_outcome=left['collision_outcome'],
                             candidate_outcome=right['collision_outcome'],
                             R1_delay_ms=left['detect_delay_ms'],
                             candidate_delay_ms=right['detect_delay_ms']))
        summary = {name: dict(min=min(r['peak_ratio_vs_R1'][name] for r in rows
                                      if r['peak_ratio_vs_R1'][name] is not None),
                             max=max(r['peak_ratio_vs_R1'][name] for r in rows
                                      if r['peak_ratio_vs_R1'][name] is not None))
                   for name in rows[0]['peak_ratio_vs_R1']}
        groups.append(dict(architecture=mode, challenger=spec['id'], rows=rows,
                           descriptive_peak_ratio_range=summary))
    return dict(schema='R3_SIGNAL_ANALYSIS.v1', source_sha256=artifacts,
                force_on_window_s=.12,
                method='collision-minus-matched-normal innovation; same applied force-on interval',
                use='descriptive only, no thresholds, selection, recalibration, or global suppression claim',
                groups=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('do not overwrite descriptive evidence')
    S._atomic_write(args.output, analyze(args.private_dir))


if __name__ == '__main__':
    main()
