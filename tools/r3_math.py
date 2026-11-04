"""Offline qualifications of the optimized R3 polarized RNE action.

No plant episodes, fitting, timing benchmark, or sealed seed is involved.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.r3_torque import _R3BlendedPRNE
from experiments.r2_prne import PRNE
from sim.nominal import Nominal


def qualify(seed=369804, samples=16):
    rng = np.random.default_rng(seed)
    empty, full = Nominal(), Nominal()
    full.set_payload(5.47, (0.01025967589081811, 0.014648133452979668,
                            0.03825206193965912))
    action = _R3BlendedPRNE(empty, full)
    endpoint0, endpoint1 = PRNE(empty), PRNE(full)
    rows = []
    for index in range(samples):
        q = rng.uniform(-0.7, 0.7, 6)
        v, u = rng.uniform(-0.35, 0.35, (2, 6))
        load = (0.0, 0.5, 1.0)[index % 3]
        matrix = np.column_stack([action.action(q, v, column, load)
                                  for column in np.eye(6)])
        expected = ((1 - load) * endpoint0.matrix(q, v)
                    + load * endpoint1.matrix(q, v))
        directional = []
        for h in (1e-5, 5e-6):
            def mass(position):
                return ((1 - load) * empty.mass_matrix(position)
                        + load * full.mass_matrix(position))
            derivative = (mass(q + h * v) - mass(q - h * v)) / (2 * h)
            directional.append(derivative)
        rows.append(dict(
            sample=index, load_fraction=load,
            endpoint_matrix_max_abs_error=float(np.max(np.abs(matrix - expected))),
            action_max_abs_error=float(np.max(np.abs(
                action.action(q, v, u, load) - matrix @ u))),
            mdot_minus_C_plus_CT_max_abs_error=float(np.max(np.abs(
                directional[1] - matrix - matrix.T))),
            mdot_h_h2_max_abs_difference=float(np.max(np.abs(
                directional[0] - directional[1]))),
        ))
    fields = [key for key in rows[0] if key.endswith('error') or key.endswith('difference')]
    worst = {key: max(row[key] for row in rows) for key in fields}
    return dict(schema='R3_PRNE_MATH.v1', episodes_run=0, seed=seed,
                scope='rigid nominal C action qualification, not elastic-plant stability',
                equations=['C(q,v)u', 'Mdot(q;v)=C(q,v)+C(q,v)^T'],
                source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ('experiments/r3_torque.py', 'experiments/r2_prne.py',
                                            'tools/r3_math.py')},
                rows=rows, worst=worst,
                passed=all(value < 1e-7 for value in worst.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('do not overwrite mathematical evidence')
    result = qualify()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['worst'], 'passed:', result['passed'])
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
