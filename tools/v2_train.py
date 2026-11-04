"""CPU numpy ridge fit and collision-blind validation threshold, offline only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def standardized(x, mean, scale):
    return np.clip((x-mean)/scale, -8., 8.)


def validation_score(x, y, active, mean, scale, weights):
    errors = np.linalg.norm(y - standardized(x, mean, scale) @ weights, axis=1)
    ema = 0.
    scores = []
    for value, gate in zip(errors, active):
        ema = ema + .1*(value-ema) if gate else 0.
        scores.append(ema)
    return np.asarray(scores)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=369401)
    args = ap.parse_args()
    np.random.seed(args.seed)  # deterministic closed form, no random batches
    manifest = json.loads((args.data_dir/'manifest.json').read_text())
    result = dict(protocol_sha256=manifest['protocol_sha256'], seed=args.seed,
                  method='train-only normalized ridge .02; disjoint no-collision validation threshold',
                  training_target=manifest['target_definition'],
                  data_sha256=hashlib.sha256((args.data_dir/'manifest.json').read_bytes()).hexdigest(),
                  models={}, validation={}, dependencies={'numpy': np.__version__},
                  official_conditions_used=False, holdout_opened=False)
    for mode in ('servo', 'torque'):
        rows = [r for r in manifest['records'] if r['mode']==mode]
        matrices, targets = [], []
        for row in rows:
            p = args.data_dir/row['file']
            assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
            if row['split'] == 'train':
                with np.load(p, allow_pickle=False) as a:
                    matrices.append(a['x'][a['active']]); targets.append(a['y'][a['active']])
        x, y = np.concatenate(matrices), np.concatenate(targets)
        mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
        # A constant column carries the intercept without standardization loss.
        mean[0], scale[0] = 0., 1.
        z = standardized(x, mean, scale)
        penalty = .02*np.eye(z.shape[1]); penalty[0, 0] = 0.
        weights = np.linalg.solve(z.T@z+penalty, z.T@y)
        scores = []
        for row in rows:
            if row['split'] == 'validation':
                with np.load(args.data_dir/row['file'], allow_pickle=False) as a:
                    score = validation_score(a['x'], a['y'], a['active'], mean, scale, weights)
                    scores.append(dict(file=row['file'], maximum_score=float(score.max())))
        maximum = max(r['maximum_score'] for r in scores)
        threshold = max(3., 1.20*maximum+.5)
        result['models'][mode] = dict(mean=mean.tolist(), scale=scale.tolist(),
                                     weights=weights.tolist(), threshold=threshold)
        result['validation'][mode] = dict(episodes=scores, threshold=threshold,
                                         train_samples=len(x),
                                         train_rmse_N=float(np.sqrt(np.mean((z@weights-y)**2))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['validation'], indent=2))


if __name__ == '__main__':
    main()
