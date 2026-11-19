"""Refit the preregistered horizon from 24 user-owned unshifted public records.

No split labels, plant parameters, or truth files are read. The caller is
responsible for providing independent, undisplaced source normals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.r17_information_prefix import estimate
from experiments.r17_information_prefix.stream import feature


def fit_files(paths, plan):
    horizons = plan['centres_ms']
    if len(paths) != 24 or len({p.resolve() for p in paths}) != 24:
        raise ValueError('24 distinct unshifted calibration records required')
    digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    if len(set(digests)) != len(digests):
        raise ValueError('duplicate calibration record bytes')
    full = [feature(p,horizons[-1]) for p in paths]
    models, cells = [], []
    for horizon in horizons:
        model, cell = estimate.fit([estimate.past(row,horizon) for row in full])
        models.append(model)
        cells.append(cell)
    i = estimate.select(cells)
    return {'schema':'r17-user-normal-horizon-fit.v1','cells':cells,
            'source_hashes':digests,'selected':None if i is None else cells[i],
            'model':None if i is None else models[i],
            'limits':'Source normals supplied by the caller. A selected horizon alone proves no performance on displaced pickups.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    if a.out.exists() or a.out.is_symlink() or not a.out.parent.is_dir():
        parser.error('new output file under an existing directory required')
    plan=json.loads(Path(__file__).with_name('plan.json').read_text())
    result=fit_files(a.samples,plan)
    with a.out.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps(result['selected'],allow_nan=False))


if __name__=='__main__':
    main()
