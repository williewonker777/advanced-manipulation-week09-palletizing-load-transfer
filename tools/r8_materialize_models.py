"""Recover exact embedded model archives into a private execution overlay only."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def materialize(root):
    root=Path(root).resolve()
    if root==ROOT or '.git' in root.parts or (root/'.git').exists():
        raise ValueError('choose a separate private execution overlay')
    source=ROOT/'experiments/r8_final/C1.py'
    tree=ast.parse(source.read_text())
    values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
            and n.targets[0].id in ('_R8_EMPTY','_R8_POSITIVE')}
    pins=json.loads((ROOT/'results/v2_protocol/R8_FREEZE.json').read_text())['pins']
    for kind in ('empty','positive'):
        rel=f'experiments/r8_final/servo_{kind}.npz'
        data=base64.b85decode(values['_R8_'+kind.upper()])
        if hashlib.sha256(data).hexdigest()!=pins[rel]:
            raise ValueError('model archive differs from immutable hash')
        p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and p.read_bytes()!=data:raise ValueError('existing model differs')
        p.write_bytes(data)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    materialize(p.parse_args().root)
