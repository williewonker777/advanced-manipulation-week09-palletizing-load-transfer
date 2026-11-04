"""Materialize self-contained registered controllers; never run a plant."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.production_audit import audit_text


def compiled_path(spec):
    return ROOT/'experiments/r3_generated'/f"{spec['id']}_{spec['architecture']}.py"


def materialize(spec, destination=None):
    template = ROOT/'experiments'/f"r3_{spec['architecture']}.py"
    text = template.read_text()
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'R3_DEFAULT_PARAMETERS' for t in n.targets)]
    if len(nodes) != 1:
        raise ValueError('exactly one default-parameter literal is required')
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = ['R3_DEFAULT_PARAMETERS = '+repr(dict(spec))+'\n']
    output = ''.join(lines)
    audit = audit_text(output, str(destination or compiled_path(spec)))
    if audit['errors']:
        raise ValueError(audit['errors'])
    target = Path(destination) if destination else compiled_path(spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_text() != output:
        raise RuntimeError('refuse to replace a materialized controller with different bytes')
    target.write_text(output)
    return dict(path=target.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(), spec=spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'results/v2_protocol/PROTOCOL_R3.json')
    args = parser.parse_args()
    records = [materialize(s) for s in json.loads(args.protocol.read_text())['candidate_grid']]
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
