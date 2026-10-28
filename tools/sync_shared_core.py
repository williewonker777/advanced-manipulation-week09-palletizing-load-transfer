"""Keep the self-contained estimator class identical in all final modules."""
import argparse
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("my_servo.py", "my_torque.py", "my_detector.py")


def extract(text):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef)
                and n.name == "SharedEstimator")
    lines = text.splitlines(keepends=True)
    return ''.join(lines[node.lineno-1:node.end_lineno]), node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    canonical, _ = extract((ROOT/'tools/shared_estimator_core.py').read_text())
    for name in FILES:
        path = ROOT/name
        source = path.read_text()
        actual, node = extract(source)
        if args.write:
            lines = source.splitlines(keepends=True)
            path.write_text(''.join(lines[:node.lineno-1])+canonical+''.join(lines[node.end_lineno:]))
        elif actual != canonical:
            raise SystemExit(f'{name}: shared core text differs')
    print('3 identical embedded classes; SHA256', hashlib.sha256(canonical.encode()).hexdigest())


if __name__ == '__main__':
    main()
