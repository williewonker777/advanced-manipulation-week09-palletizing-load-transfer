"""Integrity-gated official evaluator; forward all arguments unchanged."""
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
from check_integrity import check_integrity


def main():
    root = Path(__file__).resolve().parents[1]
    print(check_integrity(root), flush=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.call([sys.executable, str(root / "evaluate.py"), *sys.argv[1:]],
                           cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
