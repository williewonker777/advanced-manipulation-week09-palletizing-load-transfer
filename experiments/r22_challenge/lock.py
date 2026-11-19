"""Freeze the R22 detector source identity before the sealed challenge is opened."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PLAN = HERE / "plan.json"
EVALUATOR = HERE / "evaluator.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_source(raw):
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("candidate sources must be repository-relative")
    full = ROOT / path
    if full.is_symlink() or not full.is_file() or full.suffix not in (".py", ".json"):
        raise ValueError("regular candidate Python/JSON source required: " + raw)
    if not path.as_posix().startswith("experiments/r22_"):
        raise ValueError("candidate source must live in a new R22 experiment package")
    return path.as_posix(), full


def validate_python_dependencies(sources):
    """Every imported R22 local module must itself be pinned."""
    available = {name[:-3].replace("/", ".") for name in sources if name.endswith(".py")}
    available.update(name[:-12].replace("/", ".") for name in sources if name.endswith("/__init__.py"))
    missing = set()
    for name in sources:
        if not name.endswith(".py"):
            continue
        for node in ast.walk(ast.parse((ROOT / name).read_bytes(), filename=name)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if module.startswith("experiments.r22_") and not any(
                        module == item or module.startswith(item + ".") for item in available):
                    missing.add(module)
    if missing:
        raise ValueError("unpinned R22 imports: " + ", ".join(sorted(missing)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrypoint", required=True,
                        help="module:Detector, for example experiments.r22_integrated.candidate:Detector")
    parser.add_argument("--source", action="append", required=True,
                        help="repository-relative candidate dependency; repeat for every R22 Python/JSON file")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        parser.error("fresh lock output required")
    if ":" not in args.entrypoint:
        parser.error("entrypoint must be module:class")
    module, symbol = args.entrypoint.rsplit(":", 1)
    if not module.startswith("experiments.r22_") or symbol != "Detector":
        parser.error("R22 Detector entrypoint required")
    records = dict(relative_source(name) for name in args.source)
    if len(records) != len(args.source):
        parser.error("candidate source repeated")
    expected = module.replace(".", "/") + ".py"
    package = module.rsplit(".", 1)[0].replace(".", "/") + "/__init__.py"
    if expected not in records or package not in records:
        parser.error("entrypoint source and package __init__.py must be pinned")
    validate_python_dependencies(records)
    document = {
        "schema": "r22-candidate-lock.v1",
        "status": "LOCKED_BEFORE_CHALLENGE_OUTCOMES",
        "entrypoint": args.entrypoint,
        "plan_sha256": sha(PLAN),
        "evaluator_sha256": sha(EVALUATOR),
        "candidate_sources": {name: sha(path) for name, path in sorted(records.items())},
        "challenge_detail_was_not_disclosed_by_verifier_before_lock": True,
    }
    args.out.parent.resolve(strict=True)
    with args.out.open("x") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"entrypoint": args.entrypoint, "sources": len(records),
                      "lock_sha256": sha(args.out)}, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
