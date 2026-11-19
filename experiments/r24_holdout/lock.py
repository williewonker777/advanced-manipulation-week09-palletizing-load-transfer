"""Freeze an R24 entrypoint and complete repository-local source/model closure."""
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
PROTOCOL = HERE / "protocol.json"
EVALUATOR = HERE / "evaluator.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_relative(path):
    path = Path(path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("repository-relative source required")
    full = ROOT / path
    if full.is_symlink() or not full.is_file():
        raise ValueError("regular source required: " + path.as_posix())
    return path.as_posix()


def module_file(module):
    raw = Path(*module.split("."))
    choices = (raw.with_suffix(".py"), raw / "__init__.py")
    for rel in choices:
        if (ROOT / rel).is_file() and not (ROOT / rel).is_symlink():
            return rel.as_posix()
    return None


def module_name(source):
    path = Path(source)
    if path.name == "__init__.py":
        return ".".join(path.parent.parts)
    return ".".join(path.with_suffix("").parts)


def imported_modules(source):
    module = module_name(source)
    package = module if source.endswith("/__init__.py") else module.rsplit(".", 1)[0]
    tree = ast.parse((ROOT / source).read_bytes(), filename=source)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                keep = len(parts) - (node.level - 1)
                if keep < 0:
                    raise ValueError("invalid relative import in " + source)
                base = ".".join(parts[:keep])
                if node.module:
                    base = base + "." + node.module if base else node.module
            else:
                base = node.module or ""
            if base:
                found.add(base)
            for alias in node.names:
                if alias.name != "*" and base:
                    found.add(base + "." + alias.name)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "import_module" and node.args
              and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)):
            found.add(node.args[0].value)
    return found, tree


def literal_resources(tree):
    resources = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value
        if value.endswith((".json", ".npy", ".npz", ".toml", ".yaml", ".yml")):
            rel = Path(value)
            if not rel.is_absolute() and ".." not in rel.parts and (ROOT / rel).is_file():
                resources.add(rel.as_posix())
    return resources


def package_seed_files(package, entry):
    root = ROOT / Path(*package.split("."))
    if root.is_symlink() or not root.is_dir():
        raise ValueError("candidate package directory required: " + package)
    files = {entry}
    init = root / "__init__.py"
    if init.is_file() and not init.is_symlink():
        files.add(init.relative_to(ROOT).as_posix())
    # Freeze package-local model/config resources without sweeping unrelated
    # diagnostics, exposed evaluators, or tests into the deployable closure.
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts \
                or path.suffix in (".py", ".pyc", ".pyo"):
            continue
        files.add(path.relative_to(ROOT).as_posix())
    if not files:
        raise ValueError("empty candidate package")
    return files


def transitive_closure(entrypoint, protocol_sources):
    module, symbol = entrypoint.rsplit(":", 1)
    if (not module.startswith("experiments.r24_") or
            module.startswith("experiments.r24_holdout") or symbol != "Detector"):
        raise ValueError("new non-holdout experiments.r24_* Detector required")
    entry = module_file(module)
    if entry is None:
        raise ValueError("entrypoint module not found")
    candidate_package = ".".join(module.split(".")[:2])
    closure = package_seed_files(candidate_package, entry)
    queue = [name for name in closure if name.endswith(".py")]
    visited = set()
    while queue:
        source = queue.pop()
        if source in visited:
            continue
        visited.add(source)
        modules, tree = imported_modules(source)
        closure.update(literal_resources(tree))
        for imported in modules:
            local = module_file(imported)
            if local is None or local in protocol_sources:
                continue
            if local not in closure:
                closure.add(local)
                queue.append(local)
            path = Path(local)
            for parent in reversed(path.parents):
                if not parent.parts:
                    continue
                init = (parent / "__init__.py").as_posix()
                if (ROOT / init).is_file() and init not in protocol_sources and init not in closure:
                    closure.add(init)
                    queue.append(init)
    return {name: sha(ROOT / safe_relative(name)) for name in sorted(closure)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrypoint", required=True,
                        help="module:Detector under a new experiments.r24_* package")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        parser.error("fresh lock output required")
    if ":" not in args.entrypoint:
        parser.error("entrypoint must be module:class")
    protocol = json.loads(PROTOCOL.read_bytes())
    actual = {name: sha(ROOT / name) for name in protocol["source_pins"]}
    if actual != protocol["source_pins"]:
        raise ValueError("sealed protocol source differs before candidate lock")
    sources = transitive_closure(args.entrypoint, set(protocol["source_pins"]))
    document = {
        "schema": "r24-candidate-lock.v1",
        "status": "LOCKED_BEFORE_HOLDOUT_OUTCOMES",
        "entrypoint": args.entrypoint,
        "plan_sha256": sha(PLAN),
        "protocol_sha256": sha(PROTOCOL),
        "evaluator_sha256": sha(EVALUATOR),
        "transitive_sources": sources,
        "frozen_protocol_sources": protocol["source_pins"],
        "challenge_detail_was_not_disclosed_by_verifier_before_lock": True,
        "candidate_was_selected_without_holdout_output": True,
    }
    args.out.parent.resolve(strict=True)
    with args.out.open("x") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"entrypoint": args.entrypoint, "sources": len(sources),
                      "lock_sha256": sha(args.out)}, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
