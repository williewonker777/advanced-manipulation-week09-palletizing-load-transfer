"""D2/D3 source-contract audit, not a proof of arbitrary Python data flow.

Final controller/detector files may depend only on NumPy, MuJoCo and the public
trajectory/configuration. Runtime integration tests and human review supplement
these conservative AST checks. Analysis tools are intentionally out of scope.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ("my_servo.py", "my_torque.py", "my_detector.py")
FORBIDDEN_CONFIG = {"JTS_OFFSET", "JTS_DRIFT", "JTS_SEED", "TABLE_SOLREF", "TABLE_SOLIMP",
                    "SUCTION_SOLREF", "TABLE_XY", "FLOOR_Z"}
FORBIDDEN_CALLS = {"open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars"}
ORACLE_ATTRIBUTES = {"plant", "log", "true_torque", "table_force", "tool_force", "tau_link",
                     "jts_f", "jts_out", "rng", "qfrc_constraint", "efc_force", "efc_J"}


def audit_text(text, name="<source>"):
    tree = ast.parse(text, filename=name)
    errors, observations = [], []
    config_aliases, public_module_aliases = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in {"numpy", "mujoco", "sim.config", "sim.trajectory"}:
                    errors.append(f"line {node.lineno}: disallowed import {alias.name}")
                if alias.name == "sim.config":
                    config_aliases.add(alias.asname or "sim.config")
                if alias.name in {"sim.config", "sim.trajectory"}:
                    public_module_aliases.add(alias.asname or "sim")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "sim.config" and any(alias.name in FORBIDDEN_CONFIG for alias in node.names):
                errors.append(f"line {node.lineno}: forbidden configuration from-import")
            if node.module == "sim":
                if any(alias.name not in {"config", "trajectory"} for alias in node.names):
                    errors.append(f"line {node.lineno}: only sim.config/trajectory may be imported")
                config_aliases.update(alias.asname or alias.name for alias in node.names
                                      if alias.name == "config")
                public_module_aliases.update(alias.asname or alias.name for alias in node.names
                                             if alias.name in {"config", "trajectory"})
            elif node.module not in {"numpy", "mujoco", "sim.config", "sim.trajectory"}:
                errors.append(f"line {node.lineno}: disallowed from-import {node.module}")
            if any(alias.name == "*" for alias in node.names):
                errors.append(f"line {node.lineno}: star import prevents bounded input audit")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                errors.append(f"line {node.lineno}: dynamic/I/O call {node.func.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr in ORACLE_ATTRIBUTES or node.attr in FORBIDDEN_CONFIG:
                errors.append(f"line {node.lineno}: forbidden attribute {node.attr}")
            if node.attr in {"__dict__", "__globals__", "__code__"}:
                errors.append(f"line {node.lineno}: reflective access {node.attr}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_CONFIG:
            errors.append(f"line {node.lineno}: forbidden identifier {node.id}")
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str) and node.value in FORBIDDEN_CONFIG | ORACLE_ATTRIBUTES:
                errors.append(f"line {node.lineno}: forbidden dynamic lookup string {node.value}")
            if isinstance(node.value, (int, float)) and node.value in {1.42, 3.86}:
                errors.append(f"line {node.lineno}: evaluator-specific collision time {node.value}")
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            if any(isinstance(part, ast.Name) and part.id in public_module_aliases
                   for part in ast.walk(target)):
                errors.append(f"line {node.lineno}: write to read-only sim configuration/trajectory")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"append", "extend", "clear", "pop", "remove", "insert", "sort", "fill"}:
                if any(isinstance(part, ast.Name) and part.id in public_module_aliases
                       for part in ast.walk(node.func.value)):
                    errors.append(f"line {node.lineno}: mutating read-only sim configuration/trajectory")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in config_aliases:
                observations.append({"line": node.lineno, "config_field": node.attr})
    return {"file": name, "status": "pass" if not errors else "FAIL", "errors": errors,
            "config_accesses": observations,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "scope": "AST input/import checks; runtime tests and manual review also required"}


def audit_files(root=ROOT):
    return [audit_text((Path(root) / name).read_text(encoding="utf-8"), name)
            for name in PRODUCTION]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    records = audit_files(args.root)
    content = json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(content, encoding="utf-8")
    print(content, end="")
    return 0 if all(record["status"] == "pass" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
