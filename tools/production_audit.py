"""D2/D3 source-contract audit, not a proof of arbitrary Python data flow.

Final entries use NumPy, MuJoCo, public nominal/trajectory/configuration and
narrow standard-library helpers for embedded models. Runtime integration tests and human review supplement
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


def _attribute(node, owner, name):
    return (isinstance(node, ast.Attribute) and node.attr == name
            and isinstance(node.value, ast.Name) and node.value.id == owner)


def _enclosing_function(node, parents):
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def _valid_model_copy(call, parents):
    """Allow only ``nominal.m = source.m if share_model else copy(source.m)``."""
    if (len(call.args) != 1 or call.keywords or not _attribute(call.args[0], "source", "m")
            or not isinstance(parents.get(call), ast.IfExp)):
        return False
    choice = parents[call]
    assignment = parents.get(choice)
    function = _enclosing_function(call, parents)
    return (choice.orelse is call and isinstance(choice.test, ast.Name)
            and choice.test.id == "share_model" and _attribute(choice.body, "source", "m")
            and isinstance(assignment, ast.Assign) and len(assignment.targets) == 1
            and _attribute(assignment.targets[0], "nominal", "m")
            and function is not None and function.name == "_clone_nominal")


def _valid_clone_helper(function):
    args = function.args
    if (not isinstance(function, ast.FunctionDef) or function.decorator_list
            or [arg.arg for arg in args.args] != ["source", "share_model"] or args.vararg
            or args.kwarg or args.kwonlyargs or len(args.defaults) != 1
            or not isinstance(args.defaults[0], ast.Constant) or args.defaults[0].value is not False):
        return False
    owners, data_assignments, returns = [], [], []
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
            if isinstance(target, ast.Name) and target.id == "nominal":
                owners.append(isinstance(value, ast.Call) and not value.keywords
                    and len(value.args) == 1 and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "__new__" and isinstance(value.func.value, ast.Name)
                    and value.func.value.id == "Nominal" and isinstance(value.args[0], ast.Name)
                    and value.args[0].id == "Nominal")
            if _attribute(target, "nominal", "d"):
                data_assignments.append(isinstance(value, ast.Call) and not value.keywords
                    and len(value.args) == 1 and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "MjData" and isinstance(value.func.value, ast.Name)
                    and value.func.value.id == "mujoco" and _attribute(value.args[0], "nominal", "m"))
        if isinstance(node, ast.Return):
            returns.append(isinstance(node.value, ast.Name) and node.value.id == "nominal")
    return owners == [True] and data_assignments == [True] and returns == [True]


def _valid_clone_call(call, parents):
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Attribute):
        return False
    source = call.args[0]
    if not (isinstance(source.value, ast.Name) and source.value.id == "self"
            and source.attr in {"empty", "full"}):
        return False
    function = _enclosing_function(call, parents)
    owner = parents.get(function)
    if (function is None or function.name != "__init__" or not isinstance(owner, ast.ClassDef)
            or owner.name != "Membership"):
        return False
    constructs_empty = any(isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute) and node.targets[0].attr == "empty"
        and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == "self"
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "Nominal" and not node.value.args and not node.value.keywords
        for node in ast.walk(function))
    if not constructs_empty:
        return False
    if not call.keywords:
        return True
    return (source.attr == "empty" and len(call.keywords) == 1
            and call.keywords[0].arg == "share_model"
            and isinstance(call.keywords[0].value, ast.Constant)
            and call.keywords[0].value.value is True)


def audit_text(text, name="<source>"):
    tree = ast.parse(text, filename=name)
    errors, observations = [], []
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    config_aliases, public_module_aliases = set(), set()
    imports = {a.asname or a.name: a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
               for a in n.names}
    memory_only = {"io": "BytesIO", "base64": "b85decode"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in {"numpy", "mujoco", "sim.config", "sim.trajectory", *memory_only}:
                    errors.append(f"line {node.lineno}: disallowed import {alias.name}")
                if alias.name == "sim.config":
                    config_aliases.add(alias.asname or "sim.config")
                if alias.name in {"sim.config", "sim.trajectory"}:
                    public_module_aliases.add(alias.asname or "sim")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "numpy" and any(a.name == "load" for a in node.names):
                errors.append(f"line {node.lineno}: use the auditable in-memory numpy.load form")
            if node.module == "sim.config" and any(alias.name in FORBIDDEN_CONFIG for alias in node.names):
                errors.append(f"line {node.lineno}: forbidden configuration from-import")
            if node.module == "sim":
                if any(alias.name not in {"config", "trajectory"} for alias in node.names):
                    errors.append(f"line {node.lineno}: only sim.config/trajectory may be imported")
                config_aliases.update(alias.asname or alias.name for alias in node.names
                                      if alias.name == "config")
                public_module_aliases.update(alias.asname or alias.name for alias in node.names
                                             if alias.name in {"config", "trajectory"})
            elif node.module in {"sim.nominal", "collections"}:
                allowed = "Nominal" if node.module == "sim.nominal" else "deque"
                if any(a.name != allowed for a in node.names):
                    errors.append(f"line {node.lineno}: only {node.module}.{allowed} is allowed")
            elif node.module == "copy":
                if (len(node.names) != 1 or node.names[0].name != "copy"
                        or node.names[0].asname is not None):
                    errors.append(f"line {node.lineno}: only unaliased copy.copy is allowed")
            elif node.module not in {"numpy", "mujoco", "sim.config", "sim.trajectory"}:
                errors.append(f"line {node.lineno}: disallowed from-import {node.module}")
            if any(alias.name == "*" for alias in node.names):
                errors.append(f"line {node.lineno}: star import prevents bounded input audit")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                errors.append(f"line {node.lineno}: dynamic/I/O call {node.func.id}")
            if node.func.id == "copy" and not _valid_model_copy(node, parents):
                errors.append(f"line {node.lineno}: copy is limited to the owned Nominal model")
            if node.func.id == "_clone_nominal" and not _valid_clone_call(node, parents):
                errors.append(f"line {node.lineno}: Nominal clone source is not approved")
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name) and imports.get(node.func.value.id) == "numpy"
              and node.func.attr == "load"):
            arg = node.args[0] if len(node.args) == 1 else None
            memory = (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute)
                      and isinstance(arg.func.value, ast.Name)
                      and imports.get(arg.func.value.id) == "io" and arg.func.attr == "BytesIO")
            if not (memory and len(node.keywords) == 1 and node.keywords[0].arg == "allow_pickle"
                    and isinstance(node.keywords[0].value, ast.Constant)
                    and node.keywords[0].value.value is False):
                errors.append(f"line {node.lineno}: numpy.load requires in-memory bytes and allow_pickle=False")
        elif isinstance(node, ast.Attribute):
            module = imports.get(node.value.id) if isinstance(node.value, ast.Name) else None
            if module in memory_only and node.attr != memory_only[module]:
                errors.append(f"line {node.lineno}: not an approved in-memory decoder")
            if node.attr in ORACLE_ATTRIBUTES or node.attr in FORBIDDEN_CONFIG:
                errors.append(f"line {node.lineno}: forbidden attribute {node.attr}")
            if node.attr in {"__dict__", "__globals__", "__code__"}:
                errors.append(f"line {node.lineno}: reflective access {node.attr}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_CONFIG:
            errors.append(f"line {node.lineno}: forbidden identifier {node.id}")
        elif (isinstance(node, ast.Name) and node.id in {"copy", "_clone_nominal"}
              and isinstance(node.ctx, ast.Load)):
            parent = parents.get(node)
            if not (isinstance(parent, ast.Call) and parent.func is node):
                errors.append(f"line {node.lineno}: clone helper cannot escape as a value")
        elif (isinstance(node, ast.Name) and node.id in {"copy", "_clone_nominal"}
              and isinstance(node.ctx, (ast.Store, ast.Del))):
            errors.append(f"line {node.lineno}: clone helper name cannot be rebound")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == "copy":
            errors.append(f"line {node.lineno}: copy name cannot be redefined")
        elif isinstance(node, ast.arg) and node.arg in {"copy", "_clone_nominal"}:
            errors.append(f"line {node.lineno}: clone helper name cannot be shadowed")
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str) and node.value in FORBIDDEN_CONFIG | ORACLE_ATTRIBUTES:
                errors.append(f"line {node.lineno}: forbidden dynamic lookup string {node.value}")
            if isinstance(node.value, (int, float)) and node.value in {1.42, 3.86}:
                errors.append(f"line {node.lineno}: evaluator-specific collision time {node.value}")
    clone_helpers = [node for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name == "_clone_nominal"]
    copy_imports = [node for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module == "copy"]
    if copy_imports and (len(copy_imports) != 1 or len(clone_helpers) != 1
                         or not _valid_clone_helper(clone_helpers[0])):
        line = clone_helpers[0].lineno if clone_helpers else 1
        errors.append(f"line {line}: copy requires the bounded fresh-data Nominal clone helper")
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
