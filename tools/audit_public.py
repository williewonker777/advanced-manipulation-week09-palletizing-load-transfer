"""Audit a curated public staging tree; never performs network/Git operations.

Checks files, sensitive text, large copied Python blocks, and checksum coverage.
An external kit argument is optional (recommended) and is only read for the
source-similarity check. It is not bundled with the published project.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

DENIED_PARTS = {"sim", "model", ".venv", "__pycache__", "evidence"}
DENIED_NAMES = {"run.py", "evaluate.py", "install.py", "servo_controller.py",
                "torque_controller.py", "collision_detector.py", "guide_extracted.txt"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".sha256", ".html", ".yml", ".toml"}


def files_under(root):
    return sorted(p for p in root.rglob("*") if ".git" not in p.relative_to(root).parts and
                  (p.is_file() or p.is_symlink()))


def text_errors(text):
    patterns = {"machine home path": "/" + "home/",
                "private workflow reference": "agent" + "-workflow",
                "internal task identifier": r"TASK-\d{3,}",
                "internal evidence reference": "local/" + "evidence/",
                "contact/email": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
                "private key": "-----BEGIN " + r"(?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
                "API credential": r"sk-" + r"(?:proj-)?[A-Za-z0-9_-]{24,}"}
    return [label for label, pattern in patterns.items() if re.search(pattern, text)]


def normalized_lines(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def copied_blocks(public_root, kit_root, length=12):
    if kit_root is None:
        return []
    originals = {}
    for source in kit_root.rglob("*.py"):
        if ".venv" in source.parts or ".git" in source.parts:
            continue
        lines = normalized_lines(source)
        for i in range(len(lines) - length + 1):
            block = "\n".join(lines[i:i+length])
            if len(block) >= 400:
                originals[block] = source.relative_to(kit_root).as_posix()
    matches = []
    for path in files_under(public_root):
        if path.suffix != ".py":
            continue
        lines = normalized_lines(path)
        for i in range(len(lines) - length + 1):
            block = "\n".join(lines[i:i+length])
            if block in originals:
                matches.append({"file": path.relative_to(public_root).as_posix(),
                                "kit_file": originals[block], "normalized_start": i + 1})
    return matches


def audit(root, kit=None, require_checksums=True):
    root = Path(root).resolve()
    paths = files_under(root)
    errors, listing = [], []
    for path in paths:
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            errors.append(f"symlink: {name}")
            continue
        if set(path.relative_to(root).parts) & DENIED_PARTS or path.name in DENIED_NAMES:
            errors.append(f"excluded kit/environment/raw file: {name}")
        if path.suffix.lower() in {".npz", ".npy", ".stl", ".xml", ".log"} or path.stat().st_size > 8*1024**2:
            errors.append(f"raw/large/model artifact: {name}")
        listing.append({"file": name, "bytes": path.stat().st_size})
        if path.suffix in TEXT_SUFFIXES or path.name in {"README", "LICENSE", "THIRD_PARTY_NOTICES", "SHA256SUMS"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeError:
                errors.append(f"invalid UTF-8 text: {name}")
            else:
                errors.extend(f"{name}: {error}" for error in text_errors(text))
    matches = copied_blocks(root, Path(kit) if kit else None)
    errors.extend(f"large kit source block: {m['file']} from {m['kit_file']}" for m in matches)
    manifest = root / "artifacts/SHA256SUMS"
    if require_checksums:
        if not manifest.is_file():
            errors.append("missing checksum manifest")
        else:
            expected = {}
            for line in manifest.read_text().splitlines():
                try:
                    digest, name = line.split("  ", 1)
                    if (Path(name).is_absolute() or ".." in Path(name).parts or name in expected
                            or not re.fullmatch("[0-9a-f]{64}", digest)):
                        raise ValueError
                    expected[name] = digest
                except ValueError:
                    errors.append("malformed checksum manifest")
            actual = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in paths if p != manifest and not p.is_symlink()}
            if actual != expected:
                errors.append("checksum content/coverage mismatch")
    return {"status": "pass" if not errors else "FAIL", "files": listing,
            "file_count": len(listing), "total_bytes": sum(f["bytes"] for f in listing),
            "errors": errors, "copied_source_blocks": matches,
            "kit_comparison_performed": kit is not None,
            "scope": "finite path/text/checksum/12-line similarity scan plus human review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--kit", type=Path)
    parser.add_argument("--out", type=Path, help="report outside staging; avoid self-checksum cycle")
    args = parser.parse_args()
    result = audit(args.root, args.kit)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
