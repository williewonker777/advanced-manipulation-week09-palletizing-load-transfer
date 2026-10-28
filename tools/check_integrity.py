"""Verify the immutable simulation/model against the pristine source manifest.

Never import sim or create bytecode in the protected directories. Unexpected
files and symlinks are failures, not silently excluded from the check.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_integrity(root=None):
    """Return a summary, or raise ValueError if the protected trees differ."""
    root = ROOT if root is None else Path(root).resolve()
    manifest = root / "tools" / "kit_manifest.sha256"
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = Path(name)
        if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or path.is_absolute() or ".." in path.parts
                or path.parts[0] not in {"sim", "model"} or name in expected):
            raise ValueError(f"Invalid manifest entry: {name}")
        expected[name] = digest
    if not expected:
        raise ValueError("Empty kit manifest")
    actual = {}
    for name in ("sim", "model"):
        tree = root / name
        if not tree.is_dir() or tree.is_symlink():
            raise ValueError(f"Missing/unsafe protected directory: {name}")
        for path in tree.rglob("*"):
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError(f"Unexpected symlink: {rel}")
            if path.is_file():
                with path.open("rb") as stream:
                    actual[rel] = hashlib.file_digest(stream, "sha256").hexdigest()
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(name for name in set(actual) & set(expected)
                     if actual[name] != expected[name])
    if missing or extra or changed:
        raise ValueError(f"Kit integrity FAILED: missing={missing}, extra={extra}, changed={changed}")
    return {"status": "ok", "files": len(actual),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        print(json.dumps(check_integrity(args.root), sort_keys=True))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
