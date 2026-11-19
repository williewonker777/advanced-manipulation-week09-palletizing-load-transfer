"""Freeze an explicit authored-source closure for a named R27 history candidate."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.r27_history import guard


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    candidate_path = guard.source_path(args.candidate)
    if guard.sha(candidate_path) != args.candidate_sha256:
        parser.error("candidate source hash differs")
    blobs, pins = guard.base_overlay_blobs()
    names = set(args.file)
    names.add(candidate_path.relative_to(guard.ROOT).as_posix())
    package = candidate_path.parent / "__init__.py"
    if package.is_file():
        names.add(package.relative_to(guard.ROOT).as_posix())
    for name in sorted(names):
        relative = Path(name)
        source = guard.ROOT / relative
        if (relative.is_absolute() or ".." in relative.parts
                or any(part in ("sim", "model", "r27_loaded_holdout")
                       for part in relative.parts)
                or relative.name in ("evaluate.py", "run.py")
                or source.is_symlink() or not source.is_file()):
            parser.error("invalid authored closure member: " + name)
        blobs[relative.as_posix()] = source.read_bytes()
        pins[relative.as_posix()] = guard.sha(blobs[relative.as_posix()])
    document = {"schema": guard.CLOSURE_SCHEMA, "entrypoint": args.candidate,
                "candidate_sha256": args.candidate_sha256,
                "files": dict(sorted(pins.items())),
                "overlay_files": sorted(blobs),
                "kit_included": False,
                "loaded_holdout_included": False}
    guard.save(args.out, document)
    guard.load_candidate_closure(args.out, args.candidate, args.candidate_sha256)
    print(guard.sha(args.out), len(pins))


if __name__ == "__main__":
    main()
