"""Create an authenticated minimal test kit with one recorded JTS seed edit."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

FILES = ("r1_torque.py", "my_detector.py")
TREES = ("sim", "model")
EXPECTED = {
    "sim/config.py": "db2f85407992c21fb3e331acef93f3d6f91e86ad72a760da76a1b93226627735",
    "sim/plant.py": "8f4a16e1d083dc8559ac21427166c78db7261b7bedd149b8dae8f789a6245f1e",
    "sim/runner.py": "3cf99c21b93f431cc57a0929e4d05b083fb8f35f06dc52bc2d2ddcde3e55ed1f",
    "r1_torque.py": "8f66f8f03cde65fcc6d8ef2f11643691cd8fe2e96dad6eaeaa17a0c7aa3db371",
    "my_detector.py": "09f9a8573814a31b335b36ec5a78e0dedf262daa9039954568abd6bedd8dbc67",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_seed(text, seed):
    pattern = r"(?m)^JTS_SEED = 7(?=\s+#)"
    changed, count = re.subn(pattern, f"JTS_SEED = {int(seed)}", text)
    if count != 1:
        raise ValueError("expected exactly one original JTS_SEED assignment")
    return changed


def prepare(source, destination, seed):
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("fresh kit destination required")
    actual = {name: sha(source/name) for name in EXPECTED}
    if actual != EXPECTED:
        raise ValueError(f"source kit pin mismatch: {actual}")
    destination.mkdir(parents=True)
    for tree in TREES:
        shutil.copytree(source/tree, destination/tree)
    for name in FILES:
        shutil.copy2(source/name, destination/name)
    config = destination/"sim/config.py"
    original = config.read_text()
    config.write_text(replace_seed(original, seed))
    manifest = {"schema": "r18-seed-kit.v1", "seed": int(seed),
                "source_pins": actual, "copied_config_sha256": sha(config)}
    (destination/"SEED_OVERRIDE.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.source, args.destination, args.seed), indent=2))


if __name__ == "__main__":
    main()
