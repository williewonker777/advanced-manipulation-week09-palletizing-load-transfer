"""Prove and run the unmodified 8 kg example from a local course kit."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tools/kit_manifest.sha256"
MANIFEST_SHA = "7ea083e92b7fa9fd303ce91a69271f58dd6fb9ace22613039f469a84e2867294"
GUIDE_SHA = "6b1ae2d0e38b8914ee0b4b3b27d82c9c72f2e15c00d8fb7ea698cfa3330de187"
REFERENCE_PNG_SHA = "82cb22df246af8600f7c28dac1e5a901e55bd2a5e439aa0a109348bff8e6436c"
SOURCES = {
    "run.py": "7735f4a1d666f631abee10b28261d201786f9526e6892ac61235125d685c3d6b",
    "servo_controller.py": "50b5d2b05a45d540615a4af4373780d86872c60c8049c8af75dd6a4c4836e416",
    "collision_detector.py": "fe4b1101ba41ffa46e04500f7153a0c569125b20083d8cd3a9fe6d8d69bda884",
}


class CourseError(ValueError):
    pass


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def safe_paths(kit, guide, out):
    kit, guide = Path(kit), Path(guide)
    if not kit.is_dir() or kit.is_symlink():
        raise CourseError("kit must be a real directory")
    if not guide.is_file() or guide.is_symlink():
        raise CourseError("guide must be a real file")
    kit, guide = kit.resolve(), guide.resolve()
    out = Path(out)
    if out.exists() or not out.parent.is_dir():
        raise CourseError("out must be a fresh directory under an existing parent")
    out = out.parent.resolve() / out.name
    if _inside(out, kit) or _inside(out, ROOT.resolve()):
        raise CourseError("out must be external to the kit and research repository")
    return kit, guide, out


def verify_kit(kit, manifest=MANIFEST):
    if digest(manifest) != MANIFEST_SHA:
        raise CourseError("trusted protected-file manifest changed")
    expected = {}
    for line in manifest.read_text().splitlines():
        value, name = line.split("  ", 1)
        expected[name] = value
    actual = {}
    for tree_name in ("sim", "model"):
        tree = kit / tree_name
        if not tree.is_dir() or tree.is_symlink():
            raise CourseError("missing or unsafe protected tree: " + tree_name)
        for path in tree.rglob("*"):
            name = path.relative_to(kit).as_posix()
            if path.is_symlink():
                raise CourseError("unsafe protected symlink: " + name)
            if path.is_file():
                actual[name] = digest(path)
    if actual != expected:
        raise CourseError("protected course tree does not exactly match the 20-file manifest")
    for name, value in SOURCES.items():
        path = kit / name
        if not path.is_file() or path.is_symlink() or digest(path) != value:
            raise CourseError("stock course source mismatch: " + name)
    return actual


def verify_guide(guide):
    guide_digest = digest(guide)
    if guide_digest != GUIDE_SHA:
        raise CourseError("guide_authenticity=unrecognized_revision sha256=" + guide_digest)
    encoded = re.findall(rb"data:image/png;base64,([A-Za-z0-9+/=\r\n]+)", guide.read_bytes())
    hashes = [hashlib.sha256(base64.b64decode(item)).hexdigest() for item in encoded]
    if hashes.count(REFERENCE_PNG_SHA) != 1:
        raise CourseError("hash-pinned guide does not contain the expected example PNG")
    return guide_digest, len(hashes)


def run(kit, guide, out):
    kit, guide, out = safe_paths(kit, guide, out)
    protected = verify_kit(kit)
    guide_digest, image_count = verify_guide(guide)
    out.mkdir()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               MPLBACKEND="Agg", MUJOCO_GL="disable")
    command = [sys.executable, "-I", "-B", str(kit / "run.py"), "--box", "8"]
    completed = subprocess.run(command, cwd=out, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    result = out / "result.png"
    if not result.is_file() or result.is_symlink():
        raise CourseError("stock run did not create a regular result.png")
    generated = digest(result)
    verify_kit(kit)
    report = {
        "schema": "course-example-authenticity.v1",
        "source_authenticity": {
            "status": "authentic_original_course",
            "protected_files": len(protected),
            "manifest_sha256": MANIFEST_SHA,
            "stock_source_sha256": SOURCES,
            "unchanged_after_run": True,
        },
        "guide_authenticity": {"status": "known_hash_pinned_revision",
                               "sha256": guide_digest, "embedded_pngs": image_count},
        "execution": {"isolated_python": True, "headless": True, "box_kg": 8,
                      "controller": "servo_controller", "detector": "collision_detector",
                      "argv": ["run.py", "--box", "8"],
                      "stdout": completed.stdout.splitlines()},
        "render_identity": {
            "generated_sha256": generated,
            "guide_embedded_sha256": REFERENCE_PNG_SHA,
            "expected_reference_sha256": REFERENCE_PNG_SHA,
            "byte_identical_to_guide": generated == REFERENCE_PNG_SHA,
            "byte_identical_to_expected_reference": generated == REFERENCE_PNG_SHA,
            "status": "byte_identical" if generated == REFERENCE_PNG_SHA else "not_byte_identical",
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.kit, args.guide, args.out)
    except (CourseError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(2, json.dumps({"status": "rejected", "reason": str(exc)}) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
