"""Assemble the R6 detector from immutable R5 and R1 source bytes."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
R1_SOURCE = ROOT / "r1_detector.py"
R5_SOURCE = ROOT / "experiments/r5_generated/SCALE_STATE.py"
R5_FROZEN_A = ROOT / "experiments/r5_final/A_STATE.py"
R5_BUILD = ROOT / "results/v2_protocol/R5_BUILD.json"
TEST_SOURCE = ROOT / "tools/test_r6_build.py"
FINAL_DIR = ROOT / "experiments/r6_final"
BUILD_RECORD = ROOT / "results/v2_protocol/R6_BUILD.json"
OUTPUTS = {
    "A_STATE": FINAL_DIR / "A_STATE.py",
    "R1": FINAL_DIR / "R1.py",
}
EXPECTED_HASHES = {
    R1_SOURCE: "e55995db6147ff8a543b816354376f968e0027d62d87c9dd1eeb55e0d88abbb4",
    R5_SOURCE: "9627c8b21d1998e8cc2e22a273b6962c7a446cdbe79f618b3e29868c7fe4db1e",
    R5_FROZEN_A: "4f4b33c08872e8db612e1416468ea14dedc74964f05702f4aa084971f9197b44",
    R5_BUILD: "97d8dad28ebc892e7dcf7be6d726146e9e2b9571fccabde58ce9f76f4e1441b7",
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_path(path):
    return sha256_bytes(Path(path).read_bytes())


def verify_inputs():
    for path, expected in EXPECTED_HASHES.items():
        if sha256_path(path) != expected:
            raise RuntimeError("immutable R5 input differs: " + path.relative_to(ROOT).as_posix())
    record = json.loads(R5_BUILD.read_text(encoding="utf-8"))
    models = record.get("models", [])
    if len(models) != 8:
        raise RuntimeError("R5 build record must pin exactly eight models")
    for row in models:
        path = ROOT / row["path"]
        if sha256_path(path) != row["sha256"]:
            raise RuntimeError("immutable R5 model differs: " + row["path"])
    if not R5_SOURCE.read_bytes().startswith(R1_SOURCE.read_bytes()):
        raise RuntimeError("R5 SCALE_STATE no longer has the exact R1 prefix")
    return record


def assembly_bytes(enabled_modes):
    """Return an R1, singleton-mode, or dual-mode detector assembly."""
    enabled_modes = set(enabled_modes)
    if not enabled_modes:
        return R1_SOURCE.read_bytes()
    if not enabled_modes <= {"servo", "torque"}:
        raise ValueError("enabled modes must be servo and/or torque")
    source = R5_SOURCE.read_bytes()
    if enabled_modes == {"servo", "torque"}:
        return source
    suffix = """

_R5BothModeDetector = Detector
_R5_ENABLED_MODES = %r


class Detector(_R5BothModeDetector):
    def reset(self):
        super().reset()
        mode = getattr(getattr(self, "ctrl", None), "MODE", None)
        if mode not in _R5_ENABLED_MODES:
            self._r5_monitor = None
""" % (tuple(sorted(enabled_modes)),)
    return source + suffix.encode("utf-8")


def _expected_outputs():
    verify_inputs()
    return {
        OUTPUTS["A_STATE"]: assembly_bytes({"servo"}),
        OUTPUTS["R1"]: assembly_bytes(set()),
    }


def _record(outputs):
    r5 = json.loads(R5_BUILD.read_text(encoding="utf-8"))
    frozen = R5_FROZEN_A.read_bytes()
    candidate = outputs[OUTPUTS["A_STATE"]]
    old = b"_R5_ENABLED_MODES = 'servo'\n"
    new = b"_R5_ENABLED_MODES = ('servo',)\n"
    return {
        "schema": "R6_BUILD.v1",
        "immutable_inputs": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": digest}
            for path, digest in EXPECTED_HASHES.items()
        ],
        "models": [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in r5["models"]
        ],
        "model_count": 8,
        "prefix_proof": {
            "base_path": R1_SOURCE.relative_to(ROOT).as_posix(),
            "source_path": R5_SOURCE.relative_to(ROOT).as_posix(),
            "exact_R1_prefix": R5_SOURCE.read_bytes().startswith(R1_SOURCE.read_bytes()),
            "prefix_length_bytes": len(R1_SOURCE.read_bytes()),
        },
        "builder": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                    "sha256": sha256_path(Path(__file__).resolve())},
        "regression_test": {"path": TEST_SOURCE.relative_to(ROOT).as_posix(),
                            "sha256": sha256_path(TEST_SOURCE)},
        "generated": [
            {"name": name, "path": path.relative_to(ROOT).as_posix(),
             "sha256": sha256_bytes(outputs[path])}
            for name, path in OUTPUTS.items()
        ],
        "diff_proof": {
            "frozen_path": R5_FROZEN_A.relative_to(ROOT).as_posix(),
            "candidate_path": OUTPUTS["A_STATE"].relative_to(ROOT).as_posix(),
            "old_literal": old.decode().rstrip("\n"),
            "new_literal": new.decode().rstrip("\n"),
            "old_literal_count": frozen.count(old),
            "candidate_is_exact_literal_replacement": candidate == frozen.replace(old, new),
            "all_other_bytes_preserved": candidate == frozen.replace(old, new),
        },
        "runtime_file_reads": 0,
        "weights_or_threshold_updates": 0,
    }


def _write_new_or_equal(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError("refusing to replace different existing bytes: " +
                               path.relative_to(ROOT).as_posix())


def build(check=False):
    outputs = _expected_outputs()
    record = _record(outputs)
    record_bytes = (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    expected = dict(outputs)
    expected[BUILD_RECORD] = record_bytes
    if check:
        differences = [path.relative_to(ROOT).as_posix() for path, content in expected.items()
                       if not path.is_file() or path.read_bytes() != content]
        if differences:
            raise RuntimeError("R6 build differs: " + ", ".join(differences))
        return record
    for path, content in expected.items():
        _write_new_or_equal(path, content)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify outputs without writing any file")
    args = parser.parse_args()
    try:
        value = build(check=args.check)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
