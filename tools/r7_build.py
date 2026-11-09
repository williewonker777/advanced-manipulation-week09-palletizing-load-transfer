"""Copy the immutable R6 candidate and R1 fallback into the R7 freeze set."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.r7_json import write_json

SOURCE_DIR = ROOT / "experiments/r6_final"
FINAL_DIR = ROOT / "experiments/r7_final"
BUILD_RECORD = ROOT / "results/v2_protocol/R7_BUILD.json"
R6_BUILD = ROOT / "results/v2_protocol/R6_BUILD.json"
TEST_SOURCE = ROOT / "tools/test_r7_build.py"
SOURCES = {"A_STATE": SOURCE_DIR / "A_STATE.py", "R1": SOURCE_DIR / "R1.py"}
OUTPUTS = {name: FINAL_DIR / path.name for name, path in SOURCES.items()}
EXPECTED_HASHES = {
    "A_STATE": "e26c203aabc476fc51a1464877a3435e484bb8fc33244777e59968f00969c643",
    "R1": "e55995db6147ff8a543b816354376f968e0027d62d87c9dd1eeb55e0d88abbb4",
}
R6_BUILD_SHA256 = "7847fea26fa0d28d43a9d88884de68b2789c1c0dc77abe3f3ec2987b2f2d1317"


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_path(path):
    return sha256_bytes(Path(path).read_bytes())


def verify_inputs():
    if sha256_path(R6_BUILD) != R6_BUILD_SHA256:
        raise RuntimeError("immutable R6 build record differs")
    for name, path in SOURCES.items():
        if sha256_path(path) != EXPECTED_HASHES[name]:
            raise RuntimeError("immutable R6 assembly differs: " + name)
    record = json.loads(R6_BUILD.read_text(encoding="utf-8"))
    if len(record.get("models", [])) != 8:
        raise RuntimeError("R6 build record must pin exactly eight models")
    for row in record["models"]:
        if sha256_path(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError("immutable R6 model differs: " + row["path"])
    return record


def _record(outputs):
    inherited = verify_inputs()
    return {
        "schema": "R7_BUILD.v1",
        "immutable_R6_build": {"path": R6_BUILD.relative_to(ROOT).as_posix(),
                               "sha256": R6_BUILD_SHA256},
        "immutable_sources": [
            {"name": name, "path": SOURCES[name].relative_to(ROOT).as_posix(),
             "sha256": EXPECTED_HASHES[name]}
            for name in ("A_STATE", "R1")
        ],
        "models": [{"path": row["path"], "sha256": row["sha256"]}
                   for row in inherited["models"]],
        "model_count": 8,
        "builder": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                    "sha256": sha256_path(Path(__file__).resolve())},
        "regression_test": {"path": TEST_SOURCE.relative_to(ROOT).as_posix(),
                            "sha256": sha256_path(TEST_SOURCE)},
        "generated": [
            {"name": name, "path": OUTPUTS[name].relative_to(ROOT).as_posix(),
             "sha256": sha256_bytes(outputs[name]),
             "byte_equal_to_source": outputs[name] == SOURCES[name].read_bytes()}
            for name in ("A_STATE", "R1")
        ],
        "assembly_logic_changes": 0,
        "model_weight_or_threshold_updates": 0,
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
    outputs = {name: path.read_bytes() for name, path in SOURCES.items()}
    record = _record(outputs)
    if check:
        differences = [OUTPUTS[name].relative_to(ROOT).as_posix()
                       for name, content in outputs.items()
                       if not OUTPUTS[name].is_file() or OUTPUTS[name].read_bytes() != content]
        if not BUILD_RECORD.is_file():
            differences.append(BUILD_RECORD.relative_to(ROOT).as_posix())
        else:
            try:
                if json.loads(BUILD_RECORD.read_text(encoding="utf-8")) != record:
                    differences.append(BUILD_RECORD.relative_to(ROOT).as_posix())
            except (OSError, json.JSONDecodeError):
                differences.append(BUILD_RECORD.relative_to(ROOT).as_posix())
        if differences:
            raise RuntimeError("R7 build differs: " + ", ".join(differences))
        return record
    for name, content in outputs.items():
        _write_new_or_equal(OUTPUTS[name], content)
    write_json(BUILD_RECORD, record)
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
