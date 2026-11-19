"""Build the deterministic, authored-source-only R24 research study bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STAGE = "experiments/r24_portable/bundle_stage.json"
INDEPENDENT = "experiments/r24_portable/independent_closure.json"
PREFIX = "r24-study-bundle"
SOURCE = PREFIX + "/research_source"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
ARM_ENTRYPOINTS = {
    "dc": "experiments.r24_loaded.dc:Detector",
    "joint": "experiments.r24_composed.detector:Detector",
}
ARM_SOURCES = {
    "dc": "experiments/r24_loaded/dc.py",
    "joint": "experiments/r24_composed/detector.py",
}


def digest(value):
    if isinstance(value, Path):
        value = value.read_bytes()
    return hashlib.sha256(value).hexdigest()


def relative_name(name):
    path = PurePosixPath(name)
    return bool(name and not path.is_absolute() and ".." not in path.parts
                and "\\" not in name)


def stage():
    path = ROOT / STAGE
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular R24 bundle stage manifest required")
    document = json.loads(path.read_bytes())
    files = document.get("files")
    if document.get("schema") != "r24-study-bundle-stage.v1" or not isinstance(files, dict):
        raise ValueError("R24 bundle stage schema differs")
    required = {
        "experiments/r24_portable/sensor_closure.json",
        "experiments/r24_transfer_replay/closure.json",
        "experiments/r24_transfer_replay/RESULT.json",
        "experiments/r24_transfer_replay/RESULT.md",
        "tools/r24_sensor_replay.py", "tools/r24_transfer_replay.py",
        INDEPENDENT, "tools/r24_holdout_replay.py",
        "results/r24/independent_lock.json", "results/r24/exposed_choice.json",
    }
    if not required <= set(files):
        raise ValueError("R24 bundle stage is incomplete")
    for name, expected in files.items():
        source = ROOT / name
        if (not relative_name(name) or name.startswith(("sim/", "model/"))
                or name == "evaluate.py" or len(expected) != 64
                or source.is_symlink() or not source.is_file()
                or digest(source) != expected):
            raise ValueError("R24 staged authored source pin differs: " + name)
    independent = json.loads((ROOT / INDEPENDENT).read_bytes())
    if (independent.get("schema") != "r24-independent-authored-closure.v1"
            or independent.get("entrypoint") != "experiments.r24_composed.detector:Detector"
            or independent.get("choice_sha256") !=
            "5560b6c2c228e6015380e4b2d2fa82712453977f007a145f8a16a0c90ca2f822"
            or independent.get("candidate_lock_sha256") !=
            "34df722725ca185d24fd09a6582ea5e63aa7e21141da63e4ba6768f9dde98b1b"
            or any(files.get(name) != expected
                   for name, expected in independent.get("files", {}).items())):
        raise ValueError("R24 independent replay closure differs")
    return document


def disclosure(path, document):
    if path is None:
        return None, None
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular R24 research disclosure required")
    raw = path.read_bytes()
    data = json.loads(raw)
    top = {"schema", "selected_for_independent", "entrypoint",
           "exposed_choice_sha256", "independent", "default_promoted"}
    independent = {"verdict", "rows", "failed_rows", "episodes", "result_sha256"}
    arm = data.get("selected_for_independent")
    result = data.get("independent")
    if (set(data) != top or data.get("schema") != "week09-r24-research-disclosure.v1"
            or arm not in ARM_ENTRYPOINTS or data.get("entrypoint") != ARM_ENTRYPOINTS[arm]
            or data.get("default_promoted") is not False or not isinstance(result, dict)
            or set(result) != independent or result.get("verdict") not in {"PASS", "FAIL"}
            or any(type(result.get(key)) is not int or result[key] < 0
                   for key in ("rows", "failed_rows", "episodes"))
            or result["failed_rows"] > result["rows"]
            or (result["verdict"] == "PASS") != (result["failed_rows"] == 0)):
        raise ValueError("R24 research disclosure contract differs")
    for key in ("exposed_choice_sha256",):
        value = data.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("R24 disclosure SHA-256 differs")
    value = result.get("result_sha256")
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("R24 independent result SHA-256 differs")
    if ARM_SOURCES[arm] not in document["files"]:
        raise ValueError("selected R24 arm is not enrolled in pinned replay source")
    return raw, data


def readme(selection=None):
    replay_arm = "dc" if selection is None else selection["selected_for_independent"]
    selection_text = """No R24 sensor arm is selected or promoted by this bundle.  `joint` is an
additional research-only composed arm."""
    if selection is not None:
        arm = selection["selected_for_independent"]
        result = selection["independent"]
        others = ", ".join(name for name in ("cone", "response", "dc", "joint")
                           if name != arm)
        selection_text = (
            f"The disclosed finite independent research panel selected `{arm}` "
            f"(`{selection['entrypoint']}`) and returned **{result['verdict']}**: "
            f"{result['rows']} rows, {result['failed_rows']} failed rows, and "
            f"{result['episodes']} episodes.  This is research-only evidence, not "
            f"qualification or default promotion.  Unselected arms: {others}.")
    return f"""# R24 finite research replay bundle

This is an authored-source research bundle, **not** the original course demo,
a submission tree, a newly blinded evaluation, hardware validation, a
promotion decision, or a safety proof.  It intentionally contains no stock
`sim/`, `model/`, or `evaluate.py`; supply the authenticated external course
kit separately.

{selection_text}  The included physical transfer result
truthfully records that its known reproduction passed and its predeclared new challenge did not pass strictly; see `research_source/experiments/r24_transfer_replay/RESULT.md` for the disclosed inherited and delay failures.

Run from any working directory (replace paths as needed):

```sh
python -B r24-study-bundle/research_source/tools/r24_sensor_replay.py \\
  --kit /path/to/icpbl_sim --study target --arm {replay_arm} --workers 1 \\
  --out /fresh/path/target-dc.json

python -B r24-study-bundle/research_source/tools/r24_transfer_replay.py \\
  --kit /path/to/icpbl_sim --workers 1 --out /fresh/path/transfer-result

python -B r24-study-bundle/research_source/tools/r24_holdout_replay.py \\
  --kit /path/to/icpbl_sim \\
  --lock r24-study-bundle/research_source/results/r24/independent_lock.json \\
  --jobs 1 --out /fresh/path/independent-replay
```

Sensor studies: `target`, `nearby`, `disclosed`.  Sensor arms: `cone`,
`response`, `dc`, `joint`.  Python dependencies are the same as the course
environment (including NumPy/MuJoCo); the bundle adds no dependency.
"""


def receipt(document, selection_raw=None, selection=None):
    result = ROOT / "experiments/r24_transfer_replay/RESULT.json"
    result_data = json.loads(result.read_bytes())
    output = {
        "schema": "r24-study-bundle-receipt.v1",
        "classification": "finite public research replay; selection and independent evaluation pending",
        "authored_source_only": True,
        "direct_drop_in_submission": False,
        "stock_kit_included": False,
        "selection_status": "pending" if selection is None else "disclosed",
        "selected_arm": None if selection is None else selection["selected_for_independent"],
        "selected_entrypoint": None if selection is None else selection["entrypoint"],
        "independent_evaluation": selection is not None,
        "default_promoted": False,
        "sensor_arms": ["cone", "response", "dc", "joint"],
        "joint_arm_status": "research_only",
        "independent_replay_available": True,
        "independent_replay_closure_sha256": digest(ROOT / INDEPENDENT),
        "stage_sha256": digest(ROOT / STAGE),
        "source_sha256": document["files"],
        "transfer_result_sha256": digest(result),
        "transfer_known_reproduction_passed": result_data["known_reproduction_passed"],
        "transfer_new_challenge_strict_passed": result_data["new_challenge_strict_passed"],
        "limitations": [
            "external authenticated stock kit and course Python dependencies required",
            "not original demo, hidden evaluation, hardware validation, promotion, or safety proof",
        ],
    }
    if selection is not None:
        output.update({
            "selection_sha256": digest(selection_raw),
            "exposed_choice_sha256": selection["exposed_choice_sha256"],
            "independent": selection["independent"],
            "selection_validation": "strict schema and SHA syntax plus pinned replayer enrollment",
            "unselected_arms": [name for name in ("cone", "response", "dc", "joint")
                                if name != selection["selected_for_independent"]],
        })
    return output


def zip_info(name):
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build(out, selection_path=None):
    if os.path.lexists(out) or out.parent.is_symlink() or not out.parent.is_dir():
        raise ValueError("fresh bundle output and regular parent required")
    document = stage()
    selection_raw, selection = disclosure(selection_path, document)
    entries = {
        PREFIX + "/README.md": readme(selection).encode(),
        PREFIX + "/receipt.json": (json.dumps(receipt(document, selection_raw, selection), indent=2,
                                               sort_keys=True, allow_nan=False) + "\n").encode(),
        SOURCE + "/" + STAGE: (ROOT / STAGE).read_bytes(),
    }
    if selection_raw is not None:
        entries[PREFIX + "/selection.json"] = selection_raw
    for name in document["files"]:
        entries[SOURCE + "/" + name] = (ROOT / name).read_bytes()
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name in sorted(entries):
            archive.writestr(zip_info(name), entries[name], compresslevel=9)
    return document, entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh output ZIP")
    parser.add_argument("--selection", type=Path,
                        help="optional final R24 research disclosure JSON")
    args = parser.parse_args(argv)
    try:
        document, entries = build(args.out, args.selection)
        print(json.dumps({"written": True, "files": len(entries),
                          "sha256": digest(args.out)}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
