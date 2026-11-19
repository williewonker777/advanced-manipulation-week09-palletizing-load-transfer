"""Build the deterministic kit-free R25 exposed research replay bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STAGE = "experiments/r25_portable/bundle_stage.json"
PREFIX = "r25-study-bundle"
SOURCE = PREFIX + "/research_source"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


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
        raise ValueError("regular R25 bundle stage required")
    document = json.loads(path.read_bytes())
    files = document.get("files")
    required = {"tools/r25_replay.py", "experiments/r25_fragment/study.py",
                "experiments/r25_portable/exposed_closure.json"}
    selection, independent = document.get("selection"), document.get("independent")
    if (document.get("schema") != "r25-study-bundle-stage.v1"
            or not isinstance(files, dict) or not required <= set(files)
            or (selection is None) != (independent is None)):
        raise ValueError("R25 bundle stage differs")
    if selection is not None:
        if (set(selection) != {"path", "sha256"}
                or selection["path"] != "results/r25/selection.json"
                or files.get(selection["path"]) != selection["sha256"]
                or set(independent) != {"closure", "closure_sha256", "replay_tool"}
                or independent["closure"] != "experiments/r25_portable/independent_closure.json"
                or files.get(independent["closure"]) != independent["closure_sha256"]
                or independent["replay_tool"] != "tools/r25_holdout_replay.py"
                or independent["replay_tool"] not in files):
            raise ValueError("R25 final attachment pins differ")
    for name, expected in files.items():
        source = ROOT / name
        if (not relative_name(name) or name.startswith(("sim/", "model/"))
                or name == "evaluate.py"
                or len(expected) != 64 or source.is_symlink() or not source.is_file()
                or digest(source) != expected):
            raise ValueError("R25 staged source pin differs: " + name)
    return document


def disclosure(raw, document):
    data = json.loads(raw)
    top = {"schema", "selected_for_independent", "entrypoint",
           "exposed_choice_sha256", "independent", "default_promoted"}
    result_keys = {"verdict", "rows", "failed_rows", "episodes", "result_sha256"}
    result = data.get("independent")
    if (set(data) != top or data.get("schema") != "week09-r25-research-disclosure.v1"
            or not isinstance(data.get("selected_for_independent"), str)
            or not data["selected_for_independent"]
            or not isinstance(data.get("entrypoint"), str)
            or not data["entrypoint"].startswith("experiments.r25_")
            or not data["entrypoint"].endswith(":Detector")
            or data.get("default_promoted") is not False
            or not isinstance(result, dict) or set(result) != result_keys
            or result.get("verdict") not in {"PASS", "FAIL"}
            or any(type(result.get(key)) is not int or result[key] < 0
                   for key in ("rows", "failed_rows", "episodes"))
            or result["failed_rows"] > result["rows"]
            or (result["verdict"] == "PASS") != (result["failed_rows"] == 0)):
        raise ValueError("R25 research disclosure contract differs")
    module = data["entrypoint"].rsplit(":", 1)[0].replace(".", "/") + ".py"
    if module not in document["files"]:
        raise ValueError("selected R25 entrypoint is not in the pinned bundle")
    for value in (data.get("exposed_choice_sha256"), result.get("result_sha256")):
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError("R25 disclosure SHA-256 differs")
    return data


def attached_disclosure(document):
    selection = document.get("selection")
    if selection is None:
        return None, None
    path = ROOT / selection["path"]
    raw = path.read_bytes()
    if path.is_symlink() or digest(raw) != selection["sha256"]:
        raise ValueError("R25 selection attachment differs")
    return raw, disclosure(raw, document)


def readme(selection=None):
    status = """This deterministic authored-source bundle is unselected research material. It
is not the original course demo, a submission tree, an independent result,
hardware validation, qualification, default promotion, or a safety proof.
No R25 candidate is selected. The three candidate entrypoints are exposed only
for reproducible finite development comparisons."""
    independent = "Independent replay/result material is excluded until an explicit final public selection and certified closure are released."
    holdout_command = ""
    if selection is not None:
        result = selection["independent"]
        status = (f"The immutable exposed-only choice selected `{selection['selected_for_independent']}` "
                  f"(`{selection['entrypoint']}`) before independent outcomes. The finite independent "
                  f"panel returned **{result['verdict']}**: {result['rows']} rows, "
                  f"{result['failed_rows']} failed rows, and {result['episodes']} episodes. "
                  "This is research-only evidence, not qualification or default promotion.")
        independent = ("The authoritative result uses the course-required MuJoCo 3.17.0 runtime. "
                       "The earlier MuJoCo 3.11.0 result is retained only as supplemental runtime "
                       "sensitivity evidence; it did not cause reselection. The included public holdout "
                       "replay uses the frozen amended lock and authenticated external kit.")
        holdout_command = """

python -B r25-study-bundle/research_source/tools/r25_holdout_replay.py \\
  --kit /path/to/icpbl_sim \\
  --lock r25-study-bundle/research_source/results/r25/independent_lock.json \\
  --jobs 2 --out /fresh/path/r25-independent-replay
"""
    return f"""# R25 exposed finite research replay

{status}

The bundle intentionally contains no stock `sim/`, `model/`, or `evaluate.py`.
Supply the authenticated external course kit separately. Run from any working
directory, replacing paths as needed:

```sh
python -B r25-study-bundle/research_source/tools/r25_replay.py \\
  --kit /path/to/icpbl_sim --study target --candidate stream --workers 1 \\
  --out /fresh/path/r25-target-stream.json

python -B r25-study-bundle/research_source/tools/r25_replay.py \\
  --kit /path/to/icpbl_sim --study stress --candidate stream --workers 1 \\
  --out /fresh/path/r25-stress-stream.json
{holdout_command}
```

Studies: `target`, `stress`, `nearby`, `r23`, `r24`. Candidates: `stream`,
`pickup`, `joint`. Python dependencies are those of the course environment;
the bundle adds no dependency. {independent}
"""


def receipt(document, selection_raw=None, selection=None):
    output = {
        "schema": "r25-study-bundle-receipt.v1",
        "classification": "finite exposed research replay; unselected; independent pending",
        "authored_source_only": True, "stock_kit_included": False,
        "direct_drop_in_submission": False,
        "selection_status": "pending" if selection is None else "disclosed",
        "selected_candidate": None if selection is None else selection["selected_for_independent"],
        "selected_entrypoint": None if selection is None else selection["entrypoint"],
        "independent_evaluation": selection is not None,
        "default_promoted": False,
        "studies": ["target", "stress", "nearby", "r23", "r24"],
        "candidates": ["stream", "pickup", "joint"],
        "stage_sha256": digest(ROOT / STAGE), "source_sha256": document["files"],
        "limitations": ["authenticated external stock kit required",
                        ("not original demo, hardware, qualification, default promotion, or safety proof"
                         if selection is not None else
                         "not original demo, independent evidence, hardware, qualification, or safety proof")],
    }
    if selection is not None:
        context = json.loads((ROOT / "results/r25/runtime_context.json").read_bytes())
        output.update({"selection_sha256": digest(selection_raw),
                       "exposed_choice_sha256": selection["exposed_choice_sha256"],
                       "independent": selection["independent"],
                       "independent_closure_sha256": document["independent"]["closure_sha256"],
                       "independent_replay_tool": document["independent"]["replay_tool"],
                       "authoritative_runtime": context["authoritative_environment"],
                       "authoritative_result_sha256": selection["independent"]["result_sha256"],
                       "supplemental_runtime": context["earlier_noncourse_environment"],
                       "supplemental_result_sha256": context["earlier_supplemental_result_sha256"],
                       "runtime_context_sha256": digest(ROOT / "results/r25/runtime_context.json"),
                       "runtime_interpretation": "course MuJoCo 3.17.0 is authoritative; 3.11.0 is supplemental only"})
    return output


def zip_info(name):
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build(out):
    if os.path.lexists(out) or out.parent.is_symlink() or not out.parent.is_dir():
        raise ValueError("fresh R25 ZIP and regular parent required")
    document = stage()
    selection_raw, selection = attached_disclosure(document)
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
    parser.add_argument("--out", required=True, type=Path, help="fresh output ZIP")
    args = parser.parse_args(argv)
    try:
        _, entries = build(args.out)
        print(json.dumps({"written": True, "files": len(entries),
                          "sha256": digest(args.out)}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
