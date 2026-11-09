"""Package and verify the two preserved R6 comparison illustrations.

The clips are byte-identical saved-validation illustrations made for the R5
SCALE_STATE candidate.  Reuse is allowed only after R6 proves that the repaired
assembly is bit-identical to that candidate and is selected as the final servo
detector.  The clips are explanatory artifacts, not new holdout evidence.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess


SOURCE_MANIFEST_SHA256 = "38e5e7ac30c5ebc53c12411d524839e13ffe5779cf291300b9cd971ea7bcd2a0"
CANDIDATE_SHA256 = "e26c203aabc476fc51a1464877a3435e484bb8fc33244777e59968f00969c643"
SOURCE_STEMS = ("09_r5_servo_positive_comparison", "10_r5_servo_negative_comparison")
PUBLIC_STEMS = ("09_r6_servo_positive_comparison", "10_r6_servo_negative_comparison")
CASE_IDS = ("validation_collision_010", "validation_collision_011")
STREAM_IDS = ("validation-servo-018", "validation-servo-019")
MANIFEST = Path("artifacts/media/r6_manifest.json")
ADOPTION = Path("results/v2_protocol/R6_ADOPTION.json")
EQUIVALENCE = Path("results/v2_protocol/R6_SAVED_EQUIVALENCE.json")
PARITY = Path("results/v2_protocol/R6_PARITY_RESULT.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _adopted(adoption):
    return (adoption.get("schema") == "R6_ADOPTION.v1"
            and adoption.get("prefreeze_selected") == {"servo": "SCALE_STATE", "torque": "R1"}
            and adoption.get("effective_variants") == {"servo": "SCALE_STATE", "torque": "R1"}
            and adoption.get("final_assembly") == "R6"
            and adoption.get("post_official_contract_passed") is True
            and adoption.get("fallback_only_from_prefrozen_assemblies") is True
            and adoption.get("repair_or_retest_after_holdout") is False
            and adoption.get("final_files", {}).get("my_detector.py") == CANDIDATE_SHA256
            and all(_safe_digest(adoption.get(name)) for name in
                    ("holdout_review_sha256", "official_sha256", "final_contract_review_sha256")))


def _equivalence_witness(document, stream_id, raw_sha):
    if (document.get("schema") != "R6_SAVED_EQUIVALENCE.v1"
            or document.get("stream_count") != 990
            or document.get("total_samples") != 5_940_000
            or document.get("all_990_streams_bit_exact") is not True
            or len(document.get("streams", [])) != 990):
        raise ValueError("saved equivalence is incomplete")
    aggregate = document.get("aggregate", [])
    if (len(aggregate) != 8 or sum(row.get("streams", 0) for row in aggregate) != 990
            or sum(row.get("samples", 0) for row in aggregate) != 5_940_000
            or any(row.get("passed_streams") != row.get("streams")
                   or row.get("samples") != row.get("streams", 0) * 6000
                   or row.get("all_bit_comparisons_passed") is not True for row in aggregate)):
        raise ValueError("saved equivalence aggregate is incomplete")
    matches = [row for row in document.get("streams", []) if row.get("stream_id") == stream_id]
    if len(matches) != 1:
        raise ValueError("saved equivalence witness is missing")
    row = matches[0]
    if (row.get("stage") != "validation" or row.get("mode") != "servo"
            or row.get("source_raw_sha256") != raw_sha or row.get("passed") is not True
            or row.get("offline_decisions_exact") is not True
            or not _safe_digest(row.get("proof_sha256"))):
        raise ValueError("saved equivalence witness differs")
    return {key: row[key] for key in ("stream_id", "stage", "mode", "stage_index",
                                      "source_type", "source_raw_sha256", "passed",
                                      "offline_decisions_exact", "proof_sha256")}


def _parity_passed(document):
    return (document.get("schema") == "R6_PARITY_RESULT.v1"
            and document.get("stage") == "parity"
            and document.get("pair_count") == 4
            and document.get("passed") is True
            and document.get("no_exclusions") is True
            and document.get("physical_arrays") ==
                ["q", "q_d", "tau", "e_mm", "f_table", "f_tool", "qpos", "r"]
            and len(document.get("rows", [])) == 4
            and all(row.get("passed") is True for row in document["rows"]))


def _source_signal(case):
    matches = [item for item in case.get("inputs", [])
               if item.get("source_id") == "SCALE_STATE:" + case["source_case_id"] + ":signals"]
    if len(matches) != 1 or not _safe_digest(matches[0].get("sha256")):
        raise ValueError("source signal identity is missing")
    return matches[0]["sha256"]


def _verify_public_inputs(root, source_manifest):
    catalogs = {}
    for case in source_manifest.get("cases", []):
        for item in case.get("inputs", []):
            public_file = item.get("public_file")
            if public_file is None:
                continue
            path = root / public_file
            if (Path(public_file).is_absolute() or ".." in Path(public_file).parts
                    or path.is_symlink() or not path.is_file()):
                raise ValueError("public source sidecar differs: " + str(public_file))
            document = read(path)
            rows = [row for row in document.get("rows", [])
                    if row.get("case", {}).get("source_case_id") == case.get("source_case_id")]
            if len(rows) != 1:
                raise ValueError("public source case is missing: " + str(public_file))
            row = rows[0]
            prefix = item.get("source_id", "").split(":", 1)[0]
            expected_inputs = {entry.get("source_id", "").rsplit(":", 1)[-1]: entry.get("sha256")
                               for entry in case.get("inputs", [])
                               if entry.get("source_id", "").startswith(prefix + ":")}
            for kind, row_key in (("signals", "signal"), ("physical", "physical"),
                                  ("monitor", "monitor")):
                if kind in expected_inputs and row.get(row_key, {}).get("sha256") != expected_inputs[kind]:
                    raise ValueError("public source row hash differs: " + str(public_file))
            catalogs[public_file] = {"file": public_file, "sha256": sha256(path)}
    for model in source_manifest.get("models", []):
        relative = model.get("file")
        path = root / str(relative)
        if (not relative or Path(relative).is_absolute() or ".." in Path(relative).parts
                or path.is_symlink() or not path.is_file() or sha256(path) != model.get("sha256")):
            raise ValueError("public source model differs")
    return [catalogs[name] for name in sorted(catalogs)]


def build_manifest(source, root):
    source, root = Path(source).resolve(), Path(root).resolve()
    source_path = source / "r5_manifest.json"
    if source_path.is_symlink() or not source_path.is_file() or sha256(source_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("preserved source manifest differs")
    original = read(source_path)
    original_text = source_path.read_text(encoding="utf-8")
    if (original.get("schema") != "r5.media-manifest.v1"
            or tuple(case.get("source_case_id") for case in original.get("cases", [])) != CASE_IDS
            or tuple(case.get("stem") for case in original.get("cases", [])) != SOURCE_STEMS):
        raise ValueError("preserved source inventory differs")
    adoption, equivalence, parity = (read(root / path) for path in (ADOPTION, EQUIVALENCE, PARITY))
    if not _adopted(adoption):
        raise ValueError("R6 servo candidate was not finally adopted")
    if not _parity_passed(parity):
        raise ValueError("fresh full-loop parity proof did not pass")
    detector = root / "my_detector.py"
    if detector.is_symlink() or not detector.is_file() or sha256(detector) != CANDIDATE_SHA256:
        raise ValueError("current submitted detector differs from the adopted candidate")
    catalogs = _verify_public_inputs(root, original)
    cases = []
    total = 0
    for old, new_stem, stream_id in zip(original["cases"], PUBLIC_STEMS, STREAM_IDS):
        raw_sha = _source_signal(old)
        witness = _equivalence_witness(equivalence, stream_id, raw_sha)
        artifacts = []
        for item in old["artifacts"]:
            suffix = Path(item["file"]).suffix
            source_media = source / (old["stem"] + suffix)
            if (source_media.is_symlink() or not source_media.is_file()
                    or sha256(source_media) != item["sha256"]
                    or source_media.stat().st_size != item["bytes"]):
                raise ValueError("preserved media differs: " + source_media.name)
            total += item["bytes"]
            artifacts.append({**item, "file": f"artifacts/media/{new_stem}{suffix}",
                              "source_byte_sha256": item["sha256"]})
        cases.append({"source_case_id": old["source_case_id"], "stem": new_stem,
                      "source_stem": old["stem"], "selection_basis": old["selection_basis"],
                      "condition": old["condition"], "threshold": old["threshold"],
                      "timewarp": old["timewarp"], "rendered_frame_count": old["rendered_frame_count"],
                      "source_inputs": old["inputs"], "equivalence_witness": witness,
                      "artifacts": artifacts})
    return {"schema": "r6.media-manifest.v1",
            "scope_note": "Preserved validation illustrations; not a new holdout, official run, or independent efficacy result.",
            "reuse_contract": "Media bytes and rendered traces are unchanged. R6 reuse is authorized by saved-stream and fresh full-loop bit equality.",
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_manifest_canonical_sha256": canonical_sha(original),
            "source_manifest_utf8": original_text,
            "source_manifest": original,
            "public_source_catalogs": catalogs,
            "adoption": {"file": ADOPTION.as_posix(), "sha256": sha256(root / ADOPTION),
                         "effective_variants": adoption["effective_variants"]},
            "candidate": {"file": "my_detector.py", "sha256": CANDIDATE_SHA256},
            "saved_equivalence": {"file": EQUIVALENCE.as_posix(),
                                  "sha256": sha256(root / EQUIVALENCE), "streams": 990},
            "fresh_full_loop_parity": {"file": PARITY.as_posix(),
                                       "sha256": sha256(root / PARITY), "episodes": 8,
                                       "pair_count": parity["pair_count"]},
            "software_gl": original["software_gl"], "encoding": original["encoding"],
            "source_renderer": original["renderer"], "cases": cases,
            "total_media_bytes": total, "raw_traces_distributed": False,
            "new_simulation_or_rendering": False}


def reuse(source, root):
    source, root = Path(source).resolve(), Path(root).resolve()
    manifest = build_manifest(source, root)
    targets = [root / MANIFEST]
    for case in manifest["cases"]:
        targets.extend(root / item["file"] for item in case["artifacts"])
    if any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError("R6 media destination already exists")
    for case in manifest["cases"]:
        for item in case["artifacts"]:
            source_file = source / (case["source_stem"] + Path(item["file"]).suffix)
            target = root / item["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with source_file.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
                dst.flush(); os.fsync(dst.fileno())
            if sha256(target) != item["sha256"]:
                raise RuntimeError("copied media differs")
    target = root / MANIFEST
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    return manifest


def validate(root, ffprobe="ffprobe", probe_fn=None):
    root = Path(root).resolve(); path = root / MANIFEST; errors = []; report = []
    if not path.is_file() or path.is_symlink():
        return {"status": "FAIL", "errors": ["R6 media manifest is missing or unsafe"]}
    try:
        data = read(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"status": "FAIL", "errors": ["R6 media manifest is unreadable"]}
    if not isinstance(data, dict):
        return {"status": "FAIL", "errors": ["R6 media manifest root is invalid"]}
    if probe_fn is None:
        from tools.validate_media import probe as probe_fn
    if data.get("schema") != "r6.media-manifest.v1": errors.append("R6 media schema differs")
    original = data.get("source_manifest", {})
    original_text = data.get("source_manifest_utf8", "")
    try:
        parsed_original = json.loads(original_text) if isinstance(original_text, str) else None
    except json.JSONDecodeError:
        parsed_original = None
    if (data.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
            or not isinstance(original_text, str)
            or hashlib.sha256(original_text.encode("utf-8")).hexdigest() != SOURCE_MANIFEST_SHA256
            or parsed_original != original or not isinstance(original, dict)
            or data.get("source_manifest_canonical_sha256") != canonical_sha(original)
            or original.get("schema") != "r5.media-manifest.v1"):
        errors.append("R6 source manifest binding differs")
    try:
        adoption, equivalence, parity = (read(root / p) for p in (ADOPTION, EQUIVALENCE, PARITY))
        if not _adopted(adoption): errors.append("R6 adopted detector proof differs")
        if data.get("adoption") != {"file": ADOPTION.as_posix(), "sha256": sha256(root / ADOPTION),
                                    "effective_variants": adoption.get("effective_variants")}:
            errors.append("R6 adoption binding differs")
        if not _parity_passed(parity): errors.append("R6 fresh parity proof differs")
        for record, expected_path in ((data.get("saved_equivalence", {}), EQUIVALENCE),
                                      (data.get("fresh_full_loop_parity", {}), PARITY)):
            if record.get("file") != expected_path.as_posix() or record.get("sha256") != sha256(root / expected_path):
                errors.append("R6 proof hash binding differs")
        candidate = root / "my_detector.py"
        if (data.get("candidate") != {"file": "my_detector.py", "sha256": CANDIDATE_SHA256}
                or candidate.is_symlink() or not candidate.is_file() or sha256(candidate) != CANDIDATE_SHA256):
            errors.append("R6 candidate source binding differs")
        if data.get("public_source_catalogs") != _verify_public_inputs(root, original):
            errors.append("R6 public source catalog binding differs")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append("R6 proof input invalid: " + str(exc))
        equivalence = {}
    cases = data.get("cases", [])
    if not isinstance(cases, list) or any(not isinstance(row, dict) for row in cases):
        errors.append("R6 media case list is invalid"); cases = []
    if (tuple(row.get("source_case_id") for row in cases) != CASE_IDS
            or tuple(row.get("stem") for row in cases) != PUBLIC_STEMS):
        errors.append("R6 media case inventory differs")
    seen = set(); total = 0
    for index, row in enumerate(cases):
        source_rows = original.get("cases", []) if isinstance(original, dict) else []
        if not isinstance(source_rows, list): source_rows = []
        if index >= len(source_rows): errors.append("R6 source case is missing"); continue
        source_case = source_rows[index]
        if not isinstance(source_case, dict): errors.append("R6 source case is invalid"); continue
        if (row.get("source_stem") != SOURCE_STEMS[index]
                or row.get("condition") != source_case.get("condition")
                or row.get("threshold") != source_case.get("threshold")
                or row.get("timewarp") != source_case.get("timewarp")
                or row.get("source_inputs") != source_case.get("inputs")):
            errors.append(row.get("stem", "R6 case") + ": source trace metadata differs")
        try:
            expected_witness = _equivalence_witness(equivalence, STREAM_IDS[index], _source_signal(source_case))
            if row.get("equivalence_witness") != expected_witness:
                errors.append(row["stem"] + ": equivalence witness differs")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(row.get("stem", "R6 case") + ": invalid witness: " + str(exc))
        source_artifact_rows = source_case.get("artifacts", [])
        source_artifacts = ({Path(item["file"]).suffix: item for item in source_artifact_rows
                             if isinstance(item, dict) and isinstance(item.get("file"), str)}
                            if isinstance(source_artifact_rows, list) else {})
        artifact_rows = row.get("artifacts", [])
        if not isinstance(artifact_rows, list) or any(not isinstance(item, dict) for item in artifact_rows):
            errors.append(row.get("stem", "R6 case") + ": artifact list is invalid"); continue
        for item in artifact_rows:
            relative = item.get("file", ""); suffix = Path(relative).suffix
            expected = source_artifacts.get(suffix); media = root / relative
            if (relative != f"artifacts/media/{PUBLIC_STEMS[index]}{suffix}" or relative in seen
                    or suffix not in (".mp4", ".gif")):
                errors.append(row.get("stem", "R6 case") + ": artifact path differs"); continue
            seen.add(relative)
            if expected is None or any(item.get(k) != expected.get(k) for k in
                                       ("sha256", "bytes", "width", "height", "frame_count", "duration_s")):
                errors.append(relative + ": source artifact metadata differs")
            if item.get("source_byte_sha256") != item.get("sha256"):
                errors.append(relative + ": source byte identity differs")
            if media.is_symlink() or not media.is_file(): errors.append(relative + ": missing or unsafe"); continue
            raw = media.read_bytes(); total += len(raw)
            if len(raw) != item.get("bytes") or hashlib.sha256(raw).hexdigest() != item.get("sha256"):
                errors.append(relative + ": bytes or hash differ")
            try: actual = probe_fn(media, ffprobe)
            except (OSError, ValueError, KeyError, subprocess.TimeoutExpired, json.JSONDecodeError):
                errors.append(relative + ": probe failed"); continue
            for field in ("width", "height", "frame_count"):
                if actual.get(field) != item.get(field): errors.append(relative + ": " + field + " differs")
            if abs(actual.get("duration_s", 0) - item.get("duration_s", 0)) > .03:
                errors.append(relative + ": duration differs")
            expected_fps = item.get("frame_count", 0) / item.get("duration_s", 1)
            if abs(actual.get("fps", 0) - expected_fps) > .03:
                errors.append(relative + ": frame rate differs")
            expected_codec = "h264" if suffix == ".mp4" else "gif"
            if actual.get("codec") != expected_codec:
                errors.append(relative + ": codec differs")
            report.append({"file": relative, **actual})
    if len(seen) != 4 or total != data.get("total_media_bytes"):
        errors.append("R6 media coverage or total differs")
    return {"status": "pass" if not errors else "FAIL", "errors": errors,
            "files": report, "cases": len(cases), "total_media_bytes": total,
            "scope": "preserved bytes, source traces, final adoption, saved equivalence, fresh parity and ffprobe"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    reuse_parser = sub.add_parser("reuse")
    reuse_parser.add_argument("--source", type=Path, required=True)
    reuse_parser.add_argument("--root", type=Path, required=True)
    check_parser = sub.add_parser("check")
    check_parser.add_argument("--root", type=Path, required=True)
    check_parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    if args.action == "reuse":
        result = reuse(args.source, args.root)
        print("R6 preserved media packaged:", len(result["cases"]), "cases")
        return 0
    result = validate(args.root, args.ffprobe)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
