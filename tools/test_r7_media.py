"""Simulation-free tests for conditional R7 preserved-media publication."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest import mock

from tools import r7_media as M
from tools import validate_media as V
from tools import audit_public as A
from tools.r7_json import replace_json, write_json


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        replace_json(path, value)
    else:
        write_json(path, value)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(folder):
    base = Path(folder); source = base/"source"; root = base/"root"
    source.mkdir(); root.mkdir()
    candidate = Path(__file__).resolve().parents[1]/"experiments/r6_final/A_STATE.py"
    (root/"my_detector.py").write_bytes(candidate.read_bytes())
    sidecars = []
    for variant in ("R1", "SCALE_STATE"):
        path = root/f"results/v2_protocol/R5_BASE_validation_{variant}_servo.json"
        sidecars.append((variant, path))
    model = root/"models/r5/servo_SCALE_STATE.json"; dump(model, {"model": "fixed"})
    cases = []
    raw_shas = []
    for index, (case_id, stem) in enumerate(zip(M.CASE_IDS, M.SOURCE_STEMS)):
        signal = hashlib.sha256(("signal"+str(index)).encode()).hexdigest(); raw_shas.append(signal)
        inputs = []
        for variant, sidecar in sidecars:
            physical = hashlib.sha256(("physical"+str(index)).encode()).hexdigest()
            monitor = hashlib.sha256((variant+"monitor"+str(index)).encode()).hexdigest()
            inputs.extend([
                {"source_id": f"{variant}:{case_id}:signals", "sha256": signal},
                {"source_id": f"{variant}:{case_id}:physical", "sha256": physical},
                {"source_id": f"{variant}:{case_id}:monitor", "sha256": monitor},
                {"source_id": f"{variant}:{case_id}:row-sidecar",
                 "sha256": hashlib.sha256((variant+case_id+"row").encode()).hexdigest(),
                 "public_file": sidecar.relative_to(root).as_posix()},
            ])
        artifacts = []
        for suffix, frames, duration in ((".mp4", 54, 4.5), (".gif", 32, 4.57)):
            media = source/(stem+suffix); media.write_bytes((stem+suffix).encode())
            artifacts.append({"file": f"artifacts/media/{stem}{suffix}", "sha256": digest(media),
                              "bytes": media.stat().st_size, "width": 1280 if suffix==".mp4" else 640,
                              "height": 720 if suffix==".mp4" else 360,
                              "frame_count": frames, "duration_s": duration})
        cases.append({"source_case_id": case_id, "stem": stem,
                      "selection_basis": "illustrative validation pair",
                      "condition": {"mode": "servo", "index": index}, "threshold": 3.4,
                      "timewarp": [{"simulation_start_s": 0, "simulation_end_s": 6,
                                    "playback_speed": 1}], "rendered_frame_count": 54,
                      "inputs": inputs, "artifacts": artifacts})
    for variant, sidecar in sidecars:
        rows=[]
        for case in cases:
            values={item["source_id"].rsplit(":",1)[-1]:item["sha256"] for item in case["inputs"]
                    if item["source_id"].startswith(variant+":")}
            rows.append({"case":{"source_case_id":case["source_case_id"]},
                         "signal":{"sha256":values["signals"]},
                         "physical":{"sha256":values["physical"]},
                         "monitor":{"sha256":values["monitor"]}})
        dump(sidecar,{"variant":variant,"rows":rows})
        for case in cases:
            for item in case["inputs"]:
                if item.get("public_file") == sidecar.relative_to(root).as_posix():
                    item["sha256"] = hashlib.sha256((variant+case["source_case_id"]+"row").encode()).hexdigest()
    original = {"schema": "r5.media-manifest.v1", "renderer": {"file": "renderer.py", "sha256": "a"*64},
                "models": [{"file": model.relative_to(root).as_posix(), "sha256": digest(model)}],
                "software_gl": {"vendor": "Mesa", "renderer": "llvmpipe"},
                "encoding": {"fps": 12, "gif_fps": 7}, "cases": cases,
                "total_media_bytes": sum(x["bytes"] for c in cases for x in c["artifacts"])}
    dump(source/"r5_manifest.json", original)
    adoption = {"schema": "R7_ADOPTION.v1", "prefreeze_selected": {"servo": "SCALE_STATE", "torque": "R1"},
                "effective_variants": {"servo": "SCALE_STATE", "torque": "R1"},
                "final_assembly": "R7", "post_official_contract_passed": True,
                "fallback_only_from_prefrozen_assemblies": True, "repair_or_retest_after_holdout": False,
                "holdout_review_sha256": "a"*64, "official_sha256": "b"*64,
                "final_contract_review_sha256": "c"*64,
                "final_files": {"my_detector.py": M.CANDIDATE_SHA256}}
    dump(root/M.ADOPTION, adoption)
    aggregate = []
    counts = {("validation","servo"):38, ("validation","torque"):38,
              ("development","servo"):38, ("development","torque"):38,
              ("D4","servo"):385, ("D4","torque"):385,
              ("perturb","servo"):34, ("perturb","torque"):34}
    for (stage, mode), count in counts.items():
        aggregate.append({"stage": stage, "mode": mode, "streams": count,
                          "samples": count*6000, "passed_streams": count,
                          "all_bit_comparisons_passed": True})
    streams = [{"stream_id": stream, "stage": "validation", "mode": "servo",
                "stage_index": 18+index, "source_type": "study", "source_raw_sha256": raw_shas[index],
                "passed": True, "offline_decisions_exact": True, "proof_sha256": str(index+1)*64}
               for index, stream in enumerate(M.STREAM_IDS)]
    streams.extend({"stream_id": f"filler-{index:03d}", "stage": "D4", "mode": "torque",
                    "stage_index": index, "source_type": "cache", "source_raw_sha256": "d"*64,
                    "passed": True, "offline_decisions_exact": True, "proof_sha256": "e"*64}
                   for index in range(988))
    inherited = {"schema": "R6_SAVED_EQUIVALENCE.v1", "stream_count": 990,
                   "total_samples": 5_940_000, "aggregate": aggregate, "streams": streams,
                   "all_990_streams_bit_exact": True}
    dump(root/M.INHERITED_EQUIVALENCE, inherited)
    inherited_sha = digest(root/M.INHERITED_EQUIVALENCE)
    equivalence = {**inherited, "schema": "R7_SAVED_EQUIVALENCE.v1",
                   "inherited_source": {"file": M.INHERITED_EQUIVALENCE.as_posix(),
                                        "sha256": inherited_sha},
                   "no_replay_in_R7": True, "comparison_logic_changed": False,
                   "candidate_bytes_changed": False, "Plant_constructed": False,
                   "RNG_used": False}
    dump(root/M.EQUIVALENCE, equivalence)
    parity = {"schema": "R7_PARITY_RESULT.v1", "stage": "parity", "pair_count": 4,
              "passed": True, "no_exclusions": True,
              "physical_arrays": ["q","q_d","tau","e_mm","f_table","f_tool","qpos","r"],
              "rows": [{"passed": True} for _ in range(4)]}
    dump(root/M.PARITY, parity)
    return source, root


def fake_probe(path, _):
    mp4 = path.suffix == ".mp4"; frames = 54 if mp4 else 32; duration = 4.5 if mp4 else 4.57
    return {"codec": "h264" if mp4 else "gif", "width": 1280 if mp4 else 640,
            "height": 720 if mp4 else 360, "frame_count": frames,
            "duration_s": duration, "fps": frames/duration}


class R7MediaTests(unittest.TestCase):
    def test_reuse_is_create_only_and_validates_every_proof(self):
        with tempfile.TemporaryDirectory() as folder:
            source, root = fixture(folder)
            with mock.patch.object(M, "SOURCE_MANIFEST_SHA256", digest(source/"r5_manifest.json")), \
                    mock.patch.object(M, "INHERITED_EQUIVALENCE_SHA256",
                                      digest(root/M.INHERITED_EQUIVALENCE)):
                result = M.reuse(source, root)
                self.assertEqual(2, len(result["cases"]))
                self.assertEqual("pass", M.validate(root, probe_fn=fake_probe)["status"])
                with self.assertRaises(FileExistsError): M.reuse(source, root)

    def test_fallback_adoption_cannot_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            source, root = fixture(folder); adoption = M.read(root/M.ADOPTION)
            adoption["effective_variants"]["servo"] = "R1"; dump(root/M.ADOPTION, adoption)
            with mock.patch.object(M, "SOURCE_MANIFEST_SHA256", digest(source/"r5_manifest.json")), \
                    mock.patch.object(M, "INHERITED_EQUIVALENCE_SHA256",
                                      digest(root/M.INHERITED_EQUIVALENCE)):
                with self.assertRaisesRegex(ValueError, "not finally adopted"):
                    M.build_manifest(source, root)

    def test_missing_stream_or_parity_bit_proof_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            source, root = fixture(folder); equivalence = M.read(root/M.EQUIVALENCE)
            equivalence["streams"][0]["passed"] = False; dump(root/M.EQUIVALENCE, equivalence)
            with mock.patch.object(M, "SOURCE_MANIFEST_SHA256", digest(source/"r5_manifest.json")), \
                    mock.patch.object(M, "INHERITED_EQUIVALENCE_SHA256",
                                      digest(root/M.INHERITED_EQUIVALENCE)):
                with self.assertRaises(ValueError): M.build_manifest(source, root)
            equivalence["streams"][0]["passed"] = True; dump(root/M.EQUIVALENCE, equivalence)
            parity = M.read(root/M.PARITY); parity["rows"][0]["passed"] = False; dump(root/M.PARITY, parity)
            with mock.patch.object(M, "SOURCE_MANIFEST_SHA256", digest(source/"r5_manifest.json")), \
                    mock.patch.object(M, "INHERITED_EQUIVALENCE_SHA256",
                                      digest(root/M.INHERITED_EQUIVALENCE)):
                with self.assertRaisesRegex(ValueError, "full-loop"):
                    M.build_manifest(source, root)

    def test_inherited_saved_equivalence_must_match_every_scientific_field(self):
        with tempfile.TemporaryDirectory() as folder:
            source, root = fixture(folder)
            equivalence = M.read(root/M.EQUIVALENCE)
            equivalence["aggregate"][0]["samples"] += 1
            dump(root/M.EQUIVALENCE, equivalence)
            with mock.patch.object(M, "SOURCE_MANIFEST_SHA256", digest(source/"r5_manifest.json")), \
                    mock.patch.object(M, "INHERITED_EQUIVALENCE_SHA256",
                                      digest(root/M.INHERITED_EQUIVALENCE)):
                with self.assertRaisesRegex(ValueError, "payload differs"):
                    M.build_manifest(source, root)

    def test_public_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            source, root = fixture(folder)
            with mock.patch.object(M, "SOURCE_MANIFEST_SHA256", digest(source/"r5_manifest.json")), \
                    mock.patch.object(M, "INHERITED_EQUIVALENCE_SHA256",
                                      digest(root/M.INHERITED_EQUIVALENCE)):
                M.reuse(source, root)
                (root/"artifacts/media/09_r7_servo_positive_comparison.mp4").write_bytes(b"changed")
                result = M.validate(root, probe_fn=fake_probe)
            self.assertEqual("FAIL", result["status"])
            self.assertTrue(any("bytes or hash" in error for error in result["errors"]))

    def test_validate_all_requires_manifest_after_adoption(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);media=root/"artifacts/media";media.mkdir(parents=True)
            dump(media/"media_manifest.json", {})
            dump(media/"v2_manifest.json", {})
            dump(root/M.ADOPTION,{"schema":"R7_ADOPTION.v1",
                "effective_variants":{"servo":"SCALE_STATE","torque":"R1"}})
            passed={"status":"pass","errors":[]}
            with mock.patch.object(V,"validate",return_value=passed), \
                    mock.patch.object(V,"validate_v2",return_value=passed):
                result=V.validate_all(root,"ffprobe")
            self.assertIn("R7 r7_manifest.json is missing",result["errors"])
            self.assertTrue(V.r7_media_required(root))

    def test_validate_all_skips_fallback_and_dispatches_adopted_manifest(self):
        passed={"status":"pass","errors":[]}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); media=root/"artifacts/media"; media.mkdir(parents=True)
            dump(media/"media_manifest.json", {}); dump(media/"v2_manifest.json", {})
            dump(root/M.ADOPTION,{"schema":"R7_ADOPTION.v1",
                "effective_variants":{"servo":"R1","torque":"R1"}})
            with mock.patch.object(V,"validate",return_value=passed), \
                    mock.patch.object(V,"validate_v2",return_value=passed):
                fallback=V.validate_all(root,"ffprobe")
            self.assertEqual("pass",fallback["status"])
            dump(root/M.ADOPTION,{"schema":"R7_ADOPTION.v1",
                "effective_variants":{"servo":"SCALE_STATE","torque":"R1"}})
            dump(root/M.MANIFEST,{})
            with mock.patch.object(V,"validate",return_value=passed), \
                    mock.patch.object(V,"validate_v2",return_value=passed), \
                    mock.patch.object(V,"validate_r7",return_value=passed) as validate_r7:
                adopted=V.validate_all(root,"ffprobe")
            self.assertEqual("pass",adopted["status"])
            self.assertIn("r7",adopted["validated_manifests"])
            validate_r7.assert_called_once()

    def test_public_audit_requires_exact_r7_pair_only_after_adoption(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); media=root/"artifacts/media"; media.mkdir(parents=True)
            for stem in A.MEDIA_STEMS:
                for suffix in ("mp4","gif"):
                    (media/f"{stem}.{suffix}").write_bytes(b"media")
            dump(media/"media_manifest.json", {}); dump(media/"v2_manifest.json", {})
            dump(root/M.ADOPTION,{"schema":"R7_ADOPTION.v1",
                "effective_variants":{"servo":"R1","torque":"R1"}})
            with mock.patch.object(A,"media_tag_errors",return_value=[]), \
                    mock.patch.object(A,"media_metadata_bytes",return_value=b""), \
                    mock.patch.object(A,"binary_errors",return_value=[]):
                fallback=A.audit(root,ffprobe="ffprobe",require_checksums=False,require_media=True)
            self.assertEqual("pass",fallback["status"],fallback["errors"])
            dump(root/M.ADOPTION,{"schema":"R7_ADOPTION.v1",
                "effective_variants":{"servo":"SCALE_STATE","torque":"R1"}})
            with mock.patch.object(A,"media_tag_errors",return_value=[]), \
                    mock.patch.object(A,"media_metadata_bytes",return_value=b""), \
                    mock.patch.object(A,"binary_errors",return_value=[]):
                missing=A.audit(root,ffprobe="ffprobe",require_checksums=False,require_media=True)
            self.assertTrue(any("required eight" in error for error in missing["errors"]))
            for stem in A.R7_MEDIA_STEMS:
                for suffix in ("mp4","gif"):
                    (media/f"{stem}.{suffix}").write_bytes(b"media")
            dump(media/"r7_manifest.json", {})
            with mock.patch.object(A,"media_tag_errors",return_value=[]), \
                    mock.patch.object(A,"media_metadata_bytes",return_value=b""), \
                    mock.patch.object(A,"binary_errors",return_value=[]):
                adopted=A.audit(root,ffprobe="ffprobe",require_checksums=False,require_media=True)
            self.assertEqual("pass",adopted["status"],adopted["errors"])

    def test_malformed_public_manifest_fails_without_traceback(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/M.MANIFEST;path.parent.mkdir(parents=True)
            dump(path, {"schema": "r7.media-manifest.v1", "source_manifest": []})
            result=M.validate(root,probe_fn=fake_probe)
            self.assertEqual("FAIL",result["status"])


if __name__ == "__main__": unittest.main(verbosity=2)
