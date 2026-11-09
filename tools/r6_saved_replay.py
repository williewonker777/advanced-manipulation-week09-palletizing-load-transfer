"""Bit-exact R6 replay over the 990 exposed R5 saved input streams.

The index is private because it contains absolute evidence paths. Each worker
processes exactly one stream without constructing a Plant. Collection is
create-only and requires every indexed stream to pass.
"""

from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import importlib.util
import json
import os
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import config as C
from sim import trajectory as T
from sim.drive import Drive
from sim.nominal import Nominal

D = ROOT / "results/v2_protocol"
HARNESS = Path(__file__).resolve()
HARNESS_TEST = ROOT / "tools/test_r6_saved_replay.py"
SAMPLES = 6000
SIGNALS = {
    "t", "th", "thd", "tau_m", "tau_s", "grip", "r_joint", "wrench",
    "active", "r1_detect", "ref_q", "ref_qd", "ref_qdd",
}
CACHE_MAP = {
    "t": "obs_t", "th": "encoder", "thd": "encoder_velocity",
    "tau_m": "motor_torque", "tau_s": "jts", "grip": "grip",
    "r_joint": "joint_residual", "wrench": "estimated_wrench",
    "active": "active", "r1_detect": "detect",
}
RUNTIME_PINS = {
    "results/v2_protocol/R5_RUNTIME_BUNDLE.json": "5c90f590dd0356cebaa902ff4216444e71080a30c1a7299c89f3e516379cfea9",
    "experiments/r5_final/A_STATE.py": "4f4b33c08872e8db612e1416468ea14dedc74964f05702f4aa084971f9197b44",
    "experiments/r6_final/A_STATE.py": "e26c203aabc476fc51a1464877a3435e484bb8fc33244777e59968f00969c643",
    "r1_detector.py": "e55995db6147ff8a543b816354376f968e0027d62d87c9dd1eeb55e0d88abbb4",
    "r1_servo.py": "40706968fb72ed8d7a864e44e828787ae8db0365c2927e5166e4a5b2d83bba8b",
    "r1_torque.py": "8f66f8f03cde65fcc6d8ef2f11643691cd8fe2e96dad6eaeaa17a0c7aa3db371",
    "sim/trajectory.py": "1b02f05ae110de09f736d6110f5ddf7bdc7111c1fec5fc11e23752d54819bb94",
    "sim/drive.py": "ad69f9d975832494708245c3c805ebb091509c0c3f969f45e90d256a36709a3c",
    "sim/nominal.py": "08d45260e1fa90fe7919394a27e3e238c32c0f81033d1d176c066439d3a3a489",
    "sim/config.py": "db2f85407992c21fb3e331acef93f3d6f91e86ad72a760da76a1b93226627735",
    "experiments/r5_core.py": "536f315e8dc7a8ea850f8df1858f4ca620c6fb05901805e8731879e573656af9",
    "models/r5/servo_SCALE_STATE.json": "d71d4573fa021b60a75a65678234e92e04d732e51c74ce8056fe263b8dfe5195",
}
STUDY_PINS = {
    ("validation", "servo"): ("SCALE_STATE", "d93fc4dd13636d120d6fb847e9a78bcbfb830997b8744e7c2e94d3bfdd7fd931"),
    ("validation", "torque"): ("R1", "11b0a725cf888b5c3c8a49ef314cd6d1632c40f61d42b65e3f180ad30c921312"),
    ("development", "servo"): ("R1", "961dd4bd6700cb2e166ad03b56b2da9054c0f92407481722b58de2ce86386ff6"),
    ("development", "torque"): ("R1", "ef3d63b785f7034677f67199de27a2c649e82bec8f106ee7bbd5c629eacd2c68"),
    ("noise", "servo"): ("R1", "dae427e95d92553e20991c937211d657291011a07c1c06034515a20f04d820a6"),
    ("noise", "torque"): ("R1", "545041526be3a999b37a4556d3f5165d69fc8cd463b1a9164ee7765c1c16003b"),
}
MANIFEST_PINS = {
    "D4": "fe3b16d155814e1055d397fd6f9b00da8aaa02830e2dde521838b3add410f775",
    "perturb": "cf8726c8f5db7d37af789f9d9c315cf8723cc38fc2baadcf9c606006780ce6e4",
}
COMPARE_FIELDS = (
    "residual", "detect", "base_detect", "own", "score", "ema",
    "monitor_present", "packet_r_joint", "packet_wrench", "active",
    "controller_output", "applied_command",
)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_create(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def available_ram_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024.0 * 1024.0)
    raise RuntimeError("MemAvailable is unavailable")


def require_worker_resources():
    available = available_ram_gib()
    if available < 6.0:
        raise RuntimeError(f"worker requires at least 6 GiB available RAM; found {available:.3f}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("saved replay requires GPU disabled")
    return available


def array_fingerprint(value):
    array = np.asarray(value)
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def exact_array_comparison(left, right):
    left_array, right_array = np.asarray(left), np.asarray(right)
    return {
        "exact": (left_array.dtype == right_array.dtype
                  and left_array.shape == right_array.shape
                  and left_array.tobytes(order="C") == right_array.tobytes(order="C")),
        "left": array_fingerprint(left_array), "right": array_fingerprint(right_array),
    }


def validate_signal_arrays(arrays, samples=SAMPLES):
    if set(arrays) != SIGNALS:
        raise ValueError("saved study signals must contain exactly the 13 exposed fields")
    vectors = {"t", "grip", "active", "r1_detect"}
    for name in SIGNALS:
        expected = (samples,) if name in vectors else (samples, 6)
        if np.asarray(arrays[name]).shape != expected:
            raise ValueError(f"wrong signal shape for {name}: {np.asarray(arrays[name]).shape}")
    return True


def _verify_runtime_pins():
    for relative, expected in RUNTIME_PINS.items():
        if sha256_path(ROOT / relative) != expected:
            raise RuntimeError("runtime input changed: " + relative)


def _study_rows(private):
    rows = []
    for (source_stage, mode), (variant, expected_sha) in STUDY_PINS.items():
        document = D / f"R5_BASE_{source_stage}_{variant}_{mode}.json"
        if sha256_path(document) != expected_sha:
            raise RuntimeError("saved study index changed: " + document.name)
        value = read_json(document)
        expected_count = 9 if source_stage == "noise" else 38
        if (value.get("schema") != "R5_BASE_STREAMS.v1"
                or (value.get("stage"), value.get("variant"), value.get("mode"))
                != (source_stage, variant, mode)
                or len(value.get("rows", [])) != expected_count):
            raise ValueError("saved study index identity/count mismatch: " + document.name)
        for index, sidecar_value in enumerate(value["rows"]):
            sidecar = private / source_stage / variant / mode / f"{index:03d}.json"
            if read_json(sidecar) != sidecar_value:
                raise ValueError("saved study sidecar differs from index: " + str(sidecar))
            case = sidecar_value["case"]
            output_stage = "perturb" if source_stage == "noise" else source_stage
            output_index = index + 25 if source_stage == "noise" else index
            directory = sidecar.parent
            rows.append({
                "stream_id": f"{output_stage}-{mode}-{output_index:03d}",
                "stage": output_stage, "mode": mode, "stage_index": output_index,
                "source_type": "fresh_noise" if source_stage == "noise" else "study",
                "source_stage": source_stage, "source_variant": variant,
                "mass": case["mass"], "com": case["com"], "samples": sidecar_value["samples"],
                "sidecar": {"absolute_path": str(sidecar.resolve()), "sha256": sha256_path(sidecar)},
                "raw": {"absolute_path": str((directory / sidecar_value["signal"]["file"]).resolve()),
                        "sha256": sidecar_value["signal"]["sha256"]},
                "monitor": {"absolute_path": str((directory / sidecar_value["monitor"]["file"]).resolve()),
                            "sha256": sidecar_value["monitor"]["sha256"]},
            })
    return rows


def _manifest_rows(path, kind):
    path = Path(path).resolve()
    if sha256_path(path) != MANIFEST_PINS[kind]:
        raise RuntimeError(kind + " manifest differs from its frozen pin")
    value = read_json(path)
    if value.get("schema") != "task377.r5_cache_source_manifest.v1":
        raise ValueError("wrong saved-cache manifest schema")
    expected = 385 if kind == "D4" else 25
    rows = []
    for mode in ("servo", "torque"):
        selected = [row for row in value["rows"] if row["mode"] == mode]
        if (len(selected) != expected
                or [row["source_row_index"] for row in selected] != list(range(expected))):
            raise ValueError(kind + " manifest inventory/order changed")
        for row in selected:
            index = row["source_row_index"]
            rows.append({
                "stream_id": f"{kind}-{mode}-{index:03d}",
                "stage": kind, "mode": mode, "stage_index": index,
                "source_type": "cache", "source_stage": row["source_stage"],
                "source_variant": "R1", "mass": row["mass"], "com": row["com"],
                "samples": SAMPLES, "source_condition_sha256": row["source_condition_sha256"],
                "source_document": row["source_document"], "sidecar": row["sidecar"],
                "raw": row["raw_npz"], "monitor": None,
            })
    return rows


def _verify_all_paths_before_npz_open(rows):
    for row in rows:
        for field in ("raw", "monitor"):
            item = row.get(field)
            if item is not None and sha256_path(item["absolute_path"]) != item["sha256"]:
                raise RuntimeError("raw hash mismatch before open: " + item["absolute_path"])
        if sha256_path(row["sidecar"]["absolute_path"]) != row["sidecar"]["sha256"]:
            raise RuntimeError("sidecar hash mismatch: " + row["sidecar"]["absolute_path"])
        source = row.get("source_document")
        if source is not None and sha256_path(source["absolute_path"]) != source["sha256"]:
            raise RuntimeError("source document hash mismatch: " + source["absolute_path"])


def _validate_indexed_rows(rows):
    for row in rows:
        if row["samples"] != SAMPLES or row["mode"] not in ("servo", "torque"):
            raise ValueError("stream sample count or mode is invalid: " + row["stream_id"])
        if not isinstance(row["mass"], (int, float)) or len(row["com"]) != 3:
            raise ValueError("stream payload metadata is invalid: " + row["stream_id"])
        sidecar = read_json(row["sidecar"]["absolute_path"])
        if row["source_type"] == "cache":
            case = sidecar["case"]
            if (case["architecture"] != row["mode"] or case["mass"] != row["mass"]
                    or case["com"] != row["com"] or sidecar["raw_sha256"] != row["raw"]["sha256"]):
                raise ValueError("cache sidecar mode/payload/raw mismatch: " + row["stream_id"])
            with np.load(row["raw"]["absolute_path"], allow_pickle=False) as raw:
                if not set(CACHE_MAP.values()) <= set(raw.files):
                    raise ValueError("cache raw lacks exposed saved signals")
                if any(np.asarray(raw[name]).shape[0] != SAMPLES for name in CACHE_MAP.values()):
                    raise ValueError("cache raw sample count mismatch")
        else:
            if (sidecar["mode"] != row["mode"] or sidecar["case"]["mass"] != row["mass"]
                    or sidecar["case"]["com"] != row["com"]):
                raise ValueError("study sidecar mode/payload mismatch: " + row["stream_id"])
            with np.load(row["raw"]["absolute_path"], allow_pickle=False) as raw:
                validate_signal_arrays({name: raw[name] for name in raw.files})
            with np.load(row["monitor"]["absolute_path"], allow_pickle=False) as monitor:
                if set(monitor.files) != {"detect", "own", "score", "ema"}:
                    raise ValueError("study monitor schema mismatch")
                if any(np.asarray(monitor[name]).shape != (SAMPLES,) for name in monitor.files):
                    raise ValueError("study monitor sample count mismatch")


def build_index(private, d4_manifest, perturb_manifest, output):
    _verify_runtime_pins()
    private = Path(private).resolve()
    output = Path(output).resolve()
    if private.is_relative_to(ROOT):
        raise ValueError("saved raw evidence must remain outside the repository")
    if output.is_relative_to(ROOT):
        raise ValueError("private replay index must remain outside the repository")
    rows = (_study_rows(private) + _manifest_rows(d4_manifest, "D4")
            + _manifest_rows(perturb_manifest, "perturb"))
    expected_counts = {
        ("validation", "servo"): 38, ("validation", "torque"): 38,
        ("development", "servo"): 38, ("development", "torque"): 38,
        ("D4", "servo"): 385, ("D4", "torque"): 385,
        ("perturb", "servo"): 34, ("perturb", "torque"): 34,
    }
    counts = Counter((row["stage"], row["mode"]) for row in rows)
    if len(rows) != 990 or dict(counts) != expected_counts or len({row["stream_id"] for row in rows}) != 990:
        raise RuntimeError("saved replay inventory is not the registered 990 streams")
    # Deliberate barrier: no NPZ is opened until every raw hash passes.
    _verify_all_paths_before_npz_open(rows)
    _validate_indexed_rows(rows)
    value = {
        "schema": "R6_SAVED_REPLAY_PRIVATE_INDEX.v1",
        "classification": "private index; absolute raw paths; do not publish",
        "runtime_pins": RUNTIME_PINS, "manifest_pins": MANIFEST_PINS,
        "harness": {"path": "tools/r6_saved_replay.py", "sha256": sha256_path(HARNESS)},
        "synthetic_tests": {"path": "tools/test_r6_saved_replay.py",
                            "sha256": sha256_path(HARNESS_TEST)},
        "stream_count": len(rows),
        "counts": {f"{stage}:{mode}": count for (stage, mode), count in sorted(counts.items())},
        "all_raw_hashes_verified_before_any_npz_open": True,
        "streams": sorted(rows, key=lambda row: (row["stage"], row["mode"], row["stage_index"])),
    }
    write_create(output, value)
    return value


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_index_harness(index):
    if (index.get("harness", {}).get("sha256") != sha256_path(HARNESS)
            or index.get("synthetic_tests", {}).get("sha256") != sha256_path(HARNESS_TEST)):
        raise RuntimeError("private replay index was built by different harness/test bytes")


def _payload(instance, mass, com):
    instance.nom = Nominal()
    instance.nom_box = Nominal()
    instance.box_mass = float(mass)
    instance.box_com = tuple(com)
    instance.nom_box.set_payload(instance.box_mass, instance.box_com)


def _load_stream(row):
    if sha256_path(row["raw"]["absolute_path"]) != row["raw"]["sha256"]:
        raise RuntimeError("worker raw hash mismatch")
    with np.load(row["raw"]["absolute_path"], allow_pickle=False) as raw:
        if row["source_type"] == "cache":
            arrays = {target: raw[source].copy() for target, source in CACHE_MAP.items()}
            references = [T.reference(float(t)) for t in arrays["t"]]
            for offset, name in enumerate(("ref_q", "ref_qd", "ref_qdd")):
                arrays[name] = np.asarray([reference[offset] for reference in references])
        else:
            arrays = {name: raw[name].copy() for name in raw.files}
    validate_signal_arrays(arrays)
    monitor = None
    if row.get("monitor") is not None:
        if sha256_path(row["monitor"]["absolute_path"]) != row["monitor"]["sha256"]:
            raise RuntimeError("worker monitor hash mismatch")
        with np.load(row["monitor"]["absolute_path"], allow_pickle=False) as raw:
            monitor = {name: raw[name].copy() for name in raw.files}
    return arrays, monitor


def _run_replay(detector_path, mode, mass, com, signals, module_tag, samples=SAMPLES,
                instrument_r1_active=False):
    controller_module = _load_module(ROOT / f"r1_{mode}.py", module_tag + "_controller")
    detector_module = _load_module(detector_path, module_tag + "_detector")
    controller = controller_module.Controller()
    _payload(controller, mass, com)
    drive = Drive()
    if mode == "servo":
        controller.drive = drive
    controller.reset()
    detector_class = detector_module.Detector
    if instrument_r1_active:
        class TrackedNormal(detector_module.NormalResidualModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.last_active = False

            def update(self, t, mass, com, wrench, active):
                self.last_active = bool(active)
                return super().update(t, mass, com, wrench, active)

        class ObservedR1Detector(detector_class):
            def reset(self):
                super().reset()
                current_mode = getattr(getattr(self, "ctrl", None), "MODE", None)
                if self.normal_model is not None:
                    self.normal_model = TrackedNormal(**detector_module.MODEL_PARAMETERS[current_mode])

        detector_class = ObservedR1Detector
    detector = detector_class()
    _payload(detector, mass, com)
    detector.ctrl = controller
    detector.reset()
    q0 = T.reference(0.0)[0]
    pending = (np.array([q0, np.zeros(6), np.zeros(6), np.zeros(6)])
               if mode == "servo" else np.zeros(6))
    values = defaultdict(list)
    for index in range(samples):
        answer = detector.update(
            signals["t"][index], signals["th"][index].copy(), signals["thd"][index].copy(),
            signals["tau_m"][index].copy(), signals["tau_s"][index].copy(),
            int(signals["grip"][index]))
        packet = detector._shared
        monitor = getattr(detector, "_r5_monitor", None)
        values["residual"].append(np.asarray(answer[0]).copy())
        values["detect"].append(bool(answer[1]))
        values["base_detect"].append(bool(getattr(detector, "r5_base_detect", answer[1])))
        values["own"].append(bool(getattr(detector, "r5_extra", False)))
        values["score"].append(0.0 if monitor is None else float(monitor.last_score))
        values["ema"].append(0.0 if monitor is None else float(monitor.last_ema))
        values["monitor_present"].append(monitor is not None)
        values["packet_r_joint"].append(packet.r_joint.copy())
        values["packet_wrench"].append(packet.last_wrench.copy())
        normal = getattr(detector, "normal_model", None)
        if normal is not None and not hasattr(normal, "last_active"):
            raise RuntimeError("active callback was not instrumented")
        values["active"].append(bool(getattr(normal, "last_active", False)))
        reference = (signals["ref_q"][index], signals["ref_qd"][index],
                     signals["ref_qdd"][index])
        output = np.asarray(controller.compute(
            signals["t"][index], signals["th"][index].copy(), signals["thd"][index].copy(),
            signals["tau_s"][index].copy(), int(signals["grip"][index]), *reference))
        values["controller_output"].append(output.copy())
        if mode == "servo":
            applied = drive.torque(pending, signals["th"][index], signals["thd"][index])
            pending = output
        else:
            applied, pending = pending, output
        values["applied_command"].append(np.clip(applied, -C.TAU_MAX, C.TAU_MAX))
    return {name: np.asarray(items) for name, items in values.items()}


def compare_replay(row, signals, monitor, old, new):
    comparisons = {"old_vs_new:" + name: exact_array_comparison(old[name], new[name])
                   for name in COMPARE_FIELDS}
    recorded_pairs = {
        "residual": (old["residual"], signals["r_joint"]),
        "packet_r_joint": (old["packet_r_joint"], signals["r_joint"]),
        "packet_wrench": (old["packet_wrench"], signals["wrench"]),
        "active": (old["active"], signals["active"]),
        "r1_detect": (old["base_detect"], signals["r1_detect"]),
        "initial_tau_m": (np.zeros(6, dtype=signals["tau_m"].dtype), signals["tau_m"][0]),
        "delayed_applied_command": (old["applied_command"][:-1], signals["tau_m"][1:]),
    }
    for name, (left, right) in recorded_pairs.items():
        comparisons["old_vs_recorded:" + name] = exact_array_comparison(left, right)
    if monitor is not None and row["source_variant"] == "SCALE_STATE":
        for name in ("detect", "own", "score", "ema"):
            comparisons["old_vs_recorded_monitor:" + name] = exact_array_comparison(old[name], monitor[name])
    return comparisons


def offline_inheritance(row, signals, old, new):
    if row["mode"] != "servo":
        return None
    core = _load_module(ROOT / "experiments/r5_core.py", "r6_saved_offline_core")
    payload = read_json(ROOT / "models/r5/servo_SCALE_STATE.json")
    episode = {name: signals[name] for name in SIGNALS}
    episode.update(mass=row["mass"], com=np.asarray(row["com"]), mode="servo",
                   role="analysis_eval", normal_only=False)
    trace = core.batch_trace(payload, episode)
    score_delta = float(np.max(np.abs(old["score"] - trace["score"])))
    ema_delta = float(np.max(np.abs(old["ema"] - trace["ema"])))
    return {
        "offline_own_vs_frozen_R5_actual_exact": exact_array_comparison(trace["own"], old["own"]),
        "offline_detect_vs_frozen_R5_actual_exact": exact_array_comparison(trace["detect"], old["detect"]),
        "offline_own_vs_R6_actual_exact": exact_array_comparison(trace["own"], new["own"]),
        "offline_detect_vs_R6_actual_exact": exact_array_comparison(trace["detect"], new["detect"]),
        "score_max_absolute_delta_informational": score_delta,
        "EMA_max_absolute_delta_informational": ema_delta,
        "historical_vectorized_tolerance": 1e-8,
        "score_within_historical_tolerance": score_delta <= 1e-8,
        "EMA_within_historical_tolerance": ema_delta <= 1e-8,
        "score_EMA_are_not_strict_bit_gates": True,
    }


def run_worker(index_path, stream_id, output_dir):
    available = require_worker_resources()
    _verify_runtime_pins()
    index_path = Path(index_path).resolve()
    index_sha = sha256_path(index_path)
    index = read_json(index_path)
    if index.get("schema") != "R6_SAVED_REPLAY_PRIVATE_INDEX.v1" or index.get("stream_count") != 990:
        raise ValueError("wrong or incomplete private replay index")
    _verify_index_harness(index)
    matches = [row for row in index["streams"] if row["stream_id"] == stream_id]
    if len(matches) != 1:
        raise ValueError("stream id missing or duplicated")
    row = matches[0]
    signals, monitor = _load_stream(row)
    start = time.monotonic()
    old_path = ROOT / ("experiments/r5_final/A_STATE.py" if row["mode"] == "servo"
                       else "r1_detector.py")
    new_path = ROOT / "experiments/r6_final/A_STATE.py"
    old = _run_replay(old_path, row["mode"], row["mass"], row["com"], signals, "r6_old",
                      instrument_r1_active=row["mode"] == "torque")
    new = _run_replay(new_path, row["mode"], row["mass"], row["com"], signals, "r6_new")
    comparisons = compare_replay(row, signals, monitor, old, new)
    inheritance = offline_inheritance(row, signals, old, new)
    monitor_absent = bool(not np.any(new["monitor_present"])) if row["mode"] == "torque" else None
    offline_decisions_passed = (True if inheritance is None else all(
        inheritance[name]["exact"] for name in (
            "offline_own_vs_frozen_R5_actual_exact", "offline_detect_vs_frozen_R5_actual_exact",
            "offline_own_vs_R6_actual_exact", "offline_detect_vs_R6_actual_exact")))
    passed = (all(value["exact"] for value in comparisons.values())
              and monitor_absent is not False and offline_decisions_passed)
    value = {
        "schema": "R6_SAVED_REPLAY_STREAM.v1", "stream_id": stream_id,
        "stage": row["stage"], "mode": row["mode"], "stage_index": row["stage_index"],
        "source_type": row["source_type"], "source_raw_sha256": row["raw"]["sha256"],
        "private_index_sha256": index_sha, "samples": SAMPLES,
        "mass": row["mass"], "com": row["com"], "comparisons": comparisons,
        "offline_R5_inheritance": inheritance,
        "torque_R6_monitor_absent_all_samples": monitor_absent, "passed": passed,
        "elapsed_seconds": time.monotonic() - start,
        "worker_peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "available_RAM_GiB_before": available,
    }
    target = Path(output_dir).resolve() / (stream_id + ".json")
    write_create(target, value)
    if not passed:
        raise RuntimeError("bit comparison failed; preserved result: " + str(target))
    return value


def aggregate_results(index, results):
    expected = {row["stream_id"] for row in index["streams"]}
    if set(results) != expected:
        raise RuntimeError("worker result inventory is incomplete or contains extras")
    groups = defaultdict(list)
    for row in results.values():
        groups[(row["stage"], row["mode"])].append(row)
    aggregate = []
    for (stage, mode), rows in sorted(groups.items()):
        aggregate.append({
            "stage": stage, "mode": mode, "streams": len(rows),
            "samples": sum(row["samples"] for row in rows),
            "passed_streams": sum(bool(row["passed"]) for row in rows),
            "all_bit_comparisons_passed": all(row["passed"] for row in rows),
        })
    return aggregate


def collect(index_path, worker_dir, output):
    _verify_runtime_pins()
    index_path = Path(index_path).resolve()
    index_sha = sha256_path(index_path)
    index = read_json(index_path)
    _verify_index_harness(index)
    results = {}
    for path in sorted(Path(worker_dir).resolve().glob("*.json")):
        row = read_json(path)
        if (row.get("schema") != "R6_SAVED_REPLAY_STREAM.v1"
                or row.get("private_index_sha256") != index_sha
                or row["stream_id"] in results):
            raise ValueError("foreign, stale, or duplicate worker result: " + str(path))
        results[row["stream_id"]] = row
    aggregate = aggregate_results(index, results)
    if len(results) != 990 or not all(row["passed"] for row in results.values()):
        raise RuntimeError("all 990 worker results must pass before collection")
    stream_summaries = []
    for row in sorted(results.values(), key=lambda item: item["stream_id"]):
        proof = {"comparisons": row["comparisons"],
                 "offline_R5_inheritance": row["offline_R5_inheritance"],
                 "torque_R6_monitor_absent_all_samples": row["torque_R6_monitor_absent_all_samples"]}
        inheritance = row["offline_R5_inheritance"]
        stream_summaries.append({
            "stream_id": row["stream_id"], "stage": row["stage"], "mode": row["mode"],
            "stage_index": row["stage_index"], "source_type": row["source_type"],
            "source_raw_sha256": row["source_raw_sha256"], "passed": row["passed"],
            "offline_decisions_exact": (None if inheritance is None else all(
                inheritance[name]["exact"] for name in (
                    "offline_own_vs_frozen_R5_actual_exact",
                    "offline_detect_vs_frozen_R5_actual_exact",
                    "offline_own_vs_R6_actual_exact",
                    "offline_detect_vs_R6_actual_exact"))),
            "proof_sha256": hashlib.sha256(json.dumps(
                proof, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        })
    value = {
        "schema": "R6_SAVED_EQUIVALENCE.v1", "private_index_sha256": index_sha,
        "stream_count": len(results), "samples_per_stream": SAMPLES,
        "total_samples": len(results) * SAMPLES, "aggregate": aggregate,
        "streams": stream_summaries,
        "all_990_streams_bit_exact": True, "Plant_constructed": False,
        "simulation_steps_executed": 0, "RNG_used": False,
    }
    write_create(output, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--private", type=Path, required=True)
    index_parser.add_argument("--d4-manifest", type=Path, required=True)
    index_parser.add_argument("--perturb-manifest", type=Path, required=True)
    index_parser.add_argument("--output", type=Path, required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--index", type=Path, required=True)
    worker_parser.add_argument("--stream-id", required=True)
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--index", type=Path, required=True)
    collect_parser.add_argument("--worker-dir", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, default=D / "R6_SAVED_EQUIVALENCE.json")
    args = parser.parse_args()
    try:
        if args.action == "index":
            value = build_index(args.private, args.d4_manifest, args.perturb_manifest, args.output)
        elif args.action == "worker":
            value = run_worker(args.index, args.stream_id, args.output_dir)
        else:
            value = collect(args.index, args.worker_dir, args.output)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


