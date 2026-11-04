#!/usr/bin/env python3
"""Last-only R3 default-name official regression and immutable fallback check."""
import argparse
from datetime import datetime, timezone
import math
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from tools.check_integrity import check_integrity
from tools.r3_build import compiled_path
from tools import r3_extended as E
from tools import r3_select as S
from tools import r3_study as R
from tools import r3_resources as resources

D = ROOT / "results" / "v2_protocol"
PROTOCOL = D / "PROTOCOL_R3.json"
FREEZE = D / "R3_FREEZE.json"
HOLDOUT = D / "R3_HOLDOUT.json"
HOLDOUT_REVIEW = D / "R3_HOLDOUT_REVIEW.json"
HOLDOUT_MARKER = D / "R3_HOLDOUT_OPENED.json"
MARKER = D / "R3_OFFICIAL_OPENED.json"
OFFICIAL_DIR = D / "R3_OFFICIAL_DEFAULT"
RESULT = D / "R3_OFFICIAL_REGRESSION.json"
R1_EXPOSED = D / "S4_OFFICIAL.json"

CONTROL_METRICS = (("rms_mm", 0), ("max_mm", 1), ("pick_nm", 3),
                   ("place_nm", 4), ("table_n", 5))
FORCES = ("10N", "15N", "40N", "100N")
OFFICIAL_FROZEN_PATHS = (
    "tools/r3_official_regression.py",
    "tools/validate_submission.py",
    "tools/compare_baseline.py",
    "tools/r3_select.py",
    "evaluate.py",
    "results/v2_protocol/S4_OFFICIAL.json",
)


def _ratio(candidate, baseline):
    candidate, baseline = float(candidate), float(baseline)
    if candidate < 0.0 or baseline < 0.0:
        return float("inf")
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else float("inf")
    return candidate / baseline


def _delay(cell):
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def _false_alarms(cell):
    text = str(cell).strip()
    return int(text[:-1] if text.endswith("회") else text)


def _control_review(mode, current, baseline, protocol):
    errors, rows = [], []
    expected_masses = {"0.0", "4.0", "8.0"}
    if set(current) != expected_masses:
        return rows, [{"section": "control", "reason": "control masses differ",
                       "actual": sorted(current)}]
    limit = float(protocol["official"]["control_cost_ratio_vs_R1_max"])
    for mass in sorted(expected_masses, key=float):
        cells = current[mass]
        if not isinstance(cells, list) or len(cells) != 10:
            errors.append({"section": "control", "mass": float(mass),
                           "reason": "malformed evaluator control row"})
            continue
        old = baseline["control"][mass]
        ratios = {name: _ratio(cells[index], old[name])
                  for name, index in CONTROL_METRICS}
        saturation = float(cells[2])
        false_alarms = [int(value) for value in cells[6:9]]
        normal = cells[9] == "정상"
        failures = [f"{name}_ratio_gt_{limit:g}" for name, value in ratios.items()
                    if value > limit]
        if saturation != float(protocol["official"]["saturation_pct"]):
            failures.append("nonzero_saturation")
        if false_alarms != [0, 0, 0]:
            failures.append("reference_false_alarm")
        if not normal:
            failures.append("divergence")
        row = {"architecture": mode, "mass_kg": float(mass),
               "cost_ratios_vs_R1": ratios, "saturation_pct": saturation,
               "false_alarms": false_alarms, "normal_status": normal,
               "pass": not failures, "failures": failures}
        rows.append(row)
        if failures:
            errors.append({"section": "control", **row})
    return rows, errors


def _collision_review(mode, current, baseline, protocol):
    errors, rows = [], []
    expected = {(float(row["mass"]), float(row["time"])): row
                for row in baseline["collisions"]}
    actual = {(float(row["mass"]), float(row["time"])): row for row in current}
    if actual.keys() != expected.keys():
        return rows, [{"section": "collision", "reason": "collision matrix differs",
                       "missing": sorted(expected.keys() - actual.keys()),
                       "extra": sorted(actual.keys() - expected.keys())}]
    additional = float(protocol["official"]["max_additional_delay_ms"])
    caps = protocol["official"]["eight_kg_transport_delay_caps_ms"][mode]
    for key in sorted(expected):
        old, row = expected[key], actual[key]
        cells = row.get("cells")
        if not isinstance(cells, list) or len(cells) != 6:
            errors.append({"section": "collision", "mass_kg": key[0],
                           "time_s": key[1], "reason": "malformed collision row"})
            continue
        old_cells = [old["force10"], old["force15"], old["force40"],
                     old["force100"], old["false_max"], old["minimum"]]
        failures, force_reviews = [], []
        for index, force in enumerate(FORCES):
            old_delay, delay = _delay(old_cells[index]), _delay(cells[index])
            lost = old_delay is not None and delay is None
            too_slow = (old_delay is not None and delay is not None
                        and delay > old_delay + additional)
            cap = float(caps[force]) if key == (8.0, 3.0) and force in caps else None
            cap_failed = cap is not None and (delay is None or delay > cap)
            if lost:
                failures.append(f"lost_R1_{force}_detection")
            if too_slow:
                failures.append(f"{force}_delay_more_than_R1_plus_{additional:g}ms")
            if cap_failed:
                failures.append(f"{force}_delay_above_fixed_cap")
            force_reviews.append({"force": force, "R1_delay_ms": old_delay,
                                  "R3_delay_ms": delay, "fixed_cap_ms": cap,
                                  "lost_R1_detection": lost,
                                  "additional_delay_pass": not too_slow,
                                  "fixed_cap_pass": not cap_failed})
        old_fa, false_alarms = _false_alarms(old_cells[4]), _false_alarms(cells[4])
        if false_alarms > old_fa:
            failures.append("extra_collision_false_alarm")
        review = {"architecture": mode, "mass_kg": key[0], "time_s": key[1],
                  "forces": force_reviews, "R1_false_alarms": old_fa,
                  "R3_false_alarms": false_alarms, "pass": not failures,
                  "failures": failures}
        rows.append(review)
        if failures:
            errors.append({"section": "collision", **review})
    return rows, errors


def assess(records, original, protocol, frozen_selection, selected_source_sha256=None):
    """Apply the registered R3 gates independently to servo and torque."""
    selected = {spec["architecture"]: spec for spec in frozen_selection}
    modes = {}
    for mode in ("servo", "torque"):
        errors = []
        data = records.get(mode)
        baseline = original.get("combinations", {}).get(mode + "_v2_v2")
        if not isinstance(data, dict) or not isinstance(baseline, dict):
            control, collisions = [], []
            errors.append({"section": "wrapper", "reason": "missing mode/R1 result"})
        else:
            if data.get("self_contained_default_names") is not True:
                errors.append({"section": "wrapper",
                               "reason": "default-name overlay proof missing"})
            if data.get("kit_integrity", {}).get("status") != "ok":
                errors.append({"section": "wrapper",
                               "reason": "pristine overlay integrity failed"})
            if (selected_source_sha256 is not None
                    and data.get("source_sha256") != selected_source_sha256.get(mode)):
                errors.append({"section": "wrapper",
                               "reason": "evaluated controller hash differs from selection"})
            payload = data.get("control_and_collisions", {})
            try:
                control, first = _control_review(mode, payload["control"], baseline,
                                                 protocol)
                collisions, second = _collision_review(mode, payload["collisions"],
                                                       baseline, protocol)
                errors.extend(first + second)
            except (KeyError, TypeError, ValueError) as exc:
                control, collisions = [], []
                errors.append({"section": "wrapper",
                               "reason": "malformed evaluator result: " + str(exc)})
        passed = not errors
        modes[mode] = {"status": "PASS" if passed else "FAIL",
                       "frozen_selection": selected[mode],
                       "effective_selection": (selected[mode] if passed
                                               else R.baseline_spec(mode)),
                       "fallback_to_R1": not passed, "control": control,
                       "collisions": collisions, "errors": errors,
                       "no_runner_up_promotion": True, "retest_permitted": False}
    return {"schema": "R3_OFFICIAL_GATE.v1", "architectures": modes,
            "all_pass": all(row["status"] == "PASS" for row in modes.values()),
            "independent_architecture_fallback": True}


def _condition_key(row):
    payload = {name: row.get(name) for name in (
        "case_id", "mass", "com", "kind", "collision", "pick_offset",
        "phase", "direction", "force_n")}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _validate_holdout_row(row, spec):
    required = ("rms_mm", "max_mm", "saturation_pct", "diverged",
                "torque_step_pick_nm", "torque_step_place_nm",
                "table_force_place_max_n", "false_alarms", "g4", "timing",
                "private_raw_basename", "private_raw_sha256")
    if (row.get("variant") != spec["id"]
            or row.get("architecture") != spec["architecture"]
            or row.get("specification") != spec
            or row.get("stage") != "holdout"
            or row.get("official") is not False
            or any(name not in row for name in required)):
        raise RuntimeError("R3 holdout contains a malformed/dummy row")
    if not isinstance(row.get("case_id"), int) or not 0 <= row["case_id"] < 38:
        raise RuntimeError("R3 holdout case_id is invalid")
    try:
        values = [float(row["mass"]), *[float(value) for value in row["com"]],
                  *[float(row[name]) for name in required[:3]],
                  float(row["torque_step_pick_nm"]),
                  float(row["torque_step_place_nm"]),
                  float(row["table_force_place_max_n"])]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("R3 holdout numeric row fields are invalid") from exc
    if len(row["com"]) != 3 or not all(math.isfinite(value) for value in values):
        raise RuntimeError("R3 holdout has nonfinite mass/COM/metrics")
    if not isinstance(row["diverged"], bool) or not isinstance(row["false_alarms"], int):
        raise RuntimeError("R3 holdout safety fields are invalid")
    if not isinstance(row["g4"], dict) or not isinstance(row["timing"], dict):
        raise RuntimeError("R3 holdout metric packets are invalid")
    basename = row["private_raw_basename"]
    digest = row["private_raw_sha256"]
    if (not isinstance(basename, str) or Path(basename).is_absolute()
            or "/" in basename or not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())):
        raise RuntimeError("R3 holdout raw evidence reference is invalid")
    kind = row.get("kind")
    if kind == "reference":
        valid_kind = row.get("collision") is None and row.get("pick_offset") is None
    elif kind == "collision":
        collision = row.get("collision")
        valid_kind = (isinstance(collision, list) and len(collision) == 2
                      and isinstance(collision[1], list) and len(collision[1]) == 3
                      and row.get("collision_outcome") is not None)
        if valid_kind:
            try:
                valid_kind = all(math.isfinite(float(value)) for value in
                                 [collision[0], *collision[1]])
            except (TypeError, ValueError):
                valid_kind = False
    elif kind == "offset":
        offset = row.get("pick_offset")
        valid_kind = (row.get("collision") is None and isinstance(offset, list)
                      and len(offset) == 3)
        if valid_kind:
            try:
                valid_kind = all(math.isfinite(float(value)) for value in offset)
            except (TypeError, ValueError):
                valid_kind = False
    else:
        valid_kind = False
    if not valid_kind:
        raise RuntimeError("R3 holdout row kind/payload is invalid")


def _holdout_complete(document, freeze, freeze_sha256, protocol_sha256,
                      opened_marker_sha256):
    selection = freeze["chosen_specs"]
    specs = E.selected_with_r1(selection, include_r1=True)
    metadata = document.get("metadata", {})
    expected_sources = {f"{spec['architecture']}:{spec['id']}": spec for spec in specs}
    if (document.get("schema") != "R3_HOLDOUT.v1"
            or metadata.get("one_batch") is not True
            or metadata.get("chosen_specs") != selection
            or metadata.get("evaluated_specs") != specs
            or metadata.get("freeze_sha256") != freeze_sha256
            or metadata.get("protocol_sha256") != protocol_sha256
            or metadata.get("opened_marker_sha256") != opened_marker_sha256
            or metadata.get("local_commit") != freeze["local_commit"]
            or metadata.get("cases_per_source_architecture") != 38
            or set(metadata.get("source_sha256", {})) != set(expected_sources)
            or metadata.get("orchestration_sha256") != E.orchestration_hashes()):
        raise RuntimeError("R3 holdout metadata is incomplete or differs from freeze")
    for label, spec in expected_sources.items():
        if metadata["source_sha256"][label] != R.runtime_hashes(
                spec["architecture"], spec):
            raise RuntimeError("R3 holdout source hashes differ from frozen runtime")
    rows = document.get("results")
    if not isinstance(rows, list) or len(rows) != 38 * len(specs):
        raise RuntimeError("R3 holdout result count is incomplete")
    grouped = {label: [] for label in expected_sources}
    raw_basenames = set()
    for row in rows:
        label = f"{row.get('architecture')}:{row.get('variant')}"
        if label not in grouped:
            raise RuntimeError("R3 holdout contains an unexpected source")
        _validate_holdout_row(row, expected_sources[label])
        if row["private_raw_basename"] in raw_basenames:
            raise RuntimeError("R3 holdout reuses a private raw evidence basename")
        raw_basenames.add(row["private_raw_basename"])
        grouped[label].append(row)
    reference_conditions = None
    for label, group in grouped.items():
        if len(group) != 38 or {row["case_id"] for row in group} != set(range(38)):
            raise RuntimeError("R3 holdout source lacks 38 unique cases")
        counts = {kind: sum(row["kind"] == kind for row in group)
                  for kind in ("reference", "collision", "offset")}
        if counts != {"reference": 4, "collision": 30, "offset": 4}:
            raise RuntimeError("R3 holdout case-kind counts differ from protocol")
        conditions = sorted(_condition_key(row) for row in group)
        if reference_conditions is None:
            reference_conditions = conditions
        elif conditions != reference_conditions:
            raise RuntimeError("R3 holdout sources do not share matched conditions")
    chosen = {spec["architecture"]: spec for spec in selection}
    reviews = {}
    for mode in ("servo", "torque"):
        baseline = {"results": grouped[f"{mode}:R1"]}
        candidate = {"results": grouped[f"{mode}:{chosen[mode]['id']}"]}
        reviews[mode] = S.review_holdout(baseline, candidate)
    return {"specs": specs, "reviews": reviews}


def _review_effective_selection(review, selection, recomputed):
    architectures = review.get("architectures")
    if (review.get("schema") != "R3_HOLDOUT_REVIEW.v1"
            or review.get("no_tuning_permitted") is not True
            or not isinstance(architectures, dict)
            or set(architectures) != {"servo", "torque"}):
        raise RuntimeError("R3 holdout review contract is invalid")
    chosen = {spec["architecture"]: spec for spec in selection}
    effective = []
    for mode in ("servo", "torque"):
        row = architectures[mode]
        status, spec = row.get("status"), row.get("effective_selection")
        baseline = R.baseline_spec(mode)
        if (status not in {"PASS", "FAIL"}
                or row.get("review") != recomputed[mode]
                or status != recomputed[mode].get("status")):
            raise RuntimeError("R3 holdout review mode is malformed")
        if spec not in (chosen[mode], baseline):
            raise RuntimeError("R3 holdout review effective source is noncanonical")
        if status == "FAIL" and spec != baseline:
            raise RuntimeError("failed holdout mode must select frozen R1")
        effective.append(spec)
    return effective


def preflight(freeze_path=FREEZE, holdout_path=HOLDOUT,
              review_path=HOLDOUT_REVIEW, opened_marker=HOLDOUT_MARKER):
    freeze_path, holdout_path = Path(freeze_path), Path(holdout_path)
    review_path, opened_marker = Path(review_path), Path(opened_marker)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    selection = freeze.get("chosen_specs")
    if not isinstance(selection, list):
        raise RuntimeError("R3 freeze lacks chosen_specs")
    E.verify_freeze(selection, freeze_path)
    files = freeze.get("source_files", {})
    missing = [relative for relative in OFFICIAL_FROZEN_PATHS
               if files.get(relative) != R.sha(ROOT / relative)]
    if missing:
        raise RuntimeError("R3 freeze lacks current official sources: "
                           + ", ".join(missing))
    protocol_sha256 = R.sha(PROTOCOL)
    freeze_sha256, holdout_sha256 = R.sha(freeze_path), R.sha(holdout_path)
    opened_marker_sha256 = R.sha(opened_marker)
    opened = json.loads(opened_marker.read_text(encoding="utf-8"))
    if (opened.get("schema") != "R3_HOLDOUT_OPENED.v1"
            or opened.get("batch_attempt") != 1
            or opened.get("chosen_specs") != selection
            or opened.get("local_commit") != freeze["local_commit"]
            or opened.get("protocol_sha256") != protocol_sha256):
        raise RuntimeError("R3 holdout opened marker is malformed or unrelated")
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    completed = _holdout_complete(holdout, freeze, freeze_sha256,
                                  protocol_sha256, opened_marker_sha256)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (review.get("holdout_sha256") != holdout_sha256
            or review.get("freeze_sha256") != freeze_sha256
            or review.get("protocol_sha256") != protocol_sha256
            or review.get("opened_marker_sha256") != opened_marker_sha256):
        raise RuntimeError("R3 holdout review is not tied to frozen evidence")
    effective = _review_effective_selection(review, selection, completed["reviews"])
    return {"freeze": freeze, "selection": selection,
            "evaluated_specs": completed["specs"], "effective_selection": effective,
            "holdout_review": review, "holdout_review_sha256": R.sha(review_path),
            "freeze_sha256": freeze_sha256, "holdout_sha256": holdout_sha256,
            "protocol_sha256": protocol_sha256,
            "opened_marker_sha256": opened_marker_sha256}


def resolve_sources(prepared):
    files, resolved = prepared["freeze"]["source_files"], {}
    for spec in prepared["effective_selection"]:
        mode = spec["architecture"]
        source = ROOT / f"r1_{mode}.py" if spec["id"] == "R1" else compiled_path(spec)
        relative, actual = source.relative_to(ROOT).as_posix(), R.sha(source)
        if files.get(relative) != actual:
            raise RuntimeError("selected official source differs from frozen bytes")
        resolved[mode] = {"specification": spec, "source": source,
                          "relative_source": relative, "sha256": actual}
    detector = ROOT / "r1_detector.py"
    if files.get("r1_detector.py") != R.sha(detector):
        raise RuntimeError("official detector differs from frozen R1 detector")
    return resolved, detector


def stage_sources(private_dir, resolved, detector):
    private = R.require_private(private_dir)
    stage = private / "R3_OFFICIAL_STAGE"
    stage.mkdir(parents=True, exist_ok=True)
    prefix = stage / "r3"
    expected = {Path(str(prefix) + "_servo.py"): resolved["servo"]["source"],
                Path(str(prefix) + "_torque.py"): resolved["torque"]["source"],
                Path(str(prefix) + "_detector.py"): detector}
    for target, source in expected.items():
        content = source.read_bytes()
        if target.exists():
            if target.read_bytes() != content:
                raise RuntimeError("staged official source differs from frozen bytes")
            continue
        with target.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    directory = os.open(stage, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return prefix


def _exclusive_marker(path, prepared, resolved, detector=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "R3_OFFICIAL_OPENED.v1",
               "opened_utc": datetime.now(timezone.utc).isoformat(),
               "freeze_sha256": prepared["freeze_sha256"],
               "holdout_sha256": prepared["holdout_sha256"],
               "protocol_sha256": prepared["protocol_sha256"],
               "holdout_opened_marker_sha256": prepared["opened_marker_sha256"],
               "local_commit": prepared["freeze"]["local_commit"],
               "chosen_specs": prepared["selection"],
               "evaluated_specs": prepared["evaluated_specs"],
               "effective_specs": prepared["effective_selection"],
               "holdout_review_sha256": prepared["holdout_review_sha256"],
               "selected_source_sha256": {m: r["sha256"] for m, r in resolved.items()},
               "detector_sha256": R.sha(detector) if detector is not None else None,
               "batch_attempt": 1, "expected_episodes": 58,
               "tuning_or_repair_after_exposure": False}
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise RuntimeError("R3 official batch is already opened; no rerun") from exc
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return payload


def official_environment(base=None):
    env = dict(os.environ if base is None else base)
    env.pop("PYTHONPATH", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    return env


def official_command(prefix, output):
    return ["taskset", "-c", "0", sys.executable, "tools/validate_submission.py",
            "--prefix", str(prefix), "--out", str(Path(output).relative_to(ROOT))]


def _foreign_state(state, owned_group):
    foreign = []
    for pid in state["foreign_heavy_python_pids"]:
        try:
            if os.getpgid(pid) == owned_group:
                continue
        except (OSError, ProcessLookupError):
            continue
        foreign.append(pid)
    output = dict(state)
    output["foreign_heavy_python_pids"] = foreign
    return output


def run_owned_group(command, log_path):
    while True:
        before = resources.snapshot()
        if (before["mem_available_gib"] >= 6
                and not before["foreign_heavy_python_pids"]):
            break
        time.sleep(5)
    states, pauses = [], 0
    with Path(log_path).open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=official_environment(),
                                 stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        stopped = False
        try:
            while child.poll() is None:
                state = _foreign_state(resources.snapshot(), child.pid)
                states.append(state)
                safe = state["mem_available_gib"] >= 6 and not state["foreign_heavy_python_pids"]
                if not safe and not stopped:
                    os.killpg(child.pid, signal.SIGSTOP)
                    stopped, pauses = True, pauses + 1
                elif safe and stopped:
                    os.killpg(child.pid, signal.SIGCONT)
                    stopped = False
                time.sleep(0.25)
            if stopped:
                os.killpg(child.pid, signal.SIGCONT)
        except Exception:
            if stopped:
                os.killpg(child.pid, signal.SIGCONT)
                stopped = False
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
        finally:
            if child.poll() is None and stopped:
                os.killpg(child.pid, signal.SIGCONT)
    if not states:
        states.append(_foreign_state(resources.snapshot(), child.pid))
    return {"returncode": int(child.returncode), "host_before": before,
            "samples": len(states), "owned_group_pause_count": pauses,
            "min_available_RAM_GiB": min(s["mem_available_gib"] for s in states),
            "maximum_foreign_heavy_python_count": max(
                len(s["foreign_heavy_python_pids"]) for s in states),
            "foreign_processes_signalled": 0, "cpu_affinity": [0], "gpu_jobs": 0}


def _write_exclusive(path, value):
    path = Path(path)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(R.json_safe(value), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise RuntimeError("R3 official result already exists; no overwrite") from exc
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _shared_failure(reason, prepared):
    return {"schema": "R3_OFFICIAL_REGRESSION.v1", "official": True,
            "status": "FAIL", "shared_unidentified_failure": True,
            "reason": reason,
            "architectures": {mode: {"status": "FAIL",
                "frozen_selection": next(s for s in prepared["effective_selection"]
                                         if s["architecture"] == mode),
                "effective_selection": R.baseline_spec(mode),
                "fallback_to_R1": True, "no_runner_up_promotion": True,
                "retest_permitted": False} for mode in ("servo", "torque")}}


def _decorate_output(output, prepared, resolved, detector, marker, guard, records):
    output.update(source="unchanged evaluate.py; pristine default-name overlays; one A+B batch",
                  expected_episodes=58, results=records,
                  selected_source_sha256={m: r["sha256"] for m, r in resolved.items()},
                  detector_sha256=R.sha(detector), protocol_sha256=R.sha(PROTOCOL),
                  wrapper_selection={m: {"specification": r["specification"],
                      "relative_source": r["relative_source"], "sha256": r["sha256"],
                      "overlay_module": m + "_controller"}
                      for m, r in resolved.items()},
                  fallback_R1_sha256=prepared["freeze"]["fallback_R1_sha256"],
                  freeze_sha256=prepared["freeze_sha256"],
                  holdout_sha256=prepared["holdout_sha256"],
                  holdout_review_sha256=prepared["holdout_review_sha256"],
                  holdout_review=prepared["holdout_review"], marker_sha256=R.sha(MARKER),
                  marker=marker, guard=guard, tuning_after_exposure=False,
                  repair_after_exposure=False, alternative_promotion_after_failure=False,
                  failure_action=(
                      "affected mode uses frozen R1; shared failure uses both "
                      "R1; no repair/promotion/retest"))
    return output


def run_official(private_dir):
    if MARKER.exists() or RESULT.exists() or OFFICIAL_DIR.exists():
        raise RuntimeError("R3 official evidence already opened or completed")
    private = R.require_private(private_dir)
    private.mkdir(parents=True, exist_ok=True)
    check_integrity(ROOT)
    prepared = preflight()
    resolved, detector = resolve_sources(prepared)
    prefix = stage_sources(private, resolved, detector)
    marker = _exclusive_marker(MARKER, prepared, resolved, detector)
    try:
        guard = run_owned_group(official_command(prefix, OFFICIAL_DIR),
                                private / "r3_official_default.log")
    except Exception as exc:
        guard = {"launch_or_monitor_exception": type(exc).__name__,
                 "message": str(exc), "foreign_processes_signalled": 0,
                 "cpu_affinity": [0], "gpu_jobs": 0}
        output = _decorate_output(
            _shared_failure("official launch/monitor failure: " + str(exc), prepared),
            prepared, resolved, detector, marker, guard, None)
        _write_exclusive(RESULT, output)
        return output
    records, gate, shared_reason = None, None, None
    try:
        post = preflight()
        if (post["freeze_sha256"] != prepared["freeze_sha256"]
                or post["holdout_sha256"] != prepared["holdout_sha256"]
                or post["holdout_review_sha256"] != prepared["holdout_review_sha256"]
                or post["protocol_sha256"] != prepared["protocol_sha256"]
                or post["opened_marker_sha256"] != prepared["opened_marker_sha256"]):
            raise RuntimeError("frozen pre-official evidence changed after exposure")
    except Exception as exc:
        shared_reason = "post-official source/evidence check failed: " + str(exc)
    if guard["returncode"] != 0 and shared_reason is None:
        shared_reason = "unchanged official evaluator process failed"
    if guard["returncode"] == 0 and shared_reason is None:
        try:
            records = json.loads((OFFICIAL_DIR / "summary.json").read_text())
            original = json.loads(R1_EXPOSED.read_text())
            gate = assess(records, original, R.protocol(), prepared["effective_selection"],
                          {mode: row["sha256"] for mode, row in resolved.items()})
        except Exception as exc:
            shared_reason = "official result parsing/gating failed: " + str(exc)
    if shared_reason is not None:
        output = _shared_failure(shared_reason, prepared)
    else:
        output = {"schema": "R3_OFFICIAL_REGRESSION.v1", "official": True,
                  "status": "PASS" if gate["all_pass"] else "FAIL",
                  "shared_unidentified_failure": False, **gate}
    _decorate_output(output, prepared, resolved, detector, marker, guard, records)
    _write_exclusive(RESULT, output)
    check_integrity(ROOT)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_log_directory", type=Path)
    args = parser.parse_args(argv)
    result = run_official(args.private_log_directory)
    print(json.dumps({"status": result["status"], "architectures": {
        mode: row["status"] for mode, row in result["architectures"].items()}}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
