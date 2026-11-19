"""Authenticate and compare the frozen R26 candidate on the historical D4 matrix."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import hashlib
import importlib
import json

from experiments.r22_census import check
from experiments.r24_loaded.d4 import load_cases


R25 = "experiments.r25_composed.detector:Detector"
R26 = "experiments.r26_long.candidate:Detector"
MATRIX = check.MATRIX


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def constructor(key):
    module, name = key.split(":", 1)
    return getattr(importlib.import_module(module), name)


def capture(job):
    case, arm = job
    return check.old.episode_record(constructor(arm), case)


def archive_records(directory):
    records, manifest = {}, {}
    for path in sorted(Path(directory).glob("shard-*.json")):
        doc = json.loads(path.read_text())
        if doc["candidate"] != R25 or doc["matrix_sha256"] != MATRIX:
            raise ValueError("unexpected R25 D4 archive")
        manifest[path.name] = sha(path)
        for key, value in doc["records"].items():
            if int(key) in records:
                raise ValueError("duplicate archived D4 case")
            records[int(key)] = value
    if set(records) != set(range(len(load_cases()))):
        raise ValueError("incomplete archived R25 D4 matrix")
    return records, manifest


def occupied(record):
    lo, hi = record["true_window_first_index"], record["true_window_last_index"]
    if lo is None or hi is None:
        return 0
    return sum(max(0, min(end, hi + 1) - max(begin, lo))
               for begin, end in record["detect_intervals"])


def preflight(archive, out):
    records, manifest = archive_records(archive)
    case = load_cases()[0]
    fresh25 = capture((case, R25))
    fresh26 = capture((case, R26))
    gates = {"archive_full_record_exact": fresh25 == records[0],
             "fresh_candidate_physics_exact": fresh26["physical_sha256"] ==
                                                fresh25["physical_sha256"],
             "matrix_exact": MATRIX == check.MATRIX}
    document = {"schema": "r26-history-d4-preflight.v1", "case_id": 0,
                "matrix_sha256": MATRIX, "archive_manifest": manifest,
                "archived_r25": records[0], "fresh_r25": fresh25,
                "fresh_r26": fresh26, "gates": gates, "passed": all(gates.values())}
    out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"case_id": 0, "gates": gates, "passed": document["passed"]}, indent=2))
    if not document["passed"]:
        raise SystemExit(1)


def compare(archive, chunks, preflight_path, out):
    old, manifest = archive_records(archive)
    pre = json.loads(Path(preflight_path).read_text())
    if not pre["passed"] or pre["archive_manifest"] != manifest:
        raise ValueError("D4 archive authentication changed")
    new = {}
    for path in chunks:
        doc = json.loads(Path(path).read_text())
        if doc["candidate"] != R26 or doc["matrix_sha256"] != MATRIX:
            raise ValueError("mixed R26 D4 chunk")
        for key, value in doc["records"].items():
            if int(key) in new:
                raise ValueError("duplicate R26 D4 case")
            new[int(key)] = value
    cases = load_cases()
    if set(new) != set(range(len(cases))):
        raise ValueError("incomplete fresh R26 D4 matrix")
    references = {check.old.reference_key(case): case["case_id"]
                  for case in cases if case["kind"] == "reference"}
    rows, regressions, gains = [], [], []
    for case in cases:
        index = case["case_id"]
        reference = references[check.old.reference_key(case)] if case["kind"] == "collision" else index
        r25 = check.score(case, old[index], old[reference])
        r26 = check.score(case, new[index], new[reference])
        gates = check.compare(r26, r25)
        gates["latency_plus_5_ms"] = (not r25["strict"] or
            (r26["strict"] and r26["delay_ms"] <= r25["delay_ms"] + 5))
        gates["duration_plus_5_samples"] = (not r25["strict"] or
            occupied(new[index]) <= occupied(old[index]) + 5)
        gates["tool_time_plus_5_ms"] = (not r25["tool"]["detected"] or
            (r26["tool"]["detected"] and min(new[index]["tool_rise_indices"]) <=
             min(old[index]["tool_rise_indices"]) + 5))
        gates["physical_force_prefix_exact"] = (
            old[index]["physical_sha256"] == new[index]["physical_sha256"] and
            all(r26["valid"].values()))
        failed = [key for key, value in gates.items() if not value]
        identity = {key: case.get(key) for key in
                    ("controller", "detector", "mass", "com", "collision",
                     "pick_offset", "phase", "direction", "force_n")}
        row = {"id": index, "kind": case["kind"], "identity": identity,
               "r25": r25, "r26": r26,
               "r25_active_samples": occupied(old[index]),
               "r26_active_samples": occupied(new[index]), "gates": gates,
               "failures": failed}
        rows.append(row)
        if failed:
            regressions.append({"id": index, "failures": failed})
        if r26["strict"] and not r25["strict"]:
            gains.append(index)
    document = {"schema": "r26-history-d4-comparison.v1", "matrix_sha256": MATRIX,
                "cases": len(rows), "fresh_r26_episodes": len(new),
                "archive_manifest": manifest, "rows": rows,
                "regressions": regressions, "strict_gains": gains,
                "passed": not regressions}
    out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cases": len(rows), "regressions": regressions,
                      "strict_gains": gains, "passed": document["passed"]}, indent=2))
    if regressions:
        raise SystemExit(1)


def main(args):
    if args.preflight:
        preflight(args.archive, args.out)
    elif args.shard:
        index, count = args.shard
        cases = load_cases()[index::count]
        with ProcessPoolExecutor(args.workers) as pool:
            records = dict(zip([case["case_id"] for case in cases],
                pool.map(capture, [(case, R26) for case in cases]), strict=True))
        args.out.write_text(json.dumps({"candidate": R26, "matrix_sha256": MATRIX,
                                        "shard": [index, count], "records": records}) + "\n")
        print([index, count, len(records)])
    else:
        compare(args.archive, args.chunks, args.auth, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--shard", nargs=2, type=int)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--auth", type=Path)
    parser.add_argument("--chunks", nargs="*", type=Path)
    main(parser.parse_args())
