"""Authenticated full D4 comparison for the admitted frozen acquisition candidate."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from experiments.r22_census import check
from experiments.r24_loaded.d4 import load_cases
from experiments.r26_history import d4 as base


R25 = base.R25
CANDIDATE = "experiments.r26_yaw_acquisition.candidate:Detector"
CANDIDATE_SHA = "212f4e79367cb636c0143a2d1f8aa7ba5584aaaca5ecde6b412d01a4985b6784"
SOURCE = Path(__file__).with_name("candidate.py")


def assert_frozen():
    if base.sha(SOURCE) != CANDIDATE_SHA:
        raise ValueError("admitted acquisition candidate changed")


def capture(case):
    return check.old.episode_record(base.constructor(CANDIDATE), case)


def preflight(archive, out):
    old, manifest = base.archive_records(archive)
    case = load_cases()[0]
    fresh25 = base.capture((case, R25))
    fresh = capture(case)
    gates = {
        "archive_full_record_exact": fresh25 == old[0],
        "candidate_physics_exact": fresh["physical_sha256"] == fresh25["physical_sha256"],
        "matrix_exact": base.MATRIX == check.MATRIX,
    }
    document = {"schema": "r26-acquisition-d4-preflight.v1",
                "candidate": CANDIDATE, "candidate_sha256": CANDIDATE_SHA,
                "matrix_sha256": base.MATRIX, "archive_manifest": manifest,
                "case_id": 0, "archived_r25": old[0], "fresh_r25": fresh25,
                "fresh_candidate": fresh, "gates": gates, "passed": all(gates.values())}
    out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"case_id": 0, "gates": gates, "passed": document["passed"]}, indent=2))
    if not document["passed"]:
        raise SystemExit(1)


def shard(index, count, workers, out):
    cases = load_cases()[index::count]
    with ProcessPoolExecutor(workers) as pool:
        records = dict(zip([case["case_id"] for case in cases], pool.map(capture, cases), strict=True))
    out.write_text(json.dumps({"schema": "r26-acquisition-d4-shard.v1",
                               "candidate": CANDIDATE, "candidate_sha256": CANDIDATE_SHA,
                               "matrix_sha256": base.MATRIX, "shard": [index, count],
                               "records": records}) + "\n")
    print([index, count, len(records)])


def compare(archive, chunks, auth_path, out):
    old, manifest = base.archive_records(archive)
    auth = json.loads(auth_path.read_text())
    if not auth["passed"] or auth["archive_manifest"] != manifest:
        raise ValueError("D4 archive authentication changed")
    new = {}
    for path in chunks:
        document = json.loads(path.read_text())
        if (document["candidate"] != CANDIDATE or
                document["candidate_sha256"] != CANDIDATE_SHA or
                document["matrix_sha256"] != base.MATRIX):
            raise ValueError("mixed D4 shard")
        for key, value in document["records"].items():
            if int(key) in new:
                raise ValueError("duplicate D4 case")
            new[int(key)] = value
    cases = load_cases()
    if set(new) != set(range(len(cases))):
        raise ValueError("incomplete candidate D4 matrix")
    references = {check.old.reference_key(case): case["case_id"]
                  for case in cases if case["kind"] == "reference"}
    rows, regressions, gains = [], [], []
    for case in cases:
        index = case["case_id"]
        reference = references[check.old.reference_key(case)] if case["kind"] == "collision" else index
        old_score = check.score(case, old[index], old[reference])
        new_score = check.score(case, new[index], new[reference])
        gates = check.compare(new_score, old_score)
        gates.update({
            "latency_plus_5_ms": (not old_score["strict"] or
                                  (new_score["strict"] and
                                   new_score["delay_ms"] <= old_score["delay_ms"] + 5)),
            "duration_plus_5_samples": (not old_score["strict"] or
                                        base.occupied(new[index]) <= base.occupied(old[index]) + 5),
            "tool_time_plus_5_ms": (not old_score["tool"]["detected"] or
                                    (new_score["tool"]["detected"] and
                                     min(new[index]["tool_rise_indices"]) <=
                                     min(old[index]["tool_rise_indices"]) + 5)),
            "physical_force_prefix_exact": (old[index]["physical_sha256"] ==
                                             new[index]["physical_sha256"] and
                                             all(new_score["valid"].values())),
        })
        failed = [key for key, value in gates.items() if not value]
        row = {"id": index, "kind": case["kind"], "identity": case,
               "r25": old_score, "candidate": new_score,
               "r25_active_samples": base.occupied(old[index]),
               "candidate_active_samples": base.occupied(new[index]),
               "gates": gates, "failures": failed}
        rows.append(row)
        if failed:
            regressions.append({"id": index, "failures": failed})
        if new_score["strict"] and not old_score["strict"]:
            gains.append(index)
    document = {"schema": "r26-acquisition-d4-comparison.v1",
                "candidate": CANDIDATE, "candidate_sha256": CANDIDATE_SHA,
                "matrix_sha256": base.MATRIX, "cases": len(rows),
                "fresh_candidate_episodes": len(new), "archive_manifest": manifest,
                "rows": rows, "regressions": regressions, "strict_gains": gains,
                "passed": not regressions}
    out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cases": len(rows), "regressions": regressions,
                      "strict_gains": gains, "passed": document["passed"]}, indent=2))
    if regressions:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--shard", nargs=2, type=int)
    parser.add_argument("--workers", choices=(1, 2), type=int, default=2)
    parser.add_argument("--auth", type=Path)
    parser.add_argument("--chunks", nargs="*", type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    assert_frozen()
    if args.preflight:
        preflight(args.archive, args.out)
    elif args.shard:
        shard(*args.shard, args.workers, args.out)
    else:
        compare(args.archive, args.chunks, args.auth, args.out)


if __name__ == "__main__":
    main()
