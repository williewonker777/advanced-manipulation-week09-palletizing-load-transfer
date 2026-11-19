"""Compare frozen R33 against the authenticated disclosed R31 D4 history."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from experiments.r21_full_protection import runner_v2 as raw
from experiments.r22_census import check
from experiments.r33_carry_vertical.candidate import Detector


def one(case):
    return case["case_id"], raw.episode_record(Detector, case)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("transport", "full"), required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("new output required")
    prior = json.loads(args.prior.read_bytes())
    cases = check.build_cases("my_torque")
    selected_collisions = [case for case in cases if case["kind"] == "collision" and (
        args.cohort == "full" or case["phase"] == "transport")]
    needed = {raw.reference_key(case) for case in selected_collisions}
    execution = cases if args.cohort == "full" else [
        case for case in cases if case in selected_collisions or
        (case["kind"] == "reference" and raw.reference_key(case) in needed)]
    with ProcessPoolExecutor(args.jobs) as pool:
        records = dict(pool.map(one, execution))
    refs = {raw.reference_key(case): case["case_id"] for case in execution if case["kind"] == "reference"}
    rows = []
    prior_rows = {row["id"]: row for row in prior["rows"]}
    for case in selected_collisions:
        record = records[case["case_id"]]
        normal = records[refs[raw.reference_key(case)]]
        score = check.score(case, record, normal)
        old = prior_rows[case["case_id"]]["arms"]["identity"]
        rows.append({
            "id": case["case_id"], "phase": case["phase"], "force": case["collision"][1],
            "candidate": score, "parent": old["score"],
            "candidate_intervals": record["detect_intervals"],
            "parent_intervals": old["alarm_intervals"],
            "same_physics": record["physical_sha256"] == prior_rows[case["case_id"]]["physics_sha256"],
        })
    def counts(name):
        return {
            "strict": sum(row[name]["strict"] for row in rows),
            "duplicate": sum(row[name]["duplicate"] for row in rows),
            "raw_duplicate": sum(row[name]["raw_duplicate"] for row in rows),
            "false": sum(len(row[name]["false"]) for row in rows),
        }
    result = {
        "schema": "r33-carry-vertical-broad.v1", "cohort": args.cohort,
        "candidate_sha256": hashlib.sha256(Path(__file__).with_name("candidate.py").read_bytes()).hexdigest(),
        "episodes": len(execution), "collisions": len(rows),
        "counts": {name: counts(name) for name in ("parent", "candidate")},
        "gains": [row["id"] for row in rows if row["candidate"]["strict"] and not row["parent"]["strict"]],
        "losses": [row["id"] for row in rows if row["parent"]["strict"] and not row["candidate"]["strict"]],
        "new_duplicates": [row["id"] for row in rows if row["candidate"]["duplicate"] > row["parent"]["duplicate"]],
        "new_false": [row["id"] for row in rows if check.unmatched_indices(row["candidate"]["false"], row["parent"]["false"])],
        "delay_worse_over_5ms": [row["id"] for row in rows if row["parent"]["strict"] and row["candidate"]["strict"] and
                                  row["candidate"]["delay_ms"] > row["parent"]["delay_ms"] + 5],
        "physics_failures": [row["id"] for row in rows if not row["same_physics"]],
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("episodes", "collisions", "counts", "gains", "losses", "new_duplicates", "new_false", "delay_worse_over_5ms", "physics_failures")}, indent=2))


if __name__ == "__main__":
    main()
