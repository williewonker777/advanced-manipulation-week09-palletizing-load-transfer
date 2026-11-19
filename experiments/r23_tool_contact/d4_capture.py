"""Capture one bounded shard of the 385-case R23 D4 detector records."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from experiments.r21_full_protection import runner_v2 as old
from experiments.r22_census import check as census
from experiments.r23_tool_contact.d4_check import simulate, simulate_twin
from tools.d4_robustness import build_cases


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or not 0 <= args.index < args.count or args.count < 1:
        parser.error("fresh output and valid shard required")
    all_cases = build_cases("my_torque")
    if hashlib.sha256(old.canonical_cases(all_cases)).hexdigest() != census.MATRIX:
        raise ValueError("D4 matrix differs")
    cases = all_cases[args.index::args.count]
    with ProcessPoolExecutor(args.workers) as pool:
        items = list(pool.map(simulate, cases))
    offset_cases = [case for case in cases if case["kind"] == "offset"]
    with ProcessPoolExecutor(args.workers) as pool:
        twins = list(pool.map(simulate_twin, offset_cases))
    twin_by_id = {case["case_id"]: twin for case, twin in zip(offset_cases, twins, strict=True)}
    payload = {"schema": "r23-tool-contact-d4-shard.v1", "index": args.index, "count": args.count,
               "candidate_sha256": sha(HERE / "candidate.py"), "matrix_sha256": census.MATRIX,
               "rows": [{"id": case["case_id"], "item": item,
                         "same_motion_twin": twin_by_id.get(case["case_id"])}
                        for case, item in zip(cases, items, strict=True)]}
    args.out.write_text(json.dumps(payload, allow_nan=False) + "\n")
    print(json.dumps({"index": args.index, "cases": len(cases), "offset_twins": len(twins)}))


if __name__ == "__main__":
    main()
