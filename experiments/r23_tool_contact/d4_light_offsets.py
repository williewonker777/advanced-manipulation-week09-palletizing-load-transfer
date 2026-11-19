"""Capture R22 LIGHT records for the 16 exposed D4 offset cases."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from experiments.r21_full_protection import runner_v2 as old
from experiments.r22_census import check as census
from experiments.r22_light.candidate import Detector as Light
from tools.d4_robustness import build_cases


def simulate(case):
    return old.episode_record(Light, case)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("fresh output required")
    all_cases = build_cases("my_torque")
    if hashlib.sha256(old.canonical_cases(all_cases)).hexdigest() != census.MATRIX:
        raise ValueError("D4 matrix differs")
    cases = [case for case in all_cases if case["kind"] == "offset"]
    with ProcessPoolExecutor(args.workers) as pool:
        records = list(pool.map(simulate, cases))
    payload = {"schema": "r23-r22-light-offset-records.v1", "matrix_sha256": census.MATRIX,
               "rows": [{"id": case["case_id"], "record": record}
                        for case, record in zip(cases, records, strict=True)]}
    args.out.write_text(json.dumps(payload, allow_nan=False) + "\n")
    print(json.dumps({"offsets": len(cases)}))


if __name__ == "__main__":
    main()
