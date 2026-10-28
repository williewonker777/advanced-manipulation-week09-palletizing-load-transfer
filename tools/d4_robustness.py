"""Predeclared stratified stress tests; not a parameter search or official score.

Uses the unmodified kit runner and the metrics/timing wrapper in tools.sweep.
Each final architecture gets the same 180 external-force cases, 189 full-grid
empty/loaded references, and 16 position-offset cases. This is NOT the full
Cartesian product, an independent noise Monte Carlo, or a safety certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from tools import sweep

MASSES = (0., 1.3, 2., 3.7, 4., 6.25, 8.)
COMS = tuple(itertools.product((-.03, .0, .03), (-.02, .0, .02), (-.02, .015, .05)))
FORCES = (5., 10., 15., 25., 40., 100.)
DIRECTIONS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0),
              "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1),
              "+xy": (1, 1, 0), "-xy": (-1, -1, 0),
              "+xyz": (1, 1, 1), "-xyz": (-1, -1, -1)}
TIMES = {"pick": (1.38, 1.41, 1.44, 1.47, 1.50),
         "transport": (1.9, 2.3, 2.7, 3.1, 3.4),
         "place": (3.80, 3.83, 3.86, 3.89, 3.92)}
OFFSETS = ((0, 0, .01), (0, 0, .03), (0, 0, -.01), (.03, 0, 0),
           (-.03, 0, 0), (0, .03, 0), (0, -.03, 0), (.02, .02, .01))


def build_cases(controller: str, detector: str = "my_detector") -> list[dict]:
    cases = []
    def add(mass, com, collision=None, offset=None, **tags):
        cases.append(dict(controller=controller, detector=detector, case_id=len(cases),
                          mass=mass, com=com, collision=collision, pick_offset=offset, **tags))
    for mass, com in itertools.product(MASSES, COMS):
        add(mass, com, kind="reference", phase="none", direction="none", force_n=0.)
    index = 0
    for phase, force, (direction, vector) in itertools.product(TIMES, FORCES, DIRECTIONS.items()):
        unit = np.asarray(vector, float) / np.linalg.norm(vector)
        at = TIMES[phase][index % 5]
        # Coprime strides balance the pre-existing full-grid reference set.
        mass = MASSES[index % len(MASSES)]
        com = COMS[(index * 11) % len(COMS)]
        add(mass, com, (at, (unit * force).tolist()), kind="collision", phase=phase,
            direction=direction, force_n=force)
        index += 1
    for mass, offset in itertools.product((2., 8.), OFFSETS):
        add(mass, (.02, -.01, .03), offset=offset, kind="offset", phase="pick",
            direction="offset", force_n=0.)
    return cases


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("my_servo.py", "my_torque.py", "my_detector.py")}


def protocol():
    return {"design": "stratified coverage, fixed before final collision tests; no retuning",
            "mass_kg": MASSES, "com_m": COMS, "forces_n": FORCES,
            "directions": DIRECTIONS, "phase_times_s": TIMES, "pick_offsets_m": OFFSETS,
            "cases_per_controller": 385, "full_cartesian": False,
            "noise_realization": "unchanged kit default, not independent random sensor seeds",
            "classification": "evaluate.py precedence via tools.sweep.classify_collision",
            "interpretation": "finite test set only; neither exhaustive nor held-secret benchmark",
            "tuning_after_results": "forbidden; failures are reported"}


def summarize(rows):
    groups = {}
    for row in rows:
        if row["kind"] == "collision":
            key = f'{row["phase"]}/{row["direction"]}/{row["force_n"]:g}N'
            groups[key] = dict(outcome=row["collision_outcome"], delay_ms=row["detect_delay_ms"],
                               mass_kg=row["mass"], com_m=row["com"],
                               false_alarms=row["false_alarms"])
    refs = [r for r in rows if r["kind"] == "reference"]
    offsets = [r for r in rows if r["kind"] == "offset"]
    counts = {outcome: sum(r.get("collision_outcome") == outcome for r in rows)
              for outcome in ("detected", "missed", "alarm_before_collision", "indistinguishable")}
    return dict(collision_counts=counts, failure_map=groups,
                no_collision=dict(episodes=len(refs),
                    with_false_alarm=sum(r["false_alarms"] > 0 for r in refs),
                    false_alarm_rises=sum(r["false_alarms"] for r in refs),
                    diverged=sum(r["diverged"] for r in refs),
                    saturated_episodes=sum(r["saturation_pct"] > 0 for r in refs),
                    worst_tcp_mm=max(r["max_mm"] for r in refs),
                    max_table_over_mg=max(r["table_force_place_max_n"] / (9.81*r["mass"])
                                          for r in refs if r["mass"] > 0)),
                offset_contact_episodes=sum(r["tool_contact_peak_n"] > 1 for r in offsets),
                offset_contacts_missed=sum(r["tool_contact_peak_n"] > 1 and
                                          r["tool_contact_delay_ms"] is None for r in offsets),
                timing={component: {"largest_median_us": max(r["timing"][component]["median_us"] for r in rows),
                                     "largest_p99_us": max(r["timing"][component]["p99_us"] for r in rows),
                                     "observed_max_us": max(r["timing"][component]["max_us"] for r in rows)}
                        for component in ("controller", "detector", "combined")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("my_servo", "my_torque"), default="my_servo")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--register", action="store_true", help="write design, no simulation")
    args = parser.parse_args()
    path = ROOT / "results/D4_ROBUSTNESS_PROTOCOL.json"
    design = json.loads(json.dumps(protocol()))
    if args.register:
        if path.exists() and json.loads(path.read_text()) != design:
            raise SystemExit("protocol already exists with another design")
        path.write_text(json.dumps(design, indent=2) + "\n")
        print("protocol registered; no simulation")
        return
    if not path.exists() or json.loads(path.read_text()) != design:
        raise SystemExit("register the fixed protocol before running")
    memory = sweep.available_memory_bytes() / 1024**3
    if memory < 6:
        raise SystemExit("RAM below 6 GiB; refusing simulation")
    integrity = sweep.check_integrity()
    hashes = source_hashes()
    cases = build_cases(args.controller)
    start = time.perf_counter()
    if args.workers == 1:
        rows = [sweep.run_case(case) for case in cases]
    else:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            rows = []
            for row in pool.imap(sweep.run_case, cases):
                rows.append(row)
                if len(rows) % 25 == 0:
                    print(f"{len(rows)}/{len(cases)} complete", flush=True)
    if source_hashes() != hashes:
        raise RuntimeError("production source changed while experiments were running")
    sweep.classify_rows(rows)
    name = "d4-robust-" + args.controller.removeprefix("my_")
    meta = dict(name=name, controller=args.controller, detector="my_detector", workers=args.workers,
                source_sha256=hashes, protocol_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                mem_available_gib_before=memory, integrity=integrity, case_count=len(rows),
                elapsed_seconds=time.perf_counter()-start, official=False)
    sweep.write_outputs(sweep.safe_output_dir(name), meta, rows)
    summary = summarize(rows)
    (ROOT / f"results/D4_ROBUST_{args.controller}.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "failure_map"}, indent=2))


if __name__ == "__main__":
    main()
