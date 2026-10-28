"""Compare a default sweep with every deterministic displayed evaluate.py cell.

Wall/compute timings are intentionally excluded: wrapping, concurrency and OS
scheduling change them, but must not change simulation/decision metrics.
"""
import argparse
import json
from pathlib import Path
import re


def official_cells(text):
    control, collisions = {}, {}
    for line in text.splitlines():
        if not re.match(r"\s*\d+(?:\.\d+)?kg\s", line):
            continue
        cols = [part.strip() for part in line.split("|")]
        key = cols[0].split()
        mass = float(key[0].removesuffix("kg"))
        if len(key) == 1:
            control[mass] = [cols[1], cols[2], cols[3],
                             *[v.strip() for v in cols[4].split("/")], cols[5].removesuffix(" N"),
                             *[v.strip() for v in cols[7].removesuffix(" 회").split("/")], cols[8]]
        else:
            when = float(key[1].removesuffix("s"))
            collisions[mass, when] = cols[1:]
    if not control or not collisions:
        raise ValueError("Expected both [1] and [2] default evaluator tables")
    return control, collisions


def compare(official_text, sweep_data):
    control, collisions = official_cells(official_text)
    rows = sweep_data["results"]
    if not sweep_data["metadata"]["baseline"] or len(rows) != 27:
        raise ValueError("Expected --baseline sweep with 27 unique episodes")
    differences = []
    checked = 0
    for row in rows:
        if row["collision"] is not None:
            continue
        cells = [f"{row['rms_mm']:.2f}", f"{row['max_mm']:.2f}",
                 f"{row['saturation_pct']:.1f}", f"{row['torque_step_pick_nm']:.1f}",
                 f"{row['torque_step_place_nm']:.1f}", f"{row['table_force_place_max_n']:.0f}",
                 str(row["false_alarms_pick"]), str(row["false_alarms_place"]),
                 str(row["false_alarms_other"]), "발산" if row["diverged"] else "정상"]
        expected = control.pop(row["mass"])
        checked += len(cells)
        if cells != expected:
            differences.append({"section": 1, "mass": row["mass"],
                                "expected": expected, "actual": cells})
    for (mass, when), expected in collisions.items():
        group = sorted([row for row in rows if row["mass"] == mass and row["collision"]
                        and row["collision"][0] == when], key=lambda row: row["collision"][1][1])
        if len(group) != 4 or [r["collision"][1] for r in group] != [
                [0., f, 0.] for f in (10., 15., 40., 100.)]:
            raise ValueError("Unexpected default force matrix")
        cells = [row["collision_label"].removesuffix(" ms") for row in group]
        cells.append(f"{max(row['false_alarms'] for row in group)}회")
        detected = [row["collision"][1][1] for row in group if row["collision_outcome"] == "detected"]
        cells.append("—" if not detected else f"{min(detected):g} N")
        checked += len(cells)
        if cells != expected:
            differences.append({"section": 2, "mass": mass, "collision_time": when,
                                "expected": expected, "actual": cells})
    if control:
        raise ValueError("Missing no-collision cases")
    return {"status": "pass" if not differences else "FAIL", "checked_cells": checked,
            "excluded": ["controller computation time", "total wall time"],
            "differences": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official", type=Path)
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = compare(args.official.read_text(encoding="utf-8"),
                     json.loads(args.sweep.read_text(encoding="utf-8")))
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
