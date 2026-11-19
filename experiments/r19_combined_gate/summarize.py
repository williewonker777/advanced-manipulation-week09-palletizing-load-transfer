"""Summarize, without rescoring, the unchanged integrated gate outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


STAGES = ("known-normal", "exposed-impact", "fresh-normal", "fresh-impact")


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args(argv)
    if args.out.exists():
        cli.error("fresh output required")
    summary = json.loads((args.run / "summary.json").read_bytes())
    opened = [row["stage"] for row in summary["stages"]]
    if opened != list(STAGES[:len(opened)]):
        raise ValueError("non-progressive stage order")
    details = {}
    timings = []
    for stage in opened:
        doc = json.loads((args.run / (stage + ".json")).read_bytes())
        rows = doc["rows"]
        for row in rows:
            timings.extend(row["extra_measurements"])
        if stage.endswith("impact"):
            details[stage] = [{
                "id": row["id"],
                "outcome_delay_ms": {arm: {
                    "outcome": event["outcome"], "delay_ms": event["delay_ms"],
                    "false_alarm_times": event["false_alarm_times"], "prealarm": event["prealarm"],
                } for arm, event in row["arms"].items()},
                "rising_times_s": row["rising_times_s"],
                "false_event_match": row["false_event_match"],
                "failed_gates": [key for key, value in {**row["gates"], **row["extra_gates"]}.items() if not value],
            } for row in rows]
        else:
            details[stage] = [{
                "id": row["id"],
                "false_alarm_times": {arm: values["false_alarm_times"] for arm, values in row["arms"].items()},
                "false_event_match": row.get("false_event_match"),
                "failed_gates": [key for key, value in {**row["gates"], **row["extra_gates"]}.items() if not value],
            } for row in rows]
            if stage == "known-normal":
                details["fallback"] = [{
                    "id": row["id"], "kind": row["case"]["kind"],
                    "passed": row["passed"], "parity": row["parity"],
                    "failed_gates": [key for key, value in row["gates"].items() if not value],
                } for row in rows if "parity" in row]
    output = {
        "schema": "r19-combined-regression-summary.v1",
        "research_only": True,
        "independent_claim": False,
        "unchanged_scorer_summary": summary,
        "details": details,
        "timing": {
            "measurement_records": len(timings),
            "timed_cycles": sum(row.get("timed_cycles", 6000) for row in timings),
            "over_1ms": sum(row["combined_over_1ms"] for row in timings),
            "worst_case_p99_us": max(row["combined_p99_us"] for row in timings),
            "maximum_us": max(row["combined_max_us"] for row in timings),
        },
        "disposition": "regression only; hard deadline still requires zero over-1ms cycles",
    }
    with args.out.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"stages": opened, "timing": output["timing"]}, sort_keys=True))


if __name__ == "__main__":
    main()
