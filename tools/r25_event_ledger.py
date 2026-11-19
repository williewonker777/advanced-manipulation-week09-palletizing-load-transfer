"""Describe an already verified R25 result; never change its scientific verdict."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def failure_reason(event):
    """Give disjoint reasons; no attributable rise is not necessarily no signal."""
    if event["strict"]:
        return None
    if not event["attributable_rise_indices"]:
        return "no_attributable_rise"
    if event["duplicate_rises"]:
        return "multiple_rises"
    if not event["quiet_before"]:
        return "insufficient_preceding_quiet"
    return "other_non_strict"


def events(record):
    return {"single": record["event"]} if "event" in record else record.get("events", {})


def summarize(document):
    if document.get("schema") != "r25-independent-variable-pulse-result.v1":
        raise ValueError("R25 independent result required")
    rows = document["rows"]
    if sum(row["episodes"] for row in rows) != document["episodes"]:
        raise ValueError("inconsistent episode accounting")
    arms = ("r24_joint", "r25")
    stats, changed, differences, floors = {}, [], [], Counter()
    for arm in arms:
        counts, slots, reasons = Counter(), {}, Counter()
        for row in rows:
            rec = row["records"][arm]
            selected = events(rec)
            if row["kind"] == "paired":
                counts["paired_rows"] += 1
                counts["five_of_five_rows"] += all(item["strict"] for item in selected.values())
            for slot, item in selected.items():
                tally = slots.setdefault(slot, {"strict": 0, "total": 0})
                tally["strict"] += item["strict"]
                tally["total"] += 1
                counts["positive_slots"] += 1
                counts["strict_slots"] += item["strict"]
                counts["extra_rises"] += item["duplicate_rises"]
                counts["slots_with_multiple_rises"] += bool(item["duplicate_rises"])
                reason = failure_reason(item)
                if reason:
                    reasons[reason] += 1
            for twin in rec.get("twins", {"none": rec}).values():
                counts["false_rises"] += len(twin["false_times_s"])
                counts["stuck_tails"] += bool(twin["stuck_tail"])
                counts["real_tool_contacts"] += bool(twin["tool"]["present"])
                counts["tool_detections"] += bool(twin["tool"]["detected"])
        stats[arm] = {"total": dict(counts), "by_slot": slots, "miss_reasons": dict(reasons)}
    for row in rows:
        case = row["variant"] + ":" + row["id"]
        before, after = [row["records"][arm] for arm in arms]
        if before != after:
            changed.append(case)
        for guard, passed in row["incumbent_safety"].items():
            if not passed:
                floors[guard] += 1
        a, b = events(before), events(after)
        for slot, old in a.items():
            if old != b[slot]:
                differences.append({"row": case, "slot": slot, "before": old, "after": b[slot]})
    if sum(stats["r25"]["miss_reasons"].values()) != document["claims"]["absolute_miss_count"]:
        raise ValueError("ledger does not agree with independent absolute misses")
    return {"schema": "r25-secondary-event-ledger.v1",
            "scope": "Descriptive post-run accounting, not a replacement qualification.",
            "episodes": document["episodes"], "rows": len(rows), "failed_rows": len(document["failed"]),
            "protocol_claims": document["claims"], "strict_and_alarm_totals": stats,
            "changed_full_scientific_records": changed, "changed_events": differences,
            "failed_guard_row_counts": dict(floors),
            "cases_failing_any_guard": sum(not all(row["incumbent_safety"].values()) for row in rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    result = summarize(json.loads(raw))
    result["independent_result_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.out.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
