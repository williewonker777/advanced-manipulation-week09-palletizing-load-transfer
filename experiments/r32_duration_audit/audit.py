"""Authenticate and compare disclosed D4 and R26 detector observations.

This module never imports a detector or simulator.  It consumes completed,
disclosed reports and their saved per-episode records.  Alert width and force
overlap are descriptive: neither establishes a physical causal mechanism.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


EXPECTED_EPISODES = 385
EXPECTED_COLLISIONS = 180
EXPECTED_WINDOW_SAMPLES = 320
EXPECTED_PRIOR_SHA256 = (
    "7070f5f1651e07faa8701f28421082b7eaf32a16cb866dbf8e5d9ffa07018d43"
)
EXPECTED_COMPOSED_SHA256 = (
    "dab9bfe328f5f57c3db57afb1aa4b0a28665a633c84708de5f019aba499608c7"
)
EXPECTED_COMPOSED_SOURCE_SHA256 = (
    "4567628058b7284e6db1216d38e891bd182e18c415fc79613075c8a38b848a5a"
)
EXPECTED_COMPOSED_COMPARISON_SHA256 = (
    "91af2fd556319dcf1d38c2014f5cf5976475745642e8509888fbb740a1189f33"
)
EXPECTED_CONTACT_SHA256 = (
    "d15ec904019828349aa02b00c1ed111dafb9e3ccbd49f9513ea81942f484c534"
)

ARM_LABELS = {
    "c3": "C3",
    "composed": "R29 composed",
    "headroom": "R30 headroom",
    "identity": "R31 identity",
    "r32_a": "R32 A measured level",
    "c2": "R32 C2 handoff",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def occupied(spans: list[list[int]], lo: int, hi: int) -> int:
    """Count HIGH samples in an inclusive [lo, hi] window."""
    if hi < lo:
        return 0
    return sum(max(0, min(b, hi + 1) - max(a, lo)) for a, b in spans)


def whole_high(spans: list[list[int]]) -> int:
    return sum(b - a for a, b in spans)


def primary_width(arm: dict[str, Any], onset_s: float | None) -> int | None:
    """Return the strict attributed interval width on the 1 kHz grid."""
    score = arm["score"]
    if not score["strict"]:
        return None
    if onset_s is None or score["delay_ms"] is None:
        raise ValueError("strict artificial event lacks onset or delay")
    start = bisect_left(tuple(tick * 0.001 for tick in range(6000)), onset_s)
    start += int(score["delay_ms"])
    matches = [(a, b) for a, b in arm["alarm_intervals"] if a == start]
    if len(matches) != 1:
        raise ValueError("strict rise does not identify exactly one interval")
    a, b = matches[0]
    return b - a


def _validate_score(score: dict[str, Any]) -> None:
    required = {
        "strict", "duplicate", "raw_duplicate", "false", "delayed_false",
        "masked", "stuck", "delay_ms", "valid",
    }
    if not required.issubset(score):
        raise ValueError(f"incomplete score: {sorted(required - set(score))}")
    if not all(score["valid"].values()):
        raise ValueError("invalid physical/prefix/attribution score")


def _row_metrics(
    arm: dict[str, Any], rec: dict[str, Any], onset_s: float | None
) -> dict[str, Any]:
    _validate_score(arm["score"])
    spans = arm["alarm_intervals"]
    sample_count = rec["sample_count"]
    if any(not (0 <= a < b <= sample_count) for a, b in spans):
        raise ValueError("invalid alarm interval")
    lo, hi = rec["true_window_first_index"], rec["true_window_last_index"]
    on, off = rec["force_first_nonzero_index"], rec["force_last_nonzero_index"]
    result = {
        "score": arm["score"],
        "alarm_intervals": spans,
        "whole_trace_high_samples": whole_high(spans),
        "within_320ms_high_samples": 0 if lo is None else occupied(spans, lo, hi),
        "primary_width_ms": primary_width(arm, onset_s),
        "external_nonzero_samples": 0,
        "overlap_external_samples": 0,
        "post_force_high_within_320ms": 0,
        "post_force_high_whole_trace": 0,
    }
    if on is not None:
        if lo is None or off is None or hi - lo + 1 != EXPECTED_WINDOW_SAMPLES:
            raise ValueError("collision row lacks exact 320-sample attribution window")
        result.update(
            external_nonzero_samples=off - on + 1,
            overlap_external_samples=occupied(spans, on, off),
            post_force_high_within_320ms=occupied(spans, off + 1, hi),
            post_force_high_whole_trace=occupied(spans, off + 1, sample_count - 1),
        )
    return result


def _load_raw_rows(rows_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(rows_dir.glob("*.json"))
    expected = [f"{i:03d}.json" for i in range(EXPECTED_EPISODES)]
    if [p.name for p in paths] != expected:
        raise ValueError(f"raw row set is not exact 0..{EXPECTED_EPISODES - 1}")
    rows = [load_json(p) for p in paths]
    if [row["id"] for row in rows] != list(range(EXPECTED_EPISODES)):
        raise ValueError("raw row ids are not complete and ordered")
    return rows


def _validate_result(result: dict[str, Any], raw: list[dict[str, Any]]) -> None:
    if result.get("schema") != "r32-d4-physical-level-audit.v1":
        raise ValueError("unexpected D4 result schema")
    if result.get("episodes") != EXPECTED_EPISODES:
        raise ValueError("incomplete D4 result")
    if result.get("matched_force_physics_prefix") != EXPECTED_EPISODES:
        raise ValueError("D4 force/physics/prefix authentication is incomplete")
    if result.get("prior_sha256") != EXPECTED_PRIOR_SHA256:
        raise ValueError("D4 prior report digest changed")
    if [row["id"] for row in result["rows"]] != list(range(EXPECTED_EPISODES)):
        raise ValueError("D4 result row ids are not complete")
    for summary, saved in zip(result["rows"], raw, strict=True):
        rec = saved["rec"]
        if not rec["complete_fixed_grid"] or rec["sample_count"] != 6000:
            raise ValueError("raw row is not a complete 1 kHz episode")
        if saved["id"] != summary["id"]:
            raise ValueError("raw/result row id mismatch")
        if "source" in saved and saved["source"] != result["candidate_sha256"]:
            raise ValueError("raw candidate source digest differs from result")
        if "entry" in saved and saved["entry"] != result["candidate"]:
            raise ValueError("raw candidate entry differs from result")
        if rec["detect_intervals"] != summary["arms"]["new"]["alarm_intervals"]:
            raise ValueError("raw candidate intervals differ from result")
        if saved.get("side") is not None:
            if saved["side"] != summary["arms"]["identity"]["alarm_intervals"]:
                raise ValueError("saved R31 channel differs from result")
        for arm in summary["arms"].values():
            _validate_score(arm["score"])


def _composed_rows(
    observations: dict[str, Any],
    comparison: dict[str, Any],
    observation_digest: str,
) -> list[dict[str, Any]]:
    if observation_digest != EXPECTED_COMPOSED_SHA256:
        raise ValueError("R29 composed observation digest changed")
    if observations.get("source_sha256") != EXPECTED_COMPOSED_SOURCE_SHA256:
        raise ValueError("R29 composed detector source digest changed")
    rows = observations.get("rows", [])
    if observations.get("episodes") != EXPECTED_EPISODES or [r["id"] for r in rows] != list(range(EXPECTED_EPISODES)):
        raise ValueError("R29 composed observations are incomplete")
    if (
        comparison.get("old_sha256") != observation_digest
        or comparison.get("new_sha256") != EXPECTED_PRIOR_SHA256
        or comparison.get("matched_physics_prefix_and_signed") != EXPECTED_EPISODES
    ):
        raise ValueError("R29 composed authentication chain is incomplete")
    for row in rows:
        _validate_score(row["score"])
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [row["score"] for row in rows]
    delays = [int(s["delay_ms"]) for s in scores if s["strict"]]
    force_samples = sum(row["external_nonzero_samples"] for row in rows)
    overlap = sum(row["overlap_external_samples"] for row in rows)
    return {
        "episodes": len(rows),
        "strict": sum(bool(s["strict"]) for s in scores),
        "duplicate": sum(int(s["duplicate"]) for s in scores),
        "raw_duplicate": sum(int(s["raw_duplicate"]) for s in scores),
        "false_rises": sum(len(s["false"]) for s in scores),
        "delayed_false_rises": sum(len(s["delayed_false"]) for s in scores),
        "masked_rows": sum(bool(s["masked"]) for s in scores),
        "stuck_rows": sum(bool(s["stuck"]) for s in scores),
        "strict_delay_ms": {
            "count": len(delays),
            "min": min(delays) if delays else None,
            "median": statistics.median(delays) if delays else None,
            "max": max(delays) if delays else None,
            "histogram": {str(k): v for k, v in sorted(Counter(delays).items())},
        },
        "primary_width_ms": {
            "count": sum(row["primary_width_ms"] is not None for row in rows),
            "histogram": {
                str(k): v
                for k, v in sorted(
                    Counter(
                        row["primary_width_ms"]
                        for row in rows
                        if row["primary_width_ms"] is not None
                    ).items()
                )
            },
        },
        "support": {
            "whole_trace_high_samples": sum(row["whole_trace_high_samples"] for row in rows),
            "collision_320ms_high_samples": sum(row["within_320ms_high_samples"] for row in rows),
            "externally_applied_nonzero_samples": force_samples,
            "high_overlapping_external_samples": overlap,
            "external_sample_coverage": None if force_samples == 0 else overlap / force_samples,
            "post_force_high_samples_within_320ms": sum(row["post_force_high_within_320ms"] for row in rows),
            "post_force_high_samples_whole_trace": sum(row["post_force_high_whole_trace"] for row in rows),
            "collision_rows_with_any_external_overlap": sum(row["overlap_external_samples"] > 0 for row in rows),
        },
    }


def _validate_published_counts(
    result: dict[str, Any], arms: dict[str, list[dict[str, Any]]], new_label: str
) -> None:
    for published, local in (("c3", "c3"), ("headroom", "headroom"), ("identity", "identity"), ("new", new_label)):
        expected = result["counts"][published]
        actual = _aggregate(arms[local])
        checks = {
            "strict": actual["strict"],
            "duplicates": actual["duplicate"],
            "raw_duplicate": actual["raw_duplicate"],
            "false_rises": actual["false_rises"],
            "physically_overlapping_samples": actual["support"]["high_overlapping_external_samples"],
        }
        if any(expected[key] != value for key, value in checks.items()):
            raise ValueError(f"published count mismatch for {published}: {checks} vs {expected}")


def _new_indices(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> list[int]:
    return sorted(set(candidate["score"][key]) - set(baseline["score"][key]))


def compare(candidate: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    if len(candidate) != len(baseline):
        raise ValueError("arm length mismatch")
    strict_losses = []
    duplicate_regressions = []
    raw_duplicate_regressions = []
    new_false = []
    new_delayed_false = []
    mask_regressions = []
    stuck_regressions = []
    latency_failures = []
    window_high_failures = []
    delay_deltas = []
    window_deltas = []
    whole_deltas = []
    for idx, (new, old) in enumerate(zip(candidate, baseline, strict=True)):
        ns, os = new["score"], old["score"]
        if os["strict"] and not ns["strict"]:
            strict_losses.append(idx)
        if ns["duplicate"] > os["duplicate"]:
            duplicate_regressions.append(idx)
        if ns["raw_duplicate"] > os["raw_duplicate"]:
            raw_duplicate_regressions.append(idx)
        added = _new_indices(new, old, "false")
        if added:
            new_false.append({"id": idx, "indices": added})
        added = _new_indices(new, old, "delayed_false")
        if added:
            new_delayed_false.append({"id": idx, "indices": added})
        if ns["masked"] and not os["masked"]:
            mask_regressions.append(idx)
        if ns["stuck"] and not os["stuck"]:
            stuck_regressions.append(idx)
        if os["strict"]:
            if not ns["strict"] or ns["delay_ms"] > os["delay_ms"] + 5:
                latency_failures.append(idx)
            if new["within_320ms_high_samples"] > old["within_320ms_high_samples"] + 5:
                window_high_failures.append(idx)
        if os["strict"] and ns["strict"]:
            delay_deltas.append(ns["delay_ms"] - os["delay_ms"])
        window_deltas.append(new["within_320ms_high_samples"] - old["within_320ms_high_samples"])
        whole_deltas.append(new["whole_trace_high_samples"] - old["whole_trace_high_samples"])
    return {
        "baseline_strict_rows": sum(row["score"]["strict"] for row in baseline),
        "strict_losses": strict_losses,
        "duplicate_regressions": duplicate_regressions,
        "raw_duplicate_regressions": raw_duplicate_regressions,
        "new_false_rises": new_false,
        "new_delayed_false_rises": new_delayed_false,
        "mask_regressions": mask_regressions,
        "stuck_regressions": stuck_regressions,
        "latency_plus_5ms_failures": latency_failures,
        "within_320ms_high_plus_5_failures": window_high_failures,
        "joint_strict_delay_delta_ms": {
            "min": min(delay_deltas) if delay_deltas else None,
            "max": max(delay_deltas) if delay_deltas else None,
        },
        "all_row_320ms_high_delta_samples": {
            "min": min(window_deltas), "max": max(window_deltas)
        },
        "all_row_whole_trace_high_delta_samples": {
            "min": min(whole_deltas), "max": max(whole_deltas)
        },
    }


def c3_width_floor(candidate: list[dict[str, Any]], c3: list[dict[str, Any]]) -> dict[str, Any]:
    missing = []
    short = []
    comparable = 0
    for idx, (new, old) in enumerate(zip(candidate, c3, strict=True)):
        old_width = old["primary_width_ms"]
        if old_width is None:
            continue
        new_width = new["primary_width_ms"]
        if new_width is None:
            missing.append(idx)
        else:
            comparable += 1
            if new_width + 5 < old_width:
                short.append({"id": idx, "candidate_width_ms": new_width, "c3_width_ms": old_width})
    return {
        "rule": "candidate primary width + 5 ms must be at least C3 primary width",
        "c3_strict_width_rows": sum(row["primary_width_ms"] is not None for row in c3),
        "joint_strict_width_rows": comparable,
        "missing_candidate_strict_width_ids": missing,
        "shortfall_rows": short,
    }


def _build_d4_arm_rows(
    result: dict[str, Any],
    raw: list[dict[str, Any]],
    composed: list[dict[str, Any]],
    new_label: str,
) -> dict[str, list[dict[str, Any]]]:
    arms: dict[str, list[dict[str, Any]]] = {name: [] for name in ("c3", "composed", "headroom", "identity", new_label)}
    for summary, saved, composed_row in zip(result["rows"], raw, composed, strict=True):
        case = summary["case"]
        onset = None if case["collision"] is None else case["collision"][0]
        rec = saved["rec"]
        mapping = {
            "c3": summary["arms"]["c3"],
            "composed": {
                "score": composed_row["score"],
                "alarm_intervals": composed_row["alarm_intervals"],
            },
            "headroom": summary["arms"]["headroom"],
            "identity": summary["arms"]["identity"],
            new_label: summary["arms"]["new"],
        }
        for name, arm in mapping.items():
            arms[name].append(_row_metrics(arm, rec, onset))
    return arms


def _verify_shared_arms(left: dict[str, list[dict[str, Any]]], right: dict[str, list[dict[str, Any]]]) -> None:
    for name in ("c3", "composed", "headroom", "identity"):
        for idx, (a, b) in enumerate(zip(left[name], right[name], strict=True)):
            for key in ("score", "alarm_intervals", "whole_trace_high_samples", "within_320ms_high_samples"):
                if a[key] != b[key]:
                    raise ValueError(f"shared arm {name} differs at row {idx}: {key}")


def _verify_shared_physics(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> None:
    keys = (
        "physical_sha256",
        "prefix_sha256",
        "sample_count",
        "force_first_nonzero_index",
        "force_last_nonzero_index",
        "true_window_first_index",
        "true_window_last_index",
        "signed_expected_exact",
        "stock_scalar_matches_signed",
    )
    for idx, (a, b) in enumerate(zip(left, right, strict=True)):
        ar, br = a["rec"], b["rec"]
        if any(ar[key] != br[key] for key in keys):
            raise ValueError(f"contact/handoff physical record differs at row {idx}")


def _r26_summary(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if data.get("episodes") != 156 or not data.get("disclosed"):
        raise ValueError(f"not a disclosed complete R26 report: {path}")
    candidate = data["candidate"]
    c3 = "experiments.r26_yaw_acquisition.candidate:Detector"
    if c3 not in data["counts"] or candidate not in data["counts"]:
        raise ValueError("R26 report lacks C3 or candidate counts")
    losses = data.get("relative_losses", [])
    return {
        "artifact_sha256": sha256(path),
        "candidate_entry": candidate,
        "candidate_sha256": data["candidate_sha256"],
        "candidate_counts": data["counts"][candidate],
        "c3_counts": data["counts"][c3],
        "candidate_both_second_strict": data["paired_strict"]["both_second"][candidate],
        "c3_both_second_strict": data["paired_strict"]["both_second"][c3],
        "candidate_quiet": data["counts"][candidate]["false_rises"] == 0,
        "candidate_cost_losses": [loss for loss in losses if any("cost" in reason for reason in loss[2])],
        "relative_losses": losses,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _brief(report: dict[str, Any]) -> str:
    lines = [
        "# R32 disclosed duration audit",
        "",
        "This is a read-only comparison of already disclosed evidence. It is not an",
        "eligibility decision, a new protected trial, or proof that alert duration",
        "caused physical contact. Width is Boolean HIGH duration, not force duration.",
        "",
        "## Authentication",
        "",
        f"- D4 conditions: {report['scope']['d4_conditions']} complete 1 kHz episodes.",
        f"- Collision attribution windows: {report['scope']['collision_rows']} × 320 samples.",
        f"- R29 composed observations SHA-256: `{report['sources']['r29_composed_observations']['sha256']}`.",
        f"- Included candidate arms: {', '.join(report['scope']['candidate_arms'])}.",
        "",
        "## D4 aggregate",
        "",
        "| arm | strict | dup | false | mask | HIGH whole | HIGH 320 ms | force overlap | after-force/320 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, aggregate in report["d4"]["arms"].items():
        support = aggregate["support"]
        lines.append(
            f"| {ARM_LABELS[name]} | {aggregate['strict']} | {aggregate['duplicate']} | "
            f"{aggregate['false_rises']} | {aggregate['masked_rows']} | "
            f"{support['whole_trace_high_samples']} | {support['collision_320ms_high_samples']} | "
            f"{support['high_overlapping_external_samples']}/{support['externally_applied_nonzero_samples']} | "
            f"{support['post_force_high_samples_within_320ms']} |"
        )
    lines.extend(["", "## Protection comparisons", ""])
    for candidate, comparisons in report["d4"]["candidate_comparisons"].items():
        floor = report["d4"]["c3_width_floor"][candidate]
        lines.append(
            f"- **{ARM_LABELS[candidate]}**: C3 width shortfalls {len(floor['shortfall_rows'])}, "
            f"missing retained C3 widths {len(floor['missing_candidate_strict_width_ids'])}."
        )
        for baseline, item in comparisons.items():
            lines.append(
                f"  - vs {ARM_LABELS[baseline]}: strict losses {len(item['strict_losses'])}, "
                f"duplicate regressions {len(item['duplicate_regressions'])}, "
                f"new false rows {len(item['new_false_rises'])}, "
                f"latency >+5 ms {len(item['latency_plus_5ms_failures'])}, "
                f"320 ms HIGH >+5 {len(item['within_320ms_high_plus_5_failures'])}."
            )
    lines.extend([
        "",
        "## Disclosed R26 comparison",
        "",
        "| arm | strict/57 | quiet false rises | duplicate slots | both-second strict | cost losses | all relative losses |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    r26_names = {
        "r29_composed": "R29 composed",
        "r31_identity": "R31 identity",
        "r32_a": "R32 A measured level",
        "c2": "R32 C2 handoff",
    }
    first_r26 = next(iter(report["r26"].values()))
    c3_counts = first_r26["c3_counts"]
    lines.append(
        f"| C3 | {c3_counts['strict']} | {c3_counts['false_rises']} | "
        f"{c3_counts['duplicate_slots']} | {first_r26['c3_both_second_strict']} | n/a | n/a |"
    )
    for name, item in report["r26"].items():
        counts = item["candidate_counts"]
        lines.append(
            f"| {r26_names[name]} | {counts['strict']} | {counts['false_rises']} | "
            f"{counts['duplicate_slots']} | {item['candidate_both_second_strict']} | "
            f"{len(item['candidate_cost_losses'])} | {len(item['relative_losses'])} |"
        )
    lines.extend([
        "",
        "## Causality and limits",
        "",
        "- External-force overlap uses the saved true nonzero applied-force boundaries.",
        "- Prefix and attribution validation support temporal attribution only. They do not",
        "  prove that a detector's internal state was caused by physical contact.",
        "- Post-force HIGH samples are reported separately from in-force coverage.",
        "- No arbitrary promotion threshold or eligibility conclusion is introduced here.",
        "",
        "## Reproduction",
        "",
        "Set `R` to your research checkout and `E` to your saved experiment receipts.",
        "Run from `$R` with bytecode disabled:",
        "",
        "```bash",
        "PYTHONDONTWRITEBYTECODE=1 \"$R/.venv/bin/python\" -B -m experiments.r32_duration_audit.audit \\",
        "  --contact-result \"$E/r32/contact/d4/result.json\" --contact-rows \"$E/r32/contact/d4/rows\" \\",
        "  --handoff-result \"$E/r32/handoff/d4/result.json\" --handoff-rows \"$E/r32/handoff/d4/rows\" \\",
        "  --composed-observations \"$E/r31/qa/composed/observations.json\" \\",
        "  --composed-comparison \"$E/r31/qa/composed/comparison.json\" \\",
        "  --r26-contact \"$E/r32/contact/r26.json\" --r26-handoff \"$E/r32/handoff/r26.json\" \\",
        "  --r26-identity \"$E/r31/identity/full.json\" --r26-composed \"$E/r29/composed/whole.json\" \\",
        "  --out-dir NEW_EMPTY_OUTPUT",
        "```",
        "",
    ])
    return "\n".join(lines)


def assess(args: argparse.Namespace) -> dict[str, Any]:
    contact_digest = sha256(args.contact_result)
    if contact_digest != EXPECTED_CONTACT_SHA256:
        raise ValueError("R32 A contact result digest changed")
    contact_result = load_json(args.contact_result)
    contact_raw = _load_raw_rows(args.contact_rows)
    _validate_result(contact_result, contact_raw)

    observations_digest = sha256(args.composed_observations)
    comparison_digest = sha256(args.composed_comparison)
    if comparison_digest != EXPECTED_COMPOSED_COMPARISON_SHA256:
        raise ValueError("R29 composed comparison digest changed")
    observations = load_json(args.composed_observations)
    comparison_chain = load_json(args.composed_comparison)
    composed = _composed_rows(observations, comparison_chain, observations_digest)

    arms = _build_d4_arm_rows(contact_result, contact_raw, composed, "r32_a")
    _validate_published_counts(contact_result, arms, "r32_a")
    sources = {
        "r32_a_d4_result": {"sha256": contact_digest},
        "r32_a_d4_rows": {
            "count": len(contact_raw),
            "combined_sha256": hashlib.sha256(
                b"".join(args.contact_rows.joinpath(f"{i:03d}.json").read_bytes() for i in range(EXPECTED_EPISODES))
            ).hexdigest(),
        },
        "authenticated_d4_prior": {"sha256": EXPECTED_PRIOR_SHA256},
        "r29_composed_observations": {"sha256": observations_digest},
        "r29_composed_comparison": {"sha256": comparison_digest},
    }
    candidate_arms = ["r32_a"]

    if args.handoff_result is not None:
        handoff_result = load_json(args.handoff_result)
        handoff_raw = _load_raw_rows(args.handoff_rows)
        _validate_result(handoff_result, handoff_raw)
        handoff_arms = _build_d4_arm_rows(handoff_result, handoff_raw, composed, "c2")
        _validate_published_counts(handoff_result, handoff_arms, "c2")
        _verify_shared_arms(arms, handoff_arms)
        _verify_shared_physics(contact_raw, handoff_raw)
        arms["c2"] = handoff_arms["c2"]
        candidate_arms.append("c2")
        sources["c2_d4_result"] = {"sha256": sha256(args.handoff_result)}
        sources["c2_d4_rows"] = {
            "count": len(handoff_raw),
            "combined_sha256": hashlib.sha256(
                b"".join(args.handoff_rows.joinpath(f"{i:03d}.json").read_bytes() for i in range(EXPECTED_EPISODES))
            ).hexdigest(),
        }

    baselines = ("c3", "composed", "headroom", "identity")
    candidate_comparisons = {
        candidate: {baseline: compare(arms[candidate], arms[baseline]) for baseline in baselines}
        for candidate in candidate_arms
    }
    width_floors = {candidate: c3_width_floor(arms[candidate], arms["c3"]) for candidate in candidate_arms}

    collision_rows = sum(row["rec"]["force_first_nonzero_index"] is not None for row in contact_raw)
    if collision_rows != EXPECTED_COLLISIONS:
        raise ValueError("unexpected D4 collision count")

    r26 = {
        "r29_composed": _r26_summary(args.r26_composed),
        "r31_identity": _r26_summary(args.r26_identity),
        "r32_a": _r26_summary(args.r26_contact),
    }
    if args.r26_handoff is not None:
        r26["c2"] = _r26_summary(args.r26_handoff)
    baseline_counts = [value["c3_counts"] for value in r26.values()]
    if any(value != baseline_counts[0] for value in baseline_counts[1:]):
        raise ValueError("R26 C3 baselines disagree")

    return {
        "schema": "r32-disclosed-duration-comparison.v1",
        "classification": "Read-only historical/disclosed science; no eligibility or promotion decision.",
        "scope": {
            "d4_conditions": EXPECTED_EPISODES,
            "collision_rows": collision_rows,
            "reference_or_tool_rows": EXPECTED_EPISODES - collision_rows,
            "candidate_arms": candidate_arms,
            "simulations_run": 0,
            "protected_inputs_opened": 0,
        },
        "definitions": {
            "primary_width": "Width of the strict attributed Boolean interval starting at onset + reported delay on the 1 kHz grid.",
            "within_320ms_high": "Boolean HIGH samples clipped to the saved 320-sample causal attribution window.",
            "external_support": "Boolean HIGH samples overlapping saved true nonzero externally applied-force samples.",
            "after_force_tail": "Boolean HIGH samples after the last saved nonzero externally applied-force sample; not force duration.",
            "causality_limit": "Temporal overlap and authenticated prefixes do not prove an internal causal mechanism or physical contact identity.",
        },
        "sources": sources,
        "d4": {
            "arms": {name: _aggregate(rows) for name, rows in arms.items()},
            "candidate_comparisons": candidate_comparisons,
            "c3_width_floor": width_floors,
        },
        "r26": r26,
        "conclusion_boundary": "Descriptive comparison only. No arbitrary threshold, eligibility, default, safety, or promotion claim.",
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contact-result", type=Path, required=True)
    p.add_argument("--contact-rows", type=Path, required=True)
    p.add_argument("--handoff-result", type=Path)
    p.add_argument("--handoff-rows", type=Path)
    p.add_argument("--composed-observations", type=Path, required=True)
    p.add_argument("--composed-comparison", type=Path, required=True)
    p.add_argument("--r26-contact", type=Path, required=True)
    p.add_argument("--r26-handoff", type=Path)
    p.add_argument("--r26-identity", type=Path, required=True)
    p.add_argument("--r26-composed", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if (args.handoff_result is None) != (args.handoff_rows is None):
        raise ValueError("handoff result and rows must be supplied together")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("output directory must be new or empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = assess(args)
    _write_json(args.out_dir / "aggregate.json", report)
    (args.out_dir / "BRIEF.md").write_text(_brief(report) + "\n")
    receipt = {
        "schema": "r32-duration-audit-receipt.v1",
        "aggregate_sha256": sha256(args.out_dir / "aggregate.json"),
        "brief_sha256": sha256(args.out_dir / "BRIEF.md"),
        "candidate_arms": report["scope"]["candidate_arms"],
        "d4_conditions": report["scope"]["d4_conditions"],
        "simulations_run": 0,
        "assessor_sha256": sha256(Path(__file__)),
        "tests_sha256": sha256(Path(__file__).with_name("test_audit.py")),
    }
    _write_json(args.out_dir / "receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

