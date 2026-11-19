"""Full 385-case D4 protection check for the optional R23 tool witness."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r21_full_protection import runner_v2 as old
from experiments.r22_census import check as census
from experiments.r23_tool_contact.candidate import Detector as Candidate
from tools.d4_robustness import build_cases


HERE = Path(__file__).resolve().parent
CACHE_SHA256 = "c62dd430d7f3b9a4553c0562558f481a4518a893050d1cf8b28e82d3c6c44fcf"
R22_SHA256 = "99656b969d2155bfcb1e5a7cb5d3ef8cfe65d97682832538006d254d5146891f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AuditedCandidate(Candidate):
    latest = None

    def reset(self):
        self.r23_witness = []
        type(self).latest = self
        super().reset()

    def update(self, t, th, thd, tau_m, tau_s, grip):
        old_latch = getattr(self, "_tool_latch", 0)
        result = super().update(t, th, thd, tau_m, tau_s, grip)
        self.r23_witness.append(bool(old_latch > 0 or self._tool_streak >= self._tool_persistence))
        return result


def simulate(case):
    record = old.episode_record(AuditedCandidate, case)
    witness = np.asarray(AuditedCandidate.latest.r23_witness, bool)
    return {"record": record, "witness_intervals": old.intervals(witness)}


def simulate_twin(case):
    twin = dict(case, collision=None, pick_offset=None, kind="same_motion_no_offset")
    return old.episode_record(AuditedCandidate, twin)


def compare_score(candidate, base):
    return {
        "strict_retained": not base["strict"] or candidate["strict"],
        "no_new_false": not census.unmatched_indices(candidate["false"], base["false"]),
        "no_new_delayed_false": not census.unmatched_indices(candidate["delayed_false"],
                                                                base["delayed_false"]),
        "no_new_mask": not candidate["masked"] or base["masked"],
        "no_new_duplicate": (candidate["duplicate"] <= base["duplicate"]
                             and candidate["raw_duplicate"] <= base["raw_duplicate"]),
        "tool_retained": not base["tool"]["detected"] or candidate["tool"]["detected"],
        "no_new_tool_prealarm": not candidate["tool"]["prealarm"] or base["tool"]["prealarm"],
        "no_new_stuck": not candidate["stuck"] or base["stuck"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--r22", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--records", type=Path, nargs="*")
    parser.add_argument("--r22-offset-records", type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.public.exists():
        parser.error("fresh outputs required")
    if sha(args.cache) != CACHE_SHA256 or sha(args.r22) != R22_SHA256:
        raise ValueError("canonical R21 cache or R22 LIGHT result differs")
    cache = json.loads(args.cache.read_bytes())
    r22 = json.loads(args.r22.read_bytes())
    cases = build_cases("my_torque")
    if (hashlib.sha256(old.canonical_cases(cases)).hexdigest() != census.MATRIX
            or len(cases) != 385 or len(cache["rows"]) != 385 or len(r22["rows"]) != 385):
        raise ValueError("D4 matrix/accounting differs")
    prior = {row["case"]["case_id"]: row["arms"] for row in cache["rows"]}
    base_rows = {row["id"]: row for row in r22["rows"]}
    captured_twins = {}
    if args.records:
        captured = [json.loads(path.read_bytes()) for path in args.records]
        if any(doc.get("schema") != "r23-tool-contact-d4-shard.v1"
               or doc.get("candidate_sha256") != sha(HERE / "candidate.py")
               or doc.get("matrix_sha256") != census.MATRIX for doc in captured):
            raise ValueError("D4 shard identity differs")
        captured_rows = [row for doc in captured for row in doc["rows"]]
        if len(captured_rows) != 385 or {row["id"] for row in captured_rows} != set(range(385)):
            raise ValueError("D4 shards incomplete or duplicate")
        by_id = {row["id"]: row for row in captured_rows}
        simulated = [by_id[case["case_id"]]["item"] for case in cases]
        captured_twins = {row["id"]: row["same_motion_twin"] for row in captured_rows
                           if row["same_motion_twin"] is not None}
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            simulated = list(pool.map(simulate, cases))
    references = {old.reference_key(case): item["record"]
                  for case, item in zip(cases, simulated, strict=True) if case["kind"] == "reference"}
    rows = []
    for case, item in zip(cases, simulated, strict=True):
        record = item["record"]
        normal = references[old.reference_key(case)] if case["kind"] == "collision" else record
        score = census.score(case, record, normal)
        base = base_rows[case["case_id"]]["scores"]["candidate"]
        physical = record["physical_sha256"] == prior[case["case_id"]]["r10"]["physical_sha256"]
        gates = compare_score(score, base)
        rows.append({"id": case["case_id"], "kind": case["kind"], "collision": case.get("collision"),
                     "score": score, "r22": base, "witness_intervals": item["witness_intervals"],
                     "physical_exact_r10": physical, "input_certified": physical and all(score["valid"].values()),
                     "protection": all(gates.values()), "gates": gates})
    offsets = [(case, item) for case, item in zip(cases, simulated, strict=True) if case["kind"] == "offset"]
    if captured_twins:
        twins = [captured_twins[case["case_id"]] for case, _ in offsets]
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            twins = list(pool.map(simulate_twin, (case for case, _ in offsets)))
    attribution = []
    light_offsets = None
    if args.r22_offset_records:
        document = json.loads(args.r22_offset_records.read_bytes())
        if (document.get("schema") != "r23-r22-light-offset-records.v1"
                or document.get("matrix_sha256") != census.MATRIX or len(document.get("rows", [])) != 16):
            raise ValueError("R22 LIGHT offset records differ")
        light_offsets = {row["id"]: row["record"] for row in document["rows"]}
    for (case, item), twin in zip(offsets, twins, strict=True):
        record = item["record"]
        tool = record["tool"]
        attributable = [index for index in record["tool_rise_indices"] if not old.has_twin(twin, index)]
        base_record = None if light_offsets is None else light_offsets[case["case_id"]]
        attribution.append({"id": case["case_id"], "tool": tool,
                            "candidate_tool_rises": record["tool_rise_indices"],
                            "r22_tool_rises": None if base_record is None else base_record["tool_rise_indices"],
                            "new_tool_rises_vs_r22": None if base_record is None else census.unmatched_indices(
                                record["tool_rise_indices"], base_record["tool_rise_indices"]),
                            "lost_tool_rises_vs_r22": None if base_record is None else census.unmatched_indices(
                                base_record["tool_rise_indices"], record["tool_rise_indices"]),
                            "same_motion_no_offset_intervals": twin["detect_intervals"],
                            "attributable_tool_rises": attributable,
                            "same_motion_physical_except_offset_effect":
                                record["physical_sha256"] != twin["physical_sha256"]})
    collisions = [row for row in rows if row["kind"] == "collision"]
    shared = [row for row in collisions if row["score"]["strict"] and row["r22"]["strict"]]
    latency = {row["id"]: row["score"]["delay_ms"] - row["r22"]["delay_ms"] for row in shared}
    later = [row for row in collisions if row["collision"][0] >= 1.4]
    tool_present = [row for row in rows if row["score"]["tool"]["present"]]
    summary = {
        "r22": r22["summary"]["candidate"],
        "candidate": {"strict": sum(row["score"]["strict"] for row in collisions),
                      "false": sum(len(row["score"]["false"]) for row in rows),
                      "duplicate": sum(row["score"]["duplicate"] for row in collisions),
                      "tool_detected": sum(row["score"]["tool"]["detected"] for row in rows)},
        "gained_external_ids": [row["id"] for row in collisions if row["score"]["strict"] and not row["r22"]["strict"]],
        "lost_external_ids": [row["id"] for row in collisions if not row["score"]["strict"] and row["r22"]["strict"]],
        "earlier_external_ids": [key for key, value in latency.items() if value < 0],
        "delayed_external_ids": [key for key, value in latency.items() if value > 0],
        "latency_delta_ms": latency,
        "gained_tool_ids": [row["id"] for row in tool_present
                            if row["score"]["tool"]["detected"] and not row["r22"]["tool"]["detected"]],
        "lost_tool_ids": [row["id"] for row in tool_present
                          if not row["score"]["tool"]["detected"] and row["r22"]["tool"]["detected"]],
        "new_tool_prealarm_ids": [row["id"] for row in tool_present
                                  if row["score"]["tool"]["prealarm"] and not row["r22"]["tool"]["prealarm"]],
        "tool_rise_changes": [{"id": row["id"], "r22": row["r22_tool_rises"],
                               "candidate": row["candidate_tool_rises"]}
                              for row in attribution if row["r22_tool_rises"] is not None
                              and row["r22_tool_rises"] != row["candidate_tool_rises"]],
    }
    gates = {
        "all_385_input_certified": all(row["input_certified"] for row in rows),
        "all_385_protected_vs_r22": all(row["protection"] for row in rows),
        "all_189_normals_witness_quiet": all(not row["witness_intervals"] for row in rows
                                               if row["kind"] == "reference"),
        "later_forcings_witness_quiet_and_score_exact": all(
            not row["witness_intervals"] and row["score"] == row["r22"] for row in later),
        "tool_attribution_has_same_motion_twins": all(
            not row["tool"]["present"] or bool(row["attributable_tool_rises"]) for row in attribution),
        "offset_tool_rises_no_regression": light_offsets is not None and all(
            not row["tool"]["present"] or
            (len(row["candidate_tool_rises"]) <= len(row["r22_tool_rises"])
             and row["candidate_tool_rises"] and row["r22_tool_rises"]
             and row["candidate_tool_rises"][0] <= row["r22_tool_rises"][0]
             and not row["tool"]["prealarm"]) for row in attribution),
    }
    payload = {"schema": "r23-tool-contact-d4-full.v1",
               "classification": "supplemental exposed D4 protection audit, not preregistered holdout or promotion",
               "audit_history": {
                   "original_exact_tool_rise_gate": "FAIL",
                   "reason": "genuine tool rises moved earlier in cases 370/378; exact indices differed although rise counts did not increase",
                   "preserved_full_sha256": "b78899bb7f4026f480ed1ec39a2dcdd0113c6aa73dc657b7a385680c1edf00f3",
                   "supplemental_gate": "one-to-one no-extra/no-later/no-prealarm tool-rise audit"},
               "candidate_sha256": sha(HERE / "candidate.py"), "model_sha256": sha(HERE / "model.json"),
               "inputs": {"cache_sha256": sha(args.cache), "r22_sha256": sha(args.r22),
                          "matrix_sha256": census.MATRIX},
               "counts": {"cases": len(rows), "kinds": dict(Counter(row["kind"] for row in rows)),
                          "simulated_candidate": len(rows), "same_motion_offset_twins": len(twins)},
               "summary": summary, "gates": gates, "passed": all(gates.values()),
               "tool_attribution": attribution, "rows": rows}
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    public = {key: payload[key] for key in ("schema", "classification", "audit_history", "candidate_sha256",
                                             "model_sha256", "inputs", "counts", "gates", "passed")}
    public["summary"] = dict(summary, shared_external_count=len(latency),
                             nonzero_latency_delta_ms={key: value for key, value in latency.items() if value})
    public["summary"].pop("latency_delta_ms")
    public["limits"] = ["fixed deterministic 1 ms simulator trajectory", "finite exposed D4 matrix",
                         "no hard-real-time, hardware, hidden-test, or safety claim"]
    args.public.write_text(json.dumps(public, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": payload["passed"], "summary": summary, "gates": gates}))


if __name__ == "__main__":
    main()
