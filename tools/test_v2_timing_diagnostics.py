"""Pure tests for passive timing-diagnostic aggregation; no simulation runs."""
from __future__ import annotations

import unittest

from tools import v2_timing_diagnostics as D


def sample(component: str, start: int, end: int, *, cpu: int | None = None,
           t: float = 1.0, thread: int = 7, **deltas):
    value = {
        "component": component, "simulated_t_s": t, "thread_id": thread,
        "start_wall_ns": start, "end_wall_ns": end, "wall_ns": end - start,
        "thread_cpu_ns": end - start if cpu is None else cpu,
    }
    value.update({name: int(deltas.get(name, 0)) for name in D.USAGE_FIELDS})
    return value


class TimingDiagnosticTests(unittest.TestCase):
    def test_gc_overlap_requires_complete_same_thread_intersection(self):
        events = [
            {"complete": True, "thread_id": 7, "start_wall_ns": 110, "stop_wall_ns": 120},
            {"complete": True, "thread_id": 8, "start_wall_ns": 110, "stop_wall_ns": 120},
            {"complete": False, "thread_id": 7, "start_wall_ns": 115},
            {"complete": True, "thread_id": 7, "start_wall_ns": 201, "stop_wall_ns": 220},
        ]
        self.assertEqual(D.overlapping_gc_indices(events, 100, 200, 7), [0])
        self.assertEqual(D.overlapping_gc_indices(events, 120, 120, 7), [0])

    def test_gc_serialization_keeps_relative_evidence_not_thread_identifier(self):
        event = {"complete": True, "thread_id": 77, "generation": 2,
                 "start_wall_ns": 1_100, "stop_wall_ns": 1_300,
                 "start_thread_cpu_ns": 50, "stop_thread_cpu_ns": 125,
                 "collected": 4, "uncollectable": 0}
        got = D.serializable_gc_event(event, 1_000, 77)
        self.assertEqual(got["thread_role"], "measurement")
        self.assertEqual(got["start_offset_ns"], 100)
        self.assertEqual(got["wall_duration_ns"], 200)
        self.assertEqual(got["thread_cpu_duration_ns"], 75)
        self.assertNotIn("thread_id", got)

    def test_summary_pairs_calls_retains_outlier_and_gc_span(self):
        controller = [sample("controller", 10, 110, t=0.0),
                      sample("controller", 200, 1_200_300, cpu=1_100_000, t=.001)]
        detector = [sample("detector", 120, 220, t=0.0),
                    sample("detector", 1_200_310, 1_200_410, t=.001,
                           voluntary_context_switches=2)]
        events = [{"complete": True, "thread_id": 7, "generation": 1,
                   "start_wall_ns": 300, "stop_wall_ns": 500,
                   "start_thread_cpu_ns": 50, "stop_thread_cpu_ns": 80,
                   "collected": 2, "uncollectable": 0}]
        got = D.summarize_timing(controller, detector, events)
        self.assertEqual(got["paired_cycles"], 2)
        self.assertTrue(got["simulated_time_alignment"])
        self.assertEqual(len(got["outliers"]), 1)
        outlier = got["outliers"][0]
        self.assertEqual(outlier["combined"]["voluntary_context_switches"], 2)
        self.assertEqual(outlier["matching_gc_events"], events)
        self.assertIn("natural_gc_span_overlapped_measured_call",
                      outlier["passive_classification"]["observations"])

    def test_classifier_distinguishes_cpu_and_wall_gap_without_guessing_cause(self):
        base = {name: 0 for name in D.USAGE_FIELDS}
        cpu = {"combined": dict(base, wall_ns=2_000_000, thread_cpu_ns=1_900_000),
               "matching_gc_events": []}
        wall = {"combined": dict(base, wall_ns=2_000_000, thread_cpu_ns=200_000),
                "matching_gc_events": []}
        self.assertEqual(D.classify_outlier(cpu)["observations"],
                         ["cpu_time_dominated_without_recorded_gc_fault_or_context_delta"])
        self.assertEqual(D.classify_outlier(wall)["observations"],
                         ["wall_cpu_gap_without_recorded_thread_context_delta"])
        self.assertIn("no event is asserted", D.classify_outlier(wall)["causality"])

    def test_classifier_reports_all_observed_resource_deltas(self):
        combined = dict(wall_ns=3_000_000, thread_cpu_ns=1_000_000,
                        minor_page_faults=3, major_page_faults=1,
                        voluntary_context_switches=2, involuntary_context_switches=4)
        got = D.classify_outlier({"combined": combined, "matching_gc_events": []})
        self.assertEqual(set(got["observations"]), {
            "major_page_fault_delta_observed", "minor_page_fault_delta_observed",
            "voluntary_context_switch_delta_observed", "involuntary_context_switch_delta_observed",
        })

    def test_selects_worst_final_v2_per_architecture_only(self):
        def row(variant, architecture, maximum, case_id):
            return {"variant": variant, "architecture": architecture, "case_id": case_id,
                    "timing": {"combined": {"wall": {"max_us": maximum}}}}
        document = {"results": [row("v1", "servo", 999, 1), row("v2", "servo", 10, 2),
                                row("v2", "servo", 20, 3), row("v2", "torque", 30, 4)]}
        got = D.select_worst_final_cases(document)
        self.assertEqual(got["servo"]["case_id"], 3)
        self.assertEqual(got["torque"]["case_id"], 4)


if __name__ == "__main__":
    unittest.main()
