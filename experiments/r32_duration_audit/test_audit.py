from experiments.r32_duration_audit.audit import (
    _aggregate,
    c3_width_floor,
    compare,
    occupied,
    primary_width,
    whole_high,
)


def score(*, strict=True, delay=10, duplicate=0, false=(), masked=False):
    return {
        "strict": strict,
        "delay_ms": delay if strict else None,
        "duplicate": duplicate,
        "raw_duplicate": duplicate,
        "false": list(false),
        "delayed_false": [],
        "masked": masked,
        "stuck": False,
        "valid": {"physics": True, "prefix": True},
    }


def row(*, strict=True, delay=10, spans=None, width=10, duplicate=0, false=(), masked=False):
    spans = [[1010, 1020]] if spans is None else spans
    return {
        "score": score(strict=strict, delay=delay, duplicate=duplicate, false=false, masked=masked),
        "alarm_intervals": spans,
        "whole_trace_high_samples": whole_high(spans),
        "within_320ms_high_samples": whole_high(spans),
        "primary_width_ms": width if strict else None,
        "external_nonzero_samples": 120,
        "overlap_external_samples": whole_high(spans),
        "post_force_high_within_320ms": 0,
        "post_force_high_whole_trace": 0,
    }


def test_interval_accounting_and_primary_width():
    assert whole_high([[5, 9], [11, 15]]) == 8
    assert occupied([[5, 9], [11, 15]], 7, 12) == 4
    arm = {"score": score(delay=10), "alarm_intervals": [[1010, 1024]]}
    assert primary_width(arm, 1.0) == 14


def test_pairwise_guards_report_exact_regressions():
    baseline = [row(), row()]
    candidate = [
        row(delay=16, spans=[[1016, 1032]], width=16, duplicate=1, false=(3000,), masked=True),
        row(strict=False, spans=[]),
    ]
    result = compare(candidate, baseline)
    assert result["strict_losses"] == [1]
    assert result["duplicate_regressions"] == [0]
    assert result["new_false_rises"] == [{"id": 0, "indices": [3000]}]
    assert result["mask_regressions"] == [0]
    assert result["latency_plus_5ms_failures"] == [0, 1]
    assert result["within_320ms_high_plus_5_failures"] == [0]


def test_width_floor_and_aggregate_distinguish_force_support():
    c3 = [row(width=20), row(width=20)]
    candidate = [row(width=14), row(strict=False, spans=[])]
    floor = c3_width_floor(candidate, c3)
    assert floor["shortfall_rows"] == [{"id": 0, "candidate_width_ms": 14, "c3_width_ms": 20}]
    assert floor["missing_candidate_strict_width_ids"] == [1]
    aggregate = _aggregate([row(spans=[[1010, 1020]])])
    assert aggregate["support"]["whole_trace_high_samples"] == 10
    assert aggregate["support"]["externally_applied_nonzero_samples"] == 120
    assert aggregate["support"]["external_sample_coverage"] == 10 / 120

