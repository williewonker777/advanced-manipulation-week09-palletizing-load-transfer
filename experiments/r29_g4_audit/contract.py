"""Separate passive controller equivalence from alarm quality. Fail closed on unknown metrics."""
import math

PHYSICAL = frozenset(('status','cost','saturation_pct','registration','hybrid_active','support','last20_geometry'))
ALARM = frozenset(('false_alarms','false_alarm_times','tool_delay_ms'))


def physical_view(record):
    """New compact fields must be classified deliberately, not silently discarded."""
    g4 = record['g4']
    if set(g4) != PHYSICAL | ALARM:
        raise ValueError('missing/unclassified G4 fields')
    return (record['physical_sha256'], {k:g4[k] for k in PHYSICAL},
            record['ending_table_mean_N'], record['ending_table_max_N'])


def tick_times(record):
    ticks, times = record['rises'], record['rise_times_s']
    if len(ticks) != len(times) or any(type(k) is not int or not 0 <= k < 6000 for k in ticks):
        raise ValueError('complete native 1 kHz edges required')
    if ticks != sorted(set(ticks)) or any(not math.isfinite(t) or abs(t-k/1000) > 1e-8 for k,t in zip(ticks,times)):
        raise ValueError('unordered, nonfinite, or unsynchronised R29 edge')
    false = set()
    for t in record['false_times_s']:
        if not math.isfinite(t): raise ValueError('invalid false rise')
        k = round(t*1000)
        if abs(k/1000-t) > 1e-8 or k not in ticks or k in false:
            raise ValueError('false-time annotation must name exactly one measured rise')
        false.add(k)
    return set(ticks)-false, false


def unmatched(before, after, tolerance):
    """Order-preserving, one-to-one maximum-cardinality matching on a common time line."""
    a, b = sorted(before), sorted(after)
    i = j = 0
    lost, added = [], []
    while i < len(a) and j < len(b):
        if a[i] < b[j]-tolerance: lost.append(a[i]); i += 1
        elif b[j] < a[i]-tolerance: added.append(b[j]); j += 1
        else: i += 1; j += 1
    return lost+a[i:], added+b[j:]


def no_force_edges(reference, candidate, tolerance=2):
    """Tool and unclassified edges are protected. Only an explicitly false edge may disappear."""
    if type(tolerance) is not int or tolerance < 0: raise ValueError('nonnegative integer tick tolerance')
    old, old_false = tick_times(reference); new, new_false = tick_times(candidate)
    lost, added = unmatched(old, new, tolerance)
    removed_false, added_false = unmatched(old_false, new_false, tolerance)
    return dict(passed=not (lost or added or added_false), lost_protected=lost, added_unclassified=added,
                removed_certified_false=removed_false, added_false=added_false)
