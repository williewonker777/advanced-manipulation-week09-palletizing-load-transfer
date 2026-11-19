"""Score an event's end as well as its start. No labels are available to the detector."""
from __future__ import annotations

import numpy as np
from experiments.r18_paired_impulses import scoring
from experiments.r18_single_pass.paired_replay import scalar_norm_check
from tools.r14_paired_events import match_false_events, preserve_tool
from tools.r5_metrics import event_result

SLOTS = ('isolated_first', 'isolated_second', 'both_first', 'both_second')
TWINS = scoring.TWINS


def intervals(bits):
    """Integer sample intervals [start, end); do not lose an alarm still on at the end."""
    bits = np.asarray(bits)
    if bits.ndim != 1 or bits.dtype != bool or not len(bits):
        raise ValueError('nonempty one-dimensional alarm bits required')
    edges = np.diff(np.r_[False, bits, False].astype(np.int8))
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def off_samples(trace):
    first = np.flatnonzero(np.any(trace['f_ext_xyz_native'] != 0., axis=1))
    second = np.flatnonzero(np.any(trace['f_ext_xyz_second'] != 0., axis=1))
    if not len(first) or not len(second) or first[-1] >= second[0]:
        raise ValueError('two actual disjoint force components required')
    gap = np.asarray(trace['f_ext_xyz_committed'])[first[-1]+1:second[0]]
    if not len(gap) or np.any(gap):
        raise ValueError('missing applied zero-force gap')
    return len(gap)


def characterize(case, traces, quiet_ms, window_ms):
    raw = scoring.assess(case, traces, quiet_ms, window_ms)
    t = traces['both']['t']
    flags = {}
    for slot, twin, other, idx in zip(SLOTS, ('first','second','both','both'),
                                      ('none','none','second','first'), (0,1,0,1)):
        onset = float(case['impacts'][idx][0])
        k = int(np.searchsorted(t, onset))
        # Use the actual grid, not floating-point addition at the endpoint.
        end = k + int(window_ms)
        if k < quiet_ms or end >= len(t):
            raise ValueError('a complete pre-onset and clearance window is required')
        bits = np.asarray(traces[twin]['detect'], bool)
        if twin == 'both' and idx == 0:
            end = min(end, int(np.searchsorted(t, case['impacts'][1][0])))
        pre_quiet = not bool(bits[k-int(quiet_ms):k].any())
        valid = [int(round(x)) for x in raw['absolute_delays_ms'][slot]]
        flags[slot] = {
            'raw_delays_ms': [int(round((x-onset)*1000)) for x in raw['raw_event_times_s'][slot]],
            'attributable_delays_ms': valid, 'credited_delays_ms': valid if pre_quiet else [],
            'high_at_onset': bool(bits[k-1]), 'quiet_before_onset': pre_quiet,
            'no_target_twin_high_at_onset': bool(traces[other]['detect'][k-1]),
            'active_samples_in_slot': int(np.count_nonzero(bits[k:end])),
        }
    levels = {}
    for kind, trace in traces.items():
        det = np.asarray(trace['detect'], bool)
        last = None
        if kind in ('first','second'):
            onset = case['impacts'][int(kind == 'second')][0]
            last = bool(det[np.searchsorted(t, onset)+int(window_ms)])
        levels[kind] = {'on_intervals_ms':intervals(det), 'active_samples':int(det.sum()),
                        'high_at_end':bool(det[-1]), 'high_at_isolated_window_end':last}
    scalar = {tw:scalar_norm_check(traces[tw]) for tw in TWINS}
    certified = all(scalar[tw]['stock_scalar_bit_exact'] and all(
        v for k,v in raw['profiles'][tw]['gates'].items() if k!='stock_native_magnitude_exact') for tw in TWINS)
    return {'r18_gates':raw['gates'], 'r18_delays_ms':raw['absolute_delays_ms'],
            'slots':flags, 'levels':levels, 'false_s':raw['false_times_s'], 'scalar_forces':scalar,
            'signed_and_scalar_exact':bool(certified), 'actual_quiescent_force_samples':off_samples(traces['both']),
            'tool':{tw:event_result(traces[tw]['t'],traces[tw]['detect'],{},traces[tw]['f_tool'])
                    for tw in TWINS}}, raw


def compare(candidate, ref):
    keep, quiet = {}, {}
    for k in SLOTS:
        c, r = candidate['slots'][k], ref['slots'][k]
        a, b = c['attributable_delays_ms'], r['attributable_delays_ms']
        keep[k] = not b or (bool(a) and a[0] <= b[0]+5)
        quiet[k] = not r['quiet_before_onset'] or c['quiet_before_onset']
    match = {tw:match_false_events(candidate['false_s'][tw],ref['false_s'][tw])['passed'] for tw in TWINS}
    return {'keep_first_genuine':keep, 'preserve_quiet':quiet,
            'no_added_false':match,
            'preserve_tool':{tw:preserve_tool(candidate['tool'][tw],ref['tool'][tw]) for tw in TWINS},
            'no_new_stuck_tail':{tw: all(not candidate['levels'][tw][key] or ref['levels'][tw][key]
                    for key in ('high_at_end','high_at_isolated_window_end')) for tw in TWINS}}


def all_comparisons(pairs):
    return all(val for cmp in pairs for criterion in cmp.values() for val in criterion.values())


def one_per_pulse(arm):
    return all(len(a['credited_delays_ms']) == len(a['raw_delays_ms']) == 1 for a in arm['slots'].values())
