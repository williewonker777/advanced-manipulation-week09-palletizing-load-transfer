"""Audit the primary alert length without resimulating or changing any alarm.

Consumes a completed, disclosed full-d4 replay. The later single-panel minimum
width rule is applied retrospectively here; this is not an independent test.
"""
import argparse
from bisect import bisect_left
import hashlib
import json
from pathlib import Path

ARMS = ('c3', 'headroom', 'identity')
GRID = tuple(tick*.001 for tick in range(6000))


def primary(arm, onset):
    s = arm['score']
    if not all(s['valid'].values()):
        raise ValueError('incomplete physical attribution')
    if not s['strict']:
        return None
    tick = bisect_left(GRID, onset) + s['delay_ms']
    spans = [(a,b) for a,b in arm['alarm_intervals'] if a == tick and 0 <= a < b <= len(GRID)]
    if len(spans) != 1:
        raise ValueError('attributed rise must start exactly one recorded interval')
    a,b = spans[0]
    return dict(start_tick=a,stop_exclusive=b,width_ms=b-a,delay_ms=s['delay_ms'])


def audit(j):
    if (j['schema'] != 'r31-disclosed-physical-replay.v1' or j['scope'] != 'full-d4'
            or j['episodes'] != 1155 or j['arms'] != list(ARMS)
            or [r['id'] for r in j['rows']] != list(range(385))):
        raise ValueError('complete R31 full-d4 replay required')
    rows = []
    for r in j['rows']:
        if r['input']['kind'] == 'collision':
            onset = r['input']['collision'][0]
            rows.append(dict(id=r['id'], arms={n:primary(r['arms'][n],onset) for n in ARMS}))
    if len(rows) != 180:
        raise ValueError('expected 180 genuine collision inputs')
    def compare(new,base):
        pairs = [(r['id'],r['arms'][new],r['arms'][base]) for r in rows if r['arms'][new] and r['arms'][base]]
        failed = [dict(id=i,new_width_ms=a['width_ms'],old_width_ms=b['width_ms'],
                       new_delay_ms=a['delay_ms'],old_delay_ms=b['delay_ms']) for i,a,b in pairs if b['width_ms']-a['width_ms']>5]
        return dict(comparable_strict=len(pairs), shortfalls=failed)
    comparisons = {a+'_vs_'+b:compare(a,b) for a,b in (('identity','c3'),('headroom','c3'),('identity','headroom'))}
    counts = {}
    for n in ARMS:
        widths = [r['arms'][n]['width_ms'] for r in rows if r['arms'][n]]
        counts[n] = dict(strict=len(widths),exactly_4ms=widths.count(4))
    return dict(schema='r31-historical-primary-width-audit.v1',
                classification='Post-hoc length audit of 180 known collision inputs, after the independent single-panel disclosure. No new scoring attempt, simulated input, or method change.',
                rule='First attributed alert width in 1 ms samples. A primary over 5 ms shorter than a retained comparator counts as a shortfall. Width is not physical contact duration.',
                counts=counts,comparisons=comparisons,rows=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() or a.out.is_symlink() or not a.report.is_file() or a.report.is_symlink():
        p.error('regular report and fresh output required')
    data = a.report.read_bytes()
    result = audit(json.loads(data))
    result['source_report_sha256'] = hashlib.sha256(data).hexdigest()
    with a.out.open('x') as f:
        json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(result['counts'], {k:len(v['shortfalls']) for k,v in result['comparisons'].items()})
