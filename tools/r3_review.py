"""Read-only R3 stage aggregation; no simulator runs or parameter changes."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import r3_select as S

D = ROOT / 'results/v2_protocol'


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    return dict(
        rows=len(rows),
        counts={name: sum(row.get('collision_outcome') == name for row in rows)
                for name in ('detected', 'missed', 'alarm_before_collision', 'indistinguishable')},
        reference_false_alarms=sum(row['false_alarms'] for row in rows
                                   if row['kind'] == 'reference'),
        collision_false_alarms=sum(row['false_alarms'] for row in rows
                                   if row['kind'] == 'collision'),
        offset_false_alarms=sum(row['false_alarms'] for row in rows
                                if row['kind'] == 'offset'),
        saturation_episodes=sum(row['saturation_pct'] > 0 for row in rows),
        max_saturation_pct=max(row['saturation_pct'] for row in rows),
        diverged=sum(bool(row['diverged']) for row in rows),
        tool_contact_misses=sum(row.get('tool_contact_peak_n', 0) > 1
                                and row.get('tool_contact_delay_ms') is None
                                for row in rows if row['kind'] == 'offset'),
    )


def stage_review(stage):
    selection = load(D / 'R3_DIAGNOSTIC_SELECTION.json')['chosen_specs']
    result = dict(schema='R3_STAGE_REVIEW.v1', stage=stage, official=False,
                  diagnostic_only=True, no_revival=True, artifacts={}, architectures={})
    for spec in selection:
        mode, name = spec['architecture'], spec['id']
        paths = [D / f'R3_{stage.upper()}_{item}_{mode}.json' for item in ('R1', name)]
        base, candidate = map(load, paths)
        if stage == 'development':
            review = S.review_development(base, candidate)
        elif stage == 'd4':
            review = S.review_d4(base, candidate, mode)
        elif stage == 'perturb':
            review = S.review_perturbations(base, candidate, mode)
        else:
            raise ValueError('unknown stage')
        result['artifacts'].update({p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
        result['architectures'][mode] = dict(
            challenger=name, review=review,
            R1_summary=summarize(base['results']),
            candidate_summary=summarize(candidate['results']),
        )
    return result


def timing_review():
    groups = {}
    source_hashes = {}
    for pattern in ('R3_DIAGNOSTIC_R1_*.json', 'R3_VALIDATION_*.json', 'R3_DEVELOPMENT_*.json',
                    'R3_D4_*.json', 'R3_PERTURB_*.json', 'R3_HOLDOUT.json'):
        for path in sorted(D.glob(pattern)):
            document = load(path)
            rows = document.get('results')
            if not isinstance(rows, list) or not rows:
                continue
            source_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
            for row in rows:
                key = (row['architecture'], row['variant'])
                group = groups.setdefault(key, [])
                group.append(row)
    output = []
    for (mode, variant), rows in sorted(groups.items()):
        maximum_row = max(rows, key=lambda row: row['timing']['combined']['wall']['max_us'])
        outliers = [dict(stage=row['stage'], case_id=row['case_id'], **item)
                    for row in rows for item in row['timing']['outliers']]
        output.append(dict(
            architecture=mode, variant=variant, rows=len(rows),
            cycles=sum(row['timing']['combined']['wall']['samples'] for row in rows),
            largest_row_p99_us=max(row['timing']['combined']['wall']['p99_us'] for row in rows),
            maximum_wall_us=maximum_row['timing']['combined']['wall']['max_us'],
            maximum_thread_cpu_us=max(row['timing']['combined']['thread_cpu']['max_us'] for row in rows),
            maximum_case=dict(stage=maximum_row['stage'], case_id=maximum_row['case_id']),
            overrun_cycles=len(outliers), all_outliers=outliers,
            maximum_wall_outlier=(max(outliers, key=lambda item: item['combined']['wall_ns'])
                                  if outliers else None),
            global_GC_changes=False, serial_owned_simulations=1,
        ))
    replay_path = D / 'R3_REPLAY_TIMING.json'
    replay = None
    if replay_path.is_file():
        source_hashes[replay_path.relative_to(ROOT).as_posix()] = sha(replay_path)
        replay = load(replay_path)
    return dict(schema='R3_TIMING_REVIEW.v1', source_sha256=source_hashes, results=output,
                equivalent_action_replay=replay,
                p99_selection_limit_us=800, maximum_goal_us=1000,
                samples_are_per_case_not_pooled_p99=True, all_outliers_retained=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('development', 'd4', 'perturb', 'timing'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('do not overwrite a recorded stage review')
    result = timing_review() if args.stage == 'timing' else stage_review(args.stage)
    S._atomic_write(args.output, result)
    if args.stage != 'timing':
        print({mode: row['review']['status'] for mode, row in result['architectures'].items()})


if __name__ == '__main__':
    main()
