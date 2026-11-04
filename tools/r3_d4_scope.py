"""Distinguish full D4 saturation from T4 reference-only historical metrics."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'results/v2_protocol'
FIELDS = ('mass', 'com', 'kind', 'collision', 'pick_offset', 'rms_mm', 'max_mm',
          'saturation_pct', 'diverged', 'torque_step_pick_nm', 'torque_step_place_nm',
          'table_force_place_max_n', 'false_alarms', 'false_alarms_pick',
          'false_alarms_place', 'false_alarms_other', 'detect_delay_ms',
          'collision_outcome', 'alarm_before_collision', 'tool_contact_delay_ms',
          'tool_contact_peak_n')


def scope(rows):
    groups = {'all': rows, 'reference': [row for row in rows if row['kind'] == 'reference'],
              'collision': [row for row in rows if row['kind'] == 'collision'],
              'offset': [row for row in rows if row['kind'] == 'offset']}
    return {key: dict(rows=len(values), saturation_episodes=sum(
                row['saturation_pct'] > 0 for row in values),
                max_saturation_pct=max((row['saturation_pct'] for row in values), default=0),
                reference_peak_over_mg=(max((row['table_force_place_max_n'] / (9.81 * row['mass'])
                                            for row in values if row['mass'] > 0), default=0)
                                        if key == 'reference' else None))
            for key, values in groups.items()}


def main():
    target = D / 'R3_D4_SCOPE_PARITY.json'
    if target.exists():
        raise RuntimeError('do not overwrite scope evidence')
    old_path = D / 'S5_D4.json'
    old = json.loads(old_path.read_text())['results']
    output = dict(schema='R3_D4_SCOPE_PARITY.v1',
                  warning='T4 historical saturation2 is reference-only189; frozen D4 ceiling2/.15 applied conservatively to all385, not relaxed after data',
                  scope='reported non-timing matched fields only; old raw arrays unavailable; all R3 D4 cases freshly run',
                  artifacts={old_path.name: hashlib.sha256(old_path.read_bytes()).hexdigest()}, architectures={})
    for mode, challenger in [('servo', 'A_handoff_long'), ('torque', 'B_mass')]:
        paths = [D / f'R3_D4_{variant}_{mode}.json' for variant in ('R1', challenger)]
        current, candidate = [json.loads(path.read_text())['results'] for path in paths]
        previous = {row['case_id']: row for row in old
                    if row['variant'] == 'v2' and row['architecture'] == mode}
        if len(previous) != 385 or len(current) != 385:
            raise RuntimeError('D4 comparison is incomplete')
        mismatches = [dict(case_id=row['case_id'], fields=[field for field in FIELDS
                      if row.get(field) != previous[row['case_id']].get(field)]) for row in current]
        mismatches = [row for row in mismatches if row['fields']]
        output['architectures'][mode] = dict(R1=scope(current), candidate=scope(candidate),
            historical_R1_reported_field_parity=not mismatches, mismatches=mismatches)
        output['artifacts'].update({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    target.write_text(json.dumps(output, indent=2) + '\n')
    print({mode: value['historical_R1_reported_field_parity']
           for mode, value in output['architectures'].items()})


if __name__ == '__main__':
    main()
