"""Pure casewise R4 gate decisions, including irreversible earlier rejection."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import r4_study as S

D = S.D
PHYSICAL_METRICS = ('rms_mm', 'max_mm', 'saturation_pct', 'diverged',
                    'torque_step_pick_nm', 'torque_step_place_nm',
                    'table_force_place_max_n', 'g4')


def detected(row):
    return row.get('collision_outcome') == 'detected'


def review(baseline, candidate, minimum_gain):
    old = {r['case_id']: r for r in baseline['results']}
    new = {r['case_id']: r for r in candidate['results']}
    if old.keys() != new.keys() or len(old) != len(baseline['results']) or len(new) != len(candidate['results']):
        raise ValueError('incomplete or duplicate paired cases')
    failures, phases = [], {}
    gains = losses = 0
    for key, r in old.items():
        c = new[key]
        identity = ('mass', 'com', 'kind', 'collision', 'pick_offset')
        if any(r.get(k) != c.get(k) for k in identity):
            raise ValueError('unpaired condition')
        why = []
        if any(r[k] != c[k] for k in PHYSICAL_METRICS):
            why.append('physical_metric_changed')
        if c['false_alarms'] > r['false_alarms']:
            why.append('extra_outside_event_false_alarm')
        if r.get('tool_contact_delay_ms') is not None and c.get('tool_contact_delay_ms') is None:
            why.append('tool_contact_lost')
        if r['kind'] == 'collision':
            gain, loss = not detected(r) and detected(c), detected(r) and not detected(c)
            gains += gain; losses += loss
            if loss: why.append('R1_detection_lost_or_masked')
            phase = r.get('phase', 'unspecified')
            bucket = phases.setdefault(phase, dict(cases=0, R1_detected=0, candidate_detected=0))
            bucket['cases'] += 1; bucket['R1_detected'] += int(detected(r)); bucket['candidate_detected'] += int(detected(c))
        if why:
            failures.append(dict(case_id=key, mass=r['mass'], kind=r['kind'], reasons=why,
                                 R1_FA=r['false_alarms'], candidate_FA=c['false_alarms']))
    if gains < minimum_gain:
        failures.append(dict(reasons=['insufficient_added_detections'], minimum=minimum_gain, actual=gains))
    return dict(passed=not failures, additional_detected=gains, lost_R1=losses,
                R1_detected=sum(detected(r) for r in old.values() if r['kind'] == 'collision'),
                candidate_detected=sum(detected(r) for r in new.values() if r['kind'] == 'collision'),
                reference_FA_R1=sum(r['false_alarms'] for r in old.values() if r['kind'] == 'reference'),
                reference_FA_candidate=sum(r['false_alarms'] for r in new.values() if r['kind'] == 'reference'),
                outside_event_FA_R1=sum(r['false_alarms'] for r in old.values()),
                outside_event_FA_candidate=sum(r['false_alarms'] for r in new.values()),
                phases=phases, failures=failures, every_case_retained=len(old))


def evaluate_stage(stage):
    protocol = S.protocol()
    cal = S.read(D / 'R4_CALIBRATION.json')
    parity = S.read(S.equivalence_path())
    key = dict(validation='validation_min_additional_detected',
               revalidation='validation_min_additional_detected',
               development='development_min_additional_detected',
               d4='D4_min_additional_detected', perturb=None)[stage]
    minimum = protocol['detector_gates'][key] if key else 0
    entries = []
    prior = (None if stage in ('validation', 'revalidation') else
             S.read(S.validation_review_path()) if stage == 'development' else
             S.read(D / f'R4_{dict(d4="DEVELOPMENT", perturb="D4")[stage]}_REVIEW.json'))
    for mode in ('servo', 'torque'):
        for spec in protocol['candidate_grid']:
            variant = spec['id']
            if stage in ('validation', 'revalidation'):
                base = S.read(S.validation_baseline_path(variant, mode))
                data = S.read(S.validation_document_path(variant, mode))
            else:
                base = S.read(D / f'R4_BASE_{stage}_{mode}.json')
                data = S.read(D / f'R4_{stage.upper()}_{variant}_{mode}.json')
            result = review(base, data, minimum)
            audit_rows = [r for r in cal['audit'] if r['architecture'] == mode and r['variant'] == variant]
            audit_pass = len(audit_rows) == 16 and all(r['passed'] for r in audit_rows)
            online_validation = S.read(S.validation_document_path(variant, mode))
            measured = online_validation['results']
            if len(measured) != 38 or not online_validation['all_physical_bits_equal']:
                raise RuntimeError('missing full-loop validation/equivalence rows')
            p99 = max(r['timing']['combined']['wall']['p99_us'] for r in measured)
            maximum = max(r['timing']['combined']['wall']['max_us'] for r in measured)
            timing_pass = p99 <= protocol['timing']['selection_combined_p99_us_max']
            predecessor = None if prior is None else next(r for r in prior['results'] if r['architecture'] == mode and r['variant'] == variant)
            eligible_before = True if predecessor is None else predecessor['adoption_eligible']
            store_free = S.literal_store_free(spec)
            eligible = (result['passed'] and audit_pass and timing_pass and parity['passed']
                        and eligible_before and store_free)
            entries.append(dict(architecture=mode, variant=variant, family=spec['family'],
                review=result, normal_audit_pass=audit_pass, full_loop_parity_pass=parity['passed'],
                timing_p99_us=p99, timing_max_us=maximum, timing_pass=timing_pass,
                literal_store_free=store_free,
                adoption_eligible=eligible, earlier_disqualification=not eligible_before,
                permanently_disqualified=not eligible))
    if stage in ('validation', 'revalidation'):
        source_paths = ([S.validation_baseline_path(spec['id'], mode)
                         for mode in ('servo', 'torque')
                         for spec in protocol['candidate_grid']]
                        + [S.validation_document_path(spec['id'], mode)
                           for mode in ('servo', 'torque')
                           for spec in protocol['candidate_grid']])
    else:
        source_paths = [p for p in D.glob(f'R4_{stage.upper()}_*.json')
                        if not p.name.endswith('_REVIEW.json')]
    output = dict(schema='R4_STAGE_REVIEW.v1', stage=stage, protocol_sha256=S.sha(S.P), results=entries,
                  diagnostics_cannot_revive=True,
                  independent_efficacy=stage in ('validation', 'revalidation', 'development'),
                  source_files={p.name: S.sha(p) for p in source_paths})
    S.write_new(D / f'R4_{stage.upper()}_REVIEW.json', output)
    return output


def redesign_decision():
    """Contemporaneous nonuse record after validation, before development."""
    validation = S.read(D / 'R4_VALIDATION_REVIEW.json')
    if (D / 'R4_BASE_development_servo.json').exists():
        raise RuntimeError('redesign decision must precede development')
    record = dict(schema='R4_REDESIGN_DECISION.v1', validation_sha256=S.sha(D / 'R4_VALIDATION_REVIEW.json'),
                  redesigns_used=0, decisions=[])
    for family in ('W40', 'CUSUM_ACTIVE', 'CUSUM_PHASE'):
        rows = [r for r in validation['results'] if r['family'] == family]
        # Reasons are generated from data already available at this exact
        # decision point, not reconstructed after D4/holdout/official exposure.
        eligible = sum(r['adoption_eligible'] for r in rows)
        reason = ('Validation has eligible configuration(s); retain the preregistered design without a blind second grid.'
                  if eligible else
                  'Validation/normal audit does not qualify this family. No implementation failure or causal structural correction has been identified; collision-driven threshold retuning is forbidden. Retain negative configurations for diagnostic later stages only.')
        record['decisions'].append(dict(family=family, used=False, reason=reason, eligible_configurations=eligible))
    S.write_new(D / 'R4_REDESIGN_DECISION.json', record)
    return record


def select():
    reviews = {'VALIDATION': S.read(S.validation_review_path())}
    reviews.update({stage: S.read(D / f'R4_{stage}_REVIEW.json')
                    for stage in ('DEVELOPMENT', 'D4', 'PERTURB')})
    entries, chosen = {}, {}
    for mode in ('servo', 'torque'):
        eligible = [r for r in reviews['PERTURB']['results'] if r['architecture'] == mode and r['adoption_eligible']]
        def entry(stage, variant):
            return next(r for r in reviews[stage]['results'] if r['architecture'] == mode and r['variant'] == variant)
        def rank(row):
            variant = row['variant']
            order = [s['id'] for s in S.protocol()['candidate_grid']].index(variant)
            return (-entry('DEVELOPMENT', variant)['review']['additional_detected'],
                    -entry('D4', variant)['review']['additional_detected'], row['timing_p99_us'], order)
        best = min(eligible, key=rank)['variant'] if eligible else 'R1'
        chosen[mode] = best
        entries[mode] = dict(chosen=best, eligible=[r['variant'] for r in eligible],
                            all_stages_required=True, no_later_promotion=True)
    result = dict(schema='R4_PREFREEZE_DECISION.v1', chosen=chosen, architectures=entries,
                  source_evidence_sha256={
                      **{'VALIDATION': S.sha(S.validation_review_path())},
                      **{stage: S.sha(D / f'R4_{stage}_REVIEW.json')
                         for stage in ('DEVELOPMENT', 'D4', 'PERTURB')}},
                  holdout_reseal_sha256=S.sha(D / 'PROTOCOL_R4_HOLDOUT_RESEAL.json'),
                  protocol_sha256=S.sha(S.P),
                  redesign_protocol_sha256=S.sha(S.REDESIGN_PROTOCOL),
                  redesign_decision_sha256=S.sha(S.redesign_decision_path()),
                  new_video_pairs_required=any(v != 'R1' for v in chosen.values()))
    S.write_new(D / 'R4_PREFREEZE_DECISION.json', result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('validation', 'revalidation', 'development', 'd4', 'perturb', 'redesign', 'select'))
    args = ap.parse_args()
    if args.action == 'redesign': result = redesign_decision()
    elif args.action == 'select': result = select()
    else: result = evaluate_stage(args.action)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
