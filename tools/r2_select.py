"""Conservative R2 validation review; never trains or changes a controller."""
import json
from pathlib import Path
from tools.r2_study import specs, sha, PROTOCOL

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'results/v2_protocol'


def stability_name(spec, mode):
    f = spec['family']
    if f == 'C1':
        return f'C1-{mode}-f{spec["fraction"]:g}-k{spec["feedback_gain"]:g}'
    if f == 'C2':
        scale = 'small' if spec['error_scale'] < .001 else 'stroke'
        return f'C2-{mode}-{spec["geometry"]}-{scale}'
    if f == 'C3':
        ending = f'g{spec["gamma"]:g}' if spec['law'] == 'NRIC' else f'w{spec["bandwidth"]:g}'
        return f'C3-{mode}-{spec["law"]}-{ending}'
    if f == 'C4':
        return f'C4-{mode}-{spec["law"]}-r{spec["rate"]:g}-d{spec["damping_scale"]:g}'
    if f == 'C6':
        return f'C6-{mode}-{spec["variant"]}'
    return None


def review():
    baseline = json.loads((D/'R2_VALIDATION_R1.json').read_text())['results']
    refs = {(r['architecture'], r['case_id']): r for r in baseline}
    stable = json.loads((D/'R2_CANDIDATE_STABILITY_COMPACT.json').read_text())
    stability = {r['candidate']: r for r in stable['by_candidate']}
    output = []
    metrics = ('rms_mm', 'max_mm', 'torque_step_pick_nm', 'torque_step_place_nm', 'table_force_place_max_n')
    for spec in specs():
        if spec['family'] in ('base', 'v1', 'R1'):
            continue
        path = D/f'R2_VALIDATION_{spec["id"]}.json'
        if not path.exists():
            continue
        for mode in ('servo', 'torque'):
            rows = [r for r in json.loads(path.read_text())['results'] if r['architecture'] == mode]
            ratios = {key: [r[key]/refs[mode, r['case_id']][key] for r in rows] for key in metrics}
            mean_ratios = {k: sum(v)/len(v) for k, v in ratios.items()}
            g4 = sum(r['g4']['table_mean_over_mg'] >= .5 and r['g4']['table_min_over_mg'] >= .25
                     and r['g4']['place_peak_over_mg'] <= 1.1
                     and r['g4']['table_mean_over_mg']-refs[mode, r['case_id']]['g4']['table_mean_over_mg'] >= .25
                     for r in rows)
            local = stability.get(stability_name(spec, mode))
            false = sum(r['false_alarms'] for r in rows)
            p99 = max(r['timing']['combined']['wall']['p99_us'] for r in rows)
            gates = dict(no_false_alarm=false == 0,
                         no_saturation=all(r['saturation_pct'] == 0 for r in rows),
                         no_divergence=all(not r['diverged'] for r in rows),
                         cost=all(max(v) <= 1.05 for v in ratios.values()),
                         timing_p99=p99 < 800.,
                         local_radius=True if local is None else local['protocol_pass_cases'] == 6)
            if spec.get('diagnostic_only'):
                gates['not_an_adoptable_method'] = False
            if spec['family'] in ('C1', 'C2', 'C6'):
                gates['G4'] = g4 >= 3
            elif spec['family'] == 'C4':
                gates['tracking_gain'] = min(mean_ratios['rms_mm'], mean_ratios['max_mm']) <= .90
            # C3 requires additional perturbation benefit, C5 balanced detection
            # benefit, so normal validation alone cannot adopt either family.
            output.append(dict(id=spec['id'], family=spec['family'], architecture=mode,
                episodes=len(rows), false_rises=false, G4_pass_cases=g4,
                worst_ratio={k: max(v) for k, v in ratios.items()}, mean_ratio=mean_ratios,
                residual_joint_RMS_ratio=sum(r['normal_residual']['joint_RMS_Nm']/refs[mode, r['case_id']]['normal_residual']['joint_RMS_Nm'] for r in rows)/len(rows),
                combined_p99_us=p99, combined_max_us=max(r['timing']['combined']['wall']['max_us'] for r in rows),
                local_stability=local, gates=gates, normal_validation_eligible=all(gates.values()),
                remaining='development/perturbation and sweep/holdout/official gates; no adoption yet' if all(gates.values()) else 'reject: '+','.join(k for k,v in gates.items() if not v)))
    answer = dict(protocol_sha256=sha(PROTOCOL), official_used_for_selection=False,
                  holdout_opened=False, redesigns_used=0, results=output,
                  G4_unchanged=True, stability_scope=stable['uncertainty'])
    (D/'R2_VALIDATION_REVIEW.json').write_text(json.dumps(answer, indent=2)+'\n')
    return answer


if __name__ == '__main__':
    result = review()
    for row in result['results']:
        print(row['id'], row['architecture'], row['normal_validation_eligible'], row['remaining'])
