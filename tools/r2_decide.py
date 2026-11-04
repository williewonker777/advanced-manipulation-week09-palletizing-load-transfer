"""Review the frozen normal/development evidence, without tuning or deployment."""
import json
from pathlib import Path
from tools.r2_study import sha, PROTOCOL
from tools.r2_select import review

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'results/v2_protocol'
IDS = ('R1', 'C5_prne_only', 'C5_prne_plus_ridge', 'C5_ridge_only_matched')


def decide():
    validation = review()
    documents = {name: json.loads((D/f'R2_DEVELOPMENT_{name}.json').read_text()) for name in IDS}
    groups = {name: {(r['architecture'], r['case_id']): r for r in doc['results']}
              for name, doc in documents.items()}
    results = []
    for name in IDS[1:]:
        modes = []
        for mode in ('servo', 'torque'):
            ref = [r for r in groups['R1'].values() if r['architecture'] == mode and r['kind'] == 'collision']
            test = [groups[name][mode, r['case_id']] for r in ref]
            matched = [groups['C5_ridge_only_matched'][mode, r['case_id']] for r in ref]
            modes.append(dict(architecture=mode, collisions=len(ref),
                R1_detected=sum(r['collision_outcome'] == 'detected' for r in ref),
                detected=sum(r['collision_outcome'] == 'detected' for r in test),
                missed=sum(r['collision_outcome'] == 'missed' for r in test),
                lost_R1=sum(a['collision_outcome'] == 'detected' and b['collision_outcome'] != 'detected' for a,b in zip(ref,test)),
                false_rises_R1=sum(r['false_alarms'] for r in ref), false_rises=sum(r['false_alarms'] for r in test),
                matched_ridge_detected=sum(r['collision_outcome'] == 'detected' for r in matched),
                p99_us=max(r['timing']['combined']['wall']['p99_us'] for r in test)))
        gain = sum(r['detected']-r['R1_detected'] for r in modes)
        no_loss = all(r['lost_R1'] == 0 and r['false_rises'] <= r['false_rises_R1'] for r in modes)
        eligible_normal = all(r['normal_validation_eligible'] for r in validation['results'] if r['id'] == name)
        floor = max((r['residual_joint_RMS_ratio'] for r in validation['results'] if r['id'] == name), default=1.)
        gates = dict(normal=eligible_normal, no_lost_or_extra_false=no_loss,
                     benefit=(floor <= .9 or gain >= 4),
                     p99=all(r['p99_us'] < 800 for r in modes),
                     not_diagnostic=name != 'C5_ridge_only_matched')
        results.append(dict(id=name, development=modes, detected_gain=gain,
                            normal_RMS_worst_ratio=floor, gates=gates, eligible=all(gates.values())))
    return dict(schema='R2_DEVELOPMENT_REVIEW.v1', protocol_sha256=sha(PROTOCOL),
                official_used_for_selection=False, holdout_opened=False,
                control_eligible=[r for r in validation['results'] if r['family'] != 'C5' and r['normal_validation_eligible']],
                observer_results=results, no_threshold_or_gain_tuning_after_collision=True,
                matched_ablation_note='Matched new ridge is attribution-only, not an additional promotable fallback')


if __name__ == '__main__':
    answer = decide()
    (D/'R2_DEVELOPMENT_REVIEW.json').write_text(json.dumps(answer, indent=2)+'\n')
    print(json.dumps(answer, indent=2))
