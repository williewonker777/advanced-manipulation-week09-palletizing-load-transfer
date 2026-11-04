"""Select the protocol-conforming repair using fresh nonofficial data only."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'results/v2_protocol'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    validation=json.loads((D/'S4_L3_VALIDATION.json').read_text())['results']
    development=json.loads((D/'S4_L3_DEVELOPMENT.json').read_text())['results']
    old=json.loads((D/'ADOPTION_L1_DEVELOPMENT.json').read_text())
    details=[]
    for mode in ['servo','torque']:
        a=[r for r in development if r['architecture']==mode and r['variant']=='v1' and r['kind']=='collision']
        b=[r for r in development if r['architecture']==mode and r['variant']=='L3' and r['kind']=='collision']
        details.append(dict(architecture=mode,v1_detected=sum(r['collision_outcome']=='detected' for r in a),
            learned_detected=sum(r['collision_outcome']=='detected' for r in b),
            v1_false_rises=sum(r['false_alarms'] for r in a),learned_false_rises=sum(r['false_alarms'] for r in b),
            lost_v1_detections=sum(x['collision_outcome']=='detected' and y['collision_outcome']!='detected' for x,y in zip(a,b,strict=True))))
    ref_fa={v:sum(r['false_alarms'] for r in validation if r['variant']==v) for v in ['v1','L3']}
    p99=max(r['timing']['combined']['wall']['p99_us'] for r in development if r['variant']=='L3')
    gain=sum(r['learned_detected']-r['v1_detected'] for r in details)
    adopt=ref_fa['L3']==0 and gain>=4 and p99<800 and all(r['lost_v1_detections']==0 and r['learned_false_rises']<=r['v1_false_rises'] for r in details)
    physical=['rms_mm','max_mm','torque_step_pick_nm','torque_step_place_nm','table_force_place_max_n','saturation_pct','diverged']
    physical_equal=True
    for source in [validation,development]:
        base={(r['architecture'],r['case_id']):r for r in source if r['variant']=='v1'}
        physical_equal &= all(all(r[k]==base[(r['architecture'],r['case_id'])][k] for k in physical) for r in source if r['variant']=='L3')
    answer=dict(control=old['control'],learning=dict(adopt=bool(adopt),candidate='L3 geometry-only protocol repair',
        validation_false_v1=ref_fa['v1'],validation_false_learned=ref_fa['L3'],development_gain=gain,
        development=details,largest_combined_p99_us=p99,controller_metrics_bit_exact=bool(physical_equal),
        weights_and_thresholds_unchanged=True,official_compatibility_pending=True),
        L1='rejected: official empty-payload false alarms exposed wrong gate semantics',
        L2='exploratory axis-aligned training envelope; not final preferred method',
        official_exposure='official set consumed for debugging; not independent efficacy evidence',
        holdout_opened=False,post_development_numeric_tuning=False)
    (D/'ADOPTION.json').write_text(json.dumps(answer,indent=2)+'\n')
    shutil.copyfile(ROOT/('v2_detector_geometry.py' if adopt else 'v1_detector.py'),ROOT/'my_detector.py')
    files=['my_servo.py','my_torque.py','my_detector.py','v1_servo.py','v1_torque.py','v1_detector.py',
           'v2_detector_candidate.py','v2_detector_guarded.py','v2_detector_geometry.py','models/v2_residual.json',
           'tools/v2_learned_core.py','tools/v2_study.py','tools/v2_resources.py','tools/sweep.py','tools/d4_robustness.py',
           'results/v2_protocol/PROTOCOL_GATE_FIX.json']
    freeze=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=sha(D/'PROTOCOL.json'),supplement_protocol_sha256=sha(D/'PROTOCOL_GATE_FIX.json'),
        source_files={n:sha(ROOT/n) for n in files},adopted_control='v1 unchanged; C rejected',
        adopted_detector=answer['learning']['candidate'] if adopt else 'v1 unchanged; learner rejected',
        adoption_sha256=sha(D/'ADOPTION.json'),holdout_results_opened=False,
        post_development_threshold_tuning=False,official_regression_pending=True)
    (D/'FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(answer,indent=2))


if __name__=='__main__':main()
