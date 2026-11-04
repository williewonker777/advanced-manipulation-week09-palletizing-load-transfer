"""Apply preregistered adoption rules, never tune weights or thresholds."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def select(control, development):
    rows = control['results']
    c_records = []
    for architecture in ('servo','torque'):
        base = [r for r in rows if r['variant']=='v1' and r['architecture']==architecture]
        candidate = [r for r in rows if r['variant']=='C' and r['architecture']==architecture]
        conditions = []
        for a,b in zip(base,candidate,strict=True):
            g = b['g4']
            improvement = g['table_mean_over_mg']-a['g4']['table_mean_over_mg']
            conditions.append(dict(mass=b['mass'],com=b['com'],
                g4_pass=g['table_mean_over_mg']>=.5 and g['table_min_over_mg']>=.25
                        and improvement>=.25 and g['place_peak_over_mg']<=1.1,
                mean_increase_over_mg=improvement,
                tcp_ratio=b['max_mm']/a['max_mm'],
                place_step_ratio=b['torque_step_place_nm']/a['torque_step_place_nm']))
        c_records.append(dict(architecture=architecture,conditions=conditions,
                              g4_passes=sum(c['g4_pass'] for c in conditions),adopt=False))
    dev = development['results']
    details=[]
    for architecture in ('servo','torque'):
        base=[r for r in dev if r['variant']=='v1' and r['architecture']==architecture and r['kind']=='collision']
        learned=[r for r in dev if r['variant']=='L' and r['architecture']==architecture and r['kind']=='collision']
        details.append(dict(architecture=architecture,
            v1_detected=sum(r['collision_outcome']=='detected' for r in base),
            learned_detected=sum(r['collision_outcome']=='detected' for r in learned),
            lost_v1_detections=sum(a['collision_outcome']=='detected' and b['collision_outcome']!='detected'
                                   for a,b in zip(base,learned,strict=True)),
            v1_false_rises=sum(r['false_alarms'] for r in base),
            learned_false_rises=sum(r['false_alarms'] for r in learned)))
    val_base=sum(r['false_alarms'] for r in rows if r['variant']=='v1')
    val_learned=sum(r['false_alarms'] for r in rows if r['variant']=='L')
    p99=max(r['timing']['combined']['wall']['p99_us'] for r in dev if r['variant']=='L')
    gain=sum(d['learned_detected']-d['v1_detected'] for d in details)
    adopt=(gain>=4 and val_learned==0 and val_learned<=val_base and
           all(d['lost_v1_detections']==0 and d['learned_false_rises']<=d['v1_false_rises'] for d in details)
           and p99<800.)
    return dict(control=dict(adopt=False,rows=c_records,
                reason='G4 validation failed; fixed .10s return incompatible with all registered bounds over the full state box; no retuning'),
                learning=dict(adopt=bool(adopt),validation_false_v1=val_base,
                    validation_false_learned=val_learned,development_gain=gain,
                    development=details,largest_combined_p99_us=p99,
                    rule='>=4 extra, no losses or false-rise increases, validation false0, p99<800us'),
                holdout_opened=False,thresholds_changed_after_development=False)


def main():
    control_path=ROOT/'results/v2_protocol/S4_CONTROL.json'
    dev_path=ROOT/'results/v2_protocol/S4_DEVELOPMENT.json'
    answer=select(json.loads(control_path.read_text()),json.loads(dev_path.read_text()))
    answer['evidence_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [control_path,dev_path]}
    (ROOT/'results/v2_protocol/ADOPTION.json').write_text(json.dumps(answer,indent=2)+'\n')
    print(json.dumps(answer,indent=2))


if __name__=='__main__':
    main()
