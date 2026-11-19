"""Re-run the same fixed-controller impacts on the same stock plant, with no detector-derived forcing."""
import argparse
import hashlib
import json
from pathlib import Path
from types import FunctionType

from experiments.r26_holdout import evaluator as lab
from experiments.r26_yaw_acquisition.candidate import Detector as C3
from experiments.r27_disclosed import contrasts
from experiments.r27_portable import runner as signed
from experiments.r29_disjoint_events.candidate import Detector as Composed
from experiments.r29_loaded_rising_certificate.candidate import Detector as Rising
from experiments.r29_native_event_link import evaluator as paired
from experiments.r30_source_association.candidate import Detector as Geometry
from experiments.r30_yaw_headroom.candidate import Detector as Headroom

HERE = Path(__file__).resolve().parent
ARMS = {'c3':C3, 'composed':Composed, 'rising':Rising, 'geometry':Geometry, 'headroom':Headroom}
EPISODES = {'signed-stock':50, 'paired-directions':60, 'place':25}


def digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def bind(fn, **overrides):
    # Supply only our fixed arm table. Do not mutate the published evaluator or module state.
    return FunctionType(fn.__code__, {**fn.__globals__, **overrides}, fn.__name__, fn.__defaults__, fn.__closure__)


def run_signed():
    plan = json.loads(lab.PLAN.read_bytes())
    ref = json.loads((signed.HERE/'reference.json').read_bytes())
    if ref['r26_plan_sha256'] != hashlib.sha256(lab.PLAN.read_bytes()).hexdigest():
        raise ValueError('historical case plan changed')
    ids = ('s26_long_carry_signed','s26_long_place_lateral')
    cases = {c['id']:c for c in plan['single']}
    table = {**ARMS, contrasts.BASE:C3}
    family = bind(signed.family, resolve=table.__getitem__)
    by_arm = {name:[family((cases[i],plan['variants']['stock'],contrasts.BASE if name=='c3' else name,
                            'directions',ref['records'][i]['stock'],plan)) for i in ids] for name in ARMS}
    summary = {}
    for name, rows in by_arm.items():
        compare = contrasts.compare(rows,by_arm['c3'])
        # compare also enforces identical physical traces and all sample-exact pre-force prefixes.
        summary[name] = {r['case']:{k:v['event']['delays_ms'] for k,v in r['forced'].items()} for r in rows}
        summary[name]['regressions_vs_c3'] = compare['regressions']
    return by_arm, summary, sum(r['episodes'] for rows in by_arm.values() for r in rows)


def run_paired(scope):
    plan = json.loads(paired.PLAN.read_bytes())
    ref = json.loads((HERE/'anchors.json').read_bytes())
    if scope=='place':
        rows = [bind(paired.p26_case, ARMS=ARMS)(plan)]
    else:
        rows = bind(paired.stress_cases, ARMS=ARMS)(plan)
    summary = {}
    for r in rows:
        if digest(r['records']['c3']) != ref['original_c3_record_sha256'][r['id']]:
            raise ValueError('original physical-control/event record differs')
        for name, rec in r['records'].items():
            if not (rec['physical_exact_vs_c3'] and all(p['passed'] for p in rec['prefixes'].values()) and
                    all(tw['force_exact']['passed'] for tw in rec['twins'].values())):
                raise ValueError('physical, force or causal-prefix mismatch')
            summary.setdefault(name,{})[r['id']] = {slot:v['delays_ms'] for slot,v in rec['events'].items()}
    return rows, summary, sum(r['episodes'] for r in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scope',choices=EPISODES); p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    if a.out.exists() or a.out.is_symlink():
        p.error('fresh output required')
    runtime = lab.require_runtime()
    rows, delays, episodes = run_signed() if a.scope=='signed-stock' else run_paired(a.scope)
    if episodes != EPISODES[a.scope]:
        raise ValueError('incomplete physical comparison')
    obj = dict(schema='r30-disclosed-event-replay.v1', scope=a.scope, runtime=runtime,
               default_changed=False, course_original=False, controller='my_torque', arms=list(ARMS),
               episodes=episodes, delays_ms=delays, rows=rows)
    with a.out.open('x') as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')


if __name__=='__main__':
    main()
