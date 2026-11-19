"""Attribute the shared 3.922 s false alarm after the R24 independent test. No detector tuning."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.r23_detached.free import Detector as Incumbent
from experiments.r24_composed.detector import Detector as Joint
from experiments.r24_holdout import evaluator as lab


ORIGINAL_RESULT = 'dfce0da5e2ded2dd0dfb3e02a168aca48f661f715df0647fb1b46c1c87872c08'
CASE, MECHANICS = 's24_detached_offaxis_b', 'actuator_delay_gain'


def probe(base):
    class Observed(base):
        def reset(self):
            super().reset()
            self.evidence = {}

        def update(self, t, th, thd, tau_m, tau_s, grip):
            result = super().update(t, th, thd, tau_m, tau_s, grip)
            tick = round(t * 1000)
            if 3900 <= tick <= 3950:
                s = self._shared
                paths = {'r10': self.r8_base_detect, 'directional': s.detected,
                         'normal_model': bool(self.normal_model and self.normal_model.detected),
                         'horizontal': self.horizontal_detected, 'yaw': self.yaw_active,
                         'contact_witness': self.light_witness.state.active,
                         'after_release': self.detached.evidence.active,
                         'pickup_hold': getattr(self, 'pick_event_active', False)}
                self.evidence[tick] = {
                    'reported': bool(result[1]), 'active_paths': [k for k, v in paths.items() if v],
                    'grip': bool(grip), 'drive_departure': getattr(self, 'changed_drive', None),
                    'estimated_table_N': float(s.last_table_wrench[2]),
                    'yaw_Nm': None if self.yaw_value is None else float(self.yaw_value),
                    'yaw_limit_Nm': float(self.bounds[int(bool(grip))]),
                    'raw_yaw_Nm': None if getattr(self.yaw, 'raw_value', None) is None
                    else float(self.yaw.raw_value),
                    'r10_suspicion': float(s.last_score), 'r10_fast_suspicion': float(s.last_fast_score)}
            return result
    return Observed()


def run(source):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_RESULT:
        raise ValueError('requires the exact previously disclosed R24 result')
    plan = json.loads(lab.PLAN.read_bytes())
    row = next(r for r in json.loads(raw)['rows'] if r['id'] == CASE and r['variant'] == MECHANICS)
    case = next(c for c in plan['single'] if c['id'] == CASE)
    records = {}
    for arm, constructor in [('r23_free', Incumbent), ('r24', Joint)]:
        logs, records[arm] = {}, {}
        for twin, force in [('none', None), ('force', case['impact'])]:
            observer = probe(constructor)
            _, log = lab.run_episode(case, plan['variants'][MECHANICS], lambda: observer, (force, None))
            starts = [] if force is None else [lab.certified.actual_start(log['t'], force)[0]]
            common = lab.record_common(log, (force, None), starts, plan)
            if common != row['records'][arm]['twins'][twin]:
                raise ValueError('instrumented twin did not reproduce the frozen result')
            if common['false_times_s'] != [3.922] or common['false_active_samples'] != 8:
                raise ValueError('different target alarm')
            for tick, v in observer.evidence.items():
                v.update(actual_table_N=float(log['f_table'][tick]),
                         applied_force_N=float(log['f_ext'][tick]),
                         actual_tool_N=float(log['f_tool'][tick]))
            records[arm][twin] = {'actual_impact_start_ms': starts[0] if force is not None else None,
                                  'reported_ticks_in_capture': [i for i,v in observer.evidence.items() if v['reported']],
                                  'at_rise': observer.evidence[3922],
                                  'physical_sha256': lab.certified.physical_hash(log)}
            logs[twin] = log
        if lab.certified.trace_signature(logs) != row['r23_physics_sha256']:
            raise ValueError('different physics from sealed independent row')
    return {'schema':'r24-after-disclosure-false-path.v1', 'post_disclosure_diagnosis_only':True,
            'prior_result_sha256':ORIGINAL_RESULT, 'case':case, 'mechanics':MECHANICS,
            'actual_episodes':4, 'records':records,
            'scope':'Observation only. Applied/tool/table force truth is joined after simulation. No change to the candidate, scorer, or independent outcome.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    result = run(args.prior)
    with args.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print({arm:rec['none']['at_rise'] for arm,rec in result['records'].items()})
