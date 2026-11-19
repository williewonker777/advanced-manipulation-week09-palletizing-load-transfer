"""Progressively run the unchanged R17 gates plus independently measured budgets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from experiments.r17_qualification import qualification as gates
from experiments.r18_combined_gate.policy import Controller, Detector


ROOT = Path(__file__).resolve().parents[2]
STAGES = ('known-normal','exposed-impact','fresh-normal','fresh-impact')
ABS_TOL = {'q':2e-7,'q_d':0.,'qpos':2e-7,'tau':2e-5,'f_ext':0.,'f_table':2e-4,'f_tool':2e-4,'e_mm':2e-4}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True);cli.add_argument('--pins',type=Path,required=True)
    cli.add_argument('--height',type=Path,required=True);cli.add_argument('--early',type=Path,required=True)
    a=cli.parse_args(argv); dest=a.out
    if dest.exists():cli.error('new output directory required')
    pins=json.loads(a.pins.read_bytes())
    if not pins or any(sha(ROOT/n)!=v for n,v in pins.items()):cli.error('frozen overlay mismatch')
    desc=json.loads((ROOT/'experiments/r18_combined_gate/plan.json').read_bytes())
    protocol=ROOT/'experiments/r17_qualification/plan.json'
    if sha(protocol)!=desc['upstream_plan_sha256'] or sha(Path(gates.__file__))!=desc['upstream_scorer_sha256']:
        cli.error('frozen scorer or case plan changed')
    plan=json.loads(protocol.read_bytes()); height=json.loads(a.height.read_bytes());early=json.loads(a.early.read_bytes());dest.mkdir()
    side=[];timers=[];reference={};episode=gates.episode
    class Meter(Detector):
        def reset(self):
            super().reset()
            inner=self.update
            durations=np.zeros(6000,np.int64);timers.append(durations)
            tick=0
            def watch(*xs):
                nonlocal tick
                start=time.perf_counter_ns();v=inner(*xs);durations[tick]=time.perf_counter_ns()-start
                tick+=1
                return v
            self.update=watch
    gates.CANDIDATE=Controller;gates.CANDIDATE_DETECTOR=Meter
    def measured(kind,case,h,e,collision=None):
        key=collision is not None
        record,log=episode(kind,case,h,e,collision)
        if kind=='d':reference[key]={f:log[f] for f in ABS_TOL}
        if kind=='candidate':
            ns=timers.pop()
            if np.any(ns<=0):raise ValueError('incomplete measurement')
            whole=1e6*log['t_ctrl']+ns*.001
            physical=None
            if key in reference:
                physical={f:float(np.max(np.abs(log[f]-v))) for f,v in reference[key].items()}
            side.append({'phase':'forced' if key else 'normal','combined_p99_us':float(np.percentile(whole,99)),
                         'combined_max_us':float(np.max(whole)),'combined_over_1ms':int(np.count_nonzero(whole>1000.)),
                         'detector_p99_us':float(np.percentile(ns*.001,99)),'physical_max_abs_difference':physical})
        return record,log
    gates.episode=measured
    allstages=[];fresh=0
    for stage in STAGES:
        rows=[]; extras=[]; case_defs=plan[gates.STAGE_KEYS[stage]]
        if stage.startswith('fresh'):fresh+=len(case_defs)
        if stage=='known-normal':case_defs=case_defs+plan['fallback_cases']
        for case in case_defs:
            side.clear();reference.clear()
            fallback = stage=='known-normal' and case in plan['fallback_cases']
            score=gates.fallback_case if fallback else gates.normal_case if 'normal' in stage else gates.impact_case
            row=score(case,height,early); expected=2 if stage.endswith('impact') else 1
            if len(side)!=expected:raise ValueError('incomplete side measurements')
            tol_ok=all(v['physical_max_abs_difference'] is not None and all(x<=ABS_TOL[k] for k,x in v['physical_max_abs_difference'].items()) for v in side) if not fallback else row['gates']['exact_r10_nonclock_and_logs']
            row['original_r17_passed']=row['passed'];row['extra_gates']={'combined_p99':all(v['combined_p99_us']<=1000. for v in side),'D_physics_or_exact_fallback':bool(tol_ok)}
            row['extra_measurements']=list(side);row['passed']=row['passed'] and all(row['extra_gates'].values());rows.append(row)
            if not row['passed']:print(stage,case['id'],[k for k,b in {**row['gates'],**row['extra_gates']}.items() if not b],flush=True)
        if stage=='fresh-impact':extras=[gates.fresh_impact_evidence(rows,len(case_defs))]
        stage_pass=all(x['passed'] for x in rows+extras)
        with (dest/(stage+'.json')).open('x') as f:json.dump({'stage':stage,'rows':rows,'extra_evidence':extras,'passed':stage_pass},f,indent=2,allow_nan=False)
        allstages.append({'stage':stage,'count':len(rows),'passed_count':sum(x['passed'] for x in rows),'original_pass_count':sum(x['original_r17_passed'] for x in rows),'passed':stage_pass})
        print(allstages[-1],flush=True)
        if not stage_pass:break
    if any(sha(ROOT/n)!=v for n,v in pins.items()):raise ValueError('frozen source changed')
    (dest/'summary.json').write_text(json.dumps({'schema':'r18-integrated-outcome.v1','stages':allstages,
          'fresh_cases_opened':fresh, 'qualified_for_deployment':False,
          'inputs_sha256':{k:sha(p) for k,p in [('pins',a.pins),('height',a.height),('early',a.early)]}},indent=2)+'\n')


if __name__=='__main__':
    main()
