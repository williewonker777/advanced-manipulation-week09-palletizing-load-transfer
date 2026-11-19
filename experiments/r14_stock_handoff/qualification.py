"""Stage-gated stock ``sim.runner.run_episode`` evaluation; no private fixture."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

import my_detector
import r1_torque
from sim import config as C, trajectory as T
from sim.runner import run_episode
from experiments.r12_hybrid.controller import Controller as Hybrid
from experiments.r14_stock_handoff.controller import Controller
from experiments.r14_stock_handoff.detector import Detector
from tools.r11_placement import physical, score, FootprintUnverified

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
KEYS=('q','q_d','tau','e_mm','f_table','f_tool','qpos','r','detect','f_ext')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_pins():
    names=('r1_torque.py','my_detector.py','experiments/r12_hybrid/controller.py',
           'experiments/r14_online/stream.py','experiments/r14_airborne/registration.py',
           'models/r12/counterfactual_coefficients.json',
           'experiments/r14_stock_handoff/controller.py','experiments/r14_stock_handoff/detector.py',
           'experiments/r14_stock_handoff/geometry.py','experiments/r14_stock_handoff/test_controller.py',
           'experiments/r14_stock_handoff/plan.json','experiments/r14_stock_handoff/run.py')
    return {name:sha(ROOT/name) for name in names}


def calibration(path):
    raw=Path(path).read_bytes(); doc=json.loads(raw)
    model=doc['modes']['torque']['registered']
    if model.get('schema')!='r14-registration-calibration.v1':raise ValueError('stock torque calibration required')
    return model,hashlib.sha256(raw).hexdigest(),doc['pins']


def detector_for(kind): return my_detector.Detector() if kind=='r10' else Detector()

def controller_for(kind,model):
    if kind=='r10':return r1_torque.Controller()
    if kind=='hybrid':return Hybrid()
    return Controller(model)


def compact(metrics,log,ctrl,mass):
    t=np.asarray(log['t']); step=np.r_[0.,np.max(np.abs(np.diff(log['tau'],axis=0)),axis=1)]
    place=(t>=3.6)&(t<4.1); pick=(t>=1.2)&(t<1.8)
    out={'status':'diverged' if metrics['diverged'] else 'ok','cost':{
        'tcp_rms':metrics['rms_mm'],'tcp_max':metrics['max_mm'],
        'place_step':float(step[place].max()),'pick_step':float(step[pick].max()),
        'place_force':float(np.max(log['f_table'][place]))},
        'saturation_pct':metrics['saturation_pct'],'false_alarms':metrics.get('false_alarms'),
        'false_alarm_times':metrics.get('false_alarm_times'),'tool_delay_ms':metrics.get('tool_contact_delay_ms'),
        'ctrl_p99_us':float(1e6*np.percentile(log['t_ctrl'],99)),
        'registration':None,'hybrid_active':bool(getattr(ctrl,'hybrid_active',False))}
    if not metrics['diverged'] and mass > 0:
        try:out.update(physical(log,mass,T.T_RELEASE,C.DT))
        except FootprintUnverified:out['status']='footprint_unverified'
    if (kind := getattr(ctrl,'registration_result',None)) and getattr(ctrl,'registration_stream',None) is not None:
        out['registration']={'accepted':bool(ctrl.registration_accepted),
            'estimated_extra_height_mm':float(kind.get('extra_height_mm',0.)),
            'half_width_mm':float(kind.get('half_width_mm',0.)),
            'command_world_z_mm':float(1000*ctrl.registered_height_m),
            'finish_t':ctrl.registration_finish_t,
            'stream_completed_at_s':ctrl.registration_stream.completed_at_s,
            'stream_timing_us':ctrl.registration_stream.timing_us,
            'ik_max_position_error_m':float(ctrl.ik_max_position_error_m),
            'ik_max_orientation_error_rad':float(ctrl.ik_max_orientation_error_rad)}
    return out


def episode(kind,case,model,collision=None):
    ctrl=controller_for(kind,model); det=detector_for(kind)
    metrics,log=run_episode(ctrl,float(case['mass']),tuple(case['com']),det,
                            collision=collision,pick_offset=case.get('offset'))
    return compact(metrics,log,ctrl,float(case['mass'])),log


def equal_prefix(a,b,end):
    return {key:bool(np.array_equal(np.asarray(a[key])[:end],np.asarray(b[key])[:end])) for key in KEYS}


def normal_case(case,model):
    records={};logs={}
    for kind in ('r10','hybrid','registered'):
        records[kind],logs[kind]=episode(kind,case,model)
    ratios={base:{key:records['registered']['cost'][key]/max(value,1e-12)
                  for key,value in records[base]['cost'].items()} for base in ('r10','hybrid')}
    g4=score(records['r10'],records['registered'])
    prefix=equal_prefix(logs['hybrid'],logs['registered'],round(3.55/C.DT))
    reg=records['registered']['registration']; accepted=bool(reg and reg['accepted'])
    full_fallback=equal_prefix(logs['hybrid'],logs['registered'],len(logs['hybrid']['t'])) if not accepted else None
    tool_ok=(records['hybrid']['tool_delay_ms'] is None or records['registered']['tool_delay_ms'] is not None
             and records['registered']['tool_delay_ms']<=records['hybrid']['tool_delay_ms']+5.)
    gates={'status':all(records[k]['status']=='ok' for k in records),'G4':g4['all_four_g4'],
           'cost_r10':max(ratios['r10'].values())<=1.05,'cost_hybrid':max(ratios['hybrid'].values())<=1.05,
           'saturation':records['registered']['saturation_pct']<=max(records['r10']['saturation_pct'],records['hybrid']['saturation_pct']),
           'FA':records['registered']['false_alarms']<=records['hybrid']['false_alarms'],
           'tool':tool_ok,'prefix':all(prefix.values()),'timing':records['registered']['ctrl_p99_us']<=800.,
           'fallback':accepted or all(full_fallback.values())}
    return {'id':case['id'],'arms':records,'ratios':ratios,'g4':g4,'prefix':prefix,
            'full_fallback':full_fallback,'gates':gates,'passed':all(gates.values())}


def unloaded(model):
    case={'mass':0.,'com':[0.,0.,0.],'offset':None}; records={};logs={}
    for kind in ('hybrid','registered'):
        records[kind],logs[kind]=episode(kind,case,model)
    parity=equal_prefix(logs['hybrid'],logs['registered'],len(logs['hybrid']['t']))
    return {'id':'unloaded','byte_parity':parity,'passed':all(parity.values())}


def normal(plan,model,cal_sha,cal_pins):
    zero=unloaded(model);rows=[normal_case(c,model) for c in plan['normal_cases']]
    return {'schema':'r14-stock-handoff-normal.v1','stage':'normal','unloaded':zero,'rows':rows,
            'passed':zero['passed'] and all(r['passed'] for r in rows),
            'pins':{'sources':source_pins(),'calibration':cal_sha,'calibration_sources':cal_pins}}


def impact_case(case,model):
    collision=case['collision']; records={};logs={}
    for kind in ('r10','hybrid','registered'):
        records[kind],logs[kind]=episode(kind,case,model,collision)
    slim={k:{name:records[k][name] for name in ('false_alarms','false_alarm_times')} for k in records}
    for kind in records:
        # compact() intentionally omits collision delay; get it from a direct field added below is impossible.
        # Reconstruct rise relative to the frozen collision time from the retained detector trace.
        t=logs[kind]['t'];d=logs[kind]['detect'];rise=np.flatnonzero(d & ~np.r_[False,d[:-1]])
        k0=int(np.searchsorted(t,collision[0]));true=[i for i in rise if k0<=i<min(len(t),k0+320)]
        slim[kind]['delay_ms']=None if not true else float(1000*(t[true[0]]-collision[0]))
        slim[kind]['prealarm']=bool(k0>0 and d[k0-1])
    prefix=equal_prefix(logs['hybrid'],logs['registered'],round(3.55/C.DT))
    base=slim['r10'];cand=slim['registered']
    no_loss=base['delay_ms'] is None or cand['delay_ms'] is not None
    delay_ok=base['delay_ms'] is None or cand['delay_ms']<=base['delay_ms']+5.
    gates={'no_loss':no_loss,'delay':delay_ok,'no_new_prealarm':not(cand['prealarm'] and not base['prealarm']),
           'FA':cand['false_alarms']<=slim['hybrid']['false_alarms'],'prefix':all(prefix.values())}
    return {'id':case['id'],'collision':collision,'arms':slim,'prefix':prefix,'gates':gates,'passed':all(gates.values())}


def impacts(plan,model,normal_doc,normal_path,cal_sha,cal_pins):
    current=source_pins()
    if not normal_doc.get('passed') or normal_doc['pins']['sources']!=current or normal_doc['pins']['calibration']!=cal_sha:
        raise ValueError('normal gate absent, failed, or stale')
    rows=[impact_case(c,model) for c in plan['impact_cases']]
    return {'schema':'r14-stock-handoff-impact.v1','stage':'impact','rows':rows,'passed':all(r['passed'] for r in rows),
            'pins':{'sources':current,'calibration':cal_sha,'calibration_sources':cal_pins,
                    'normal_result':sha(normal_path)}}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('normal','impact'),required=True)
    p.add_argument('--calibration',type=Path,required=True);p.add_argument('--normal-result',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    if a.out.exists() or a.out.is_symlink():p.error('fresh output required')
    plan=json.loads((HERE/'plan.json').read_text());model,cal_sha,cal_pins=calibration(a.calibration)
    if a.stage=='normal':result=normal(plan,model,cal_sha,cal_pins)
    else:
        if a.normal_result is None:p.error('--normal-result required for impact stage')
        result=impacts(plan,model,json.loads(a.normal_result.read_text()),a.normal_result,cal_sha,cal_pins)
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'stage':a.stage,'passed':result['passed'],'failed':[r['id'] for r in result.get('rows',[]) if not r['passed']]}))

if __name__=='__main__':main()
