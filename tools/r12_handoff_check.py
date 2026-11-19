"""Stage physical load transfer, false alarms, and release-impact preservation together."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'results/r12/check_plan.json'


def no_new_false_events(reference, candidate, tolerance=.020):
    """One-to-one, chronological tolerance matching; a displaced FA is new."""
    a=sorted(reference['false_alarm_times'])
    b=sorted(candidate['false_alarm_times'])
    if len(a)!=reference['false_alarms'] or len(b)!=candidate['false_alarms']:
        return False
    i=0
    for event in b:
        while i<len(a) and a[i]<event-tolerance-1e-10:
            i+=1
        if i==len(a) or abs(a[i]-event)>tolerance+1e-10:
            return False
        i+=1
    return True


def normal_checks(a,b,geometry_score):
    if a.get('status')!='ok' or b.get('status')!='ok':
        return {'pass':False,'g4':None,'reason':'incomplete or unscorable normal pair'}
    g4=geometry_score(a,b)
    costs={k:(v/b0 if b0>0 else (1. if v==0. else None))
           for k,v in b['cost'].items() for b0 in [a['cost'][k]]}
    e0,e1=a['event'],b['event']
    gates={'g4':g4['all_four_g4'] is True,'cost':all(v is not None and v<=1.05 for v in costs.values()),
           'saturation':b['sat_pct']<=a['sat_pct'],'alarms':no_new_false_events(e0,e1),
           'tool':(not e0['tool_detected'] or (e1['tool_present'] and e1['tool_detected']))
                  and (not e1['tool_prealarm'] or e0['tool_prealarm'])}
    return {'pass':all(gates.values()),'gates':gates,'g4':g4,'cost_ratios':costs}


def impact_checks(a,b):
    if a.get('status')!='ok' or b.get('status')!='ok':
        return {'pass':False,'reason':'diverged or unverified impact pair'}
    e0,e1=a['event'],b['event']
    gates={'prefix':a['normal_prefix_equal'] and b['normal_prefix_equal'],
           'distinguishable':e1['outcome']=='detected',
           'preserve_delay':(e0['outcome']!='detected' or
                             (e1['outcome']=='detected' and e1['delay_ms']<=e0['delay_ms']+5.+1e-9)),
           'prealarm':not e1['prealarm'] or e0['prealarm'],
           'false_alarms':no_new_false_events(e0,e1),
           'tool':(not e0['tool_detected'] or (e1['tool_present'] and e1['tool_detected']))
                  and (not e1['tool_prealarm'] or e0['tool_prealarm'])}
    return {'pass':all(gates.values()),'gates':gates}


def collect(mode,plan,stage):
    import importlib
    from geometry_eval import physical, FootprintUnverified
    from events import event_result
    from sim.runner import run_episode
    from collision_detector import Detector
    controller=importlib.import_module(mode+'_controller').Controller
    if controller.MODE!=mode:
        raise ValueError('unexpected controller mode')
    rows=[]
    for case in plan['conditions']:
        m=case['mass']
        kwargs={'box_mass':m,'box_com':tuple(case['com']),'pick_offset':case.get('offset')}
        meta,log=run_episode(controller(),detector=Detector(),**kwargs)
        if meta['diverged']:
            rows.append({'id':case['id'],'status':'diverged'});continue
        t=log['t']
        if stage=='normal':
            try:
                result=physical(log,m,3.9,.001)
            except FootprintUnverified:
                rows.append({'id':case['id'],'status':'placing footprint unverified'});continue
            dtau=np.r_[0.,np.max(np.abs(np.diff(log['tau'],axis=0)),axis=1)]
            pick=(t>=1.2)&(t<1.8)
            place=(t>=3.6)&(t<4.1)
            rec={**result,'cost':{'tcp_rms':meta['rms_mm'],'tcp_peak':meta['max_mm'],
                 'pick_step':float(dtau[pick].max()),'place_step':float(dtau[place].max()),
                 'placing_force_peak':float(log['f_table'][place].max())},
                 'sat_pct':meta['saturation_pct'],'event':event_result(t,log['detect'],{},log['f_tool'])}
        else:
            onset,force=case['probe']
            coll=(onset,tuple(force))
            pert,forced=run_episode(controller(),detector=Detector(),collision=coll,**kwargs)
            if pert['diverged']:
                rows.append({'id':case['id'],'status':'diverged'});continue
            # The zero-amplitude ramp endpoint at onset cannot affect earlier samples.
            prefix=t<onset
            if not np.isclose(np.max(forced['f_ext']),np.linalg.norm(force)) or np.any(forced['f_ext'][prefix]):
                raise ValueError('requested release pulse was not realized')
            rec={'event':event_result(t,forced['detect'],{'collision':coll},forced['f_tool'],log['detect']),
                 'normal_prefix_equal': all(log[k][prefix].tobytes()==forced[k][prefix].tobytes()
                                            for k in ('q','tau','detect'))}
        rows.append({'id':case['id'],'status':'ok',**rec})
    return rows


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def check(kit,src,mode,plan=PLAN):
    sys.path.insert(0,str(ROOT))
    from tools.validate_submission import prepare_overlay
    from tools.r11_placement import score
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    env.pop('PYTHONPATH',None)
    from tools.r12_force_audit import snapshot_files, install_snapshot
    sources={'reference_ctrl':ROOT/('r1_'+mode+'.py'),'reference_det':ROOT/'experiments/r10_scratch/CANDIDATE.py',
             'candidate_ctrl':src/('my_'+mode+'.py'),'candidate_det':src/'my_detector.py',
             'scorer':Path(__file__),'plan':plan,'geometry':ROOT/'tools/r11_placement.py',
             'events':ROOT/'tools/r5_metrics.py','kit_manifest':ROOT/'tools/kit_manifest.sha256'}
    blobs,pins=snapshot_files(sources)
    design=json.loads(blobs['plan'])
    record={'schema':'r12-handoff-check.v1','mode':mode,'pins':pins}
    for stage,fn in [('normal',lambda a,b:normal_checks(a,b,score)),('impact',impact_checks)]:
        variants={}
        for name in ('reference','candidate'):
            with tempfile.TemporaryDirectory(prefix='handoff-check-') as td:
                dest=Path(td)
                prepare_overlay(kit,dest,'my',mode,archived_r9=True)
                if digest(dest/'tools/kit_manifest.sha256')!=pins['kit_manifest']:
                    raise ValueError('kit manifest differs from frozen assessment input')
                for key,filename in [(name+'_ctrl',mode+'_controller.py'),(name+'_det','collision_detector.py'),
                        ('geometry','geometry_eval.py'),('events','events.py'),('scorer','handoff.py'),('plan','design.json')]:
                    install_snapshot(blobs[key],pins[key],dest/filename)
                code=('import json,sys,handoff;from pathlib import Path;'
                      'json.dump(handoff.collect(sys.argv[1],json.loads(Path("design.json").read_text()),'
                      'sys.argv[2]),sys.stdout,allow_nan=False)')
                proc=subprocess.run([sys.executable,'-c',code,mode,stage],cwd=dest,env=env,
                                    text=True,stdout=subprocess.PIPE,check=True)
                variants[name]=json.loads(proc.stdout)
        rows=[]
        for c,a,b in zip(design['conditions'],variants['reference'],variants['candidate'],strict=True):
            if not c['id']==a['id']==b['id']:
                raise ValueError('unmatched case IDs')
            rows.append({'id':c['id'],'reference':a,'candidate':b,'assessment':fn(a,b)})
        record[stage]=rows
        if stage=='normal' and not all(r['assessment']['pass'] for r in rows):
            record.update(decision='normal_failed_impacts_not_run',impact=None)
            return record
    record['decision']=('finite_cases_passed' if all(r['assessment']['pass'] for r in record['impact'])
                        else 'impact_qualification_failed')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--candidate-dir',type=Path,required=True)
    p.add_argument('--mode',choices=['torque','servo'],default='torque')
    args=p.parse_args()
    kit,src,out=[v.resolve() for v in (args.kit,args.candidate_dir,args.out)]
    if args.out.is_symlink() or out.exists() or not out.parent.is_dir() or out.is_relative_to(kit):
        p.error('new result file outside the kit required')
    if not all((src/f'my_{n}.py').is_file() and not (src/f'my_{n}.py').is_symlink()
               for n in ('torque','servo','detector')):
        p.error('three regular self-contained candidate modules required')
    sys.dont_write_bytecode=True
    result=check(kit,src,args.mode)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(result['decision'],out)
