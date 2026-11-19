"""Public-sensor R17 information-prefix stock reproduction helpers."""
from __future__ import annotations
import hashlib,json
import numpy as np

VARIANTS=('none','explicit_zero','xy','up','down')

def offset(case,variant):
    if variant=='none': return None
    if variant=='explicit_zero': return (0.,0.,0.)
    x,y,z=map(float,case['shift'])
    return (x,y,{'xy':0.,'up':z,'down':-z}[variant])

def feature_from_samples(samples,case,last_centre):
    from experiments.r17_information_prefix.stream import InformationStream
    from tools.r13_grasp_estimator import DT
    arrays={key:np.asarray(samples[key]) for key in ('t','encoder','jts','grip')}
    count=int(last_centre)+24
    if any(value.shape[0]<count for value in arrays.values()): raise ValueError('public prefix ends before decision')
    stream=InformationStream(float(case['mass']),np.asarray(case['com'],dtype=float))
    for values in zip(*(arrays[key][:count] for key in ('t','encoder','jts','grip')),strict=True):
        stream.observe(*values)
    result=stream.information(int(last_centre))
    if result['completed_at_s']!=(count-1)*DT: raise ValueError('information deadline mismatch')
    return result

def metrics(rows):
    norm=lambda key:np.array([np.linalg.norm(row[key]) for row in rows])
    truth,error,action=norm('true_xy_mm'),norm('point_error_mm'),norm('action_error_mm')
    tiny,large=truth<1.,truth>=3.
    misses=sum(row['accept'] and not row['covered'] for row in rows)
    signed=sum(bool(flag and row['accept'] and np.dot(row['true_xy_mm'],row['model_action_xy_mm'])>0) for flag,row in zip(large,rows))
    rms=float(np.sqrt(np.mean(error**2))); maximum=float(action.max())
    return {'count':len(rows),'vector_rmse_mm':rms,'max_action_error_mm':maximum,
            'accepted':int(sum(row['accept'] for row in rows)),'accepted_interval_misses':int(misses),
            'false_tiny_actions':int(sum(flag and row['accept'] for flag,row in zip(tiny,rows))),
            'big_n':int(large.sum()),'big_signed':int(signed),'unsupported':int(sum(not row['mass_interpolated'] or not row['signature_supported'] for row in rows)),
            'qualified':bool(rms<=1.5 and maximum<=2. and misses==0 and not any(flag and row['accept'] for flag,row in zip(tiny,rows)) and signed==int(large.sum()))}

def projection(row,fields): return {key:row[key] for key in fields}

def validate(calibration,reference,plan):
    if calibration.get('schema')!='r17-normal-information-calibration.v1' or calibration.get('last_centre')!=2920 or calibration.get('completed_at_s')!=2.943: raise ValueError('frozen 2.943 s calibration required')
    provenance=calibration.get('portable_provenance',{})
    if provenance.get('source_normal_count')!=24 or provenance.get('truth_used_in_fit') is not False or provenance.get('raw_records_shipped') is not False or 'cannot independently regenerate' not in provenance.get('stock_regeneration',''): raise ValueError('transparent pooled normal-only provenance required')
    if reference.get('schema')!='r17-information-prefix-portable-stock-reference.v1' or len(reference.get('rows',()))!=15 or reference.get('metrics',{}).get('count')!=15: raise ValueError('frozen 15-row stock reference required')
    if tuple(plan.get('variants',()))!=VARIANTS or len(plan.get('holdout',()))!=3: raise ValueError('frozen ep1/ep2/ep3 plan required')

def run_stock(calibration,reference,plan):
    from experiments.r17_information_prefix.estimate import decide
    from sim.runner import run_episode
    import collision_detector,torque_controller
    from r13_grasp_audit import capture,score_capsule,observed_geometry,compact_metrics
    validate(calibration,reference,plan)
    if torque_controller.Controller.MODE!='torque': raise ValueError('R1 torque required')
    rows,capsules=[],[]
    for case in plan['holdout']:
        for variant in VARIANTS:
            ctrl,samples=capture(torque_controller.Controller)
            run,log=run_episode(ctrl,float(case['mass']),tuple(case['com']),collision_detector.Detector(),pick_offset=offset(case,variant))
            if run['diverged']: raise ValueError('diverged stock observation')
            feature=feature_from_samples(samples,case,calibration['last_centre'])
            prediction=decide(feature,calibration)
            rows.append({'id':case['id'],'variant':variant,**prediction,'run':compact_metrics(run)})
            capsules.append({'geometry':score_capsule(log),'com':np.asarray(case['com'],dtype=float)})
    predicted=[projection(row,reference['row_fields'][:10]) for row in rows]
    seal=hashlib.sha256(json.dumps(predicted,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    baselines={}
    for row,item in zip(rows,capsules,strict=True):
        actual=np.asarray(observed_geometry(item['geometry'],item['com'])['local_com_shift_mm'])[:2]
        if row['variant']=='none': baselines[row['id']]=actual
        truth=actual-baselines[row['id']]; point=np.asarray(row['relative_xy_mm'])-truth; action=np.asarray(row['model_action_xy_mm'])-truth
        row.update(true_xy_mm=truth.tolist(),point_error_mm=point.tolist(),action_error_mm=action.tolist(),covered=bool(np.linalg.norm(point)<=row['halfwidth_radius_mm']))
    projected=[projection(row,reference['row_fields']) for row in rows]
    parity=projected==reference['rows']; measured=metrics(rows); metric_parity=measured==reference['metrics']
    if not parity or not metric_parity: raise AssertionError('stock reproduction differs from frozen research scoring')
    return {'rows':rows,'metrics':measured,'prediction_seal_sha256':seal,'predictions_sealed_before_offline_score':True,'research_row_parity':parity,'research_metric_parity':metric_parity}
