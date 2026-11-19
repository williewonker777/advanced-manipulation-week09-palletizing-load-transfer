"""Revised torque registration using an empty-motion normal reference."""
from __future__ import annotations
import numpy as np
from sim import config as C
from tools import r13_grasp_estimator as old
from experiments.r14_airborne import registration as base

ALPHA=.25

def _x(empty):return np.asarray(empty['empty_vector'],float)
def _regress(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-9);z=(x-mean)/scale;design=np.c_[np.ones(len(z)),z];ridge=np.diag([0.]+[ALPHA]*z.shape[1]);coef=np.linalg.solve(design.T@design+ridge,design.T@y);return mean,scale,coef
def _predict(empty,model):return float(np.r_[1.,(_x(empty)-model['empty_mean'])/model['empty_scale']]@model['empty_coef'])

def _fit(rows):
 feats=[r['features'] for r in rows];sensor=old.fit_normal(feats,C.JTS_NOISE);targets=[base._raw(f,sensor,False) for f in feats];mean,scale,coef=_regress([_x(r['empty']) for r in rows],targets);model={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in sensor.items()};model.update(empty_mean=mean.tolist(),empty_scale=scale.tolist(),empty_coef=coef.tolist(),empty_alpha=ALPHA);return model

def calibrate(rows):
 if len(rows)!=12:raise ValueError('exactly 12 preregistered normal rows required')
 model=_fit(rows);loo=[]
 for i,row in enumerate(rows):
  smaller=_fit([r for j,r in enumerate(rows) if j!=i]);target=base._raw(row['features'],smaller,False);loo.append(float(target-_predict(row['empty'],smaller)))
 model.update(schema='r14-empty-reference-calibration.v1',empty_guard_mm=float(max(abs(x) for x in loo)),empty_loo_signed_mm=loo)
 return model

def decide(features,empty,model):
 if model.get('schema')!='r14-empty-reference-calibration.v1':raise ValueError('unknown calibration')
 coef,blocks,info=base._raw(features,model);absolute=float(features['prior_bottom_mm'])+float(features['vertical']@coef[:3]);normal=_predict(empty,model);point=absolute-normal;width=.75+float(model['empty_guard_mm'])+max(abs((float(features['prior_bottom_mm'])+b-normal)-point) for b in blocks);mass=float(features['mass']);domain=(model['range'][0]<=mass<=model['range'][1] and np.linalg.norm(coef[:3])<=15. and info>1e-4);accept=bool(domain and width<=2.5 and abs(point)>width)
 return dict(extra_height_mm=point,absolute_bottom_mm=absolute,expected_normal_bottom_mm=normal,prior_bottom_mm=float(features['prior_bottom_mm']),half_width_mm=float(width),interval_mm=[point-width,point+width],accept=accept,applied_mm=point if accept else 0.,soft_interval_action_mm=float(np.sign(point)*max(0.,abs(point)-width)) if domain else 0.,independent_height_information_per_mm=info,fitted_displacement_within_domain=bool(domain),block_delete_mm=[float(features['prior_bottom_mm'])+b-normal for b in blocks],empty_feature=empty)
