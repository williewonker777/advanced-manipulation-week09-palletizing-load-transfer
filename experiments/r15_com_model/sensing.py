"""Pre-landing horizontal payload-COM registration; public sensor dynamics only."""
import numpy as np

from experiments.r14_airborne.registration import _system
from tools import r13_grasp_estimator as old


def _solutions(features, dynamics, deleted=True):
    grams, rhs, scales, _ = _system(features, dynamics)
    G, b = grams.sum(axis=0), rhs.sum(axis=0)
    eye = np.eye(len(b))
    prior = np.asarray(features['prior_com_mm'], dtype=float)
    if prior.shape != (3,) or not np.all(np.isfinite(prior)):
        raise ValueError('expected finite nominal latch COM translation')
    def fitted(gram, right):
        return prior + (np.linalg.solve(gram+eye, right)*scales)[:3]
    point = fitted(G,b)
    folds = np.array([fitted(G-g,b-h) for g,h in zip(grams,rhs)]) if deleted else None
    return point,folds


def _normal_model(fits,noise):
    dynamics=old.fit_normal(fits,noise)
    observed=np.array([_solutions(f,dynamics,False)[0][:2] for f in fits])
    masses=np.array([float(f['mass']) for f in fits])
    design=np.c_[np.ones(len(fits)),masses-float(dynamics['center'])]
    baseline=np.linalg.lstsq(design,observed,rcond=None)[0]
    return dynamics,baseline


def _expected(mass,dynamics,baseline):
    return np.asarray(baseline)[0]+(float(mass)-float(dynamics['center']))*np.asarray(baseline)[1]


def calibrate(fits,noise):
    if len(fits)<6:
        raise ValueError('need six or more registered normal pickups')
    dynamics,baseline=_normal_model(fits,noise)
    errors=[]
    for i,f in enumerate(fits):
        d,b=_normal_model([x for j,x in enumerate(fits) if j!=i],noise)
        errors.append(_solutions(f,d,False)[0][:2]-_expected(f['mass'],d,b))
    guard=float(max(np.linalg.norm(e) for e in errors))
    return {'schema':'r15-xy-calibration.v1','dynamics':{k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in dynamics.items()},
            'baseline_xy':baseline.tolist(),'normal_guard_mm':guard,'loo_vectors_mm':np.array(errors).tolist()}


def decide(features,model):
    if model.get('schema')!='r15-xy-calibration.v1':
        raise ValueError('unknown horizontal registration model')
    dynamics=model['dynamics']
    absolute,folds=_solutions(features,dynamics)
    normal=_expected(features['mass'],dynamics,model['baseline_xy'])
    xy=absolute[:2]-normal
    length=float(np.linalg.norm(xy))
    guard=float(model['normal_guard_mm'])
    if not np.isfinite(guard) or guard<0:
        raise ValueError('invalid calibration envelope')
    radius=float(.75 + guard + np.max(np.linalg.norm(folds[:,:2]-absolute[:2],axis=1)))
    domain=(dynamics['range'][0] <= float(features['mass']) <= dynamics['range'][1]
            and np.linalg.norm(absolute-np.asarray(features['prior_com_mm']))<=20.
            and np.isfinite(length) and length<=20.)
    accepted=bool(domain and length>radius and radius<=2.5)
    action=xy*min(1.,15./max(1.e-15,length)) if accepted else np.zeros(2)
    return dict(relative_xy_mm=xy.tolist(),absolute_com_offset_mm=absolute.tolist(),expected_normal_xy_mm=normal.tolist(),
                deleted_relative_xy_mm=(folds[:,:2]-normal).tolist(),halfwidth_radius_mm=radius,
                in_domain=bool(domain),accept=accepted,model_action_xy_mm=action.tolist())
