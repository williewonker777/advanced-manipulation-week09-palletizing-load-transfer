"""Estimate change of box-bottom height in the taught release frame, not residual COM alone.

Calibration uses known nominal pickups and allowed signals. No scoring geometry is
accepted by this module. Empirical guards describe finite trials, not coverage.
"""
import numpy as np
from tools import r13_grasp_estimator as old


def _system(feat, model):
    mass = float(feat['mass'])
    bias = np.asarray(model['bias'])
    sigma = np.asarray(model['sigma'])
    y = (feat['y'] - bias[0] - (mass-float(model['center']))*bias[1]) / sigma
    if y.shape != (len(old.CENTRES), 6) or not (np.isfinite(y).all() and np.all(sigma>0)):
        raise ValueError('invalid features or calibration')
    n, j = y.shape
    t = (old.DT*old.CENTRES - 2.55)
    eye = np.broadcast_to(np.eye(j),(n,j,j))
    x = np.concatenate((feat['jac'], feat['stiffness'][:,:,None], eye,
                        eye*t[:,None,None]), axis=-1)/sigma[None,:,None]
    if not np.all(np.isfinite(x)):
        raise ValueError('nonfinite feature matrix')
    reg = np.r_[np.full(3,15.), .065, np.maximum(sigma*3.,.06), np.maximum(sigma*.8,.02)]
    # Accumulate only five compact blocks. Adding an identity after column
    # scaling is the same physical ridge as the original augmented least squares.
    x = x * reg
    v = np.asarray(feat['vertical'])
    if v.shape != (3,) or not np.isclose(v@v,1.,rtol=0,atol=1e-8):
        raise ValueError('expected a unit placement vertical')
    fold = (old.CENTRES//80)%5
    grams, rhs = [], []
    for i in range(5):
        xb = x[fold==i].reshape(-1,x.shape[-1])
        target = y[fold==i].ravel()
        grams.append(xb.T@xb)
        rhs.append(xb.T@target)
    return np.stack(grams), np.stack(rhs), reg, v


def _raw(feat, model, detail=True):
    grams, rhs, scales, v = _system(feat,model)
    gt, bt = grams.sum(axis=0), rhs.sum(axis=0)
    eye = np.eye(len(scales))
    def solve(g,b):
        return np.linalg.solve(g+eye,b)*scales
    coef = solve(gt,bt)
    if not detail:
        return float(v@coef[:3]) + float(feat['prior_bottom_mm'])
    deleted = [float(v@solve(gt-g, bt-b)[:3]) for g,b in zip(grams,rhs)]
    # Normalize independently of priors. For a full Gram matrix project the
    # vertical column on all other columns via a Schur complement. Pseudoinverse
    # handles a rank-deficient nuisance design without inventing information.
    unscaled = gt / (scales[:,None]*scales[None,:])
    orth = np.linalg.svd(v[None,:],full_matrices=True)[2][1:].T
    ns = np.eye(len(scales))[:,1:].copy()
    ns[:3,:2] = orth
    unit = np.r_[v, np.zeros(len(scales)-3)]
    cov = ns.T@unscaled@ns
    cross = ns.T@unscaled@unit
    remaining = unit@unscaled@unit - cross@np.linalg.lstsq(cov,cross,rcond=1e-14)[0]
    info = float(np.sqrt(max(0.,remaining)/(len(old.CENTRES)*6)))
    return coef, deleted, info


def _model(fits,noise):
    model = old.fit_normal(fits,noise)
    masses = np.array([float(row['mass']) for row in fits])
    values = np.array([_raw(row,model,False) for row in fits])
    a = np.c_[np.ones(len(masses)),masses-float(model['center'])]
    model['normal_bottom_intercept_slope_mm'] = np.linalg.lstsq(a,values,rcond=None)[0]
    return model


def normal_expectation(mass,model):
    a,b = model['normal_bottom_intercept_slope_mm']
    return float(a + b*(float(mass)-float(model['center'])))


def calibrate(fits,noise):
    if len(fits)<5:
        raise ValueError('at least five normal pickups required')
    model = _model(fits,noise)
    loo = []
    for i, row in enumerate(fits):
        smaller = _model([q for j,q in enumerate(fits) if j!=i],noise)
        loo.append(_raw(row,smaller,False)-normal_expectation(row['mass'],smaller))
    model['normal_guard_mm'] = float(np.max(np.abs(loo)))
    model['loo_signed_mm'] = loo
    model['schema'] = 'r14-registration-calibration.v1'
    return {key: val.tolist() if isinstance(val,np.ndarray) else val for key,val in model.items()}


def decide(features, calibration_dict):
    """Public single-case sensing. Caller may pass the normal calibration as plain JSON."""
    m = calibration_dict
    if m.get('schema') != 'r14-registration-calibration.v1':
        raise ValueError('unknown calibration format')
    coef, blocks, info = _raw(features,m)
    mass = float(features['mass'])
    prior = float(features['prior_bottom_mm'])
    centre = normal_expectation(mass,m)
    residual = float(features['vertical']@coef[:3])
    point = (prior + residual) - centre
    width = .75+float(m['normal_guard_mm'])+max(abs(d-residual) for d in blocks)
    domain = (m['range'][0] <= mass <= m['range'][1] and np.linalg.norm(coef[:3])<=15. and info>1e-4)
    accepted = bool(domain and width<=2.5 and abs(point)>width)
    return dict(extra_height_mm=point, absolute_bottom_mm=prior+residual,
                expected_normal_bottom_mm=centre, prior_bottom_mm=prior, com_residual_height_mm=residual,
                half_width_mm=width, interval_mm=[point-width,point+width], accept=accepted,
                applied_mm=point if accepted else 0., soft_interval_action_mm=float(np.sign(point)*max(0.,abs(point)-width)) if domain else 0.,
                independent_height_information_per_mm=info, fitted_displacement_within_domain=bool(domain),
                block_delete_mm=[prior+b-centre for b in blocks])
