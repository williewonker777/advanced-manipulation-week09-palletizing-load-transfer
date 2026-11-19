"""Select a causal observation horizon on source normals, not on test cases."""
from __future__ import annotations

import numpy as np

from experiments.r16_robust_prefix import robust
from tools.r13_grasp_estimator import DT

MARGIN = robust.PHYSICAL_MARGIN_MM
MAX_RADIUS = robust.MAX_RADIUS_MM
SCHEMA = 'r17-normal-information-calibration.v1'


def past(row, final_centre):
    centres = robust.USED_CENTRES[robust.USED_CENTRES <= final_centre]
    if not len(centres) or centres[-1] != final_centre:
        raise ValueError('horizon must coincide with an observed centre')
    n = len(centres)
    values = {k:np.asarray(row[k])[:n] for k in ('y','jac','stiffness')}
    if any(len(v) != n for v in values.values()):
        raise ValueError('truncated measurement matrix')
    return {**values, **{k:row[k] for k in ('mass','prior_com_mm','signature')},
            'centres':centres, 'completed_at_s': (final_centre+23)*DT}


def solve(row, model, blocks):
    coord = robust._coordinate(row, model)
    sigma = np.asarray(model['sigma'])
    bias = np.einsum('k,ktj->tj', coord, model['bias'])
    y = (row['y']-bias)/sigma
    n, joints = y.shape
    eye = np.broadcast_to(np.eye(joints), (n,joints,joints))
    centres = row['centres']
    t = (centres-centres.mean())*DT
    design = np.concatenate((row['jac'],row['stiffness'][:,:,None],eye,eye*t[:,None,None]),axis=-1)/sigma[None,:,None]
    scales = np.r_[np.full(3,15.),.065,np.maximum(sigma*3,.06),np.maximum(sigma*.8,.02)]
    folds = np.arange(n)*5//n

    def subset(mask):
        matrix = design[mask].reshape(-1,design.shape[-1])
        target = y[mask].ravel()
        coef = np.linalg.solve(matrix.T@matrix+np.diag(1/scales**2),matrix.T@target)
        return np.asarray(row['prior_com_mm'])+coef[:3]

    full = subset(np.ones(n,bool))
    deletes = np.array([subset(folds != i) for i in range(5)]) if blocks else None
    return full,deletes


def _fit(rows):
    model = robust._dynamics(rows)
    x, _ = robust._coordinates(rows, model['normalization'])
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError('normal mass/motion coordinates not identifiable')
    points = np.stack([solve(row,model,False)[0][:2] for row in rows])
    return model, np.linalg.lstsq(x,points,rcond=None)[0]


def fit(rows):
    if len(rows) < 12:
        raise ValueError('diverse unshifted source normals required')
    model,normal = _fit(rows)
    loo, deletion = [],[]
    for i,row in enumerate(rows):
        part,other_normal=_fit(rows[:i]+rows[i+1:])
        point,deleted=solve(row,part,True)
        expected=robust._coordinate(row,part)@other_normal
        loo.append(point[:2]-expected)
        deletion.append(float(np.max(np.linalg.norm(deleted[:,:2]-point[:2],axis=1))))
    guard=float(np.max(np.linalg.norm(loo,axis=1)))
    radii=MARGIN+guard+np.asarray(deletion)
    doc={'schema':SCHEMA,'last_centre':int(rows[0]['centres'][-1]),
         'completed_at_s':float(rows[0]['completed_at_s']),'normal_guard_mm':guard,
         'normalization':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in model['normalization'].items()},
         'bias':model['bias'].tolist(),'sigma':model['sigma'].tolist(),'baseline_xy':normal.tolist()}
    summary={'last_centre_ms':doc['last_centre'],'decision_s':doc['completed_at_s'],
             'loo_guard_mm':guard,'worst_normal_radius_mm':float(max(radii)),
             'spurious_normals':int(np.sum(np.linalg.norm(loo,axis=1)>radii)),
             'normal_count':len(rows)}
    summary['passes']=bool(max(radii)<=MAX_RADIUS and summary['spurious_normals']==0)
    return doc,summary


def select(cells):
    for i in range(len(cells)-1):
        if cells[i]['passes'] and cells[i+1]['passes']:
            return i
    return None


def decide(row,doc):
    if doc['schema']!=SCHEMA or not np.array_equal(row['centres'], robust.USED_CENTRES[robust.USED_CENTRES<=doc['last_centre']]):
        raise ValueError('wrong causal grid or calibration')
    model={'normalization':{k:np.asarray(v) if isinstance(v,list) else v for k,v in doc['normalization'].items()},
           'bias':np.asarray(doc['bias']),'sigma':np.asarray(doc['sigma'])}
    coord=robust._coordinate(row,model)
    point,deletions=solve(row,model,True)
    rel=point[:2]-coord@np.asarray(doc['baseline_xy'])
    radius=float(MARGIN+doc['normal_guard_mm']+max(np.linalg.norm(deletions[:,:2]-point[:2],axis=1)))
    length=float(np.linalg.norm(rel))
    if not np.isfinite(np.r_[rel,radius,coord]).all():
        raise ValueError('nonfinite sensing result')
    norm=model['normalization']
    in_mass=bool(norm['mass_range'][0]<=float(row['mass'])<=norm['mass_range'][1])
    in_sig=bool(np.linalg.norm(coord[2:])<=norm['signature_radius_max'])
    n=len(row['centres'])
    t=(row['centres']-row['centres'].mean())*DT
    dc=np.broadcast_to(np.eye(6),(n,6,6))
    z=np.concatenate((row['stiffness'][:,:,None],dc,dc*t[:,None,None]),axis=2).reshape(-1,13)
    jac=row['jac'].reshape(-1,3)
    rank=int(np.linalg.matrix_rank(jac-z@np.linalg.lstsq(z,jac,rcond=None)[0]))
    accept=bool(rank==3 and in_mass and in_sig and radius<=MAX_RADIUS and length>radius)
    action=rel*min(1.,robust.MAX_ACTION_MM/max(length,1e-12)) if accept else np.zeros(2)
    return {'relative_xy_mm':rel.tolist(),'model_action_xy_mm':action.tolist(),'halfwidth_radius_mm':radius,
            'accept':accept,'mass_interpolated':in_mass,'signature_supported':in_sig,'projected_rank':rank,
            'completed_at_s':float(row['completed_at_s'])}
