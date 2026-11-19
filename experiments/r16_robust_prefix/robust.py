"""Measurement-conditioned normal-only horizontal COM estimator for R16."""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from experiments.r14_online.stream import Stream
from sim import config as C
from sim import trajectory as T
from sim.nominal import Nominal
from tools.r13_grasp_estimator import CENTRES, DT, HALF, WEIGHT, filtered_input

LAST_CENTRE = 3424
DEADLINE_INDEX = LAST_CENTRE + HALF + 3
MASK = CENTRES <= LAST_CENTRE
USED_CENTRES = CENTRES[MASK]
PRE_CENTRES = np.arange(300, 901, 20)
PHYSICAL_MARGIN_MM = .75
MAX_RADIUS_MM = 2.5
MAX_ACTION_MM = 15.


def _local(table, centres):
    ix = centres[:, None] + np.arange(-HALF, HALF+1)
    values = table[ix]
    avg = lambda a: np.einsum('s,nsj->nj', WEIGHT, a)
    return (avg(values), avg(table[ix+1]-table[ix-1])/(2*DT),
            avg((table[ix+1]-values)-(values-table[ix-1]))/(DT**2))


def _pregrasp_signature(t, encoder, jts, grip):
    nominal = Nominal()
    enc, disp, tau, _ = filtered_input(t, encoder, jts, grip, C.DT, nominal.K,
                                       C.D, nominal.elastic, C.JTS_CUTOFF)
    q, qd, qdd = _local(enc-disp, PRE_CENTRES)
    dq, dvd, dad = _local(disp, PRE_CENTRES)
    observed = _local(tau, PRE_CENTRES)[0]
    base = np.array([nominal.inverse_dynamics(a, b, c) for a, b, c in zip(q, qd, qdd)])
    plus = np.array([nominal.inverse_dynamics(a-.05*d, b-.05*e, c-.05*f)
                     for a, b, c, d, e, f in zip(q, qd, qdd, dq, dvd, dad)])
    minus = np.array([nominal.inverse_dynamics(a+.05*d, b+.05*e, c+.05*f)
                      for a, b, c, d, e, f in zip(q, qd, qdd, dq, dvd, dad)])
    stiffness = (plus-minus)/.10
    residual = observed-base
    taught = np.array([T.reference(float(k)*DT)[0] for k in PRE_CENTRES])
    taught_velocity = np.array([T.reference(float(k)*DT)[1] for k in PRE_CENTRES])
    tracking = _local(enc, PRE_CENTRES)[0]-taught
    dynamic = tracking-tracking.mean(axis=0)
    velocity = taught_velocity-taught_velocity.mean(axis=0)
    coulomb = np.tanh(taught_velocity/C.FRICTION_EPS)
    coulomb -= coulomb.mean(axis=0)
    project = lambda a, b: float(np.sum(a*b)/(np.sum(b*b)+1e-12))
    eye = np.broadcast_to(np.eye(6), (len(PRE_CENTRES), 6, 6))
    design = np.concatenate((stiffness[:, :, None], eye), axis=2).reshape(-1, 7)
    target = residual.ravel()
    scales = np.r_[.065, np.full(6, .15)]
    coef = np.linalg.solve(design.T@design+np.diag(1/scales**2), design.T@target)
    return np.array([coef[0], project(dynamic, velocity), project(dynamic, coulomb),
                     float(np.sqrt(np.mean(dynamic**2)))])


def feature(path):
    """Extract only causal public features through the fixed 3.327 s deadline."""
    with np.load(path, allow_pickle=False) as z:
        t = np.asarray(z['t']); encoder = np.asarray(z['encoder']); jts = np.asarray(z['jts'])
        grip = np.asarray(z['grip']); mass = float(z['mass']); com = np.asarray(z['com'])
        if len(t) <= DEADLINE_INDEX:
            raise ValueError('public record ends before R16 deadline')
        signature = _pregrasp_signature(t, encoder, jts, grip)
        stream = Stream(mass, com)
        for k in range(DEADLINE_INDEX+1):
            stream.observe(float(t[k]), encoder[k], jts[k], bool(grip[k]))
        if stream._next_centre <= int(np.flatnonzero(MASK)[-1]):
            raise ValueError('carried prefix incomplete')
        return {'mass': np.array(mass), 'y': stream._y[MASK].copy(),
                'jac': stream._jac[MASK].copy(), 'stiffness': stream._stiffness[MASK].copy(),
                'prior_com_mm': stream._prior_com_mm.copy(), 'signature': signature,
                'completed_at_s': np.array(DEADLINE_INDEX*DT)}


def save_feature(path, value):
    np.savez_compressed(path, **{key: np.asarray(item) for key, item in value.items()})


def load_feature(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def _coordinates(rows, normalization=None):
    mass = np.array([float(r['mass']) for r in rows])
    signatures = np.stack([r['signature'] for r in rows])
    if normalization is None:
        center = signatures.mean(axis=0)
        scale = np.maximum(signatures.std(axis=0), np.array([1e-4, 1e-5, 1e-5, 1e-5]))
        normalization = {'mass_center': float(mass.mean()), 'signature_center': center,
                         'signature_scale': scale, 'mass_range': [float(mass.min()), float(mass.max())],
                         'signature_radius_max': float(np.max(np.linalg.norm((signatures-center)/scale, axis=1)))}
    x = np.column_stack((np.ones(len(rows)), mass-normalization['mass_center'],
                         (signatures-normalization['signature_center'])/normalization['signature_scale']))
    return x, normalization


def _dynamics(rows, normalization=None):
    x, normalization = _coordinates(rows, normalization)
    ys = np.stack([r['y'] for r in rows])
    bias = np.linalg.lstsq(x, ys.reshape(len(rows), -1), rcond=None)[0].reshape(x.shape[1], *ys.shape[1:])
    error = ys-np.einsum('nk,ktj->ntj', x, bias)
    sigma = np.maximum(np.asarray(C.JTS_NOISE)*.22, np.median(abs(error), axis=(0, 1))*1.4826)
    return {'normalization': normalization, 'bias': bias, 'sigma': sigma}


def _coordinate(row, model):
    n = model['normalization']; sig = np.asarray(row['signature'])
    return np.r_[1., float(row['mass'])-n['mass_center'],
                 (sig-n['signature_center'])/n['signature_scale']]


def _solutions(row, model, deleted=True):
    coordinate = _coordinate(row, model)
    bias = np.einsum('k,ktj->tj', coordinate, model['bias'])
    sigma = np.asarray(model['sigma'])
    y = (row['y']-bias)/sigma
    n, joints = y.shape
    eye = np.broadcast_to(np.eye(joints), (n, joints, joints))
    time = (USED_CENTRES-USED_CENTRES.mean())*DT
    design = np.concatenate((row['jac'], row['stiffness'][:, :, None], eye,
                             eye*time[:, None, None]), axis=-1)/sigma[None, :, None]
    scales = np.r_[np.full(3, 15.), .065, np.maximum(sigma*3, .06),
                   np.maximum(sigma*.8, .02)]
    folds = np.arange(n)*5//n
    def solve(keep):
        matrix = design[keep].reshape(-1, design.shape[-1]); target = y[keep].ravel()
        coef = np.linalg.solve(matrix.T@matrix+np.diag(1/scales**2), matrix.T@target)
        return np.asarray(row['prior_com_mm'])+coef[:3]
    full = solve(np.ones(n, bool))
    deletions = np.array([solve(folds != fold) for fold in range(5)]) if deleted else None
    return full, deletions


def _fit(rows):
    dynamics = _dynamics(rows)
    coordinates, _ = _coordinates(rows, dynamics['normalization'])
    absolute = np.array([_solutions(row, dynamics, False)[0][:2] for row in rows])
    baseline = np.linalg.lstsq(coordinates, absolute, rcond=None)[0]
    return dynamics, baseline


def calibrate(rows):
    if len(rows) < 12:
        raise ValueError('at least 12 source-normal rows required')
    dynamics, baseline = _fit(rows)
    loo = []
    for index, row in enumerate(rows):
        subset = [item for j, item in enumerate(rows) if j != index]
        smaller, normal = _fit(subset)
        point = _solutions(row, smaller, False)[0][:2]
        loo.append(point-_coordinate(row, smaller)@normal)
    n = dynamics['normalization']
    return {'schema': 'r16-robust-prefix-calibration.v1',
            'dynamics': {'bias': dynamics['bias'].tolist(), 'sigma': dynamics['sigma'].tolist()},
            'normalization': {key: value.tolist() if isinstance(value, np.ndarray) else value
                              for key, value in n.items()},
            'baseline_xy': baseline.tolist(), 'normal_guard_mm': float(max(np.linalg.norm(v) for v in loo)),
            'loo_vectors_mm': np.asarray(loo).tolist(), 'deadline_s': DEADLINE_INDEX*DT}


def _model(document):
    if document.get('schema') != 'r16-robust-prefix-calibration.v1':
        raise ValueError('calibration schema')
    normalization = {key: np.asarray(value) if isinstance(value, list) else value
                     for key, value in document['normalization'].items()}
    return {'normalization': normalization, 'bias': np.asarray(document['dynamics']['bias']),
            'sigma': np.asarray(document['dynamics']['sigma'])}


def decide(row, document):
    model = _model(document); absolute, deleted = _solutions(row, model)
    coordinate = _coordinate(row, model); normal = coordinate@np.asarray(document['baseline_xy'])
    relative = absolute[:2]-normal; length = float(np.linalg.norm(relative))
    deletion = np.linalg.norm(deleted[:, :2]-absolute[:2], axis=1)
    radius = PHYSICAL_MARGIN_MM+float(document['normal_guard_mm'])+float(deletion.max())
    n = model['normalization']; signature = np.asarray(row['signature'])
    signature_radius = float(np.linalg.norm((signature-n['signature_center'])/n['signature_scale']))
    support = (n['mass_range'][0] <= float(row['mass']) <= n['mass_range'][1]
               and signature_radius <= n['signature_radius_max'])
    identifiable = bool(np.linalg.matrix_rank(np.stack(row['jac']).reshape(-1, 3)) == 3)
    accept = bool(support and identifiable and radius <= MAX_RADIUS_MM and length > radius)
    action = relative*min(1., MAX_ACTION_MM/max(length, 1e-12)) if accept else np.zeros(2)
    return {'relative_xy_mm': relative.tolist(), 'absolute_com_offset_mm': absolute.tolist(),
            'expected_normal_xy_mm': normal.tolist(), 'halfwidth_radius_mm': float(radius),
            'normal_guard_mm': float(document['normal_guard_mm']), 'block_delete_mm': deletion.tolist(),
            'measured_signature': signature.tolist(), 'mass_interpolated': bool(n['mass_range'][0] <= float(row['mass']) <= n['mass_range'][1]),
            'signature_standardized_radius': signature_radius,
            'signature_supported': bool(signature_radius <= n['signature_radius_max']),
            'identifiable': identifiable, 'accept': accept, 'model_action_xy_mm': action.tolist(),
            'completed_at_s': float(row['completed_at_s'])}


def dump(path: Path, document):
    with path.open('x') as stream:
        json.dump(document, stream, indent=2, allow_nan=False); stream.write('\n')
