"""Own derivation/tests of wrench-origin power duality; no teaching assets."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.check_integrity import check_integrity
from sim.nominal import Nominal
from sim import trajectory as T
import v1_detector


def skew(p):
    x,y,z = p
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def audit():
    rng = np.random.default_rng(369401)
    power_errors = []
    for _ in range(100):
        axis = rng.normal(size=3);axis /= np.linalg.norm(axis)
        angle = rng.uniform(-2.,2.)
        s = skew(axis);rotation = np.eye(3)+np.sin(angle)*s+(1-np.cos(angle))*(s@s)
        p = rng.normal(size=3)
        a = np.block([[rotation,skew(p)@rotation],[np.zeros((3,3)),rotation]])
        velocity, wrench = rng.normal(size=(2,6))
        moved_velocity = a@velocity
        moved_wrench = np.linalg.solve(a.T,wrench)
        power_errors.append(abs(float(velocity@wrench-moved_velocity@moved_wrench)))
    nom, full = Nominal(), Nominal(); full.set_payload(5.35, (.013,-.009,.027))
    core = v1_detector.SharedEstimator(nom,full,5.35,(.013,-.009,.027))
    origin_errors = []
    virtual_errors = []
    for t in (.2,1.39,1.51,2.45,3.81,4.2):
        q = T.reference(t)[0]
        jacobian, point, _ = core._point_jacobian(q)
        body = nom.m.body('link_6').id
        flange = nom.d.xpos[body].copy()
        jp,jr = np.zeros((3,6)),np.zeros((3,6))
        mujoco.mj_jac(nom.m,nom.d,jp,jr,flange,body)
        flange_jacobian = np.vstack((jp,jr))
        for force in (np.array([12.,-9.,5.]),np.array([0.,0.,30.])):
            bottom_wrench = np.r_[force,np.cross(flange-point,force)]
            tau = flange_jacobian.T@np.r_[force,np.zeros(3)]
            origin_errors.append(float(np.max(abs(tau-jacobian.T@bottom_wrench))))
            qd = rng.normal(size=6)
            virtual_errors.append(abs(float(tau@qd-bottom_wrench@(jacobian@qd))))
    q = T.reference(T.T_RELEASE)[0]
    jacobian,point,rotation = core._point_jacobian(q)
    direction = np.linalg.solve(jacobian,np.array([0.,0.,1.,0.,0.,0.]))
    geometry = []
    for displacement in (-.0015,-.0005,.0005,.0015):
        _, p1, r1 = core._point_jacobian(q+direction*displacement)
        geometry.append(dict(delta_m=displacement,
            tangent_error_m=float(np.linalg.norm((p1-point)[:2])),
            normal_linearization_error_m=float(abs(p1[2]-point[2]-displacement)),
            rotation_error_rad=float(np.arccos(np.clip((np.trace(rotation.T@r1)-1)/2,-1,1)))))
    # An instantaneous wrench has no source label. A vertical force shifted
    # along its own action line has exactly the same moment and joint torques.
    force = np.array([0.,0.,30.]); r = np.array([0.,0.,.2])
    table_wrench = np.r_[force,np.zeros(3)]
    collision_wrench = np.r_[force,np.cross(r,force)]
    equal = bool(np.array_equal(table_wrench,collision_wrench))
    result = dict(official=False, derived_from='public nominal model and own algebra; no plant truth',
                  power_duality_max_error=float(max(power_errors)),
                  shifted_origin_joint_torque_max_error=float(max(origin_errors)),
                  virtual_power_max_error=float(max(virtual_errors)),
                  local_normal_linearization=geometry,
                  identical_normal_wrench_counterexample=equal,
                  scope='instantaneous source non-identifiability; not a claim of all dynamic histories',
                  references='Park/Choi/Chung manuscript2024 sections2.2,6.1,7.2; Choi lecture2024 section11.5')
    assert max(power_errors)<1e-10 and max(origin_errors)<1e-10 and max(virtual_errors)<1e-10 and equal
    return result


def main():
    check_integrity(ROOT)
    out = audit()
    (ROOT/'results/v2_protocol/GEOMETRY_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
