"""Metric-compatible partial RNE and R2 momentum-observer candidate.

The spatial convention is angular-first ``[omega, linear_velocity]``.  For
actual velocity ``v`` and design velocity ``u``, :class:`PRNE` computes the
bilinear action ``C(q, v) u`` in one forward/backward pass.  ``transpose`` is
the reverse-mode product ``C(q, v).T @ y``; it does not form six basis
columns.  Fixed tool/payload bodies are folded into their actuated parent
inertia, so the recursion has exactly six moving links.

This is a metric-compatible Coriolis convention, not a claim that the unique
Christoffel convention is used.  Its defining checks are
``Mdot = C + C.T`` and ``C(v)v = bias-gravity``.
"""
from __future__ import annotations

import numpy as np
import mujoco

import r1_detector


def _skew(x):
    x = np.asarray(x, dtype=float)
    return np.array(((0.0, -x[2], x[1]),
                     (x[2], 0.0, -x[0]),
                     (-x[1], x[0], 0.0)))


def _ad(twist):
    """Motion cross-product matrix for angular-first twists."""
    twist = np.asarray(twist, dtype=float)
    out = np.zeros((6, 6))
    out[:3, :3] = _skew(twist[:3])
    out[3:, :3] = _skew(twist[3:])
    out[3:, 3:] = out[:3, :3]
    return out


def _motion_transform(parent_rotation, parent_position,
                      child_rotation, child_position):
    """Map a parent-body twist to the child body origin/frame."""
    rotation = child_rotation.T @ parent_rotation
    offset_parent = parent_rotation.T @ (child_position - parent_position)
    out = np.zeros((6, 6))
    out[:3, :3] = rotation
    out[3:, 3:] = rotation
    out[3:, :3] = -rotation @ _skew(offset_parent)
    return out


def _body_inertia(model, body):
    """Angular-first spatial inertia about a MuJoCo body origin."""
    mass = float(model.body_mass[body])
    centre = np.asarray(model.body_ipos[body], dtype=float)
    rotation_flat = np.empty(9)
    mujoco.mju_quat2Mat(rotation_flat, model.body_iquat[body])
    rotation = rotation_flat.reshape(3, 3)
    central = rotation @ np.diag(model.body_inertia[body]) @ rotation.T
    cross = _skew(centre)
    out = np.zeros((6, 6))
    out[:3, :3] = central - mass * cross @ cross
    out[:3, 3:] = mass * cross
    out[3:, :3] = -mass * cross
    out[3:, 3:] = mass * np.eye(3)
    return out


class PRNE:
    """Six-link O(n) pRNE action and reverse transpose action."""

    def __init__(self, nominal):
        self.nominal = nominal
        self.model = nominal.m
        if self.model.nv != 6 or self.model.njnt != 6:
            raise ValueError("R2 pRNE requires the public six-hinge nominal model")
        self.bodies = tuple(int(self.model.jnt_bodyid[j]) for j in range(6))
        if any(int(self.model.jnt_type[j]) != mujoco.mjtJoint.mjJNT_HINGE
               for j in range(6)):
            raise ValueError("R2 pRNE supports hinge joints only")
        if any(int(self.model.body_parentid[self.bodies[j]]) !=
               (0 if j == 0 else self.bodies[j - 1]) for j in range(6)):
            raise ValueError("R2 pRNE requires a serial six-link chain")
        self.motion_subspaces = tuple(
            np.r_[self.model.jnt_axis[j],
                  -np.cross(self.model.jnt_axis[j], self.model.jnt_pos[j])]
            for j in range(6)
        )
        self.inertias = self._fold_fixed_inertias()

    def _fold_fixed_inertias(self):
        """Fold fixed descendants (tool and supplied payload) into link six."""
        data = mujoco.MjData(self.model)
        data.qpos[:] = 0.0
        mujoco.mj_forward(self.model, data)
        inertias = [_body_inertia(self.model, body) for body in self.bodies]
        body_to_link = {body: index for index, body in enumerate(self.bodies)}
        # This kit has only immediate fixed children.  Walking ancestors keeps
        # the code correct if another fixed frame is inserted below a link.
        for body in range(1, self.model.nbody):
            if body in body_to_link or self.model.body_jntnum[body]:
                continue
            ancestor = int(self.model.body_parentid[body])
            while ancestor not in body_to_link and ancestor != 0:
                ancestor = int(self.model.body_parentid[ancestor])
            if ancestor == 0:
                continue
            link = body_to_link[ancestor]
            parent_rotation = data.xmat[ancestor].reshape(3, 3)
            child_rotation = data.xmat[body].reshape(3, 3)
            transform = _motion_transform(
                parent_rotation, data.xpos[ancestor],
                child_rotation, data.xpos[body])
            inertias[link] += transform.T @ _body_inertia(self.model, body) @ transform
        return tuple(inertias)

    def _kinematics(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("q must be a finite six-vector")
        self.nominal._update(q, np.zeros(6))
        data = self.nominal.d
        transforms = []
        parent_rotation = np.eye(3)
        parent_position = np.zeros(3)
        for body in self.bodies:
            rotation = data.xmat[body].reshape(3, 3).copy()
            position = data.xpos[body].copy()
            transforms.append(_motion_transform(
                parent_rotation, parent_position, rotation, position))
            parent_rotation, parent_position = rotation, position
        return transforms

    @staticmethod
    def _vector(value, name):
        value = np.asarray(value, dtype=float)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be a finite six-vector")
        return value

    def _forward(self, q, velocity, design):
        velocity = self._vector(velocity, "velocity")
        design = self._vector(design, "design")
        transforms = self._kinematics(q)
        twists, designs, partials, parent_twists = [], [], [], []
        parent_twist = np.zeros(6)
        parent_design = np.zeros(6)
        parent_partial = np.zeros(6)
        for index, (transform, subspace) in enumerate(
                zip(transforms, self.motion_subspaces, strict=True)):
            transported = transform @ parent_twist
            twist = transported + subspace * velocity[index]
            design_twist = transform @ parent_design + subspace * design[index]
            # Corrected partial derivative: the transport term uses only the
            # parent contribution, not the full child twist.
            partial = (transform @ parent_partial
                       + _ad(transported) @ (subspace * design[index]))
            parent_twists.append(transported)
            twists.append(twist)
            designs.append(design_twist)
            partials.append(partial)
            parent_twist, parent_design, parent_partial = (
                twist, design_twist, partial)
        return transforms, twists, designs, partials, parent_twists

    def action(self, q, velocity, design):
        """Return ``C(q, velocity) @ design`` without forming ``C``."""
        transforms, twists, designs, partials, _ = self._forward(
            q, velocity, design)
        forces = [
            inertia @ partial - _ad(twist).T @ (inertia @ design_twist)
            for inertia, twist, design_twist, partial in zip(
                self.inertias, twists, designs, partials, strict=True)
        ]
        result = np.zeros(6)
        for index in range(5, -1, -1):
            result[index] = self.motion_subspaces[index] @ forces[index]
            if index:
                forces[index - 1] += transforms[index].T @ forces[index]
        return result

    def transpose(self, q, velocity, dual):
        """Return ``C(q, velocity).T @ dual`` by reverse recursion."""
        dual = self._vector(dual, "dual")
        # The forward design can be zero because all Jacobians below are
        # bilinear in design; actual twists and transforms still depend on v.
        transforms, twists, _, _, parent_twists = self._forward(
            q, velocity, np.zeros(6))
        force_adjoints = []
        parent = np.zeros(6)
        for index, (transform, subspace) in enumerate(
                zip(transforms, self.motion_subspaces, strict=True)):
            parent = transform @ parent + subspace * dual[index]
            force_adjoints.append(parent.copy())
        partial_adjoints = [
            inertia @ adjoint
            for inertia, adjoint in zip(self.inertias, force_adjoints, strict=True)
        ]
        design_adjoints = [
            -inertia @ (_ad(twist) @ adjoint)
            for inertia, twist, adjoint in zip(
                self.inertias, twists, force_adjoints, strict=True)
        ]
        result = np.zeros(6)
        for index in range(5, -1, -1):
            subspace = self.motion_subspaces[index]
            result[index] = (
                (_ad(parent_twists[index]) @ subspace) @ partial_adjoints[index]
                + subspace @ design_adjoints[index]
            )
            if index:
                partial_adjoints[index - 1] += (
                    transforms[index].T @ partial_adjoints[index])
                design_adjoints[index - 1] += (
                    transforms[index].T @ design_adjoints[index])
        return result

    def matrix(self, q, velocity):
        """Test/oracle helper; runtime paths should use action/transpose."""
        return np.column_stack([
            self.action(q, velocity, np.eye(6)[column])
            for column in range(6)
        ])


class PRNESharedEstimator(r1_detector.SharedEstimator):
    """R1 estimator with only the momentum-observer beta term replaced."""

    def reset(self):
        super().reset()
        self._prne_empty = PRNE(self.nom)
        self._prne_payload = PRNE(self.nom_box)

    def _observer(self, q, v, tau, nom, grip):
        mass = nom.mass_matrix(q)
        momentum = mass @ v
        self.last_grip = bool(grip)
        if self.integral is None:
            self.integral = momentum.copy()
            self.mass_prev = mass.copy()
        self.mass_prev = mass.copy()  # retained for state/schema parity
        prne = self._prne_payload if nom is self.nom_box else self._prne_empty
        # h - Mdot*v = g + C*v - (C+C.T)*v = g - C.T*v.
        beta = nom.gravity(q) - prne.transpose(q, v, v)
        self.integral += (tau - beta + self.r_joint) * self.dt
        self.r_joint = self.observer_gain * (momentum - self.integral)


def make_detector(parameters=None):
    """Build a controller-independent pRNE detector class.

    ``variant`` is ``"prne_only"`` or ``"prne_plus_ridge"``.  Optional
    ``ridge`` is a mapping by architecture with ``mean/scale/weights/threshold``
    and is intended for root's collision-blind normal-only recalibration.
    """
    parameters = {} if parameters is None else dict(parameters)
    variant = parameters.get("variant", "prne_only")
    if variant not in {"prne_only", "prne_plus_ridge"}:
        raise ValueError("variant must be prne_only or prne_plus_ridge")
    ridge = parameters.get("ridge", r1_detector.MODEL_PARAMETERS)

    class Detector(PRNESharedEstimator):
        USE_SHARED = False
        R2_VARIANT = variant

        def reset(self):
            super().reset()
            mode = getattr(getattr(self, "ctrl", None), "MODE", None)
            known = variant == "prne_plus_ridge" and mode in ridge
            self.normal_model = (
                r1_detector.NormalResidualModel(**ridge[mode]) if known else None)

        def update(self, t, th, thd, tau_m, tau_s, grip):
            del tau_m
            packet = PRNESharedEstimator.update(
                self, t, th, thd, tau_s, grip)
            extra = False
            if self.normal_model is not None:
                extra = self.normal_model.update(
                    t, self.box_mass, self.box_com, packet.last_wrench,
                    r1_detector.outside_table_height_band(packet, grip))
            return packet.r_joint.copy(), bool(packet.detected or extra)

    Detector.__name__ = "R2PRNEDetector"
    return Detector


Detector = make_detector()


__all__ = ["PRNE", "PRNESharedEstimator", "make_detector", "Detector"]
