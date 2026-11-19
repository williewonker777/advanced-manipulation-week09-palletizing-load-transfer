"""Open-gripper approach-contact witness composed with frozen observation memory."""
from __future__ import annotations

import mujoco
import numpy as np

from experiments.r28_contact_memory.candidate import Detector as Memory
from sim import trajectory as T


MIN_COMMAND_SPEED_M_S = 0.01
ONSET_FORCE_N = 5.0
RELEASE_FORCE_N = 2.5
PERSISTENCE = 3
REPORT_SAMPLES = 8


def reference_tcp_velocity(model, t):
    q, qd, _ = T.reference(float(t))
    model.d.qpos[:] = q
    mujoco.mj_kinematics(model.m, model.d)
    mujoco.mj_comPos(model.m, model.d)
    jacobian = np.zeros((3, 6))
    angular = np.zeros((3, 6))
    mujoco.mj_jacSite(model.m, model.d, jacobian, angular, model.m.site("tcp").id)
    return jacobian @ qd


class ApproachContactEpisode:
    """One report for a signed force opposing a live commanded approach."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.armed = True
        self.over = 0
        self.pulse = 0

    def feed(self, force, command_velocity, open_gripper, incumbent):
        force = np.asarray(force, float) if force is not None else None
        velocity = np.asarray(command_velocity, float)
        finite = (force is not None and force.shape == (3,) and velocity.shape == (3,)
                  and np.isfinite(force).all() and np.isfinite(velocity).all())
        speed = float(np.linalg.norm(velocity)) if finite else 0.0
        eligible = bool(open_gripper and finite and speed >= MIN_COMMAND_SPEED_M_S)
        if not eligible:
            self.over = 0
            self.pulse = 0
            self.armed = True
            return False

        opposing = -float(force @ velocity) / speed
        if opposing <= RELEASE_FORCE_N:
            self.armed = True
            self.over = 0
        elif self.armed and opposing > ONSET_FORCE_N:
            self.over += 1
            if self.over >= PERSISTENCE:
                self.armed = False
                self.over = 0
                if not incumbent:
                    self.pulse = REPORT_SAMPLES
        else:
            self.over = 0

        extra = self.pulse > 0
        self.pulse = max(0, self.pulse - 1)
        return extra


class Detector(Memory):
    """Memory unchanged, plus an optional direction-conditioned approach witness."""

    def reset(self):
        super().reset()
        self.approach_episode = ApproachContactEpisode()
        self.approach_force_n = None
        self.approach_opposing_n = None
        self.approach_extra = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
        force = self.grip_force_value
        velocity = reference_tcp_velocity(self.nom, t)
        speed = float(np.linalg.norm(velocity))
        self.approach_force_n = None if force is None else np.asarray(force, float).copy()
        self.approach_opposing_n = (
            None if force is None or speed < MIN_COMMAND_SPEED_M_S
            else -float(np.asarray(force, float) @ velocity) / speed
        )
        self.approach_extra = self.approach_episode.feed(
            force, velocity, not bool(grip), bool(incumbent))
        return residual, bool(incumbent or self.approach_extra)
