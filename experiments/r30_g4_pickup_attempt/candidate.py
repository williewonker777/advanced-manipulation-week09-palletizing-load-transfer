"""Initial-pickup ownership for the unchanged R30 approach-contact witness."""
from __future__ import annotations

import mujoco
import numpy as np

from experiments.r28_contact_memory.candidate import Detector as Memory
from sim import trajectory as T


MIN_COMMAND_SPEED_M_S = 0.01
REVERSE_PROGRESS_M_S = -0.005
REVERSE_PERSISTENCE = 3
ONSET_FORCE_N = 5.0
RELEASE_FORCE_N = 2.5
PERSISTENCE = 3
REPORT_SAMPLES = 8


def tcp_velocities(model, t, encoder, encoder_velocity):
    q_ref, qd_ref, _ = T.reference(float(t))
    jp = np.zeros((3, 6)); jw = np.zeros((3, 6))
    model.d.qpos[:] = q_ref
    mujoco.mj_kinematics(model.m, model.d); mujoco.mj_comPos(model.m, model.d)
    mujoco.mj_jacSite(model.m, model.d, jp, jw, model.m.site("tcp").id)
    command = jp @ qd_ref
    model.d.qpos[:] = encoder
    mujoco.mj_kinematics(model.m, model.d); mujoco.mj_comPos(model.m, model.d)
    mujoco.mj_jacSite(model.m, model.d, jp, jw, model.m.site("tcp").id)
    measured = jp @ encoder_velocity
    return command, measured


class PickupOwner:
    """At most one optional notice in the first uninterrupted open-grip motion."""

    def reset(self):
        self.started = False
        self.retired = False
        self.consumed = False
        self.reverse = 0
        self.armed = True
        self.over = 0
        self.pulse = 0

    def __init__(self):
        self.reset()

    def feed(self, force, command, measured, grip, incumbent):
        command = np.asarray(command, float); measured = np.asarray(measured, float)
        force = None if force is None else np.asarray(force, float)
        kinematics_finite = (command.shape == (3,) and measured.shape == (3,)
                             and np.isfinite(command).all() and np.isfinite(measured).all())
        force_finite = (force is not None and force.shape == (3,)
                        and np.isfinite(force).all())
        speed = float(np.linalg.norm(command)) if kinematics_finite else 0.0
        progress = (float(measured @ command / speed)
                    if kinematics_finite and speed > 0 else 0.0)

        if grip or not kinematics_finite:
            self.retired = True
        elif not self.started:
            self.started = speed >= MIN_COMMAND_SPEED_M_S and progress >= REVERSE_PROGRESS_M_S
        else:
            if speed < MIN_COMMAND_SPEED_M_S:
                self.retired = True
            self.reverse = self.reverse + 1 if progress < REVERSE_PROGRESS_M_S else 0
            if self.reverse >= REVERSE_PERSISTENCE:
                self.retired = True
        if incumbent and self.started and not self.retired:
            self.consumed = True

        if self.retired:
            self.pulse = 0
            return False
        if self.pulse:
            self.pulse -= 1
            return True
        if not self.started or self.consumed:
            return False
        if not force_finite:
            self.over = 0
            return False

        opposing = -float(force @ command) / speed
        if opposing <= RELEASE_FORCE_N:
            self.armed = True
            self.over = 0
        elif self.armed and opposing > ONSET_FORCE_N:
            self.over += 1
            if self.over >= PERSISTENCE:
                self.armed = False
                self.over = 0
                self.consumed = True
                self.pulse = REPORT_SAMPLES - 1
                return True
        else:
            self.over = 0
        return False


class Detector(Memory):
    def reset(self):
        super().reset()
        self.pickup_owner = PickupOwner()
        self.pickup_extra = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
        command, measured = tcp_velocities(self.nom, t, th, thd)
        self.pickup_extra = self.pickup_owner.feed(
            self.grip_force_value, command, measured, bool(grip), bool(incumbent))
        return residual, bool(incumbent or self.pickup_extra)
