"""Physically deferred one-shot notice for an otherwise unowned pickup pulse."""
from __future__ import annotations

import numpy as np

from experiments.r28_contact_memory.candidate import Detector as Memory
from experiments.r30_g4_pickup_attempt.candidate import tcp_velocities


MIN_COMMAND_SPEED_M_S = 0.01
REVERSE_PROGRESS_M_S = -0.005
REVERSE_PERSISTENCE = 3
ONSET_FORCE_N = 5.0
RELEASE_FORCE_N = 2.5
PERSISTENCE = 3
REPORT_SAMPLES = 8


class DeferredPickupNotice:
    def __init__(self):
        self.reset()

    def reset(self):
        self.started = False
        self.retired = False
        self.native_owned = False
        self.reverse = 0
        self.pending = False
        self.over = 0
        self.below = 0
        self.pulse = 0

    def feed(self, force, command, measured, grip, incumbent):
        if self.pulse:
            self.pulse -= 1
            return True
        command = np.asarray(command, float); measured = np.asarray(measured, float)
        force = None if force is None else np.asarray(force, float)
        kinematics = (command.shape == (3,) and measured.shape == (3,)
                      and np.isfinite(command).all() and np.isfinite(measured).all())
        observed = force is not None and force.shape == (3,) and np.isfinite(force).all()
        speed = float(np.linalg.norm(command)) if kinematics else 0.0
        progress = float(measured @ command / speed) if kinematics and speed > 0 else 0.0

        if grip or not kinematics:
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
            self.native_owned = True

        if self.retired:
            self.pending = False; self.pulse = 0
            return False
        if not self.started or not observed:
            self.over = 0
            return False

        opposing = -float(force @ command) / speed
        if not self.pending:
            if self.native_owned:
                return False
            self.over = self.over + 1 if opposing > ONSET_FORCE_N else 0
            if self.over >= PERSISTENCE:
                self.pending = True
                self.over = 0
                self.below = 0
            return False

        self.below = self.below + 1 if opposing <= RELEASE_FORCE_N else 0
        if self.below < PERSISTENCE:
            return False
        self.pending = False
        self.retired = True
        if self.native_owned:
            return False
        self.pulse = REPORT_SAMPLES - 1
        return True


class Detector(Memory):
    def reset(self):
        super().reset()
        self.deferred_pickup = DeferredPickupNotice()
        self.unload_extra = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
        command, measured = tcp_velocities(self.nom, t, th, thd)
        self.unload_extra = self.deferred_pickup.feed(
            self.grip_force_value, command, measured, bool(grip), bool(incumbent))
        return residual, bool(incumbent or self.unload_extra)
