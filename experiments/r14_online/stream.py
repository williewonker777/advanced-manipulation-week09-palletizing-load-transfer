"""Causal incremental equivalent of ``r13_grasp_estimator.public_features``."""
from __future__ import annotations

import time

import numpy as np

from sim import config as C, trajectory as T
from sim.nominal import Nominal
from tools.r13_grasp_estimator import CENTRES, DT, HALF, WEIGHT, flange


class Stream:
    """Accumulate public samples and finalize each fitting centre causally.

    A centre is evaluated when its final rigid-axis JTS reconstruction sample
    is available (centre + HALF + 3), so the last centre completes at 3.527 s.
    Nine independent public nominal models are constructed once and each has
    ``set_payload`` called exactly once.
    """

    CAPACITY = 3551

    def __init__(self, mass, com):
        self.mass = float(mass)
        self.com = np.asarray(com, dtype=float).copy()
        if (not np.isfinite(self.mass) or self.mass <= 0
                or self.com.shape != (3,) or not np.isfinite(self.com).all()):
            raise ValueError("known positive finite payload and 3-D COM required")
        self.geometry = Nominal()
        self.models = [Nominal() for _ in range(9)]
        self.encoder = np.zeros((self.CAPACITY, 6))
        self.jts = np.zeros_like(self.encoder)
        self.grip = np.zeros(self.CAPACITY, dtype=bool)
        self.sensor = np.zeros((self.CAPACITY-1, 6))
        self.tau = np.zeros((self.CAPACITY-2, 6))
        self.disp = np.zeros_like(self.tau)
        self.count = 0
        self.latch = None
        self.rising_edges = 0
        self._spring_previous = None
        self._disp_state = None
        self._models_ready = False
        self._next_centre = 0
        n = len(CENTRES)
        self._y = np.zeros((n, 6))
        self._jac = np.zeros((n, 6, 3))
        self._stiffness = np.zeros((n, 6))
        self._prior_bottom_mm = None
        self._prior_com_mm = None
        self._vertical = None
        self._observe_ns = []
        self._finish_ns = None
        self.completed_at_s = None
        self.inverse_dynamics_calls = 0
        self.payload_set_calls = 0
        self._inverse = 1./np.expm1(2*np.pi*C.JTS_CUTOFF*DT)
        elastic = np.asarray(self.geometry.elastic, bool)
        self._elastic = elastic
        self._tconst = np.asarray(C.D)[elastic]/np.asarray(self.geometry.K)[elastic]
        if np.any(self._tconst <= 0):
            raise ValueError("positive elastic time constants required")
        self._memory = np.exp(-DT/self._tconst)
        self._ramp = 1.-(self._tconst/DT)*(1.-self._memory)

    def _initialize_models(self):
        if self._models_ready or self.latch is None or self.latch >= len(self.disp):
            return
        pick, rot = flange(self.geometry, T.reference(T.T_GRASP)[0])
        meas, actual = flange(
            self.geometry, self.encoder[self.latch]-self.disp[self.latch]
        )
        _, placed = flange(self.geometry, T.reference(T.T_RELEASE)[0])
        box = np.asarray(C.BOX_POS)
        half = .5*np.asarray(C.BOX_SIZE)
        rel = actual.T@rot
        centre = actual.T@(pick+rot@box-meas)
        nominal_offset = centre+rel@self.com-box-self.com
        prior_bottom = (placed[2]@(centre-box)
                        - (abs(placed[2]@rel)-abs(placed[2]))@half)
        shifts = [np.zeros(3)]
        for axis in range(3):
            delta = np.zeros(3);delta[axis] = .001
            shifts.extend((delta, -delta))
        shifts.extend((np.zeros(3), np.zeros(3)))
        for model, shift in zip(self.models, shifts, strict=True):
            model.set_payload(self.mass, self.com+nominal_offset+shift)
            self.payload_set_calls += 1
        self._prior_bottom_mm = np.array(1000*prior_bottom)
        self._prior_com_mm = 1000*nominal_offset
        self._vertical = placed[2].copy()
        self._models_ready = True

    @staticmethod
    def _local_kinematics(table, centre):
        ix = centre+np.arange(-HALF, HALF+1)
        values = table[ix]
        avg = lambda values: np.einsum("s,sj->j", WEIGHT, values)
        return (avg(values), avg(table[ix+1]-table[ix-1])/(2*DT),
                avg((table[ix+1]-values)-(values-table[ix-1]))/(DT**2))

    def _process_centre(self, slot, centre):
        q0, v0, a0 = self._local_kinematics(
            self.encoder[:len(self.disp)]-self.disp, centre
        )
        dq, dv, da = self._local_kinematics(self.disp, centre)
        tau_mean = self._local_kinematics(self.tau, centre)[0]
        values = []
        for index, model in enumerate(self.models):
            rho = .05 if index == 7 else -.05 if index == 8 else 0.
            values.append(model.inverse_dynamics(
                q0-rho*dq, v0-rho*dv, a0-rho*da
            ))
        self.inverse_dynamics_calls += 9
        base = values[0]
        self._y[slot] = tau_mean-base
        for axis in range(3):
            self._jac[slot, :, axis] = .5*(values[1+2*axis]-values[2+2*axis])
        self._stiffness[slot] = (values[7]-values[8])/.10

    def observe(self, t, encoder, jts, grip):
        start = time.perf_counter_ns()
        try:
            k = self.count
            encoder = np.asarray(encoder, dtype=float)
            jts = np.asarray(jts, dtype=float)
            if k >= self.CAPACITY:
                raise ValueError("stream capacity exceeded")
            if (not np.isscalar(t) or not np.isfinite(t)
                    or not np.isclose(float(t), k*DT, rtol=0., atol=1e-12)
                    or encoder.shape != (6,) or jts.shape != (6,)
                    or not np.isfinite(encoder).all() or not np.isfinite(jts).all()
                    or grip not in (0, 1, False, True)):
                raise ValueError("expected sequential finite 1 ms public samples")
            self.encoder[k] = encoder
            self.jts[k] = jts
            self.grip[k] = bool(grip)
            rising = bool(grip) and (k == 0 or not self.grip[k-1])
            if rising:
                self.rising_edges += 1
                if self.rising_edges > 1:
                    raise ValueError("one prior grasp required")
                self.latch = k
            if k >= 1:
                self.sensor[k-1] = jts+self._inverse*(jts-self.jts[k-1])
            if k >= 2:
                j = k-2
                self.tau[j] = self.sensor[j]
                rigid = ~self._elastic
                self.tau[j, rigid] = .5*(self.sensor[j, rigid]+self.sensor[j+1, rigid])
                spring = self.tau[j, self._elastic]/np.asarray(self.geometry.K)[self._elastic]
                if j == 0:
                    self._disp_state = spring.copy()
                else:
                    previous = self._spring_previous
                    self._disp_state = (self._memory*self._disp_state
                                        +(1.-self._memory)*previous
                                        +self._ramp*(spring-previous))
                self._spring_previous = spring.copy()
                self.disp[j, self._elastic] = self._disp_state
                if self.latch is not None and j >= self.latch:
                    self._initialize_models()
                while (self._models_ready and self._next_centre < len(CENTRES)
                       and j >= CENTRES[self._next_centre]+HALF+1):
                    centre = int(CENTRES[self._next_centre])
                    if not self.grip[centre]:
                        raise ValueError("carried payload required in free-flight windows")
                    self._process_centre(self._next_centre, centre)
                    self._next_centre += 1
                    if self._next_centre == len(CENTRES):
                        self.completed_at_s = float(t)
            self.count += 1
            return self._next_centre
        finally:
            self._observe_ns.append(time.perf_counter_ns()-start)

    def finish(self):
        start = time.perf_counter_ns()
        try:
            if self.count < 3550:
                raise ValueError("public stream ended before the 3.55 s deadline")
            if (self.rising_edges != 1 or self.latch is None
                    or not 0 < self.latch < CENTRES[0]-HALF-2):
                raise ValueError("one prior grasp required")
            if self._next_centre != len(CENTRES):
                raise ValueError("public stream ended before all fitting centres")
            if np.any(~self.grip[CENTRES]):
                raise ValueError("carried payload required in free-flight windows")
            return {"mass": np.array(self.mass), "y": self._y.copy(),
                    "jac": self._jac.copy(), "stiffness": self._stiffness.copy(),
                    "prior_bottom_mm": self._prior_bottom_mm.copy(),
                    "prior_com_mm": self._prior_com_mm.copy(),
                    "vertical": self._vertical.copy()}
        finally:
            self._finish_ns = time.perf_counter_ns()-start

    @property
    def feature(self):
        return self.finish()

    @property
    def timing_us(self):
        values = np.asarray(self._observe_ns, dtype=float)/1000.
        return {"p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "max": float(values.max(initial=0.)),
                "finish": None if self._finish_ns is None else self._finish_ns/1000.}

    @property
    def state_bytes(self):
        arrays = (self.encoder, self.jts, self.grip, self.sensor, self.tau,
                  self.disp, self._y, self._jac, self._stiffness)
        return int(sum(array.nbytes for array in arrays))
