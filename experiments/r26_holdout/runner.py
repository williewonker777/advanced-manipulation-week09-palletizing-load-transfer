"""R26 variable-duration signed force injector on the authenticated stock plant."""
from __future__ import annotations

import numpy as np

import sim.runner as stock
from sim.plant import Plant as StockPlant


def pulse(t, impact):
    if impact is None:
        return np.zeros(3)
    onset, duration_ms, vector = impact
    duration = float(duration_ms) / 1000.0
    if not 0.035 <= duration <= 0.23:
        raise ValueError("registered pulse duration outside 35--230 ms")
    ramp = min(0.018, duration / 4.0)
    age = float(t) - float(onset)
    weight = min(age / ramp, 1.0, (duration - age) / ramp)
    return max(weight, 0.0) * np.asarray(vector, dtype=float)


def expected(t, impacts):
    first = np.stack([pulse(value, impacts[0]) for value in t])
    second = np.stack([pulse(value, impacts[1]) for value in t])
    return {
        "first": first,
        "second": second,
        "committed": first + second,
        "scalar": np.linalg.norm(first, axis=1),
    }


def instrumented(impacts, variant):
    kind = variant["kind"]

    class R26Plant(StockPlant):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.r26_tick = 0
            self.r26_first = []
            self.r26_second = []
            self.r26_committed = []
            self.r26_gain = np.asarray(variant.get("joint_gain", np.ones(6)), dtype=float)
            self.r26_bias = np.asarray(variant.get("bias_Nm", np.zeros(6)), dtype=float)
            self.r26_deadzone = float(variant.get("deadzone_Nm", 0.0))
            if kind == "place_surface_shift":
                pick = self.m.geom("table_pick").id
                place = self.m.geom("table_place").id
                self.m.geom_friction[pick, 0] *= variant["pick_friction_scale"]
                self.m.geom_friction[place, 0] *= variant["place_friction_scale"]
                self.m.geom_solref[[pick, place], 0] *= variant["table_solref_time_scale"]
                self.m.dof_damping[:] *= variant["damping_scale"]

        def push(self, ignored_native):
            del ignored_native
            t = self.r26_tick * stock.C.DT
            first = pulse(t, impacts[0])
            second = pulse(t, impacts[1])
            committed = first + second
            super().push(committed)
            body = self.m.body("link_6").id
            self.r26_first.append(first.copy())
            self.r26_second.append(second.copy())
            self.r26_committed.append(self.d.xfrc_applied[body, :3].copy())
            self.r26_tick += 1

        def step(self, tau):
            adjusted = np.asarray(tau, dtype=float) * self.r26_gain + self.r26_bias
            if kind == "actuator_deadzone":
                adjusted = np.where(np.abs(adjusted) < self.r26_deadzone, 0.0, adjusted)
            return super().step(adjusted)

    return R26Plant
