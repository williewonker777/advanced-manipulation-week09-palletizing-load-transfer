"""Closure-local variable-duration signed force injector on the stock plant."""
from __future__ import annotations

import numpy as np

import sim.runner as stock
from sim.plant import Plant as StockPlant


def pulse(t, impact):
    if impact is None:
        return np.zeros(3)
    onset, duration_ms, vector = impact
    duration = float(duration_ms) / 1000.0
    if not 0.04 <= duration <= 0.24:
        raise ValueError("registered pulse duration outside 40--240 ms")
    ramp = min(0.02, duration / 4.0)
    age = float(t) - float(onset)
    weight = min(age / ramp, 1.0, (duration - age) / ramp)
    return max(weight, 0.0) * np.asarray(vector, dtype=float)


def expected(t, impacts):
    first = np.stack([pulse(value, impacts[0]) for value in t])
    second = np.stack([pulse(value, impacts[1]) for value in t])
    return {"first": first, "second": second, "committed": first + second,
            "scalar": np.linalg.norm(first, axis=1)}


def instrumented(impacts, variant):
    kind = variant["kind"]

    class VariablePulsePlant(StockPlant):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.r25_tick = 0
            self.r25_first = []
            self.r25_second = []
            self.r25_committed = []
            self.r25_gain = np.asarray(variant.get("joint_gain", np.ones(6)), dtype=float)
            if kind == "contact_shift":
                tables = [self.m.geom(name).id for name in ("table_pick", "table_place")]
                if self.has_box:
                    box = self.m.body("box_free").id
                    tables.extend(np.flatnonzero(self.m.geom_bodyid == box).tolist())
                self.m.geom_friction[tables, 0] *= variant["table_friction_scale"]
                self.m.geom_solref[tables, 0] *= variant["table_solref_time_scale"]
                if self.m.neq:
                    self.m.eq_solref[:, 0] *= variant["suction_solref_time_scale"]

        def push(self, ignored_native):
            t = self.r25_tick * stock.C.DT
            first = pulse(t, impacts[0])
            second = pulse(t, impacts[1])
            committed = first + second
            super().push(committed)
            body = self.m.body("link_6").id
            self.r25_first.append(first.copy())
            self.r25_second.append(second.copy())
            self.r25_committed.append(self.d.xfrc_applied[body, :3].copy())
            self.r25_tick += 1

        def step(self, tau):
            return super().step(np.asarray(tau, dtype=float) * self.r25_gain)

    return VariablePulsePlant
