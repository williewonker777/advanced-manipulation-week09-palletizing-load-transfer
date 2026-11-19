"""Deterministically generate the sealed fresh loaded pair panel."""
from __future__ import annotations

import random


SEED = 310_131


def jitter_vector(rng, base):
    return [round(value + rng.uniform(-0.35, 0.35), 2) for value in base]


def build_cases():
    rng = random.Random(SEED)
    templates = [
        ("orthogonal_close", 1.72, 92, [22.0, 2.0, 0.8], [-2.0, -24.0, 1.2], (1.2, 2.4)),
        ("signed_opposite", 2.05, 116, [0.8, 26.0, -1.0], [-0.7, -25.0, 1.3], (2.6, 3.8)),
        ("near_collinear", 2.39, 136, [20.0, 8.0, 1.4], [17.0, 13.0, -1.1], (4.0, 5.1)),
        ("oblique_reversal", 2.74, 151, [-24.0, 8.0, 1.0], [16.0, -21.0, -1.4], (5.3, 6.4)),
        ("mixed_3d", 3.08, 164, [12.0, 16.0, 10.0], [-15.0, 9.0, -12.0], (6.6, 7.7)),
    ]
    cases = []
    for index, (family, base_time, base_gap, first_base, second_base, mass_range) in enumerate(templates):
        start = round(base_time + rng.uniform(-0.018, 0.018), 3)
        gap_ms = base_gap + rng.randint(-4, 4)
        first_duration = rng.randint(43, 59)
        second_duration = rng.randint(42, 56)
        first = jitter_vector(rng, first_base)
        second = jitter_vector(rng, second_base)
        continuous = [round(value * 0.92, 2) for value in first]
        span = gap_ms + second_duration
        if span > 230 or first_duration >= gap_ms:
            raise AssertionError("fresh pair support design differs")
        cases.append({
            "id": f"r31f_{index}_{family}",
            "family": family,
            "mass": round(rng.uniform(*mass_range), 2),
            "com": [round(rng.uniform(-0.032, 0.032), 3),
                    round(rng.uniform(-0.031, 0.031), 3),
                    round(rng.uniform(0.013, 0.051), 3)],
            "offset": [round(rng.uniform(-0.0017, 0.0017), 4),
                       round(rng.uniform(-0.0017, 0.0017), 4),
                       round(rng.uniform(0.0003, 0.0012), 4)],
            "long": [start, span, continuous],
            "impacts": [[start, first_duration, first],
                        [round(start + gap_ms / 1000.0, 3), second_duration, second]],
        })
    return cases
