import numpy as np

from experiments.r30_g4_tool_innovation.candidate import (
    ApproachContactEpisode,
    REPORT_SAMPLES,
)


def test_signed_opposition_requires_persistence_and_rearms_after_release():
    event = ApproachContactEpisode()
    velocity = np.array([0.0, 0.0, -0.02])
    force = np.array([0.0, 0.0, 6.0])
    assert not event.feed(force, velocity, True, False)
    assert not event.feed(force, velocity, True, False)
    assert event.feed(force, velocity, True, False)
    assert sum(event.feed(force, velocity, True, False) for _ in range(20)) == REPORT_SAMPLES - 1
    assert not event.feed(np.zeros(3), velocity, True, False)
    assert not event.feed(force, velocity, True, False)
    assert not event.feed(force, velocity, True, False)
    assert event.feed(force, velocity, True, False)


def test_wrong_direction_closed_grip_and_incumbent_do_not_add_report():
    velocity = np.array([0.0, 0.0, -0.02])
    for force, open_gripper, incumbent in (
        (np.array([0.0, 0.0, -20.0]), True, False),
        (np.array([0.0, 0.0, 20.0]), False, False),
        (np.array([0.0, 0.0, 20.0]), True, True),
    ):
        event = ApproachContactEpisode()
        assert not any(event.feed(force, velocity, open_gripper, incumbent) for _ in range(6))


def test_invalid_or_stationary_input_fails_closed():
    event = ApproachContactEpisode()
    assert not event.feed(None, np.zeros(3), True, False)
    assert not event.feed(np.array([np.nan, 0.0, 0.0]), np.ones(3), True, False)
