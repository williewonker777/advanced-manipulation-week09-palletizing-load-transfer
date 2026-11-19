import numpy as np

from experiments.r30_g4_unload_notice.candidate import DeferredPickupNotice, REPORT_SAMPLES


COMMAND = np.array([0.0, 0.0, -0.02])
PROGRESS = np.array([0.0, 0.0, -0.01])
OVER = np.array([0.0, 0.0, 6.0])
BELOW = np.array([0.0, 0.0, 2.0])


def start_pending(item):
    item.feed(np.zeros(3), COMMAND, PROGRESS, False, False)
    for _ in range(3): assert not item.feed(OVER, COMMAND, PROGRESS, False, False)
    assert item.pending


def test_unowned_pulse_reports_only_at_physical_release():
    item = DeferredPickupNotice(); start_pending(item)
    assert not any(item.feed(OVER, COMMAND, PROGRESS, False, False) for _ in range(20))
    assert not item.feed(BELOW, COMMAND, PROGRESS, False, False)
    assert not item.feed(BELOW, COMMAND, PROGRESS, False, False)
    assert item.feed(BELOW, COMMAND, PROGRESS, False, False)
    assert sum(item.feed(BELOW, COMMAND, PROGRESS, False, False) for _ in range(20)) == REPORT_SAMPLES - 1


def test_native_during_pending_consumes_release_notice():
    item = DeferredPickupNotice(); start_pending(item)
    assert not item.feed(OVER, COMMAND, PROGRESS, False, True)
    assert not any(item.feed(BELOW, COMMAND, PROGRESS, False, False) for _ in range(6))


def test_prior_native_consumes_all_later_pickup_pulses():
    item = DeferredPickupNotice()
    assert not item.feed(np.zeros(3), COMMAND, PROGRESS, False, True)
    assert not any(item.feed(OVER, COMMAND, PROGRESS, False, False) for _ in range(10))


def test_grip_and_invalid_kinematics_retire_without_notice():
    for grip, command in ((True, COMMAND), (False, np.full(3, np.nan))):
        item = DeferredPickupNotice(); start_pending(item)
        assert not item.feed(OVER, command, PROGRESS, grip, False)
        assert item.retired
