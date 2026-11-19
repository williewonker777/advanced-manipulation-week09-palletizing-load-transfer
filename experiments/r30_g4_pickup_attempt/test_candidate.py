import numpy as np

from experiments.r30_g4_pickup_attempt.candidate import PickupOwner, REPORT_SAMPLES


COMMAND = np.array([0.0, 0.0, -0.02])
PROGRESS = np.array([0.0, 0.0, -0.01])
CONTACT = np.array([0.0, 0.0, 6.0])


def test_weak_contact_reports_once_and_consumes_pickup():
    owner = PickupOwner()
    assert not owner.feed(np.zeros(3), COMMAND, PROGRESS, False, False)
    assert not owner.feed(CONTACT, COMMAND, PROGRESS, False, False)
    assert not owner.feed(CONTACT, COMMAND, PROGRESS, False, False)
    assert owner.feed(CONTACT, COMMAND, PROGRESS, False, False)
    assert sum(owner.feed(CONTACT, COMMAND, PROGRESS, False, False) for _ in range(20)) == REPORT_SAMPLES - 1
    assert not any(owner.feed(np.zeros(3), COMMAND, PROGRESS, False, False) for _ in range(10))


def test_native_event_consumes_later_optional_contact():
    owner = PickupOwner()
    assert not owner.feed(np.zeros(3), COMMAND, PROGRESS, False, False)
    assert not owner.feed(CONTACT, COMMAND, PROGRESS, False, True)
    assert not any(owner.feed(CONTACT, COMMAND, PROGRESS, False, False) for _ in range(10))


def test_grip_or_sustained_reversal_permanently_retires_owner():
    for transition in ("grip", "reverse"):
        owner = PickupOwner()
        owner.feed(np.zeros(3), COMMAND, PROGRESS, False, False)
        if transition == "grip":
            owner.feed(np.zeros(3), COMMAND, PROGRESS, True, False)
        else:
            reverse = -PROGRESS
            for _ in range(3): owner.feed(np.zeros(3), COMMAND, reverse, False, False)
        assert not any(owner.feed(CONTACT, COMMAND, PROGRESS, False, False) for _ in range(10))


def test_invalid_input_fails_closed():
    owner = PickupOwner()
    assert not owner.feed(None, COMMAND, PROGRESS, False, False)
    assert owner.started and not owner.retired
    assert not owner.feed(np.zeros(3), np.full(3, np.nan), PROGRESS, False, False)
    assert owner.retired
