from experiments.r26_long.candidate import SignedYawOwnership


def test_one_event_bridges_route_dropout_and_reacquisition():
    state = SignedYawOwnership()
    support = [-3.0, 1.0]
    outputs = [state.feed(observed, value, support, True) for observed, value in [
        (True, -4.0), (True, -3.6), (False, -3.0), (False, -2.5), (True, -3.8)]]
    assert outputs == [False, False, True, True, False]
    assert state.active and state.mean < 0


def test_zero_closer_sample_releases_immediately():
    state = SignedYawOwnership()
    state.feed(True, 4.0, [2.0, 0.0], True)
    assert not state.feed(False, 1.9, [2.0, 0.0], True)
    assert not state.active


def test_opposite_sign_is_a_new_event_not_a_bridge():
    state = SignedYawOwnership()
    state.feed(True, -4.0, [-2.0, 0.0], True)
    assert not state.feed(False, 3.0, [-2.0, 0.0], True)
    assert not state.active
    state.feed(True, 3.2, [2.0, 0.0], True)
    assert state.active and state.mean > 0


def test_grip_switch_and_pickup_block_release():
    state = SignedYawOwnership()
    state.feed(True, -4.0, [-2.0, 0.0], False)
    assert not state.feed(False, -3.5, [-2.0, 0.0], True)
    assert not state.active
    state.feed(True, -4.0, [-2.0, 0.0], True)
    assert not state.feed(False, -3.5, [-2.0, 0.0], True, blocked=True)
    assert not state.active


def test_independent_support_must_agree_at_route_handoff():
    state = SignedYawOwnership()
    state.feed(True, -4.0, [-3.0, 0.0], True)
    assert not state.feed(False, -3.0, [2.0, 0.0], True)
    assert not state.active


def test_support_noise_does_not_split_established_identity():
    state = SignedYawOwnership()
    state.feed(True, -4.0, [-3.0, 0.0], True)
    assert state.feed(False, -3.0, [-2.5, 0.0], True)
    assert state.feed(False, -2.8, [3.0, 0.0], True)
