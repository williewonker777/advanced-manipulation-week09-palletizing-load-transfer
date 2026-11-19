from experiments.r20_event_bridge.state import HandoffState


def test_overlap_bridges_until_existing_clearance():
    state = HandoffState()
    assert state.update(True, True, False, 0, 8)
    assert state.update(False, False, False, 7, 8)
    assert not state.update(False, False, False, 8, 8)


def test_evidence_outside_event_does_not_start_bridge():
    state = HandoffState()
    assert not state.update(False, True, False, 0, 8)


def test_incumbent_takeover_ends_bridge_without_dropping_incumbent():
    state = HandoffState()
    assert state.update(True, True, False, 0, 8)
    assert not state.update(False, True, True, 0, 8)


def test_disarm_resets_bridge():
    state = HandoffState()
    assert state.update(True, True, False, 0, 8)
    assert not state.update(False, False, False, 0, 8, armed=False)
