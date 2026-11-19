import numpy as np

from experiments.r31_servo_release.candidate import (
    DeferredEpisode, REPORT_SAMPLES, coherent_horizontal,
)


def test_coherence_requires_two_aligned_observers_and_no_support():
    assert coherent_horizontal(np.array([0., 6., 0.]), np.array([0., 7., 0., 0., 0., 0.]), False)[0]
    assert not coherent_horizontal(np.array([0., 6., 0.]), np.array([0., -7., 0., 0., 0., 0.]), False)[0]
    assert not coherent_horizontal(np.array([0., 6., 0.]), np.array([0., 7., 0., 0., 0., 0.]), True)[0]


def test_unowned_episode_reports_only_after_valid_release():
    item=DeferredEpisode()
    for _ in range(3): assert not item.feed(True,True,False)
    assert item.pending
    for _ in range(10): assert not item.feed(True,True,False)
    for _ in range(4): assert not item.feed(False,False,False)
    assert item.pending
    assert not item.feed(False,True,False);assert not item.feed(False,True,False)
    assert item.feed(False,True,False)
    assert sum(item.feed(False,True,False) for _ in range(20))==REPORT_SAMPLES-1


def test_native_during_episode_consumes_notice():
    item=DeferredEpisode()
    for _ in range(3): item.feed(True,True,False)
    assert not item.feed(True,True,True)
    assert not any(item.feed(False,True,False) for _ in range(5))
