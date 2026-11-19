import numpy as np

from experiments.r21_full_protection import runner_v2 as raw
from experiments.r31_single_counterfactual.scorer import arbitrary_onset


def test_arbitrary_synthetic_prefix_and_restore():
    previous=raw.ONSET_INDICES
    data=np.arange(31,dtype=np.float64)
    expected=raw.array_sha(data,18)
    with arbitrary_onset(17):
        assert raw.ONSET_INDICES==(17,)
        assert raw.array_sha(data,17+1)==expected
    assert raw.ONSET_INDICES is previous


def test_invalid_onset_rejected():
    try:
        with arbitrary_onset(6000):pass
    except ValueError:return
    raise AssertionError('invalid onset accepted')
