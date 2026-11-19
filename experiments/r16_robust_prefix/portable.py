"""In-memory public-signal adapter for the frozen R16 robust-prefix estimator."""
import numpy as np

from experiments.r14_online.stream import Stream
from experiments.r16_robust_prefix.robust import (
    DEADLINE_INDEX, MASK, _pregrasp_signature,
)


def feature_from_samples(samples, case):
    """Return the frozen R16 feature without writing a raw sensor capture."""
    required = ('t', 'encoder', 'jts', 'grip')
    if not all(key in samples for key in required):
        raise ValueError('complete public sample stream required')
    t, encoder, jts, grip = (np.asarray(samples[key]) for key in required)
    mass = float(case['mass']); com = np.asarray(case['com'], float)
    if (len(t) <= DEADLINE_INDEX or encoder.shape != (len(t), 6)
            or jts.shape != encoder.shape or grip.shape != t.shape
            or not np.array_equal(t, np.arange(len(t))*.001)
            or not np.isfinite(encoder).all() or not np.isfinite(jts).all()
            or mass <= 0 or com.shape != (3,) or not np.isfinite(com).all()):
        raise ValueError('regular finite public stock record through 3.447 s required')
    signature = _pregrasp_signature(t, encoder, jts, grip)
    stream = Stream(mass, com)
    for index in range(DEADLINE_INDEX+1):
        stream.observe(float(t[index]), encoder[index], jts[index], bool(grip[index]))
    if stream._next_centre <= int(np.flatnonzero(MASK)[-1]):
        raise ValueError('carried prefix incomplete')
    return {'mass': np.array(mass), 'y': stream._y[MASK].copy(),
            'jac': stream._jac[MASK].copy(), 'stiffness': stream._stiffness[MASK].copy(),
            'prior_com_mm': stream._prior_com_mm.copy(), 'signature': signature,
            'completed_at_s': np.array(DEADLINE_INDEX*.001)}
