"""A two-hypothesis continuation of an existing learned horizontal-force alarm; never initiates one."""
import numpy as np


class SignedEvent:
    """Compare each independent signal with its measured event and zero-load models.
    
    Means are estimated only while the original alarm is present. Under equal
    covariance and prior odds, ||x-m||² < ||x||² iff 2 x·m > m·m.
    A sharp disagreement or change of grip ends the carry; it never mutes an alarm.
    """
    def __init__(self, predict=False, require_stationary=False):
        self.predict = predict
        self.require_stationary = require_stationary
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = None
        self.bridge = self.observed = False
        self.last_grip = None
        self.history = []

    def feed(self, observed, ref_signal, force, grip, width, lag, bound):
        if self.last_grip != bool(grip):
            self.reset()
        self.last_grip = bool(grip)
        if force is None or ref_signal is None or not (np.isfinite(force).all() and np.isfinite(ref_signal).all()):
            self.n = 0
            self.mean = None
            self.bridge = self.observed = False
            self.history.clear()
            return False
        value = np.stack((ref_signal[:2],force[:2]))
        self.history.append(value)
        if len(self.history)>2*width:
            del self.history[0]
        if observed:
            if not self.observed and not self.bridge:
                self.n = 0
                self.mean = None
            self.n += 1
            if self.mean is None:
                self.mean = value.copy()
            else:
                self.mean += (value-self.mean)/self.n
            self.observed = True
            self.bridge = False
            return False
        if not (self.bridge or self.observed) or self.mean is None or len(self.history)<2*width:
            self.bridge = self.observed = False
            return False
        self.observed=False
        # Refuse an event defined only by noise in one estimator or opposite directions.
        m=self.mean;energy=np.sum(m*m,axis=1)
        valid=bool(min(energy)>0 and np.dot(m[0],m[1])>0)
        if valid:
            avg=np.mean(self.history[-width:],axis=0)
            if self.predict or self.require_stationary:
                before=np.mean(self.history[:width],axis=0)
                # A least-assumption first-order prediction for a delayed smoother.
                # Do not amplify a growing force; only extrapolate a fall toward zero.
                slope=np.sum((avg-before)*m,axis=1)/width
            proj=np.sum(avg*m,axis=1)
            if self.predict:proj=proj+lag*np.minimum(0.,slope)
            valid=bool(np.all(2.*proj>energy))
            if self.require_stationary:
                # When the primary model already explains the fall as a ramp ending,
                # hold only if both redundant estimates still exceed its frozen decision level.
                valid=valid and bool(np.all(slope>=0) or np.all(proj/np.sqrt(energy)>bound))
        self.bridge=valid
        return valid
