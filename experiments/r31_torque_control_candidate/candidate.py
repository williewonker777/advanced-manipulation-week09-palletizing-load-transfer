"""Conservative disjoint composition of the frozen held and detached lifetimes."""
from experiments.r30_detached_lifetime.candidate import Detector as DetachedLifetime
from experiments.r29_native_event_link.candidate import Detector as HeldPlateau


class Detector(DetachedLifetime, HeldPlateau):
    """One cooperative C3 update; held-grip and detached-open ownership are disjoint.

    The MRO is R31 -> R30 lifetime -> R29 detached -> R29 held -> C3.  R29
    detached calls ``super().update`` exactly once, which enters held; held calls
    the physical C3 base exactly once.  No case, time, mass, or optional early
    headroom branch is added here.
    """
