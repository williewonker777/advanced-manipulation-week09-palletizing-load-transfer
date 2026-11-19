"""Compose two mechanisms with disjoint measured-gripper ownership; no new gate or threshold."""
from experiments.r29_detached_release.candidate import Detector as Detached
from experiments.r29_native_event_link.candidate import Detector as Held


class Detector(Detached, Held):
    """Held calls the physical C3 update once. Detached can retire only an already
    delivered native event with the gripper open. With the gripper open the held
    observer returns no pair, resets, and passes the native result unchanged.
    """
