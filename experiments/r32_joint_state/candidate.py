"""Add verified detached readback outside C2 without changing its parent event path."""
from dataclasses import dataclass
from experiments.r32_detached_readback.candidate import RetirementReadback
from experiments.r32_native_handoff.candidate import Detector as HeldC2

class ParentCompatibleReadback(RetirementReadback):
    """Call authoritative readback once, but expose only legacy output to the parent."""
    def update(self,*args):
        super().update(*args)
        return self.legacy_output

@dataclass(frozen=True)
class JointDecision:
    event_started:bool
    event_output:bool
    measured_load:bool
    linked_native:bool
    detached_native:bool
    parent_alarm_level:bool
    alarm_level:bool

class Detector(HeldC2):
    def reset(self):
        super().reset();self.detached_endpoint=ParentCompatibleReadback()
        self.joint_decision=JointDecision(False,False,False,False,False,False,False)
    def update(self,t,th,thd,tm,ts,grip):
        residual,parent=super().update(t,th,thd,tm,ts,grip);held=self.decision
        detached=bool(self.detached_endpoint.readback_active and not grip)
        level=bool(parent or detached)
        self.held_decision=held
        self.joint_decision=JointDecision(held.event_started,held.event_output,held.measured_load,
            bool(held.linked_native or detached),detached,held.alarm_level,level)
        self.decision=self.joint_decision
        return residual,level
