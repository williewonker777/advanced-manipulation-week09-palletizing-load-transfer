"""Read-only telemetry wrappers; returned residual and Boolean are unchanged."""
from experiments.r31_held_identity.candidate import Detector as Identity
from experiments.r32_measured_level.candidate import Detector as Level
from experiments.r32_native_handoff.candidate import Detector as Handoff

class IdentityChannels(Identity):
    latest=None
    def reset(self):super().reset();self.channel_log=[];type(self).latest=self
    def update(self,*args):
        notices,width=self.planar_notices,self.pending_width
        residual,out=super().update(*args)
        self.channel_log.append({'output':bool(out),'notice':bool(width or self.planar_notices>notices),
                                 'native':bool(self.r29_active_routes),'routes':tuple(sorted(self.r29_active_routes))})
        return residual,out

class LevelChannels(Level):
    latest=None
    def reset(self):super().reset();self.channel_log=[];type(self).latest=self
    def update(self,*args):
        residual,out=super().update(*args);d=self.decision
        self.channel_log.append({'output':bool(out),'event':d.event_output,'continuation':d.measured_continuation})
        return residual,out

class HandoffChannels(Handoff):
    latest=None
    def reset(self):super().reset();self.channel_log=[];type(self).latest=self
    def update(self,*args):
        residual,out=super().update(*args);d=self.decision
        self.channel_log.append({'output':bool(out),'event_output':d.event_output,'event_started':d.event_started,
                                 'measured_load':d.measured_load,'linked_native':d.linked_native,'alarm_level':d.alarm_level})
        return residual,out
