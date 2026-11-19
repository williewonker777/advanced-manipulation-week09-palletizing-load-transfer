"""Expose alarm-notice and measured-native channels without changing output."""
from dataclasses import dataclass
from experiments.r31_held_identity.candidate import Detector as HeldDetector

@dataclass(frozen=True)
class EventChannels:
    output_level:bool
    output_rise:bool
    notice_level:bool
    notice_rise:bool
    native_level:bool
    native_rise:bool
    native_routes:tuple[str,...]
    native_routes_complete:bool
    native_consumed:bool
    source_present:bool
    native_source_linked:bool

class Detector(HeldDetector):
    """Bit-identical detector with an additional read-only ``event_channels`` view.

    ``notice_level`` is an optional reporting envelope, not force duration.
    ``native_level`` is internal measured detector evidence, not physical truth.
    """
    latest=None
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);type(self).latest=self
        if not hasattr(self,'channel_log'):self._reset_channels()
    def _reset_channels(self):
        self.channel_log=[];self.event_channels=None;self._channel_output=False;self._channel_notice=False;self._channel_native=False
    def reset(self):
        super().reset();self._reset_channels();type(self).latest=self
    def update(self,t,th,thd,tau_m,tau_s,grip):
        notices=self.planar_notices;width=self.pending_width;linked=self.associated_native_samples
        residual,reported=super().update(t,th,thd,tau_m,tau_s,grip)
        notice=bool(width or self.planar_notices>notices)
        routes=tuple(sorted(self.r29_active_routes));native=bool(routes)
        channels=EventChannels(bool(reported),bool(reported and not self._channel_output),notice,
            bool(notice and not self._channel_notice),native,bool(native and not self._channel_native),routes,
            bool(self.r29_routes_complete),bool(self.r29_native_consumed),self.source is not None,
            self.associated_native_samples>linked)
        self.event_channels=channels;self.channel_log.append(channels)
        self._channel_output=bool(reported);self._channel_notice=notice;self._channel_native=native
        return residual,reported
