from experiments.r31_event_channels.adapter import Detector,EventChannels
from experiments.r31_held_identity.candidate import Detector as HeldDetector

def test_adapter_is_read_only_subclass_and_schema_is_explicit():
    assert issubclass(Detector,HeldDetector)
    row=EventChannels(False,False,True,True,False,False,(),True,False,True,False)
    assert row.notice_level and row.notice_rise and not row.native_level
    assert tuple(row.__dataclass_fields__)==('output_level','output_rise','notice_level','notice_rise','native_level','native_rise','native_routes','native_routes_complete','native_consumed','source_present','native_source_linked')
