from experiments.r31_held_identity.candidate import Detector as Identity
from experiments.r32_measured_level.candidate import Detector as Level
from experiments.r32_level_replay.channels import IdentityChannels,LevelChannels
from experiments.r32_native_handoff.candidate import Detector as Handoff
from experiments.r32_level_replay.channels import HandoffChannels
def test_wrappers_only_subclass_frozen_methods():
 assert issubclass(IdentityChannels,Identity) and issubclass(LevelChannels,Level) and issubclass(HandoffChannels,Handoff)
