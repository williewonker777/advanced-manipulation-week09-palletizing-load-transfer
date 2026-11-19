import inspect
from experiments.r32_joint_state.candidate import Detector,ParentCompatibleReadback
from experiments.r32_native_handoff.candidate import Detector as C2
def test_structural_contract():
 assert issubclass(Detector,C2)
 source=inspect.getsource(ParentCompatibleReadback.update)
 assert source.count('super().update')==1 and 'legacy_output' in source
 update=inspect.getsource(Detector.update)
 assert 'and not grip' in update and 'parent or detached' in update and 'self.held_decision=held' in update
def test_parent_compatible_endpoint_returns_only_legacy():
 e=ParentCompatibleReadback();high=(2.,0.);low=(0.,0.)
 assert e.update(True,True,high,1.,False,True)
 assert e.update(True,True,low,1.,False,True)
 assert e.update(True,True,low,1.,False,True)
 answer=e.update(True,True,low,1.,True,True)
 assert e.readback_active and not e.legacy_output and answer is False
