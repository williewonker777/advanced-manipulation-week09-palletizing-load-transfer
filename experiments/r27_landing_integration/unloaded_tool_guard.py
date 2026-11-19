"""Tool guard adapter for unresolved-payload candidate."""
from experiments.r27_landing_integration import support_tool_guard
from experiments.r27_landing_integration.unloaded_aware import Detector
support_tool_guard.Detector=Detector
if __name__=='__main__':support_tool_guard.main()
