"""Separated normal-audit adapter for unresolved-payload candidate."""
from experiments.r27_landing_integration import support_audit
from experiments.r27_landing_integration.unloaded_aware import Detector
support_audit.Candidate=Detector
if __name__=='__main__':support_audit.main()
