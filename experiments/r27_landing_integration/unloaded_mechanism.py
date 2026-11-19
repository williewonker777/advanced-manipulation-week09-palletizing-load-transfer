"""Signed/pair adapter for unresolved-payload candidate."""
from experiments.r27_landing_integration import mechanism
from experiments.r27_landing_integration.unloaded_aware import Detector
mechanism.Candidate=Detector
mechanism.FROZEN={}
if __name__=='__main__':mechanism.main()
