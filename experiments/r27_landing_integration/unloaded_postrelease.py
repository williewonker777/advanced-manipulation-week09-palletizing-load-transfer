"""Late signed X/Z protection adapter for unresolved-payload candidate."""
from experiments.r26_delivery_pair import study
from experiments.r27_landing_integration.unloaded_aware import Detector
study.Pair=Detector
if __name__=='__main__':study.main()
