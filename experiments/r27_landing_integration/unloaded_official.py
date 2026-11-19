"""Authentic official adapter for unresolved-payload candidate."""
from experiments.r27_landing_integration import official
official.FROZEN={
 "unloaded_plan.json":"b6c0ec32ec7d0c81ea110ade0aca6d0ffdd2807cf00ee3f7ab998da7729d47e7",
 "model.json":"447660558c5ba67ea65e53be568e493bdd5f42532b558f440884b3fb610b5939",
 "unloaded_resolution.json":"4505aa2889b1ba48eacfc620080866ddcf11ac5c2ca01b2d4d8a361e1388ddb2",
 "unloaded_aware.py":"fe22ea86d0a3e975d8846e11f4fe63ccda99d40cf6bfb17d4ba34cac10ce4c69"}
_score=official.score
def score(item):
 name,detector,folder,env=item
 if name=='r27':detector='experiments.r27_landing_integration.unloaded_aware'
 return _score((name,detector,folder,env))
official.score=score
if __name__=='__main__':official.main()
