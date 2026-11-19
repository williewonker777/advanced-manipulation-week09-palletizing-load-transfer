"""The frozen nearby development cases, shared by both evaluation frontends."""

def tasks(module, plan, arms):
    return [(kind, case, variant, arms, plan)
            for kind in ("negative", "single", "paired")
            for case in plan[kind] for variant in plan["variants"].values()]


def heldback_plan(base_plan, registered):
    plan = dict(base_plan)
    cases = []
    for time_value in registered["heldback_development"]["times_s"]:
        for vector in registered["heldback_development"]["force_vectors_N"]:
            for mass, com in registered["heldback_development"]["payloads"]:
                cases.append({"id": f"held_{len(cases):02d}", "phase": "measured_loaded",
                              "mass": mass, "com": com, "offset": None,
                              "impact": [time_value, vector]})
    plan["negative"], plan["paired"], plan["single"] = [], [], cases
    plan["variants"] = {key: base_plan["variants"][key]
                        for key in registered["heldback_development"]["mechanics"]}
    return plan
