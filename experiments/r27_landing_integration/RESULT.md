# R27 landing integration pilot

Status: **rejected after freeze**. This is disclosed MuJoCo 3.17 development evidence, not blind or deployment evidence.

The preregistered gripped inverse-force witness was fit on seven G4 normal episodes and frozen at a 14.78106064144941 N horizontal bound. It passed a separate six-normal audit with byte-identical C1 rises, active samples, false times, tool delay, cost, and physics.

Authentic `evaluate.py` improved the actual G4+C1 result from 15/24 to 17/24 detected collision slots. Both gains were the previously missed 15 N horizontal pickup stratum (mass 0 and 8 kg); official control cells and false counts were unchanged. A bounded signed/two-pulse check recovered pickup Y15 at 40 ms, preserved pickup X15 and transport Y15, did not claim the still-missed Z15, and preserved both second impulses without duplicates.

The candidate is not eligible: the disclosed `tool_high` genuine-contact row retained the real tool detection (9 ms, no prealarm) but added a late rise at sample 3857 before C1's sample 3903 rise. That violates the preregistered no-new-rise gate. Candidate/model bytes were not retuned.

## Separate support-aware attempt

A post-v1 public-signal diagnosis found a real physical distinction: the recovered pickup edge had valid arm-support fraction `1.0`, zero observed table normal, and zero handoff; the tool-high 3.857 s edge had arm-support fraction `0.3047`, observed table normal `27.28 N`, and handoff `0.8369`. A distinct proposal was preregistered before its outcomes: admit the optional source only when the actual G4 estimator is valid and the arm owns at least half the payload support. The half-weight boundary was physical and fixed, not selected from collision force or time.

The support-aware source (`e40865f5c78851fb02f361d0cac9709f338e3a2ae016e71575e57bb111ee2bc3`) passed the separated 12-episode normal audit exactly. It then failed its first challenged gate: authentic official performance was only 16/24, below the preregistered requirement to retain at least v1's 17/24. It retained the mass-8 pickup gain but correctly failed open for the mass-0 case, where payload-support ownership is unavailable. Evaluation stopped at that falsifier; no threshold alternative or post-outcome retuning was attempted.

## Unresolved-payload opt-in candidate

The zero-payload failure was repaired in a third, separately preregistered source without changing either failed candidate. Payload weight is treated continuously. A normal-only mass-zero G4 episode measured `0 N` maximum admissible table normal; the preregistered rule added a `0.5 N` sensor reserve. Loads above this resolution retain the valid half-support rule. Loads at or below it use a physically closed grip plus measured table normal no greater than the frozen resolution; invalid or unknown state produces no optional report.

Frozen candidate `experiments.r27_landing_integration.unloaded_aware:Detector` (`fe22ea86d0a3e975d8846e11f4fe63ccda99d40cf6bfb17d4ba34cac10ce4c69`) passed:

- separated G4 normal audit: 12/12 exact against C1;
- authentic official comparison: C1 15/24, candidate 17/24, identical control cells and false counts;
- known and preregistered unseen tool offsets: 10/10 episodes exact, including real contacts at 25, 30, and 35 mm high offsets;
- signed/pair panel: exact physics/forces, pickup Y gained, X/transport and both second pulses preserved, no duplicates;
- late post-release X/Z panel: 4/4 strict with all registered guards passing;
- preregistered 0.1, 0.35, 0.7, and 1.0 kg panel: all quiet twins passed, all four missing +Y impacts gained at 39–40 ms, and all four inherited −Y impacts preserved at 32–41 ms.

This is a meaningful **opt-in integrated research system**, not an automatic default or safety qualification.

Frozen pins:

- `plan.json`: `4d37047e107c9881f09ba90ecca55fc263ad0bc5338485852b8c10c4fed79c8b`
- `model.json`: `447660558c5ba67ea65e53be568e493bdd5f42532b558f440884b3fb610b5939`
- `candidate.py`: `ffde18b42b8ccd8250eae184147f7e498003356cfdb1a7cd46ccc5edfce0fb95`

Reproduction (from the project root, after sourcing the evidence `env.sh`; every invocation also sets `PYTHONDONTWRITEBYTECODE=1` and single-thread BLAS):

```text
.venv/bin/python -B -m experiments.r27_landing_integration.calibrate --out <fresh-fit-dir>
.venv/bin/python -B -m experiments.r27_landing_integration.audit --out <fresh-audit.json>
.venv/bin/python -B -m experiments.r27_landing_integration.official --out <fresh-official-dir>
.venv/bin/python -B -m experiments.r27_landing_integration.mechanism --out <fresh-mechanism.json>
.venv/bin/python -B -m experiments.r27_landing_integration.tool_guard --out <fresh-tool.json>
.venv/bin/python -B -m experiments.r27_landing_integration.summarize --evidence <evidence-dir> --out <fresh-summary.json>
```
