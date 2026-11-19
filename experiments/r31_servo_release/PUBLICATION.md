# R31 servo release publication set

This experiment is a **rejected post-disclosure research result**, not a
deployable detector.  The known official panel improved R10 servo from 23/24
to 24/24, but the prelocked prospective panel produced duplicate notices and
no candidate-only strict gain.

## Recommended portable source set

Publish these files together:

- `__init__.py`
- `candidate.py`
- `test_candidate.py`
- `plan.json`
- `prospective_plan.json`
- `PORTABLE_TRUTH.json`
- `RESULT.md`
- `PUBLICATION.md`

The diagnostic and simulation runners (`diagnose.py`,
`episode_diagnose.py`, `known.py`, `official.py`, and `prospective.py`) are
useful for an authenticated local reproduction, but depend on the separately
supplied course simulator/kit.  They must not be represented as a standalone
public simulator closure, and the kit itself is not included.

## Baseline and result caveats

- The direct baseline is the shipped R10 `my_detector:Detector` in SERVO mode.
- The candidate is an additive optional notice and never masks the baseline.
- Official inputs were already disclosed development inputs; the 24-episode
  prospective panel was prelocked after the known result but is small and is
  not a blind deployment holdout.
- The official improvement is insufficient for qualification: prospective
  empty -Y10 and light +Y10 contacts gained duplicate rises, while empty +Y9
  produced two candidate rises but no strict candidate-only gain.
- Status is permanently `REJECTED_NO_ENROLLMENT`; no post-result threshold or
  arbitration change was made.

`PORTABLE_TRUTH.json` is the compact, machine-path-free truth record.  It pins
the detector and plans, records the physical inputs and alarm edges, and
contains only disclosed result summaries rather than raw licensed-kit data.
