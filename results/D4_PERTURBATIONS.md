# D4 private perturbation study (supplemental)

These are **analysis-only** results from a private `PerturbedPlant` subclass and
an owned runner copy. They are not official `evaluate.py` figures and no
perturbed value entered production control/detection. The unchanged frozen
sources are identified in `D4_PERTURBATIONS.json`.

## Protocol

- 50 sequential episodes, one simulation process.
- A/B noise sensitivity: JTS RNG seeds 3/7/11 at 0, 4.3 and 8 kg.
- Friction: viscous and Coulomb scales 0.8/1.2 at 4.3 and 8 kg.
- Elasticity: K scales 0.8/1.2 with D scaled by `sqrt(K scale)`, re-equilibrated
  initial deflection and box settling.
- Two 8 kg friction/stiffness corners per controller.
- 8 kg transport collisions: 40 N along each of ±x, ±y and ±z.
- Nominal subclass parity is bit-exact against the kit runner for all non-timing
  metrics and logged physical arrays.

## Non-collision envelope

| controller | cases | worst RMS / max TCP mm | max pick / place step N m | max table force N | saturation / divergence | false alarms |
|---|---:|---:|---:|---:|---:|---:|
| A servo | 19 | 1.89 / 5.28 | 9.66 / 16.31 | 86.69 | 0% / 0 | 4 |
| B torque | 19 | 9.97 / 17.26 | 9.17 / 14.67 | 138.46 | 0% / 0 | 7 |

Noise-seed-only cases were stable with zero false alarms. Servo tracking and
steps stayed within their original target envelope; its 8 kg low-stiffness case
reached 86.69 N, narrowly above the 86.33 N `1.1 mg` target. Detector false
alarms appeared under both stiffness changes and both combined corners.

Torque control was substantially friction-sensitive. Friction scale 0.8 caused
16.7--16.9 mm maximum TCP error. At scale 1.2, 8 kg table force reached 115 N;
the high-friction/low-stiffness corner reached 138 N. The opposite corner
reached 17.26 mm maximum TCP error. These are explicit robustness failures,
although no case saturated or diverged.

## Collision directions

All twelve 40 N transport collisions were detected without an alarm already in
progress. Delays ranged 26--44 ms for servo and 26--49 ms for torque. The torque
`-z` case had one additional evaluator rise outside the collision window; all
other collision rows had zero such rises.

## Interpretation and limits

- The frozen nominal acceptance results remain valid; this study maps sensitivity
  and does not replace official evaluation.
- The detector calibration varied mass/COM and both baseline controllers, but
  did not include plant K/friction perturbations. The observed stiffness/corner
  false alarms expose that boundary.
- The torque controller's model-based friction feedforward is the dominant
  sensitivity in this small grid. This is evidence for a limitation, not a basis
  for post-freeze tuning.
- Only three noise seeds, two scale endpoints and two corners were tested. This
  is a small generalization grid, not a probability or hardware guarantee.
