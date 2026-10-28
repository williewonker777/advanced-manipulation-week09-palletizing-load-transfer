# Palletizing load transfer: elastic-joint control and collision detection

Individual advanced-manipulation simulation study, M1013, fixed 1 ms sampling.
**Our code and results only. The separately supplied course kit is required and
is NOT included. Full task acceptance is NOT achieved.**

## Same-PC official result (8 kg)

| Approach | TCP RMS/max mm | Pick/place step N m | Place table N | False pick/place/other |
|---|---:|---:|---:|---|
| Baseline servo A | 6.14/14.00 | 57.5/73.8 | 101 | 5/2/0 |
| Our servo A + detector | 1.17/3.91 | 9.7/14.6 | 79 | 0/0/0 |
| Baseline torque B | 7.04/13.18 | 57.2/74.8 | 295 | 5/4/0 |
| Our torque B + detector | 3.23/8.44 | 9.0/13.1 | 80 | 0/0/0 |

Zero saturation at these official conditions. Nominal control targets pass;
minimum transport detection is 15 N, not the requested <15 N. Some delays
regress, and the B collision evaluation includes one extra false rise.

770 stratified stress episodes: no false alarms in 189 no-collision references
per architecture; 94/180 A and 91/180 B force cases detected; all 12 actual
offset tool contacts detected. This is finite coverage, NOT a detection rate
for a population. A has two saturation cases and light-mass force-ratio failures.
Observed stress timing maxima exceed 3 ms despite sub-ms p99; no hard-RT claim.
Separate 50-case parameter perturbations expose B friction sensitivity.

## Run inside a separately obtained unchanged kit

Copy our three `my_*.py` files and `tools/` into a disposable kit work copy.
Keep its `sim/`, `model/`, baseline modules and evaluator unchanged. Do not
overwrite the supplied kit README with this README. Install dependencies only
in that work copy's virtual environment; see [reproduction](report/REPRODUCE.md).

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_integrity.py
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/evaluate_checked.py --ctrl my_servo --detector my_detector --out results/evaluation_A.txt
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/evaluate_checked.py --ctrl my_torque --detector my_detector --out results/evaluation_B.txt
```

The wrapper invokes the original commands `evaluate.py --ctrl my_servo
--detector my_detector` / `evaluate.py --ctrl my_torque --detector my_detector`
after the integrity check. Our three final modules are self-contained: only
NumPy, MuJoCo and public trajectory/config imports; each works without the other.
They embed an identical timestamp-cached estimator for optional sharing.

## Evidence and presentation

- [Full integration, ablations, goal decisions](report/RESULTS.md)
- [16-slide presentation](results/presentation.pdf)
- [Method and equations](docs/METHOD.md), [limits](docs/LIMITATIONS.md), [primary references](docs/REFERENCES.md)
- [Control A](results/figures/control_servo.png), [control B](results/figures/control_torque.png)
- [A failure map](results/figures/failure_map_servo.png), [B failure map](results/figures/failure_map_torque.png)
- [Analysis-only perturbations](results/D4_PERTURBATIONS.md)
- [Checksums](artifacts/SHA256SUMS), `tools/audit_public.py` for package checks

Sharing on/off is numerically identical and cuts estimator advances in half;
physical effects of support fusion are reported separately. Earlier direction-
confounded thresholds were rejected; final symmetric thresholds were trained
using no-collision data only with disjoint holdout validation and frozen before
collision testing. No safety-stop response or real-robot validation is included.

## License

Our original code: BSD-3-Clause. External course-kit/model/runtime code is not
redistributed or licensed by us; see [third-party notices](THIRD_PARTY_NOTICES).
