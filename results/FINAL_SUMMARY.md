# Final results: load-transfer-aware palletizing

**Control targets pass at nominal 8 kg; full acceptance is not achieved.**

Official figures below come only from the unchanged evaluator through the integrity gate.
All official 0/4/8 kg rows are in `FINAL_OFFICIAL.json` and evaluation text files.

## Official same-PC 8 kg integration

| Combination (8 kg) | RMS / max mm | Pick / place N m | Table N | Sat % | False pick/place/other |
|---|---:|---:|---:|---:|---|
| servo_base_base | 6.14 / 14.00 | 57.5 / 73.8 | 101 | 0 | 5/2/0 |
| servo_base_final | 6.14 / 14.00 | 57.5 / 73.8 | 101 | 0 | 0/0/0 |
| servo_final_base | 1.17 / 3.91 | 9.7 / 14.6 | 79 | 0 | 3/7/0 |
| servo_final_final | 1.17 / 3.91 | 9.7 / 14.6 | 79 | 0 | 0/0/0 |
| torque_base_base | 7.04 / 13.18 | 57.2 / 74.8 | 295 | 0 | 5/4/0 |
| torque_base_final | 7.04 / 13.18 | 57.2 / 74.8 | 295 | 0 | 0/0/0 |
| torque_final_base | 3.23 / 8.44 | 9.0 / 13.1 | 80 | 0 | 5/9/3 |
| torque_final_final | 3.23 / 8.44 | 9.0 / 13.1 | 80 | 0 | 0/0/0 |

## Official collision delays (ms) at 8 kg

| Combination | Time s | 10 N | 15 N | 40 N | 100 N | Minimum | False max |
|---|---:|---|---|---|---|---|---|
| servo_base_base | 1.42 | 구분 불가 | 구분 불가 | 구분 불가 | 구분 불가 | — | 4회 |
| servo_base_base | 3.0 | 놓침 | 68 | 26 | 17 | 15 N | 8회 |
| servo_base_base | 3.86 | 경보 중 | 경보 중 | 경보 중 | 경보 중 | — | 6회 |
| servo_base_final | 1.42 | 놓침 | 놓침 | 34 | 22 | 40 N | 0회 |
| servo_base_final | 3.0 | 놓침 | 77 | 24 | 18 | 15 N | 0회 |
| servo_base_final | 3.86 | 놓침 | 64 | 42 | 27 | 15 N | 0회 |
| servo_final_base | 1.42 | 경보 중 | 경보 중 | 경보 중 | 경보 중 | — | 9회 |
| servo_final_base | 3.0 | 놓침 | 놓침 | 26 | 17 | 40 N | 13회 |
| servo_final_base | 3.86 | 경보 중 | 경보 중 | 경보 중 | 경보 중 | — | 4회 |
| servo_final_final | 1.42 | 놓침 | 놓침 | 36 | 24 | 40 N | 0회 |
| servo_final_final | 3.0 | 놓침 | 83 | 27 | 18 | 15 N | 0회 |
| servo_final_final | 3.86 | 놓침 | 69 | 33 | 21 | 15 N | 0회 |
| torque_base_base | 1.42 | 경보 중 | 경보 중 | 경보 중 | 경보 중 | — | 8회 |
| torque_base_base | 3.0 | 97 | 68 | 27 | 17 | 10 N | 12회 |
| torque_base_base | 3.86 | 경보 중 | 경보 중 | 경보 중 | 경보 중 | — | 7회 |
| torque_base_final | 1.42 | 놓침 | 놓침 | 42 | 24 | 40 N | 0회 |
| torque_base_final | 3.0 | 놓침 | 85 | 29 | 19 | 15 N | 0회 |
| torque_base_final | 3.86 | 놓침 | 61 | 40 | 25 | 15 N | 0회 |
| torque_final_base | 1.42 | 경보 중 | 경보 중 | 경보 중 | 경보 중 | — | 16회 |
| torque_final_base | 3.0 | 놓침 | 68 | 26 | 17 | 15 N | 19회 |
| torque_final_base | 3.86 | 구분 불가 | 구분 불가 | 구분 불가 | 구분 불가 | — | 9회 |
| torque_final_final | 1.42 | 놓침 | 놓침 | 36 | 24 | 40 N | 0회 |
| torque_final_final | 3.0 | 놓침 | 83 | 26 | 18 | 15 N | 1회 |
| torque_final_final | 3.86 | 놓침 | 91 | 42 | 22 | 15 N | 0회 |

`놓침`=miss; `경보 중`=alarm already active; `구분 불가`=indistinguishable, not a detection.

## Historical official control ablation (8 kg)

These are preserved stage snapshots, not reruns of wrappers against the final shared class.
All use the unchanged baseline detector. Sharing and support fusion are evaluated separately below.

| Stage | RMS/max mm | Pick/place N m | Table N | Saturation % |
|---|---:|---:|---:|---:|
| A baseline | 6.14/14.00 | 57.5/73.8 | 101 | 0.0 |
| A C1 | 1.17/3.37 | 107.7/106.6 | 79 | 0.0 |
| A C1+ramp | 1.17/3.27 | 9.7/17.7 | 79 | 0.0 |
| A C1+C2 | 1.20/3.45 | 9.7/17.7 | 79 | 0.0 |
| A C1+C2+C3 | 1.18/3.91 | 9.7/17.9 | 79 | 0.0 |
| B baseline | 7.04/13.18 | 57.2/74.8 | 295 | 0.0 |
| B B1 | 2.24/4.45 | 97.9/96.7 | 79 | 0.0 |
| B B1+B2 | 2.31/4.25 | 104.1/103.2 | 79 | 0.0 |
| B +ramp B3 | 2.41/4.31 | 9.0/17.2 | 80 | 0.0 |
| B +B3 | 3.12/7.82 | 9.0/16.0 | 80 | 0.0 |

## Historical official detector ablation (baseline servo, 8 kg)

| Stage | False pick/place/other | Transport 10/15/40/100 N delay ms |
|---|---|---|
| Baseline | 5/2/0 | 놓침 / 68 / 26 / 17 |
| +V | 5/2/0 | 놓침 / 68 / 22 / 17 |
| +V+B | 3/3/0 | 놓침 / 52 / 22 / 16 |
| +V+B+W, instantaneous T | 0/0/0 | 놓침 / 53 / 22 / 15 |
| D3 symmetric T | 0/0/0 | 놓침 / 85 / 25 / 18 |
| Final FAST/mass gate | 0/0/0 | 놓침 / 77 / 24 / 18 |

Historical W and T stages use the documented D3 threshold snapshot; the final FAST protocol
uses a new disjoint no-collision sample and mass-zero gate. This last row is a combined change,
not a claim that persistence alone caused every difference. V alone reduced peak pickup Fz
45.21 to 36.23 N, but oracle force RMSE worsened; no universal V-accuracy claim is made.

## Sharing and support fusion (supplemental)

| Mode | Support blend | Shared | RMS/max mm | Pick/place N m | Last20ms mean table N | Median/p99/max us |
|---|---:|---|---:|---:|---:|---:|
| servo | 0.0 | True | 1.175/3.914 | 9.66/16.84 | 0.17 | 109.1/204.4/328.1 |
| servo | 0.0 | False | 1.175/3.914 | 9.66/16.84 | 0.17 | 192.0/375.9/523.8 |
| servo | 0.1 | True | 1.174/3.914 | 9.66/14.63 | 0.16 | 112.3/215.2/312.7 |
| servo | 0.1 | False | 1.174/3.914 | 9.66/14.63 | 0.16 | 181.2/363.3/574.7 |
| torque | 0.0 | True | 3.115/7.817 | 8.98/15.08 | 8.76 | 118.3/214.0/382.3 |
| torque | 0.0 | False | 3.115/7.817 | 8.98/15.08 | 8.76 | 185.9/351.9/616.6 |
| torque | 0.1 | True | 3.232/8.444 | 8.98/13.13 | 8.75 | 116.9/207.4/351.2 |
| torque | 0.1 | False | 3.232/8.444 | 8.98/13.13 | 8.75 | 195.4/358.0/531.3 |

All four share-on/off parity comparisons are exact for physical arrays, residuals and alarms.
Sharing advances the estimator 6000 instead of 12000 times; improvement in physical outputs
comes from the estimator/load algorithm, not object reuse. Full support fusion was rejected:
it increased servo place step and torque-controller tracking error; selected blend is 0.10.

## Robustness (supplemental, frozen sources; no retuning)

189 mass/COM references + 180 stratified force cases + 16 offset cases per architecture.
7 masses including nonintegers, 27 COM positions, 3 phases with 5 times each, 6 force norms
5–100 N, ±x/±y/±z and four diagonal directions. Not the full Cartesian product.

| Mode | No-collision FA episodes / 189 | Collision detected/missed/prior/indist. | Tool contacts missed / present | Worst max TCP mm | Worst p99/max us |
|---|---:|---|---:|---:|---:|
| servo | 0/189 | 94/86/0/0 | 0/6 | 3.94 | 309.0/3267.7 |
| torque | 0/189 | 91/89/0/0 | 0/6 | 9.10 | 357.1/3205.1 |

Failure-map figures vary mass/COM/time across cells and must not be read as fixed-condition
minimum detectable force curves. Default-noise finite coverage does not establish an event probability.
Separate 50-case noise/friction/stiffness study: `D4_PERTURBATIONS.md`; its perturbed model
figures are not official. Three noise seeds had no false alarms; stiffness/corner cases did.
Low-mass force-ratio and any saturation failures are listed in `D4_CONTROL_FAILURES.csv`.
All 12 genuine offset tool contacts across A/B were detected; other offset cases had no tool
contact but some had false alarms. With unchanged baseline controllers, the historical 30 mm
horizontal-offset misses became 12 ms (servo, 86.06 N contact) and 19 ms (torque, 29.66 N)
with the final FAST detector; each also had one extra false rise.
Torque friction/corner sensitivity reached 17.26 mm and 138.46 N; no perturbed case diverged or saturated.

## Acceptance

### servo
- PASS: pick_reduction_75pct
- PASS: place_reduction_75pct
- PASS: table_le_1p1mg
- PASS: tcp_max_le_baseline
- PASS: zero_saturation
- PASS: zero_false_alarms_all_official_masses
- PASS: no_prior_or_indistinguishable
- MISS: transport_minimum_below_15n
- MISS: same_pc_delay_no_regression
- MISS: strict_observed_every_cycle_below_1ms
- MISS: sustained pre-release handover; low-force sensitivity and some delay targets.
- Overall full acceptance: **not achieved**.
### torque
- PASS: pick_reduction_75pct
- PASS: place_reduction_75pct
- PASS: table_le_1p1mg
- PASS: tcp_max_le_baseline
- PASS: zero_saturation
- PASS: zero_false_alarms_all_official_masses
- PASS: no_prior_or_indistinguishable
- MISS: transport_minimum_below_15n
- MISS: same_pc_delay_no_regression
- MISS: strict_observed_every_cycle_below_1ms
- MISS: sustained pre-release handover; low-force sensitivity and some delay targets.
- Overall full acceptance: **not achieved**.

## Attribution, rejected attempts and limits

- Historical C1/B1 alone improved tracking but worsened torque steps; continuous load FF greatly reduced switching-induced steps.
- New velocity candidates failed a predeclared no-collision oracle-error guard; retained V is not universally more accurate.
- FAST thresholds: 24 training + 12 disjoint holdout episodes, zero false alarms; no collision data used for tuning.
- An earlier direction-confounded detector candidate was discarded and is not represented as a successful final result.
- Early settle/overtravel/two-stage placement candidates worsened force/saturation/tracking and were rejected.
- Offset-associated false alarms and support-line vertical ambiguities remain. No safety-stop response is implemented.
- Finite empirical stability and desktop wall-clock timings are not passivity or hard-real-time proofs.
- See `docs/METHOD.md`, `docs/LIMITATIONS.md`, `docs/REFERENCES.md` and presentation.pdf.
