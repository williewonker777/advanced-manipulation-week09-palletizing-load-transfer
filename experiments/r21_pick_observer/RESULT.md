# R21 normal-prediction pick observer

Status: **opt-in research candidate; R10 submission unchanged**.

The R20 horizontal channel could not lower its scalar threshold without crossing normal load-transfer residuals. R21 instead predicts the normal horizontal residual from public time, known mass and known COM. It subtracts that signed two-axis prediction, averages four causal samples and requires three consecutive bound exceedances. The threshold is `1.05 × 7.55495 = 7.93270 N`, where `7.55495 N` is the maximum leave-one-COM-group-out error over all 20 normal-only calibration episodes. No collision label, applied force, plant state, or evaluation result enters fitting or threshold selection. R20 and R10 alarms remain an OR.

## Results

- Authenticated original evaluator, torque: R20 **20/24** -> R21 **22/24** recognized collision cells. The two added cells are the previously missed `0 kg / 1.42 s / 10 N` at **47 ms** and `8 kg / 1.42 s / 10 N` at **49 ms**.
- Original no-collision 0/4/8 kg rows remain zero false alarms. The existing one false alarm in the 8 kg / 3.0 s collision row remains one; no official row loses a detection.
- Predeclared post-fit six-context four-trial set: all physical/applied-force logs exact, no added false or baseline loss. R20 is strict in 5/6 and R21 in 6/6. One case is candidate-only (+46 ms); R21 also lowers latency on the five R20 positives.
- JTS-seed-only fixture, seeds 11/29/41/53 at 0/8 kg: 32 runs. R20 misses all eight weak impacts; R21 detects all at **43–50 ms**, with no extra no-force or collision-run false alarm. This is finite local robustness, not a probability or hardware claim.
- Three fixed teaching-offset controls: the only real tool contact has no R21 prealarm and remains detected; two no-contact offsets stay quiet.

The official cases are exposed evidence. The post-fit six-context set is separate but finite. The per-tick model may remain sensitive to untested dynamics, calibration drift, sensor bias, and COM outside the declared interpolation box. The JSON model is an explicit runtime dependency. Do not promote without the separately owned independent challenge and product replay.

Import: `experiments.r21_pick_observer.detector.Detector`. Frozen candidate/model SHA-256: `1f03ed49…e9e3e8` / `499b2577…9c6bf`.
