# R21 optional torque release gate

All three artifacts are **installable research opt-ins**, not promoted defaults.
The submitted `my_detector.py`, `my_torque.py`, and `my_servo.py` remain unchanged.

| Variant | Standalone SHA-256 | Original official torque | Protection |
|---|---|---:|---|
| R20 horizontal | `30cc98154c7a37c75f3da52e6f41771b1ba268179e1994cb7a17a8bbf961e8b1` | 20/24 | 106 episodes, 0 failures, gain 211 |
| R21 pick observer | `65d53612279667250e0240f623fba608b051f6169d8f35a358b4d1f7838a8500` | 22/24 | 106 episodes, 0 failures, gain 200 |
| R21 yaw balance | `f28a4b6da0939e435fc97130ee9be7db7b98b7c2956346f778914788162492ad` | 24/24 | 106 episodes, 0 failures, gains 199/200/201/202/210 |

Each official audit used the pristine licensed simulator with the detector under
the default filename. Servo fallback matched R10 exactly, normal trials had no
false alarm, and the protected kit manifest was unchanged. The R21 gain case 200
is a concrete R20-missed-to-R21-caught 10 N event (49 ms, no pre-alarm).

The protection selection is a reused, exposed D4 subset, not an independent
holdout: 7 references, 12 pick impacts (10/15/40 N in +/-x and +/-y), and 16
offset cases. It includes six modular-versus-standalone parity cases per variant;
each parity case has 6000 samples and all detection, residual, physical, delay,
and false-alarm outputs are exact.

Build the portable two-source yaw archive with `tools/r21_torque_package.py`. Its
fresh original-evaluator audit digest is
`3ff9e1da6e647abb6aa183f4659a6e5b23a7615011e5454be2fe2c3ae0bf0650`.
The [portable archive receipt](../../results/r21/package.json) records the archive
hash and the later **failed** full-D4 protection result (385 cases per arm).
The small exposed D4 screen above is not the complete grid or an independent holdout.

Default promotion is rejected: the full grid confirmed a new no-force alarm
(case 174) and more duplicate rises in cases 330, 339 and 349. Other limitations
include duplicate-at-release responses at 40/129 ms and misses near 6--6.5 N.
Education and simulation only, not hardware.
