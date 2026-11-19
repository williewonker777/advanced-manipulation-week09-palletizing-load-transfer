# R21 servo yaw witness — rejected

Status: **rejected after fresh false-alarm evidence; not a submission or option**.

This lane used the already frozen `YawBalance` mechanics with plain R10 servo control and plain R10 detection. Sixteen servo normal-only runs were calibrated with exactly the torque study's 60 bins × 100 ms, adjacent-bin maximum, leave-one-COM reserve and 5% margin. Candidate and model were pinned before any force result.

The unmodified original official evaluator improved from R10 **23/24** to candidate **24/24**, recovering `0 kg / 3.86 s / 10 N` at **88 ms**. All three no-force control rows stayed at zero false alarms. However, an official `8 kg / 3.0 s / 100 N` run gained a delayed false rise at **3.862 s** (R10 none), so even the official table contains a regression.

The predeclared physically different set confirms the veto. All six weak and two strong impacts remain detected and physical logs are exact; there is no weak candidate-only gain. The strong `-y` case at 3.515 s gains a delayed candidate-only false rise at **3.861 s**. The candidate therefore fails `no_added_force_false` and is not retuned after seeing it.

Files remain only as negative evidence. R10, the original course kit and `experiments/r21_yaw_impulse` are unchanged.
