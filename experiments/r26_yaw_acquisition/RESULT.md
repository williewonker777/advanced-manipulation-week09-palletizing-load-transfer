# R26 yaw reacquisition targeted verdict

Frozen candidate: `experiments.r26_yaw_acquisition.candidate:Detector`, SHA256
`212f4e79367cb636c0143a2d1f8aa7ba5584aaaca5ecde6b412d01a4985b6784`.

The candidate is eligible for broad historical QA. It exposes every native R25
fall and consumes only a confidently same-source yaw reacquisition when the
complete effective R25 route reconstruction says yaw is the sole live route.
The inferred continuous episode and consumption decision are telemetry, not a
bridged Boolean alarm. Route uncertainty fails open.

Targeted evidence under `E/r26/acquisition/`:

- Long signed target, three mechanics: one strict rise at the unchanged first
  time in all rows; zero duplicate/false/tail; exact force and physics.
- D4 case 299: exact R25 interval `[1919, 2085)`, 166 active samples, unchanged
  +19 ms delay. The rejected long candidate had extended this to 301 samples.
- R23 `p23_transfer_to_post_detach`, three mechanics: exact R25 event records
  and all safety/physics guards. The two actuator-skew first slots absent in R25
  remain absent rather than being claimed as a gain.
- R25 close same/opposed pairs, corrected short second, and pickup: 62 episodes,
  all protection claims pass.

No sealed R26 independent input was opened and the full D4 matrix was not rerun.
The compact receipt is `E/r26/acquisition/summary.json`.

Reproduction uses MuJoCo 3.17 and the project environment:

```sh
source "$E/env.sh"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

.venv/bin/python -B -m experiments.r26_yaw_acquisition.target \
  --archive "$E/r25/holdout/course-first.json" --out /tmp/r26-acquisition-target.json
.venv/bin/python -B -m experiments.r26_yaw_acquisition.history \
  --candidate experiments.r26_yaw_acquisition.candidate:Detector \
  --out /tmp/r26-acquisition-history.json
.venv/bin/python -B -m experiments.r26_yaw_acquisition.protect \
  --r25 "$E/r25/holdout/course-first.json" \
  --pickup "$E/r25/pickup/h1_targeted.json" \
  --out /tmp/r26-acquisition-protection.json
```
