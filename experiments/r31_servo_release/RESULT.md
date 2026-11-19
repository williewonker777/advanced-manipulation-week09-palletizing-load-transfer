# R31 servo release notice — rejected

Status: **rejected by the prelocked prospective panel; no enrollment**.

The frozen experiment addressed the shipped R10 servo detector's solitary
official miss (`0 kg / 3.86 s / +Y 10 N`) without modifying `my_servo.py`,
`my_detector.py`, defaults or the course kit.

## Mechanism and disclosed diagnosis

The R21 servo-yaw option recovered the official miss but created delayed
3.861/3.862 s false rises after earlier strong impacts.  R31 did not retune that
yaw threshold.  It required agreement between two causal public channels:

- the delayed inverse-dynamics horizontal force estimate; and
- the existing shared-estimator horizontal wrench.

Both had to exceed the fixed 5 N physical-resolution bound, agree in direction
(cosine >= 0.8), and occur where table support was not admissible.  The source
opened a pending episode, emitted nothing at onset, survived the observer-invalid
grip-model transition, and reported only after three valid samples without full
coherent evidence if no native R10 alarm owned the episode.

On disclosed traces the distinction was genuine:

- target +Y10: coherent episode `3933..3988`, no native edge;
- +Y15/+Y40: coherent episodes with native edges `3938`/`3891`;
- historical yaw-false prior +Y100 and -Y40: no coherent release episode near
  the false `3862`/`3861` edge;
- matched no-force traces: no coherent episode.

## Known and official results

The 16 known executions passed.  Only the target changed, from no edge to one
eight-sample notice at tick `3991` (+131 ms).  +Y15/40/100, both historical
recoil counterexamples, and matched no-force traces retained exact R10 edges,
active occupancy and physics.

The authentic original official evaluator also passed its preregistered gate:

- R10 servo: **23/24**, one miss, zero no-impact false rises;
- candidate: **24/24**, zero misses, zero no-impact false rises;
- the only displayed collision-cell change was
  `0 kg / 3.86 s / 10 N: miss -> 131 ms`;
- all control cells were identical.

This official result alone does not qualify the candidate.

## Prospective failure

The candidate and six load/COM/direction cases were pinned before any output.
All 24 registered executions were retained.  The panel failed:

- R10 strict: **5/6**;
- candidate strict: **3/6**;
- candidate-only strict gains: **0**;
- candidate no-force edges: **0**.

Concrete failures:

- empty +Y9: R10 `[]`, candidate `[3896, 3975]` — two rises, not a valid gain;
- empty -Y10: R10 `[3928]`, candidate `[3928, 4002]` — duplicate;
- light +Y10: R10 `[3912]`, candidate `[3912, 3988]` — duplicate.

The +X, mid-load -X and heavy oblique rows retained their native edges.  All
physical arrays matched.  The counterexamples show that the coherent source can
release and reacquire across the physical release transition, while a brief
native servo rise does not necessarily own the earlier pending segment.  Thus
the design can split one real contact even though it fixes the official target.
Per the frozen falsifier, no threshold, lifetime or arbitration revision was
made after this result.

## Pins and evidence

- plan SHA-256: `2ddeb22ffadc56d6f11440e5d0a57ad3889f0f341e9a0575bc07a16e48f36e1b`
- candidate SHA-256: `2c1af9274f310dd84a9a8909987be7fe41e28fd89faf477be07087a7c2738303`
- prospective plan SHA-256: `55f271aee39fb20258bdf6e28e7e016c85320ddce24277a201c69710547b954d`
- known result SHA-256: `71c2d9b367d6d7ca81a210b6730764b601329d7f4319fd7cf6e6a080f1431c25`
- official result SHA-256: `6ecfee642d6e4c3c0e18eae3b8a77561259708c9a6448c97fbe59787056d99a4`
- prospective result SHA-256: `46bb7a3158c382e8b98b23357cd44c42e8d1fa0bc2149504a03a4148b09ea498`

## Reproduction

```bash
. "$E/env.sh"
cd "$R"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

.venv/bin/python -B -m experiments.r31_servo_release.diagnose \
  --out /fresh/path/diagnosis.json
.venv/bin/python -B -m experiments.r31_servo_release.episode_diagnose \
  --out /fresh/path/episodes.json
.venv/bin/python -B -m experiments.r31_servo_release.known \
  --out /fresh/path/known.json
.venv/bin/python -B -m experiments.r31_servo_release.official \
  --out /fresh/path/official
.venv/bin/python -B -m experiments.r31_servo_release.prospective \
  --out /fresh/path/prospective.json
```
