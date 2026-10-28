# Limits and rejected approaches

- A vertical collision along the table normal through a feasible contact COP
  can produce the same wrench as table support. These inputs cannot identify
  its source. An additional contact model, vacuum/force sensing or a different
  process would be needed; no such hardware was validated here.
- Low force sensitivity, short tool strikes, other directions and some timing
  conditions remain failure modes. Inspect the failure maps rather than a
  single best-case delay. Alarm-before and indistinguishable are not detections.
- Nominal payload mass/COM are assumed known, as required by the kit. Unknown
  payload identification was not tested. Wrench inversion depends on Jacobian
  conditioning; the solve/lstsq fallback is not a global observability guarantee.
- The physical support cone and taught-height gate are approximations, not an
  exact contact-state classifier. A 35 mm height band can explain support-like
  forces before actual contact. The mass-zero guard prevents nonexistent payload
  support from being subtracted.
- The JTS direct-load projection saturates near one around release. Bounded
  fusion and scheduled unloading improve torque transients, but do not establish
  sustained transfer to the table before release. Tested overtravel/hold and
  two-stage slow approaches were rejected after worse force/saturation/TCP error.
- Alternative causal velocity estimators were rejected by a predeclared
  no-collision oracle-error guard. The retained estimator still has dynamic
  model/noise error. Oracle signals were used only in offline analysis.
- Thresholds were collision-blind, but finite no-collision training/holdout and
  default sensor-noise realization do not prove a universal zero false-alarm
  rate. Separate noise-seed tests remain finite sensitivity evidence.
- Strict every-cycle 1 ms and passivity/safety certification must not be inferred
  from medians, p99 or finite gain sweeps. Results are for a shared desktop host,
  not a real-time controller, industrial robot or certified safety channel.
- No collision-triggered stop or retreat is implemented. The kit scores detection
  while motion continues; detection results are not injury-risk evaluations.
