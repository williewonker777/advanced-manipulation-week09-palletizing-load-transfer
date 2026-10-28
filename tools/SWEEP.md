# Sweep runner

`sweep.py` calls `sim.runner.run_episode` directly. It does not patch `sim/`,
controllers, or detectors. Run the integrity check first (the runner repeats it
automatically), and always remove inherited ROS/PRISM Python paths:

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_integrity.py
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/sweep.py \
  --name baseline-servo --baseline --workers 4
```

The baseline preset is the exact default matrix from `evaluate.py`: control and
false-alarm metrics at 0/4/8 kg, plus 0/8 kg collisions at 1.42/3.0/3.86 s and
10/15/40/100 N in +y. It intentionally rejects mass, COM, collision, and pickup
offset overrides. Its physical metrics and collision classifications use the
same definitions as official evaluation; timing columns are additional.

## Custom grid

Repeat `--com` and `--collision` to form a grid. Every collision has a matching
no-collision reference for the same mass, COM, and pickup offset. A collision is
`time,fx,fy,fz` in world coordinates. Use the `--option=value` form when the
first vector component is negative so argument parsing cannot interpret it as
another option.

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/sweep.py \
  --name diagonal-example --controller servo_controller \
  --detector collision_detector --masses 2.5,8 \
  --com 0.02,-0.01,0.03 --com=-0.02,0.01,0.03 \
  --collision 1.42,20,20,0 --collision=3.86,0,0,-40 \
  --pick-offset 0.002,0,0 --workers 4
```

Outputs are limited to `results/sweeps/<name>/`:

- `results.json`: metadata and nested per-case metrics;
- `results.csv`: flattened per-case metrics;
- `SUMMARY.md`: no-collision table, timing worst cases, and collision outcomes.

Controller `compute` and detector `update` are timed separately on every cycle;
their paired sum is also reported. Each has median, p99, and maximum latency in
microseconds. These are wall-clock `perf_counter` measurements, so OS scheduling,
other simulator jobs, CPU frequency, and multiprocessing contention can inflate
tails. Compare timing runs on the same idle host and do not treat a single
maximum as a deterministic real-time bound.

When combining pickup offset with an artificial collision, the original
`collision_eval` merges tool-contact and artificial-force true windows. Its
earliest true-window alarm can then precede the artificial force and yield a
negative raw `detect_delay_ms`. This runner preserves that official behavior;
use `tool_contact_delay_ms` and the collision outcome/reference trace rather
than interpreting a negative delay as prediction. Interface-smoke records
include such a case; no simulator/evaluator logic was patched.

The command refuses to start below 6 GiB `MemAvailable` and accepts at most four
workers. Module arguments are Python module names, not file paths. Experiment
names cannot contain path separators, and symlinks cannot redirect the output
outside the work copy.
