# Sweep: final-baseline-tool-torque

- Controller / detector: `torque_controller` / `my_detector`
- Workers: 1
- Cases: 1 (1 no-collision references, 0 collisions)
- Collision outcomes: {'detected': 0, 'missed': 0, 'alarm_before_collision': 0, 'indistinguishable': 0}

## No-collision metrics

| mass kg | COM m | RMS / max mm | pick / place step N m | place table N | false pick/place/other | saturation % |
|---:|---|---:|---:|---:|---:|---:|
| 8 | (0.02, -0.01, 0.03) | 7.88 / 14.42 | 57.8 / 72.2 | 333 | 0 / 1 / 0 | 0.0 |

## Timing

Timing distributions for every case are in JSON/CSV (median, p99, max in microseconds).

- Worst case controller: p99 32.93 us, max 73.53 us
- Worst case detector: p99 173.35 us, max 309.99 us
- Worst case combined: p99 201.51 us, max 341.47 us
