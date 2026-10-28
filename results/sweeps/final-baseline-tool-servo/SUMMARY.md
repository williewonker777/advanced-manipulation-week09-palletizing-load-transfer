# Sweep: final-baseline-tool-servo

- Controller / detector: `servo_controller` / `my_detector`
- Workers: 1
- Cases: 1 (1 no-collision references, 0 collisions)
- Collision outcomes: {'detected': 0, 'missed': 0, 'alarm_before_collision': 0, 'indistinguishable': 0}

## No-collision metrics

| mass kg | COM m | RMS / max mm | pick / place step N m | place table N | false pick/place/other | saturation % |
|---:|---|---:|---:|---:|---:|---:|
| 8 | (0.02, -0.01, 0.03) | 6.88 / 18.38 | 57.7 / 74.0 | 196 | 0 / 1 / 0 | 0.0 |

## Timing

Timing distributions for every case are in JSON/CSV (median, p99, max in microseconds).

- Worst case controller: p99 8.97 us, max 53.71 us
- Worst case detector: p99 166.73 us, max 302.11 us
- Worst case combined: p99 172.22 us, max 311.93 us
