# 과제 제출 파일

이 폴더의 세 파일(`my_servo.py`, `my_torque.py`, `my_detector.py`)이 Week09 최종 제출본이다. 수업 키트 폴더(`icpbl_sim`)에 복사한 뒤 원본 평가기로 확인한다.

```bash
python evaluate.py --ctrl my_servo --detector my_detector
python evaluate.py --ctrl my_torque --detector my_detector
```

| 모드 | 수평 충돌 검출 | 무충돌 오경보 | 이전 제출(R10) |
|---|---:|---:|---:|
| 서보 | **24/24** | 0회 | 23/24 |
| 토크 | **24/24** | 0회 | 18/24 |

- 제어기는 R1 그대로다.
- 감지기는 연결된 제어기를 보고 방식을 고른다. 토크면 R31, 서보면 R32, 그 밖의 제어기면 R10이다.
- 조합 근거, 검증, 재현 방법과 한계는 [docs/FINAL_SUBMISSION.md](../docs/FINAL_SUBMISSION.md)에 있다.
