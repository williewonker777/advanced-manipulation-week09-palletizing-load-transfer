# 수업 예제와 저장소의 비교 영상은 무엇이 다른가?

| 무엇인가 | 실제 실행·출처 | 어떻게 확인했나 |
|---|---|---|
| 수업 자료의 8 kg 예제 | 별도 제공된 `icpbl_sim/run.py --box 8`. 제공 서보 제어기와 제공 감지기, 별도 외력·파지 오차 없음. 스크립트를 인자 없이 실행하면 **6 kg**다. | 원본 파일을 고치지 않고 다시 실행한 결과 그래프가 수업 가이드 HTML 내장 그래프와 바이트 단위로 일치했다. RMS 6.14 mm, 최대 14.00 mm, 오경보 7회. |
| 저장소의 첫 비교 영상 | 초기 코드 커밋은 `8c83ebe`, 첫 네 영상이 추가된 커밋은 `05091fc`. **수업 제어기와 당시 작성한 학생 제어기를 비교**해 따로 제작했다. | 네 MP4가 해당 커밋의 Git 파일과 지금도 바이트 단위로 같다. 이 영상을 현재 R10이나 후속 연구의 녹화라고 다시 이름 붙이지 않는다. |
| R14 이후 연구 | 물리 키트에 연구용 제어기·감지기와 검증 조건을 조합해 새로 실행했다. | 정상-외력 대조·힘·시간을 따로 기록했다. 원본 데모 영상도, 수업 기본 제어기의 결과도 아니다. [R24 하중·충격 실험](R24_LOADED_TRANSFER.md)도 여기에 해당한다. |

R24 검증 때도 원본 8 kg 예제를 별도로 다시 실행했다. 수업 가이드의 그림과 다시 일치했고, R10 기본 제출은 58개 공식 에피소드를 재검증했다. [원본 그림 지문과 실행 기록 요약](../results/r24/course_identity.json).

R29에서도 기본 파일을 지정하지 않은 `run.py --box 8`로 재확인했다. 출력 PNG와 교안 내장 PNG의 SHA-256은 둘 다 `82cb22df…`였고, 제공 키트 28파일은 실행 전후 같았다. [R29 당시 원본·R10 검증 기록](../results/r29/handoff.json).

R32에서도 제공 서보와 제공 감지기의 8 kg 원본을 별도로 재실행했다. 생성 PNG와 교안 그림의 SHA-256은 다시 같았다. **R32의 하중·경보 시간표는 그 원본 그림이나 비교 영상이 아니다.** [이번 원본·제출본 검증](../results/r32/original-and-release.json)

R33에서도 이 8 kg 원본의 생성 PNG가 교안과 바이트 단위로 일치했다. [R33 약한 수직 충격 그래프](R33_VERTICAL_IMPACTS.md)는 **연구용 토크 실행**이며 원본 화면이 아니다. [원본 검증표](../results/r33/original-and-submission.json)

### 저장소의 첫 네 비교 영상 (기존 파일 그대로)

1. [8 kg 서보: 외부 충돌 없음](../artifacts/media/01_servo_no_collision.mp4)
2. [8 kg 토크: 외부 충돌 없음](../artifacts/media/02_torque_no_collision.mp4)
3. [서보: 놓기 직전 3.86초에 수평 15 N](../artifacts/media/03_servo_place_collision.mp4)
4. [서보: 운반 중 3.00초에 수평 15 N](../artifacts/media/04_servo_transport_collision.mp4)

같은 네 조건을 제공 `run.py`로 **새로** 실행해 현재 R10과도 대조했다. 무충격 서보·토크의 TCP RMS는 수업 예제 6.14/7.04 mm, R10 1.17/3.23 mm였다. 놓기 직전 15 N에서 수업 감지기는 이미 경보 중이었고 R10은 67 ms에 구분했다. 운반 중 15 N은 각각 68 ms와 37 ms였다. 이는 위 영상 자체에서 읽어 낸 수치가 아니다. [조건 및 재실행 지표](../results/r17/demo_provenance.json).

수업 예제의 출처를 확인하려면 이 저장소의 인증 도구를 쓴다. 도구는 알려진 원본 키트·가이드 해시와 보호 파일을 확인하고, 존재하지 않는 새 출력 폴더에서 제공 `run.py --box 8`을 헤드리스로 실행한다.

```bash
KIT=/path/to/original/icpbl_sim
GUIDE=/path/to/IC-PBL_시뮬레이션_실습가이드.html
python -B tools/course_example.py \
  --kit "$KIT" --guide "$GUIDE" --out /path/to/new-course-example
```

인증한 실행에서는 생성 PNG가 가이드 내장 PNG와 바이트 단위로 일치했다. 자세한 수치와 R20 연구 결과는 [R20 문서](R20_ORIGINAL_AND_DETECTION.md)에 있다.

3D로 직접 보려면 그래픽 환경에서 **출력 전용 폴더를 현재 디렉터리로 두고** 아래 명령을 실행한다. 이 수동 뷰어 명령은 위 인증 도구의 격리 실행을 대신하지 않는다. 창을 닫으면 반복 재생이 끝난다.

```bash
python -B /path/to/original/icpbl_sim/run.py --box 8 --viewer
```

수업 원본 키트나 가이드의 내장 이미지는 이 공개 저장소에 복사하지 않는다.
