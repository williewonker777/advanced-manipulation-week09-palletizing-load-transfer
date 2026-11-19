# 수업 원본과 R26 착지 연구를 따로 실행하기

수업에서 받은 **원본 키트와 HTML 가이드가 별도로 필요**하다. 이 묶음에는 원본 코드, 로봇 모델, 원본 결과 이미지가 없다. 수업 키트의 가상환경(Python + MuJoCo **3.17.0**)에서 압축을 푼 폴더로 이동한다. 기존 결과나 키트를 덮어쓰지 않도록 출력은 매번 **존재하지 않는 외부 디렉터리**를 지정한다.

```bash
KIT=/path/to/icpbl_sim
GUIDE=/path/to/IC-PBL_시뮬레이션_실습가이드.html
OUT=$(mktemp -d)
# 1. 수업 원본: 제공 서보+제공 감지기, 8 kg, 추가 외력 없음.
python -B tools/r26_landing_replay.py original --kit "$KIT" --guide "$GUIDE" --out "$OUT/original" --box 8
# 결과 그림: "$OUT/original/result.png". report.json이 교안 내장 PNG와의 바이트 일치를 기록한다.

# 2. 연구 비교: 같은 보정 착지 제어기에서 제공 키트의 물리를 사용하되 감지기는 직접 작성한 것.
python -B tools/r26_landing_replay.py replay --kit "$KIT" --out "$OUT/xplus" \
  --scenario signed-x-positive --arm r24 --arm registered-c1
```

원본은 실제로 `run.py --box 8`만 호출한다. 인자 없이 실행하는 6 kg 예제, 저장소의 비교 영상, `registered-c1`은 원본 8 kg 데모가 아니다. `describe`는 키트 없이도 연구 시나리오와 이름을 보여준다. `normal-sh5`, `signed-x-negative`, `signed-z-positive`, `signed-z-negative`도 같은 방법으로 실행할 수 있다.

연구 출력에서 `result.json`은 조건별 실제 경보와 대조 실험, `receipt.json`은 키트·소스·실행 환경 지문이다. `baseline`은 초기 토크 제어기와 기존 연구 감지기이며, 수업 원본도 R10 기본 제출 조합도 아니다. R26 연구는 실기 안전장치가 아니다. 경보가 짧아졌다고 해서 접촉이 그때 끝났다는 뜻이 아니다.
