# Wheelance (휠런스)

앉은 자세와 바퀴 미는 동작을 함께 보는 휠체어 사용자 자세 선별 모바일 웹앱 (대학 과제용 프로토타입, 의료 진단 도구 아님)

- 대상: 양손으로 추진하는 수동 휠체어 사용자와 보호자
- 흐름: 질문 5개 → 측면 영상 1개 (3초 가만히 → 밀기) → 가만히 있을 때 vs 밀 때 상체 각도 비교 + "먼저 확인해 볼 것"(몸 / 미는 기술 / 휠체어 세팅) → 운동 → 지난번과 비교
- 선택: 바퀴 테 3점을 찍으면 손 궤적 패턴·접촉 각도까지 분석
- 주의: "먼저 확인해 볼 것"의 점수 규칙은 가설이며 검증된 분류법이 아님 (`screening.py`, `config.SCREENING_PARAMS`)
- 근거: Consortium for Spinal Cord Medicine (2005). *Preservation of upper limb function following spinal cord injury.* J Spinal Cord Med, 28(5), 433–470.

---

## 1. 로컬에서 실행하기

Python **3.11**이 필요합니다. (3.12 이상에서는 mediapipe 0.10.14 설치가 실패할 수 있어요.)

```bash
cd wheelance
python3.11 -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Firebase 설정(2번)을 마친 뒤:

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`이 열립니다.

## 2. Firebase 설정 (처음 한 번)

1. [Firebase 콘솔](https://console.firebase.google.com)에서 **프로젝트 만들기**
2. **Authentication → 시작하기 → 로그인 방법 → 이메일/비밀번호** 사용 설정
3. **Firestore Database → 데이터베이스 만들기** (위치: `asia-northeast3 (서울)` 권장, 프로덕션 모드)
4. **프로젝트 설정(톱니바퀴) → 일반 → 웹 API 키** 복사
5. **프로젝트 설정 → 서비스 계정 → 새 비공개 키 생성** → JSON 파일 다운로드
6. **자동으로 만들기 (추천):** wheelance 폴더에서 아래를 실행하고, 물어보면 웹 API 키를 붙여넣습니다.
   ```bash
   python make_secrets.py 다운로드한파일.json
   ```
   `.streamlit/secrets.toml`이 만들어져요. (손으로 만들려면 `secrets.toml.example`을 복사해 채우면 됩니다.)
   - 웹 API 키가 안 보이면: 프로젝트 설정 → 일반 → 내 앱에서 `</>`(웹) 앱을 등록하면 `apiKey` 값이 나와요.

> `secrets.toml`은 비밀번호와 같습니다. `.gitignore`에 들어 있으니 GitHub에 절대 올리지 마세요.

Firestore 보안 규칙은 기본값(모두 거부)으로 두면 됩니다. 이 앱은 서버에서 관리자 권한(firebase-admin)으로 접근하고,
코드에서 로그인한 사용자 본인의 `users/{uid}` 아래만 읽고 쓰도록 제한합니다.

## 3. Streamlit Community Cloud 배포 (폰으로 접속하기)

1. 이 폴더를 GitHub 저장소에 올립니다. (`secrets.toml`이 빠졌는지 확인)
2. [share.streamlit.io](https://share.streamlit.io) → **Create app** → 저장소와 `app.py` 선택
3. **Advanced settings → Python version: 3.11** 선택
4. **Secrets** 칸에 `secrets.toml` 내용을 그대로 붙여넣기 → Deploy
5. 생성된 주소를 QR코드로 만들어 발표 슬라이드에 넣으면 누구나 폰으로 접속할 수 있어요.

- `packages.txt`는 OpenCV가 클라우드에서 필요로 하는 시스템 라이브러리 목록입니다. 지우지 마세요.
- 실시간 웹캠 코칭은 클라우드에서 TURN 서버가 필요해서 **로컬 실행에서만** 켜집니다. 클라우드에서는 안내 문구만 보입니다.

## 4. 휴대폰 홈 화면에 추가하기 (앱처럼 쓰기)

- **안드로이드(Chrome):** 사이트 접속 → 오른쪽 위 ⋮ → **홈 화면에 추가**
- **아이폰(Safari):** 사이트 접속 → 아래 공유 버튼 → **홈 화면에 추가**

## 5. 촬영 가이드 (분석 정확도에 가장 큰 영향)

- 카메라 배치 (측면): 바퀴에서 **130±30cm**, 높이는 **바퀴 반지름 정도(약 30±10cm)**, 렌즈가 **바퀴 면과 평행(±5°)**
  - 출처: Nourbakhsh S, Archambault PS, Martinez Rocha J, Monacelli E. Computer Vision-Based Classification of Manual Wheelchair Propulsion Patterns. *IEEE Trans Neural Syst Rehabil Eng*. 2026;34:2786–2794. doi:10.1109/TNSRE.2026.3697633
  - 재지 않고 맞추는 법: 큰 걸음 두 걸음 거리, 무릎 높이, 화면에서 **바퀴가 동그란 원**으로 보이면 평행 (타원이면 비스듬함)
- 평평한 바닥, 밝은 곳
- 측면: 몸의 정확한 옆면, 카메라를 향한 팔이 잘 보이게. 앱에서는 **앉은 사람 기준**으로 보이는 쪽을 고릅니다 (왼쪽 몸이 보이면 사람이 화면 왼쪽을 바라봄). 보호자가 나란히 걸으며 찍거나 삼각대 고정
- 평소 속도로 **5회 이상** 밀기, 영상 **5~20초**
- 측면 분석 전에 첫 화면에서 **림(바퀴 손잡이 테) 위의 서로 멀리 떨어진 3점**을 찍습니다
- 자세 사진: 정면은 **좌석 2점**, 측면은 **등받이 위·아래 2점**을 찍으면 폰이 기울어도 휠체어 기준으로 각도를 잽니다

## 6. 내가 채워야 할 TODO

| 위치 | 내용 | 비워두면 |
| --- | --- | --- |
| `config.py` → `THRESHOLDS` | 항목별 정상/주의/경고 기준값과 `source`(출처) | "판정 보류"로 표시 (앱은 정상 작동) |
| `config.py` → `THRESHOLDS["elbow_angle"]` | 팔꿈치 권장 각도 (가이드라인 원문 확인) | 판정 보류 |
| `config.py` → `THRESHOLDS["modified_cva"]` | 거북목 기준 (서 있는 자세 연구값이면 "좌식 참고치"로 표기) | 판정 보류 |
| `data/exercises.json` → `youtube_url` | 운동별 실제 영상 링크 | "운동 영상 보기" 버튼이 숨겨짐 |
| `config.py` → `ALGORITHM_PARAMS` | 실제 영상으로 조정할 알고리즘 값 (의학 기준 아님) | 기본값으로 작동 |

**ALGORITHM_PARAMS 조정 방법:** 사이드바 **디버그 모드**를 켜고 실제 영상을 분석하면,
프레임별 밀기/복귀 구분과 스트로크별 패턴 판단 근거가 보입니다.
스트로크 수도 두 방식(기존: 림 접촉+앞쪽 이동 / 대안: 손목 x좌표가 평균선을 3번 지나면 1회, Nourbakhsh et al. 2026)으로 비교해 보여줍니다.
밀기가 잘 안 잡히면 `rim_contact_tolerance_ratio`를 올리고, 잡음이 스트로크로 잡히면 `min_push_frames`를 올리세요.

## 7. 테스트

```bash
python tests/test_algo.py          # 합성 궤적으로 패턴 4종·접촉 각도·추진 빈도 검증
python tests/test_pipeline.py      # 실제 영상 파일로 전체 분석 흐름 (카메라 이동·노이즈 포함)
python tests/test_posture_live.py  # 자세 사진 기준선 보정, 실시간 코칭 프레임 처리
python tests/test_app.py           # 앱 화면 실행 (Firebase는 가짜로 대체)
```

## 8. 파일 구조

```
wheelance/
├── app.py               # 화면 (로그인·동의·5개 탭·사이드바)
├── config.py            # 임계값, 출처, 알고리즘 설정, 코칭 문구
├── propulsion.py        # 추진 영상 분석 (측면/정면)
├── posture.py           # 정지 자세 사진 분석 (휠체어 기준선)
├── live_coach.py        # 실시간 웹캠 코칭 (로컬 전용)
├── pose_utils.py        # MediaPipe 공통 처리, 좌표·각도 계산
├── overlay.py           # 스켈레톤·궤적·림 합성
├── auth.py              # Firebase 로그인 (REST API)
├── database.py          # Firestore 저장/조회/삭제
├── timer_component.py   # 체중 이동 알림 타이머
├── make_secrets.py      # Firebase 키 파일로 secrets.toml 자동 생성
├── styles.py            # 모바일·접근성 CSS
├── data/exercises.json  # 재활 운동 목록
├── tests/               # 검증 스크립트
├── .streamlit/          # 테마, secrets 예시
├── requirements.txt
└── packages.txt         # 클라우드용 시스템 라이브러리
```

## 9. 한계 (발표 때 먼저 말하면 좋은 것)

- MediaPipe는 손이 아니라 **손목**을 찍고, 견봉·이주·C7 같은 해부학 지점이 없어서 모든 값은 **근사치**입니다.
- 손 궤적 패턴은 가이드라인의 4가지 정의를 규칙으로 옮긴 **영상 기반 추정**이며, 추진력(힘)은 측정하지 않습니다.
- 판정 기준값은 출처를 확인해 직접 입력해야 하며, 입력 전에는 "판정 보류"로 표시됩니다.
- 원본 영상·사진은 저장하지 않고 측정 수치만 저장합니다.

## 10. GitHub에 올리기 전 체크리스트

Firebase 서비스 계정 JSON과 `.streamlit/secrets.toml`은 절대 저장소에 올리지
마세요. 이 파일들은 `.gitignore`에 등록되어 있습니다. 공개하기 전에 아래 명령으로
추적 상태를 확인할 수 있습니다.

```powershell
git status --short
git check-ignore -v .streamlit/secrets.toml
git check-ignore -v *firebase-adminsdk*.json
```

GitHub에서 빈 저장소를 만든 뒤, 아직 원격 저장소가 없다면 다음처럼 연결합니다.

```powershell
git init
git add .
git status
git commit -m "Initial Wheelance release"
git branch -M main
git remote add origin https://github.com/<사용자명>/<저장소명>.git
git push -u origin main
```

`git status`에 비밀 파일이 보이면 커밋하거나 push하지 말고 `.gitignore` 설정을
먼저 확인하세요. Streamlit Community Cloud 배포 시에는 저장소에 키를 추가하지
말고, 앱 설정의 **Secrets**에 `secrets.toml` 내용을 입력합니다.
