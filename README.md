# Wheelance (휠런스)

앉은 자세와 휠체어 밀기 동작을 함께 확인하는 모바일 웹앱입니다.
영상에서 상체 자세와 바퀴 추진 패턴을 분석하고, 사용자가 확인할 수 있는 운동과 참고 정보를 제공합니다.

> Wheelance는 의료 진단 도구가 아니며, 대학 과제용 프로토타입입니다.
> 분석 결과는 전문 의료진의 평가를 대신할 수 없습니다.

## 주요 기능

- 앉은 자세와 추진 동작의 간단한 측면 분석
- 가만히 앉은 상태와 밀 때 상체 변화 비교
- 추진 패턴과 접촉 각도의 영상 기반 추정
- 모바일 화면에 맞춘 자세 측정 화면
- 운동 안내와 지난 기록 비교
- Firebase를 통한 로그인 및 사용자별 기록 저장

## 로컬 실행

Python 3.11을 권장합니다.

```powershell
cd wheelance
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

실행 후 브라우저에서 `http://localhost:8501`에 접속합니다.

## Firebase 설정

이 앱은 로그인과 기록 저장을 위해 Firebase를 사용합니다.
처음 사용하는 경우 아래 설정을 마친 뒤 실행하세요.

1. Firebase 프로젝트를 생성합니다.
2. Authentication에서 이메일/비밀번호 로그인을 활성화합니다.
3. Firestore Database를 생성합니다.
4. 프로젝트의 웹 앱 API 키를 준비합니다.
5. Firebase 서비스 계정의 새 비공개 키 JSON을 다운로드합니다.
6. 다운로드한 JSON 파일로 Secrets를 생성합니다.

```powershell
python make_secrets.py 다운로드한-서비스계정.json
```

생성된 `.streamlit/secrets.toml`에 Firebase 설정이 저장됩니다.
자세한 항목은 `.streamlit/secrets.toml.example`을 참고하세요.

> `.streamlit/secrets.toml`과 `*firebase-adminsdk*.json`은 비밀 파일입니다.
> GitHub이나 공개 문서에 올리지 마세요.

## Streamlit Community Cloud 배포

1. GitHub 저장소에서 `app.py`를 지정해 앱을 생성합니다.
2. Python 버전은 3.11을 선택합니다.
3. 앱의 **Secrets**에 `.streamlit/secrets.toml` 내용을 입력합니다.
4. `Deploy`를 실행합니다.

실시간 웹캠 코칭은 로컬 환경에서만 권장되며,
클라우드 환경에서는 안내 화면만 제공됩니다.

## 촬영 시 참고

- 측면 카메라와 바퀴 사이 거리: 약 130±30cm
- 카메라 높이: 바퀴 반지름 정도
- 카메라를 바퀴 면과 가급적 평행하게 배치
- 밝고 넓은 평평한 바닥 사용
- 평소 속도로 5회 이상 반복하고 5~20초 내외로 촬영
- 정면 사진은 좌석 기준점 2개, 측면 사진은 등받이 기준점 2개를 표시

## 테스트

```powershell
.venv\Scripts\python.exe tests/test_algo.py
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe tests/test_app.py
```

## 한계

- MediaPipe는 손목과 관절 근처 좌표를 추정하므로 모든 수치는 근사치입니다.
- 추진 패턴은 영상 규칙으로 추정한 것이며 실제 추진력은 측정하지 않습니다.
- 원본 영상과 사진은 저장하지 않고 측정 수치만 저장합니다.
