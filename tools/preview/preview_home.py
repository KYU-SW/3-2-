"""로그인 없이 앱 화면을 미리 보는 하네스 (Firebase는 가짜). 화면 확인용, 저장소에 넣지 않음."""
import runpy
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[2]  # 이 파일은 tools/preview/ 안에 있다
sys.path.insert(0, str(APP_DIR))
import auth  # noqa: E402
import database  # noqa: E402

auth.is_authenticated = lambda: True
database.get_user_document = lambda uid: {"consent_agreed": True, "profile": {}}
from datetime import datetime, timedelta, timezone  # noqa: E402

_now = datetime.now(timezone.utc)
import base64  # noqa: E402
_photo = base64.b64encode(open(str(APP_DIR / "videos" / "sample_posture_photo.jpg"), "rb").read()).decode()
_fake = [  # 화면 확인용 가짜 기록 (최신이 먼저). 사진은 실제 영상 1개뿐이라 두 번 다 같은 사진·같은 각도
    {"view": "side", "created_at": _now - timedelta(days=2), "metrics": {"trunk_change": 11.0, "pattern": "semicircular", "head_cva_static": 71.0}, "judgments": {}, "is_demo": False, "posture_photo": _photo},
    {"view": "side", "created_at": _now - timedelta(days=30), "metrics": {"trunk_change": 18.0, "pattern": "arc", "head_cva_static": 71.0}, "judgments": {}, "is_demo": False, "posture_photo": _photo},
    {"view": "front", "created_at": _now - timedelta(days=2), "metrics": {"shoulder_tilt_static": 2.6}, "judgments": {}, "is_demo": False},
    {"view": "front", "created_at": _now - timedelta(days=30), "metrics": {"shoulder_tilt_static": 6.8}, "judgments": {}, "is_demo": False},
]
database.list_propulsion_sessions = lambda uid: list(_fake)
database.list_stretch_logs = lambda uid: []
st.session_state.setdefault("uid", "preview")
st.session_state.setdefault("email", "preview@example.com")
if "page" in st.query_params and "_page_from_url" not in st.session_state:  # ?page=exercise 처럼 바로 열기 (캡처용)
    st.session_state["page"] = st.query_params["page"]
    st.session_state["_page_from_url"] = True
runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")
