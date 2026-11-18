"""로그인 없이 '찍어둔 영상 올리기' 뒤 화면을 미리 보는 하네스. ?v=영상이름 으로 실제 분석 결과를 넣는다. 저장소에 넣지 않음."""
import runpy
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[2]  # 이 파일은 tools/preview/ 안에 있다
sys.path.insert(0, str(APP_DIR))
import auth  # noqa: E402
import database  # noqa: E402
import propulsion  # noqa: E402

auth.is_authenticated = lambda: True
database.get_user_document = lambda uid: {"consent_agreed": True, "profile": {}}
database.list_propulsion_sessions = lambda uid: []
database.list_stretch_logs = lambda uid: []
database.save_propulsion_session = lambda uid, **kw: "x"
st.session_state.setdefault("uid", "preview")
st.session_state.setdefault("email", "preview@example.com")
st.session_state.setdefault("page", "exam")
st.session_state.setdefault("side_source", "찍어둔 영상 올리기")
st.session_state.setdefault("front_source", "찍어둔 영상 올리기")
name = st.query_params.get("v")
if name and st.session_state.get("_demo_video") != name:
    view = "front" if name.startswith(("flat", "asym")) or name.endswith("f.mp4") else "side"
    st.session_state["exam_view"] = view
    st.session_state[f"{view}_result"] = propulsion.analyze_video_bytes(
        open(APP_DIR / "videos" / ("broken_samples" if name.startswith(("rot", "shake", "tilt", "far", "dark", "oblique")) else "validation") / name, "rb").read(), view=view, filmed_side=None)
    st.session_state[f"{view}_saved"] = False
    st.session_state["_demo_video"] = name
runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")
