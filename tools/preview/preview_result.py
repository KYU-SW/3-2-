"""로그인 없이 검사 결과 화면을 미리 보는 하네스 (Firebase·분석 결과 모두 가짜). 저장소에 넣지 않음."""
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
st.session_state.setdefault("uid", "preview")
st.session_state.setdefault("email", "preview@example.com")
if "side_result" not in st.session_state:
    result = propulsion.PropulsionAnalysisResult("side", "left", 12.0, 0.95)
    result.metrics = {
        "trunk_static": propulsion.MetricResult(2.0, "pending", None),
        "trunk_peak": propulsion.MetricResult(16.0, "pending", None),
        "trunk_change": propulsion.MetricResult(14.0, "pending", None),
        "stroke_rate": propulsion.MetricResult(42.0, "pending", None),
        "head_cva_static": propulsion.MetricResult(40.0, "pending", None),
    }
    real = propulsion.analyze_video_bytes(open(APP_DIR / "videos" / "validation" / "arc_02.mp4", "rb").read(), view="side", filmed_side="left")
    result.still_frame, result.still_time = real.still_frame, real.still_time
    result.debug = {"head_cva": real.debug["head_cva"], "head_cva_time": real.debug["head_cva_time"]}
    result.pattern = "arc"
    result.stroke_count = 7
    st.session_state["side_result"] = result
    st.session_state["side_saved"] = False
    for qid, value in {"trunk_control": "yes", "pain": "none", "training": "no", "fit": "recent", "fatigue": "fine"}.items():
        st.session_state[f"answer_{qid}"] = value
runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")
