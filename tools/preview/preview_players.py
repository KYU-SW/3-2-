"""마네킹 확인용 페이지. ?id=운동id 로 하나만 볼 수 있음."""
import json
import sys
from pathlib import Path

import streamlit as st

APP = Path(__file__).resolve().parents[2]  # 이 파일은 tools/preview/ 안에 있다
sys.path.insert(0, str(APP))
import mannequin  # noqa: E402

if "photo" in st.query_params:
    import base64, styles
    from datetime import datetime, timezone, timedelta
    import app as wl
    styles.inject_custom_css()
    b = base64.b64encode(open(str(APP / "videos" / "sample_posture_photo.jpg"), "rb").read()).decode()
    now = datetime.now(timezone.utc)
    wl._photo_compare_card("🧍 가만히 있을 때 목 각도 (거북목)", b, b, "65°", "71°", "목 각도가 커졌어요 · 머리가 덜 나와요", 43.0, [now, now - timedelta(days=14)])
    st.stop()
st.set_page_config(layout="centered")
if "fig" in st.query_params:
    import posture_figure as f
    for t, s in [("neck", f.neck_angle(37, 46, 43)), ("trunk", f.trunk_lean(18, 11, 10.6)), ("trunk2", f.trunk_lean(7, 15, 10.6)), ("shoulder", f.shoulder_tilt(6.5, 2.1))]:
        st.markdown(f"<div style='border:1px solid #555;border-radius:12px;padding:8px;margin:8px 0'><b>{t}</b>{s}{f.legend('기준')}</div>", unsafe_allow_html=True)
    st.stop()

exercises = json.loads((APP / "data" / "exercises.json").read_text(encoding="utf-8"))
names = {item["id"]: item["name"] for item in exercises}
all_ids = [item["id"] for item in exercises if mannequin.has_motion(item["id"])]
ids = [st.query_params["id"]] if "id" in st.query_params else all_ids
st.title(f"관절 마네킹 {len(ids)}개")
for exercise_id in ids:
    st.markdown(f"### {names.get(exercise_id, exercise_id)}")
    mannequin.show_motion(exercise_id)
