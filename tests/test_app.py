"""Streamlit AppTest로 app.py 화면을 실제로 실행해본다 (Firebase는 가짜로 대체)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
import database, overlay, propulsion

saved = []
database.get_user_document = lambda uid: {"consent_agreed": True, "profile": {"rim_diameter_inch": 24.0, "dominant_hand": "both"}}
database.save_propulsion_session = lambda uid, **kw: saved.append(kw) or "doc1"
database.list_propulsion_sessions = lambda uid: []

ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(("OK  " if cond else "FAIL"), name, extra)
def errors(at):
    return [e.value for e in at.exception] + [e.value for e in at.error]

# 1) 로그인 전 화면
at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
check("login page renders", not at.exception and any("로그인" in t.label for t in at.tabs), errors(at))

# 2) 로그인 + 동의 완료 상태 → 5개 탭
at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
at.session_state["uid"] = "u1"; at.session_state["refreshToken"] = "r"; at.session_state["email"] = "a@b.c"
at.run()
check("home: menu buttons, no tabs", not at.exception and not at.tabs and all(any(b.label == x for b in at.button) for x in ("운동", "기록")), (errors(at), [b.label for b in at.button]))
[b for b in at.button if b.key == "home_start"][0].click().run()
check("exam page: back button + side/front choice", not at.exception and any(b.label == "← 홈" for b in at.button) and at.radio(key="exam_view").value == "side", errors(at))
question_radios = [r for r in at.radio if r.key and str(r.key).startswith("answer_")]
check("5 questions shown, none preselected", len(question_radios) == 5 and all(r.value is None for r in question_radios), [r.label for r in at.radio])

# 3) 분석 결과가 있을 때: 결과 표시 → 저장 버튼 → 실제 저장 호출
result = propulsion.PropulsionAnalysisResult("side", "left", 12.0, 0.95)
result.metrics = {
    "contact_angle": propulsion.MetricResult(95.0, "caution", "contact_angle_low"),
    "stroke_rate": propulsion.MetricResult(40.0, "pending", None),
    "pattern": propulsion.judge_pattern("arc"),
    "trunk_static": propulsion.MetricResult(2.0, "pending", None),
    "trunk_peak": propulsion.MetricResult(16.0, "pending", None),
    "trunk_change": propulsion.MetricResult(14.0, "pending", None),
    "trunk_sweep": propulsion.MetricResult(15.0, "pending", None),
}
result.pattern = "arc"; result.stroke_count = 7
result.trajectory = [(0, -60, "push"), (40, -50, "push"), (40, -40, "recovery"), (0, -55, "recovery")]
result.rim = (0.0, 0.0, 60.0)
result.debug = {"time": [0, 1], "visibility": [.9, .9], "wrist_x": [1, 2], "wrist_y": [3, 4], "forward_lean": [2.0, 16.0], "baseline_seconds": [3.0], "phase": ["push", "recovery"], "pattern_votes": ["s1: arc"], "stroke_count_compare": [7, 6], "frame_time": [0, 1, 2], "frame_visibility": [.9, .9, .1]}
at.session_state["side_result"] = result
at.session_state["side_saved"] = False
at.session_state["side_previous_change"] = 18.0
at.session_state["debug_mode"] = True
for qid, value in {"trunk_control": "yes", "pain": "3", "training": "no", "fit": "never", "fatigue": "fine"}.items():
    at.session_state[f"answer_{qid}"] = value
at.run()
metric_labels = [m.label for m in at.metric]
check("result metrics shown in Korean", "접촉 각도" in metric_labels and "추진 빈도" in metric_labels, metric_labels)
warnings_shown = [w.value for w in at.warning]
check("screening: technique first (result card)", any("먼저 확인해 볼 것" in m.value and "미는 기술" in m.value for m in at.markdown), [m.value for m in at.markdown if "hero" in m.value])
check("reference compare: trunk band + me + pattern", any("wl-ref" in m.value and "4.1~10.6°" in m.value and "나 +14°" in m.value and "Symonds" in m.value for m in at.markdown)
      and any("권장 <b>반원형</b>" in m.value and "호형" in m.value for m in at.markdown), [m.value[:120] for m in at.markdown if "wl-ref" in m.value])
check("result summary cards", all(x in metric_labels for x in ["밀 때 더 숙여진 정도", "미는 패턴", "1분에 미는 횟수"]), metric_labels)
markdowns = [m.value for m in at.markdown]
check("screening: video + answer reasons", any("**영상**" in m and "14°" in m for m in markdowns) and any("**답변**" in m and "배운 적이 없어요" in m for m in markdowns), [m for m in markdowns if "영상" in m or "답변" in m])
check("screening: pain notice", any("통증이 있었다고" in i.value for i in at.info), [i.value for i in at.info])
check("still vs moving cards", all(x in metric_labels for x in ["가만히 있을 때", "밀 때 (가장 숙였을 때)", "차이 (더 숙여진 정도)"]), metric_labels)
change_card = [m for m in at.metric if m.label == "차이 (더 숙여진 정도)"][0]
check("still vs moving: value and delta vs last time", change_card.value == "14.0°" and "-4.0°" in str(change_card.delta), (change_card.value, change_card.delta))
check("questions folded when all answered", any("답변 완료" in e.label for e in at.expander), [e.label for e in at.expander])
check("recommended pattern banner on top", any("권장 패턴(반원형)이 아니에요" in w.value for w in at.warning), [w.value for w in at.warning])
captions = [c.value for c in at.caption] + [m.value for m in at.markdown if "wl-fine" in m.value]
check("detail pattern as reference", any("참고(영상 기반 추정)" in c and "호형" in c for c in captions), captions)
check("stroke rate reference caption", any("50회/분" in c and "판정 기준 아님" in c and "67" not in c for c in captions), captions)
check("camera guide with source", any("동그란 원" in m.value for m in at.markdown) and any("wl-fine" in m.value and "130±30cm" in m.value and "Nourbakhsh" in m.value for m in at.markdown))
check("removed features gone", not any(n.label == "림 지름(인치)" for n in at.number_input) and not any("웹캠" in t.label or "자세 사진" in t.label for t in at.tabs), [t.label for t in at.tabs])
check("pending note shown once", sum("기준값이 아직 정해지지 않아" in c for c in captions) == 1 and not any("판정 보류(기준값 미설정)" in c for c in captions), captions)
guide_image = overlay.draw_camera_guide_example()
check("camera guide example image", guide_image.shape == (230, 340, 3) and guide_image.min() < 255, guide_image.shape)
for example_view in ("side", "front"):
    example = overlay.draw_reference_example(example_view)
    check(f"reference example image {example_view}", example.shape == (300, 340, 3) and example.min() < 255, example.shape)
check("stroke count comparison table", any("평균선" in str(t.value) for t in at.table), [str(t.value) for t in at.table])
check("coaching shown", any("뒤쪽에서 잡고" in w.value for w in at.warning) and any("림 아래로" in w.value for w in at.warning))
check("debug view renders without crash", not at.exception, errors(at))
save = [b for b in at.button if b.label == "측정 결과 저장"]
check("save button present", len(save) == 1)
save[0].click().run()
check("save actually called", len(saved) == 1 and saved[0]["metrics"]["contact_angle"] == 95.0 and saved[0]["judgments"]["pattern"] == "caution"
      and saved[0]["metrics"]["trunk_change"] == 14.0 and saved[0]["answers"].get("training") == "no", saved)
check("saved message", any("저장했어요" in s.value for s in at.success) and not at.exception, errors(at))
check("after save: go to exercise button", any(b.label == "맞춤 운동 보러 가기" for b in at.button), [b.label for b in at.button])
at.session_state["page"] = "exercise"; at.run()
check("exercise tab: result-based reasons (arc pattern; trunk 14° w/o fatigue not a keyword) and push technique first",
      any("내 검사 결과로 고른 운동" in m.value and "호형" in m.value and "더 숙여지고" not in m.value for m in at.markdown)
      and "휠체어 반원 밀기" in [h.value for h in at.subheader][:3],
      [h.value for h in at.subheader])
subs_now = [h.value for h in at.subheader]
check("only matching exercises shown", "휠체어 반원 밀기" in subs_now and not any(
    name in subs_now for name in ("팔 W 만들기", "옆구리 늘이기", "어깨 높이 맞추기")
), subs_now)
check("exercise evidence shown", any("wl-fine" in m.value and "Consortium" in m.value for m in at.markdown))

# 3-1) 정면 결과가 함께 있을 때: 어깨 기울기 표시 + 세팅 근거로 반영 + 따로 저장
at.session_state["page"] = "exam"
front = propulsion.PropulsionAnalysisResult("front", None, 14.0, 1.0)
front.metrics = {name: propulsion.MetricResult(value, "pending", None) for name, value in {
    "symmetry_index": 0.9, "timing_diff": 0.1, "shoulder_tilt": 7.2, "lateral_lean": 1.0, "lateral_lean_max": 3.0,
    "shoulder_tilt_static": 7.4, "shoulder_tilt_moving": 8.1, "shoulder_tilt_sway": 6.0,
    "lateral_lean_static": -1.2, "lateral_lean_moving": -2.0, "lateral_lean_sway": 5.0}.items()}
front.debug = {"tilt_time": [0.0, 1.0, 2.0], "shoulder_tilt_series": [7.0, 7.4, 8.0], "lateral_lean_series": [-1.0, -1.2, -2.0], "baseline_seconds": [1.0]}
at.session_state["front_result"] = front
at.session_state["front_saved"] = False
at.run()
check("front result renders", not at.exception and any("앞에서 본 결과" in m.value for m in at.markdown) and any("옆에서 본 결과" in m.value for m in at.markdown), errors(at))
front_cards = [(m.label, m.value) for m in at.metric if m.label in ("가만히 있을 때", "밀고 올 때", "밀 때 흔들린 폭")]
check("front cards say which side", ("가만히 있을 때", "왼쪽 7.4°") in front_cards and ("가만히 있을 때", "오른쪽 1.2°") in front_cards, front_cards)
markdowns = [m.value for m in at.markdown]
check("front sentence; lateral lean shown as reference only", any("왼쪽 어깨가 7° 낮아요" in m for m in markdowns) and not any("몸통이" in m and "기울어 있어요" in m for m in markdowns)
      and any("참고로만 보여드려요" in m for m in markdowns), [m for m in markdowns if "어깨" in m or "몸통" in m])
check("front finding listed under setup", any("영상: 가만히 있을 때 왼쪽 어깨가 7° 낮아요" in m for m in markdowns), [m for m in markdowns if "영상" in m])
check("two save buttons until saved (front only now)", len([b for b in at.button if b.label == "측정 결과 저장"]) == 1)
[b for b in at.button if b.label == "측정 결과 저장"][0].click().run()
check("front saved separately", len(saved) == 2 and saved[1]["view"] == "front" and saved[1]["metrics"]["shoulder_tilt_static"] == 7.4, saved[-1])
at.session_state["front_result"] = None

# 4) 경고가 있는 결과는 저장 버튼 없음
bad = propulsion.PropulsionAnalysisResult("side", "left", 3.0, 0.2); bad.warnings = ["관절이 잘 보이는 프레임이 20%뿐이에요"]
at.session_state["side_result"] = bad; at.session_state["side_saved"] = False
at.run()
check("warning blocks save", not [b for b in at.button if b.label == "측정 결과 저장"] and not at.exception, errors(at))
print("ALL PASS" if ok else "SOME FAILED")

# ---- 5·6단계 화면 ----
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
store = {"prop": [], "stretch": [], "posture": []}
def save_prop(uid, **kw):
    store["prop"].append({"id": f"p{len(store['prop'])}", "created_at": kw.get("created_at") or now, "metrics": kw["metrics"], "judgments": kw["judgments"], "is_demo": kw.get("is_demo", False), "view": kw.get("view", "side")})
    return "x"
database.save_propulsion_session = save_prop
database.list_propulsion_sessions = lambda uid: list(reversed(store["prop"]))
database.list_posture_sessions = lambda uid: store["posture"]
database.save_stretch_log = lambda uid, ex, created_at=None: store["stretch"].append({"created_at": created_at or now, "exercise_id": ex}) or "s"
database.list_stretch_logs = lambda uid: store["stretch"]
database.delete_demo_data = lambda uid: 0

at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
at.session_state["uid"] = "u1"; at.session_state["refreshToken"] = "r"; at.session_state["email"] = "a@b.c"
at.run()
check("home/exam/exercise/records render", not at.exception, errors(at))
check("home explains the flow", any("3초 가만히" in m.value for m in at.markdown))
[b for b in at.button if b.key == "home_to_exercise"][0].click().run()
check("exercise cards: push technique shown w/o history", any("휠체어 반원 밀기" in h.value for h in at.subheader))

demo = [b for b in at.button if b.label == "데모 데이터 불러오기"][0]
demo.click().run()
check("demo data saved: side+front x 4 dates", len(store["prop"]) == 8 and len({str(p["created_at"])[:10] for p in store["prop"]}) == 4 and all(p["is_demo"] for p in store["prop"]), [str(p["created_at"])[:10] for p in store["prop"]])
at.session_state["page"] = "records"; at.run()
at.radio(key="statistics_period").set_value("월간").run()
check("stats monthly charts render", not at.exception and any("손 궤적 패턴 변화" in h.value for h in at.subheader), errors(at))

check("records: neck + shoulder + trunk cards from demo", all(any(x in m.value for m in at.markdown) for x in ("거북목 · 목 각도", "어깨 높이 차이", "밀 때 상체가 더 숙여지는 정도")), errors(at))
check("records: compare card from demo data", any("지난 검사와 비교" in m.value for m in at.markdown) and any("wl-compare" in m.value and "→" in m.value for m in at.markdown)
      and any("예시 데이터" in m.value for m in at.markdown), errors(at))
at.session_state["page"] = "exercise"; at.run()
stretch = [b for b in at.button if b.label == "완료하기"][0]
stretch.click().run()
check("exercise log saved, card shows done", len(store["stretch"]) == 1 and not at.exception and any("오늘 완료했어요" in m.value for m in at.markdown), errors(at))

# 판별된 키워드가 있으면 해당 운동만 표시
store["prop"].append({
    "id": "neck",
    "created_at": now,
    "metrics": {"head_cva_static": 50.0, "pattern": "semicircular"},
    "judgments": {},
    "answers": {},
    "is_demo": False,
    "view": "side",
})
at.run()
subs = [h.value for h in at.subheader]
check(
    "exercise cards match detected keyword only",
    not at.exception
    and all(name in subs for name in ("팔 W 만들기", "10초 거북목 리셋", "목 앞쪽 늘이기"))
    and not any(name in subs for name in ("휠체어 반원 밀기", "어깨 높이 맞추기", "옆구리 늘이기")),
    subs,
)

# 경고 판정이 있는 최근 기록 → 관련 운동이 위로
store["prop"].append({"id": "w", "created_at": now, "metrics": {}, "judgments": {"modified_cva": "warning"}, "is_demo": False})
at.run()
subs = [h.value for h in at.subheader]
check("legacy warning record does not crash exercise tab", not at.exception and "휠체어 반원 밀기" in subs, subs)
print("ALL PASS" if ok else "SOME FAILED")
