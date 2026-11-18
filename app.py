"""Wheelance Streamlit 앱 진입점과 인증·동의·추진 분석 화면."""

from __future__ import annotations

import hashlib
import html
import io
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

import auth
import camera_recorder
import database
import keywords
import mannequin
import overlay
import posture_figure
import propulsion
import screening
from config import (
    APP_NAME,
    APP_TAGLINE,
    CAMERA_GUIDE_DETAIL,
    CAMERA_GUIDE_STEPS,
    CAMERA_GUIDE_TIP,
    COACHING_MESSAGES,
    FRONT_GUIDE_STEPS,
    FRONT_GUIDE_TIP,
    JUDGMENT_LABELS,
    JUDGMENT_PENDING,
    LANDMARK_NOTICE,
    EXERCISE_PROGRAM,
    MEDICAL_DISCLAIMER,
    POSTURE_KEYWORDS,
    REFERENCE_COMPARISONS,
    SCREENING_NOTICE,
    SCREENING_PARAMS,
    SCREENING_QUESTIONS,
    SOURCE_CV_PATTERN,
    STROKE_RATE_REFERENCE,
    SUPPORTED_USERS_NOTICE,
)
from styles import inject_custom_css


st.set_page_config(
    page_title=f"{APP_NAME} | 휠체어 자세 선별",
    page_icon="♿",
    layout="centered",
    initial_sidebar_state="collapsed",
)
inject_custom_css()
auth.initialize_auth_state()


UploadType = Any
ProgressCallback = Callable[[float], None]


# 측정 항목 한국어 이름과 단위 (화면 표시용)
METRIC_LABELS: dict[str, tuple[str, str]] = {
    "contact_angle": ("접촉 각도", "°"),
    "stroke_rate": ("추진 빈도", "회/분"),
    "pattern": ("손 궤적 패턴", ""),
    "trunk_lean_mean": ("상체 숙임 평균", "°"),
    "trunk_lean_max": ("상체 숙임 최대", "°"),
    "trunk_lean_range": ("상체 숙임 변동폭", "°"),
    "elbow_angle": ("팔꿈치 각도", "°"),
    "symmetry_index": ("좌우 대칭 지수", ""),
    "timing_diff": ("좌우 타이밍 차이", "초"),
    "shoulder_tilt": ("어깨 기울기", "°"),
    "lateral_lean": ("측방 기울기 평균", "°"),
    "lateral_lean_max": ("측방 기울기 최대", "°"),
    "pelvic_tilt": ("골반 경사", "°"),
    "modified_cva": ("거북목 각도(CVA)", "°"),
    "trunk_flexion": ("체간 굴곡", "°"),
    "trunk_static": ("가만히 있을 때 상체 각도", "°"),
    "trunk_peak": ("밀 때 가장 숙였을 때", "°"),
    "trunk_change": ("밀 때 더 숙여진 정도", "°"),
    "trunk_sweep": ("밀 때 상체가 오간 폭", "°"),
    "shoulder_tilt_static": ("어깨 기울기 (가만히)", "°"),
    "shoulder_tilt_moving": ("어깨 기울기 (밀고 올 때)", "°"),
    "shoulder_tilt_sway": ("어깨가 흔들린 폭", "°"),
    "lateral_lean_static": ("몸통 옆 기울기 (가만히)", "°"),
    "lateral_lean_moving": ("몸통 옆 기울기 (밀고 올 때)", "°"),
    "lateral_lean_sway": ("몸통이 옆으로 흔들린 폭", "°"),
    "head_cva_static": ("목 각도 (가만히, 클수록 바름)", "°"),
    "head_cva_moving": ("목 각도 (밀 때)", "°"),
    "head_tilt_static": ("머리 옆 기울기 (가만히)", "°"),
    "head_tilt_moving": ("머리 옆 기울기 (밀고 올 때)", "°"),
}
# 기록 탭 추이 그래프: 자세(가만히) → 움직임(밀 때) 순서
TREND_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("🧍 자세 (가만히 있을 때)", ("head_cva_static", "shoulder_tilt_static", "lateral_lean_static", "head_tilt_static")),
    ("🧑‍🦽 움직임 (밀 때)", ("trunk_change", "shoulder_tilt_moving", "stroke_rate")),
)
# 각 수치가 무엇을 뜻하는지 (카드의 ? 도움말)
METRIC_HELP: dict[str, str] = {
    "trunk_static": "처음 가만히 앉아 있을 때의 상체 각도예요. 0°는 똑바로 세운 자세, +는 앞으로 숙임, −는 뒤로 기댐이에요.",
    "trunk_peak": "미는 동안 상체가 가장 앞으로 숙여졌을 때의 각도예요. (잠깐 튄 값은 빼고 계산해요)",
    "trunk_change": "가만히 있을 때보다 밀 때 상체가 얼마나 더 숙여지는지예요. 클수록 밀 때 상체를 많이 써요.",
    "trunk_sweep": "미는 동안 상체가 앞뒤로 움직인 폭이에요.",
    "contact_angle": "한 번 밀 때 손이 바퀴 테를 따라 이동한 각도예요. 클수록 길게 미는 거예요.",
    "stroke_rate": "1분에 몇 번 미는지예요. 같은 속도라면 적게 미는 쪽이 어깨 부담이 적어요.",
    "elbow_angle": "밀다가 손이 바퀴 맨 위에 왔을 때 팔꿈치가 펴진 정도예요. 180°면 완전히 편 상태예요.",
    "trunk_lean_mean": "영상 전체에서 상체가 수직에서 기운 정도의 평균이에요. (앞뒤 방향 구분 없음)",
    "trunk_lean_max": "영상 전체에서 상체가 수직에서 가장 많이 기운 각도예요.",
    "trunk_lean_range": "영상 전체에서 상체 기울기가 변한 폭이에요.",
    "shoulder_tilt_static": "출발 전 가만히 있을 때, 양쪽 어깨를 이은 선이 수평에서 기운 각도예요. 낮은 쪽 어깨를 알려줘요.",
    "shoulder_tilt_moving": "밀고 오는 동안의 어깨 기울기(가운데 값)예요.",
    "shoulder_tilt_sway": "밀고 오는 동안 어깨선이 좌우로 흔들린 폭이에요. 클수록 밀 때 어깨가 번갈아 많이 움직여요.",
    "lateral_lean_static": "출발 전 가만히 있을 때, 골반 가운데에서 어깨 가운데로 이은 선이 수직에서 기운 각도예요.",
    "lateral_lean_moving": "밀고 오는 동안의 몸통 옆 기울기(가운데 값)예요.",
    "lateral_lean_sway": "밀고 오는 동안 몸통이 좌우로 흔들린 폭이에요.",
    "symmetry_index": "양손이 위아래로 움직인 폭의 비율이에요. 1에 가까울수록 양손이 비슷하게 움직여요.",
    "timing_diff": "왼손과 오른손이 가장 높이 올라간 시점의 차이예요. 작을수록 양손을 같이 써요.",
}
# 결과 화면에서 크게 보여줄 핵심 항목 (나머지는 "나머지 측정값 보기"로 접는다)
PRIMARY_METRICS: tuple[str, ...] = ("contact_angle", "stroke_rate")
# 정지 vs 동작 비교 블록에서 따로 보여주므로 카드 목록에서는 빼는 항목
COMPARISON_METRICS: tuple[str, ...] = (
    "trunk_static", "trunk_peak", "trunk_change",
    "shoulder_tilt_static", "shoulder_tilt_moving", "shoulder_tilt_sway",
    "lateral_lean_static", "lateral_lean_moving", "lateral_lean_sway",
    # 정면의 예전 방식 값(방향 구분 없는 평균)은 위 값과 겹쳐서 숨긴다
    "shoulder_tilt", "lateral_lean", "lateral_lean_max",
)
PATTERN_NAMES: dict[str, str] = {
    "semicircular": "반원형 (권장)",
    "arc": "호형",
    "single_looping_over": "단일 루프형",
    "double_looping_over": "이중 루프형",
    "unknown": "판정 보류",
}
# 측면 영상에서 카메라 쪽 몸(왼쪽/오른쪽)은 propulsion.detect_filmed_side()가 자동으로 고른다.
CHART_PURPLE = "#8A5CC7"  # 그래프 선 색: 밝은·어두운 화면 모두에서 보이는 보라
RIM_CLICK_DISPLAY_WIDTH = 340  # 림 클릭 이미지 표시 폭(px). 모바일 화면에 들어가는 크기.


def _safe_float(value: Any) -> float | None:
    """수치 값을 화면 표시용 float로 변환합니다."""
    try:
        if value is None:
            return None
        converted = float(value)
        return converted if converted == converted else None
    except (TypeError, ValueError):
        return None


def _metric_value(value: Any, unit: str = "") -> str:
    """측정값을 단위와 함께 과도한 소수점 없이 표시합니다."""
    if isinstance(value, str):
        return PATTERN_NAMES.get(value, value)
    numeric = _safe_float(value)
    if numeric is None:
        return "–"
    return f"{numeric:.1f}{unit}" if unit else f"{numeric:.2f}"


def _judgment_label(judgment: str | None) -> str:
    """판정 키를 설정된 접근성 문구로 변환합니다."""
    return JUDGMENT_LABELS.get(str(judgment), JUDGMENT_LABELS["pending"])


def _fine(text: str) -> None:
    """출처·참고 설명처럼 찾는 사람만 보면 되는 문장: 작고 흐린 글씨."""
    st.markdown(f"<p class='wl-fine'>{html.escape(str(text))}</p>", unsafe_allow_html=True)


def _show_disclaimer() -> None:
    """모든 화면 공통 면책 문구를 표시합니다."""
    st.markdown(f'<div class="wheelance-disclaimer">{MEDICAL_DISCLAIMER}</div>', unsafe_allow_html=True)


def _show_error(message: str) -> None:
    """앱을 중단하지 않고 오류를 표시합니다."""
    st.error(message)


def _show_login_page() -> None:
    """비로그인 상태의 소개·로그인·회원가입 화면을 표시합니다."""
    st.markdown(f"<p class='wl-brand'>{APP_NAME}</p>", unsafe_allow_html=True)
    st.markdown("<p class='wl-login-subtitle'>휠체어 사용자를 위한 자세 체크</p>", unsafe_allow_html=True)

    login_tab, signup_tab = st.tabs(["로그인", "회원가입"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("이메일", key="login_email", placeholder="name@example.com")
            password = st.text_input("비밀번호", type="password", key="login_password")
            submitted = st.form_submit_button("로그인")
        if submitted:
            try:
                auth.sign_in(email, password)
                st.rerun()
            except auth.AuthError as error:
                _show_error(str(error))

    with signup_tab:
        with st.form("signup_form"):
            email = st.text_input("이메일", key="signup_email", placeholder="name@example.com")
            password = st.text_input("비밀번호", type="password", key="signup_password")
            password_confirm = st.text_input("비밀번호 확인", type="password", key="signup_password_confirm")
            submitted = st.form_submit_button("회원가입")
        if submitted:
            if password != password_confirm:
                _show_error("비밀번호가 서로 달라요.")
            else:
                try:
                    session = auth.sign_up(email, password)
                    try:
                        database.ensure_user_document(session.uid, session.email)
                    except database.DatabaseError:
                        pass
                    st.rerun()
                except auth.AuthError as error:
                    _show_error(str(error))


def _load_user_document() -> dict[str, Any]:
    """현재 로그인 사용자의 문서를 읽고 실패 시 빈 문서를 반환합니다."""
    uid = str(st.session_state.get("uid", ""))
    try:
        document = database.get_user_document(uid)
        return document or {}
    except database.DatabaseError as error:
        st.warning(str(error))
        return {}


def _needs_consent(user_document: dict[str, Any]) -> bool:
    """민감정보 수집 동의를 아직 받지 않았는지 확인합니다."""
    return user_document.get("consent_agreed") is not True


def _show_consent_page() -> None:
    """첫 로그인 후 민감정보 수집 동의 화면을 표시합니다."""
    st.markdown("# 데이터 수집 동의")
    st.markdown(
        "Wheelance는 추진·자세 분석을 위해 아래와 같이 측정 수치를 처리합니다."
    )
    st.markdown(
        """
        <div class="wheelance-card">
        <strong>수집하는 항목</strong><br>
        상체 각도(가만히 있을 때·밀 때), 접촉 각도, 추진 빈도, 손 궤적 패턴, 팔꿈치 각도 등
        분석 결과의 측정 수치<br>
        질문 5개에 대한 답변 (통증 여부, 상체 조절, 휠체어 사용 경험 등 건강 관련 정보 포함)
        </div>
        <div class="wheelance-card">
        <strong>선택해서 저장하는 항목</strong><br>
        결과 화면에서 직접 고른 경우에만: 얼굴을 흐리게 처리하고 머리~가슴만 잘라 낸 자세 사진 1장 (기록 비교용)
        </div>
        <div class="wheelance-card">
        <strong>수집하지 않는 항목</strong><br>
        원본 영상, 얼굴이 보이는 사진, 영상 프레임 파일
        </div>
        <div class="wheelance-card">
        <strong>수집 목적</strong><br>
        앉은 자세와 미는 동작의 변화를 확인하고, 먼저 확인해 볼 것을 안내하기 위해 사용합니다.<br><br>
        <strong>삭제 방법</strong><br>
        로그인 후 사이드바의 “내 데이터 전체 삭제” 기능으로 삭제할 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
    agreed = st.checkbox("위 내용을 읽었으며 측정 수치와 질문 답변 수집에 동의합니다.", key="consent_checkbox")
    if st.button("동의하고 시작하기", disabled=not agreed, use_container_width=True):
        try:
            database.set_consent(str(st.session_state["uid"]), True)
            st.rerun()
        except database.DatabaseError as error:
            _show_error(str(error))
    _show_disclaimer()


def _render_header() -> None:
    """로그인 후 공통 헤더: 홈에서는 앱 이름만, 다른 화면에서는 '← 홈' 버튼과 화면 이름."""
    page = st.session_state.get("page", "home")
    if page == "home":
        st.markdown(f"<p class='wl-brand'>{APP_NAME}</p>", unsafe_allow_html=True)
        return
    with st.container(key="page_bar"):
        back, title = st.columns([1, 3], vertical_alignment="center")
        back.button("← 홈", key="back_home", on_click=_go_to_tab, args=("홈",))
        title.markdown(f"<p class='wl-page-title'>{PAGE_TITLES[page]}</p>", unsafe_allow_html=True)


def _render_metric_card(name: str, metric: Any) -> None:
    """MetricResult 하나를 카드로 표시합니다."""
    label, unit = METRIC_LABELS.get(name, (name, ""))
    value = getattr(metric, "value", None)
    judgment = getattr(metric, "judgment", "pending")
    st.metric(label, _metric_value(value, unit), help=METRIC_HELP.get(name))
    # 판정 보류는 카드마다 반복하지 않고 _render_metric_cards()가 맨 아래에 한 번만 안내한다.
    if judgment != JUDGMENT_PENDING:
        st.caption(_judgment_label(judgment))
    if name == "stroke_rate":
        _fine(STROKE_RATE_REFERENCE)


def _render_metric_cards(items: list[tuple[str, Any]]) -> None:
    """측정값 카드를 2열로 그립니다."""
    for start in range(0, len(items), 2):
        columns = st.columns(2)
        for column, (name, metric) in zip(columns, items[start : start + 2]):
            with column:
                _render_metric_card(name, metric)


def _render_pending_note(items: list[tuple[str, Any]]) -> None:
    """판정 보류 항목이 있으면 안내를 한 번만 표시합니다."""
    if any(getattr(metric, "judgment", JUDGMENT_PENDING) == JUDGMENT_PENDING for _, metric in items):
        _fine("판정 표시가 없는 항목은 기준값이 아직 정해지지 않아 수치만 보여드려요. 각 수치 옆 ?를 누르면 뜻을 볼 수 있어요.")


def _render_recommended_pattern(result: Any) -> None:
    """측면 결과 맨 위: 권장 패턴(semicircular) 여부를 크게, 세부 패턴은 참고로 작게 표시합니다."""
    pattern = str(getattr(result, "pattern", "unknown"))
    if pattern == "semicircular":
        st.success("### ✓ 권장 패턴(반원형)으로 밀고 있어요")
    elif pattern == "unknown":
        st.info("### – 권장 패턴 여부 판정 보류")
    else:
        st.warning("### ! 권장 패턴(반원형)이 아니에요")
    _fine(
        f"참고(영상 기반 추정): 세부 패턴 {PATTERN_NAMES.get(pattern, pattern)} · "
        f"스트로크 {getattr(result, 'stroke_count', 0)}회 기준 · "
        "손이 돌아올 때 밀 때보다 낮게 오면 반원형, 같은 높이면 호형, 높게 오면 루프형으로 봐요."
    )


def _metric_number(result: Any, name: str) -> float | None:
    """결과에서 숫자 측정값 하나를 꺼냅니다 (없거나 NaN이면 None)."""
    metric = (getattr(result, "metrics", {}) or {}).get(name)
    return _safe_float(getattr(metric, "value", None))


def _current_answers() -> dict[str, str]:
    """화면에서 고른 질문 답변을 모읍니다 (아직 안 고른 질문은 빠짐)."""
    answers: dict[str, str] = {}
    for question in SCREENING_QUESTIONS:
        value = st.session_state.get(f"answer_{question['id']}")
        if value is not None:
            answers[str(question["id"])] = str(value)
    return answers


def _previous_session() -> dict[str, Any] | None:
    """가장 최근에 저장한 실제(데모 아님) 측면 기록을 가져옵니다. 없으면 None."""
    try:
        sessions = database.list_propulsion_sessions(str(st.session_state["uid"]))
    except database.DatabaseError:
        return None
    for session in sessions:
        if not session.get("is_demo") and (session.get("metrics") or {}).get("trunk_change") is not None:
            return session
    return None


def _render_screening(side_result: Any, front_result: Any = None) -> None:
    """결과 맨 위: 옆·앞 영상과 답변을 합쳐 "먼저 확인해 볼 것"과 그 근거를 보여줍니다."""
    answers = _current_answers()
    measured: dict[str, Any] = {}
    if side_result is not None:
        measured["trunk_change"] = _metric_number(side_result, "trunk_change")
        measured["pattern"] = getattr(side_result, "pattern", "unknown")
    if front_result is not None:
        measured["shoulder_tilt_static"] = _metric_number(front_result, "shoulder_tilt_static")
        measured["lateral_lean_static"] = _metric_number(front_result, "lateral_lean_static")
    outcome = screening.recommend(answers, measured)
    for notice in outcome.notices:
        st.info(notice)
    top = outcome.top
    # 결론은 보라 큰 카드 하나로 보여준다 (근거는 카드 안, 다음 순서는 접어 둠).
    with st.container(key="result_hero"):
        if top is None:
            st.markdown(
                "<p class='hero-small'>검사 결과</p>"
                "<p class='hero-title'>지금은 먼저 확인할 것이<br>눈에 띄지 않아요</p>",
                unsafe_allow_html=True,
            )
            st.markdown("정기적으로 다시 측정해서 변화를 살펴보세요.")
        else:
            st.markdown(
                f"<p class='hero-small'>먼저 확인해 볼 것</p><p class='hero-title'>{top.label}</p>",
                unsafe_allow_html=True,
            )
            for reason in top.video_reasons:
                st.markdown(f"- **영상**: {reason}")
            for reason in top.answer_reasons:
                st.markdown(f"- **답변**: {reason}")
            st.markdown(top.guide)
    if top is not None:
        others = [lever for lever in outcome.levers[1:] if lever.score > 0]
        if others:
            with st.expander("그다음으로 볼 것: " + ", ".join(lever.label for lever in others)):
                for lever in others:
                    st.markdown(f"**{lever.label}**")
                    for reason in lever.video_reasons:
                        st.markdown(f"- 영상: {reason}")
                    for reason in lever.answer_reasons:
                        st.markdown(f"- 답변: {reason}")
                    st.caption(lever.guide)
    _fine(SCREENING_NOTICE)


def _render_trunk_comparison(result: Any) -> None:
    """가만히 있을 때 vs 밀 때의 상체 각도를 나란히 보여주고, 지난 기록과 비교합니다."""
    static = _metric_number(result, "trunk_static")
    peak = _metric_number(result, "trunk_peak")
    change = _metric_number(result, "trunk_change")
    if peak is None:
        return
    st.markdown("#### 가만히 있을 때 vs 밀 때")
    # 지난 기록은 분석할 때 미리 담아둔 값을 쓴다 (저장한 뒤 자기 자신과 비교하지 않도록).
    previous_change = _safe_float(st.session_state.get("side_previous_change"))
    columns = st.columns(3)
    with columns[0]:
        st.metric("가만히 있을 때", _metric_value(static, "°"), help=METRIC_HELP["trunk_static"])
    with columns[1]:
        st.metric("밀 때 (가장 숙였을 때)", _metric_value(peak, "°"), help=METRIC_HELP["trunk_peak"])
    with columns[2]:
        delta = None
        if change is not None and previous_change is not None:
            delta = f"{change - previous_change:+.1f}° (지난번 대비)"
        # 숙임이 커지는 쪽을 빨강으로 보이게 한다 (delta_color="inverse").
        st.metric("차이 (더 숙여진 정도)", _metric_value(change, "°"), delta=delta, delta_color="inverse", help=METRIC_HELP["trunk_change"])
    if change is None:
        st.caption("처음에 가만히 있는 구간이 없어서 차이를 계산하지 못했어요.")
    else:
        level = screening.trunk_change_level(change)
        text = {
            "small": "밀 때도 상체가 거의 그대로예요.",
            "moderate": "밀 때 상체가 조금 더 숙여져요.",
            "notable": "밀 때 상체가 많이 숙여져요.",
        }[level]
        st.markdown(f"**{text}** 가만히 있을 때보다 {change:.0f}° 더 숙여져요.")
    _fine(
        "상체 각도: 0°는 똑바로 세운 자세, +는 앞으로 숙임, −는 뒤로 기댐이에요. "
        f"차이가 {SCREENING_PARAMS['trunk_change_min_detectable_deg']:.0f}°보다 작으면 측정 흔들림과 구분하기 어려워요."
    )
    _render_trunk_timeline(result)


def _render_frontal_comparison(result: Any) -> None:
    """앞에서 본 결과: 어깨 기울기와 몸통 옆 기울기를 가만히 있을 때 / 밀고 올 때로 나눠 보여줍니다."""
    limit = SCREENING_PARAMS["frontal_notable_deg"]
    shown = False
    for static_key, moving_key, sway_key, title, level_text in (
        ("shoulder_tilt_static", "shoulder_tilt_moving", "shoulder_tilt_sway", "어깨 높이", "좌우 어깨 높이가 거의 같아요"),
        ("lateral_lean_static", "lateral_lean_moving", "lateral_lean_sway", "몸통 옆 기울기", "몸통이 거의 똑바로 서 있어요"),
    ):
        static = _metric_number(result, static_key)
        moving = _metric_number(result, moving_key)
        sway = _metric_number(result, sway_key)
        if static is None and moving is None:
            continue
        shown = True
        st.markdown(f"#### {title}")
        columns = st.columns(3)
        with columns[0]:
            st.metric("가만히 있을 때", _side_angle(static), help=METRIC_HELP[static_key])
        with columns[1]:
            st.metric("밀고 올 때", _side_angle(moving), help=METRIC_HELP[moving_key])
        with columns[2]:
            st.metric("밀 때 흔들린 폭", _metric_value(sway, "°"), help=METRIC_HELP[sway_key])
        if static_key == "lateral_lean_static":
            # 실제 영상에서 똑바로 민 사람도 −3~−5°로 치우쳐 나와(휴대폰 기울기 영향) 단정하는 문장은 쓰지 않는다.
            _fine("몸통 옆 기울기는 휴대폰이 조금만 기울어도 값이 달라져서, 참고로만 보여드려요.")
        elif static is not None:
            if abs(static) >= limit:
                st.markdown(f"**{screening.frontal_sentence(static_key, static)}**")
            else:
                st.markdown(f"**{level_text}.** (차이 {limit:.0f}° 미만)")
    if not shown:
        return
    _fine(
        "왼쪽·오른쪽은 앉은 사람 기준이에요. 휴대폰이 좌우로 기울어 있으면 그만큼 값이 치우쳐요. "
        f"{limit:.0f}°는 측정 오차보다 큰 차이만 보려고 정한 값이고, 의학적 기준이 아니에요."
    )
    debug = getattr(result, "debug", None) or {}
    times, tilt, lean = debug.get("tilt_time") or [], debug.get("shoulder_tilt_series") or [], debug.get("lateral_lean_series") or []
    if times and len(times) == len(tilt) == len(lean):
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=times, y=tilt, mode="lines", name="어깨 기울기", line={"color": CHART_PURPLE, "width": 3}))
        figure.add_trace(go.Scatter(x=times, y=lean, mode="lines", name="몸통 옆 기울기", line={"color": "#D85A30", "width": 2}))
        baseline_seconds = float((debug.get("baseline_seconds") or [0.0])[0])
        if baseline_seconds > 0:
            figure.add_vrect(x0=times[0], x1=times[0] + baseline_seconds, fillcolor="#8B95A1", opacity=0.18, line_width=0,
                             annotation_text="가만히", annotation_position="top left")
        figure.update_layout(height=260, margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis_title="시간(초)",
                             yaxis_title="기울기(°, 왼쪽 +)", legend={"orientation": "h"})
        st.plotly_chart(figure, use_container_width=True)


def _side_angle(value: float | None) -> str:
    """부호 있는 좌우 각도를 "왼쪽 7.0°"처럼 씁니다 (+ = 왼쪽, − = 오른쪽)."""
    if value is None:
        return "–"
    if abs(value) < 0.05:
        return "0.0°"
    return f"{'왼쪽' if value > 0 else '오른쪽'} {abs(value):.1f}°"


def _render_trunk_timeline(result: Any) -> None:
    """시간에 따른 상체 각도 그래프 (회색 구간 = 가만히 있던 구간)."""
    debug = getattr(result, "debug", None) or {}
    times, lean = debug.get("time") or [], debug.get("forward_lean") or []
    if not times or len(times) != len(lean):
        return
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=times, y=lean, mode="lines", name="상체 각도", line={"color": CHART_PURPLE, "width": 3}))
    baseline_seconds = float((debug.get("baseline_seconds") or [0.0])[0])
    if baseline_seconds > 0:
        figure.add_vrect(x0=times[0], x1=times[0] + baseline_seconds, fillcolor="#8B95A1", opacity=0.18, line_width=0,
                         annotation_text="가만히", annotation_position="top left")
    figure.update_layout(height=240, margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis_title="시간(초)",
                         yaxis_title="상체 각도(°, 앞으로 +)", showlegend=False)
    st.plotly_chart(figure, use_container_width=True)


def _render_result_metrics(result: Any) -> None:
    """추진 분석 측정값을 핵심 2개만 크게 보여주고, 나머지는 접어 둡니다.

    손 궤적 패턴과 정지 vs 동작 비교값은 위쪽 블록에서 따로 보여주므로 카드에서 뺀다.
    """
    metrics = getattr(result, "metrics", {}) or {}
    if not metrics:
        st.info("표시할 측정값이 아직 없어요.")
        return
    # 예전 방식의 상체 숙임 3종(방향 구분 없는 값)은 정지 vs 동작 비교와 겹쳐서 화면에는 보이지 않는다.
    hidden = ("pattern", "trunk_lean_mean", "trunk_lean_max", "trunk_lean_range") + COMPARISON_METRICS
    items = [(name, metric) for name, metric in metrics.items() if name not in hidden and _safe_float(getattr(metric, "value", None)) is not None]
    if not items:
        return
    primary = [item for item in items if item[0] in PRIMARY_METRICS]
    others = [item for item in items if item[0] not in PRIMARY_METRICS]
    _render_metric_cards(primary or items)
    if primary and others:
        with st.expander(f"나머지 측정값 {len(others)}개 보기"):
            _render_metric_cards(others)
    _render_pending_note(items)


def _render_coaching(result: Any) -> None:
    """주의·경고 판정과 연결된 코칭 메시지를 표시합니다."""
    metrics = getattr(result, "metrics", {}) or {}
    shown: set[str] = set()
    for metric in metrics.values():
        judgment = getattr(metric, "judgment", "pending")
        coaching_key = getattr(metric, "coaching_key", None)
        if judgment in {"caution", "warning"} and coaching_key and coaching_key not in shown:
            message = COACHING_MESSAGES.get(coaching_key)
            if message:
                st.warning(message)
                shown.add(coaching_key)


def _render_trajectory(result: Any) -> None:
    """손목 궤적과 림 원을 Plotly 그래프로 표시합니다."""
    trajectory = getattr(result, "trajectory", None) or []
    if not trajectory:
        return
    if getattr(result, "view", "side") == "front":
        _render_front_wrist_heights(trajectory)
        return
    st.markdown("#### 손이 움직인 길")
    _fine("골반을 기준으로 본 손목의 움직임이에요. 초록은 앞으로 밀 때, 갈색은 뒤로 돌아올 때예요.")
    rows = []
    for x, y, phase in trajectory:
        rows.append({"x": float(x), "y": -float(y), "phase": str(phase)})
    data = pd.DataFrame(rows)
    figure = go.Figure()
    for phase, color in (("push", "#1D9E75"), ("recovery", "#B7791F")):
        subset = data[data["phase"] == phase]
        if not subset.empty:
            figure.add_trace(
                go.Scatter(
                    x=subset["x"],
                    y=subset["y"],
                    mode="lines+markers",
                    name="밀기" if phase == "push" else "복귀",
                    line={"color": color, "width": 3},
                    marker={"size": 5},
                )
            )
    rim = getattr(result, "rim", None)
    if rim:
        center_x, center_y, radius = rim
        theta = [index * 6.283185307 / 80 for index in range(81)]
        figure.add_trace(
            go.Scatter(
                x=[center_x + radius * math.cos(value) for value in theta],
                y=[-(center_y + radius * math.sin(value)) for value in theta],
                mode="lines",
                name="림",
                line={"color": "#B45FC6", "dash": "dash"},
            )
        )
    figure.update_layout(
        height=330,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title="골반 기준 가로 위치(px)",
        yaxis_title="골반 기준 높이(px)",
        legend={"orientation": "h"},
        yaxis={"scaleanchor": "x", "scaleratio": 1},
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_front_wrist_heights(trajectory: list[tuple[float, float, str]]) -> None:
    """정면 영상: 시간에 따른 양손 높이를 그려 좌우 차이를 보여줍니다."""
    figure = go.Figure()
    for hand, color, name in (("left", "#1D9E75", "왼손"), ("right", "#D85A30", "오른손")):
        points = [(t, -y) for t, y, label in trajectory if label == hand]
        if points:
            figure.add_trace(go.Scatter(x=[p[0] for p in points], y=[p[1] for p in points], mode="lines", name=name, line={"color": color, "width": 3}))
    figure.update_layout(height=280, margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis_title="시간(초)", yaxis_title="손 높이(px, 위가 +)", legend={"orientation": "h"})
    st.plotly_chart(figure, use_container_width=True)


def _render_representative_frames(result: Any) -> None:
    """스켈레톤이 합성된 대표 프레임을 표시합니다."""
    frames = getattr(result, "representative_frames", None) or []
    if not frames:
        return
    with st.expander("분석된 장면 보기"):
        columns = st.columns(min(3, len(frames)))
        for column, frame in zip(columns, frames):
            with column:
                st.image(frame, use_container_width=True)


def _render_debug(result: Any) -> None:
    """디버그 모드: 프레임별 시계열과 패턴 판단 근거를 표시합니다.

    debug 사전에는 길이가 다른 목록이 섞여 있으므로 time과 길이가 같은 것만 표로 묶는다.
    """
    debug = getattr(result, "debug", None) or {}
    if not debug:
        return
    with st.expander("디버그 시계열", expanded=True):
        st.caption(f"유효 프레임 비율 {getattr(result, 'valid_frame_ratio', 0):.0%} · 스트로크 {getattr(result, 'stroke_count', 0)}회")
        length = len(debug.get("time", []))
        series = {key: values for key, values in debug.items() if isinstance(values, list) and len(values) == length}
        if series:
            frame = pd.DataFrame(series)
            numeric_columns = [column for column in ("visibility", "wrist_x", "wrist_y", "right_wrist_y", "forward_lean") if column in frame]
            if numeric_columns:
                st.line_chart(frame.set_index("time")[numeric_columns])
            if "phase" in frame:
                st.write(frame["phase"].value_counts(dropna=False))
            st.dataframe(frame, use_container_width=True, hide_index=True)
        compare = debug.get("stroke_count_compare")
        if compare and len(compare) == 2:
            st.markdown("**스트로크 수 비교**")
            st.table(pd.DataFrame(
                {"방식": ["기존 (림 접촉 + 앞쪽 이동)", "대안 (손목 x 평균선 3회 교차 = 1회)"], "스트로크 수": compare}
            ))
        for key in ("pattern_votes", "left_tops", "right_tops"):
            if debug.get(key):
                st.markdown(f"**{key}**")
                st.write(debug[key])
        if debug.get("frame_time"):
            st.markdown("**전체 프레임 visibility** (분석에서 빠진 프레임 포함)")
            st.line_chart(pd.DataFrame({"time": debug["frame_time"], "visibility": debug["frame_visibility"]}).set_index("time"))


@st.cache_data(show_spinner=False, max_entries=4)
def _first_frame_rgb(video_bytes: bytes) -> Any:
    """림 클릭용 첫 화면(RGB)을 영상별로 한 번만 추출합니다."""
    frame = propulsion.extract_first_frame(video_bytes)
    return None if frame is None else frame[:, :, ::-1].copy()


def _show_rim_click_flow(video_bytes: bytes, key_prefix: str) -> list[tuple[float, float]]:
    """첫 프레임에서 림 3점을 클릭받아 분석 프레임 픽셀 좌표로 반환합니다.

    이미지는 모바일 폭(RIM_CLICK_DISPLAY_WIDTH)으로 줄여 보여주므로,
    클릭 좌표 × (원래 폭 ÷ 표시 폭)으로 분석 프레임 좌표로 되돌린다.
    """
    rgb_frame = _first_frame_rgb(video_bytes)
    if rgb_frame is None:
        st.error("영상의 첫 프레임을 읽지 못했어요. 다른 영상을 선택해주세요.")
        return []

    video_id = hashlib.md5(video_bytes).hexdigest()[:10]
    points_key = f"{key_prefix}_rim_points"
    if st.session_state.get(f"{key_prefix}_rim_video") != video_id:
        # 새 영상이면 이전에 찍은 점을 비운다.
        st.session_state[f"{key_prefix}_rim_video"] = video_id
        st.session_state[points_key] = []
        st.session_state[f"{key_prefix}_rim_round"] = 0
    round_number = int(st.session_state.get(f"{key_prefix}_rim_round", 0))

    points: list[tuple[float, float]] = st.session_state.setdefault(points_key, [])
    frame_height, frame_width = rgb_frame.shape[:2]
    scale = frame_width / RIM_CLICK_DISPLAY_WIDTH
    st.caption("바퀴 손잡이 테(림) 위에서 서로 멀리 떨어진 3점을 차례로 눌러주세요.")
    preview = overlay.draw_rim_click_points(rgb_frame, points)
    clicked = streamlit_image_coordinates(
        Image.fromarray(preview),
        width=RIM_CLICK_DISPLAY_WIDTH,
        key=f"{key_prefix}_rim_frame_{video_id}_{round_number}",
    )
    if clicked and len(points) < 3:
        point = (float(clicked["x"]) * scale, float(clicked["y"]) * scale)
        last_click = st.session_state.get(f"{key_prefix}_rim_last_click")
        if last_click != (clicked["x"], clicked["y"]):
            st.session_state[f"{key_prefix}_rim_last_click"] = (clicked["x"], clicked["y"])
            points.append(point)
            st.rerun()
    st.write(f"선택된 점: {len(points)}/3")
    if st.button("림 점 다시 찍기", key=f"{key_prefix}_reset_rim"):
        st.session_state[points_key] = []
        st.session_state[f"{key_prefix}_rim_round"] = round_number + 1
        st.session_state.pop(f"{key_prefix}_rim_last_click", None)
        st.rerun()
    return list(points)


def _analyze_uploaded_video(uploaded_file: UploadType, *, view: str, filmed_side: str | None, rim_points: list[tuple[float, float]] | None, rim_diameter: float, key_prefix: str) -> None:
    """업로드 영상을 분석하고 결과를 session_state에 보관합니다.

    Streamlit은 버튼을 누를 때마다 스크립트를 처음부터 다시 실행한다.
    그래서 결과 표시와 "저장" 버튼은 분석 버튼 안이 아니라 _show_saved_result()에서 그린다.
    (분석 버튼 안에 저장 버튼을 두면, 저장을 누르는 순간 분석 버튼이 False가 되어 저장 코드가 실행되지 않는다.)
    """
    if uploaded_file is None:
        return
    video_bytes = uploaded_file.getvalue()
    progress = st.progress(0.0, text="분석 준비 중...")

    def progress_callback(value: float) -> None:
        """추진 분석 진행률을 Streamlit progress에 반영합니다."""
        bounded = max(0.0, min(1.0, float(value)))
        progress.progress(bounded, text=f"분석 중... {int(bounded * 100)}%")

    try:
        result = propulsion.analyze_video_bytes(
            video_bytes,
            view=view,
            filmed_side=filmed_side,
            rim_points=rim_points,
            rim_diameter_inch=float(rim_diameter),
            progress_callback=progress_callback,
        )
    except Exception as error:  # 분석 중 예상 못 한 오류도 앱을 멈추지 않게 한다.
        progress.empty()
        st.error(f"분석에 실패했어요. 촬영 조건을 확인하고 다시 시도해주세요. ({error})")
        return
    progress.empty()
    previous = _previous_session()
    st.session_state[f"{key_prefix}_previous_change"] = ((previous or {}).get("metrics") or {}).get("trunk_change")
    st.session_state[f"{key_prefix}_result"] = result
    st.session_state[f"{key_prefix}_saved"] = False


def _render_filming_checks(result: object) -> None:
    """촬영 점검 결과(거리·밝기·관절 보임·가만히 3초·미는 동작)를 한눈에 보여줍니다."""
    checks = getattr(result, "filming_checks", None) or []
    if not checks:
        return
    problems = [check for check in checks if check[0] != "ok"]
    title = "📷 촬영 점검: 잘 찍혔어요" if not problems else f"📷 촬영 점검: 고치면 좋은 점 {len(problems)}개"
    with st.expander(title, expanded=bool(problems)):
        for status, label, message in checks:
            icon = "✅" if status == "ok" else "⚠️"
            st.markdown(f"{icon} **{label}** · {message}")


CVA_PAIR_STEADY_SD = 3.0  # 앱 − 탭 차이의 표준편차가 이 안이면 "차이가 일정" (2D 영상 측정 오차 ±3~4° 수준)


def _render_c7_check(result: Any, key_prefix: str) -> None:
    """검증 도구: 가만히 있는 장면 사진에 C7·귀 앞 돌기를 탭해 논문 방식 CVA를 재고, 앱 자동 값과 비교합니다.

    여러 사람의 (논문 방식, 앱 자동) 짝을 모아 차이가 일정하면, 논문의 정상 기준에 평균 차이를 더해 앱 기준으로 쓸 수 있다.
    """
    frame = getattr(result, "still_frame", None)
    still_time = getattr(result, "still_time", None)
    if frame is None or still_time is None:
        return
    with st.expander("🎯 거북목 정밀 측정 비교 (C7 찍기 · 검증용)"):
        st.caption("가만히 있는 장면이에요. ① 목 뒤 튀어나온 뼈(C7) → ② 귀 앞 작은 돌기(귀 구슬) 순서로 눌러주세요. C7에 스티커를 붙이고 찍으면 정확해요.")
        points_key = f"{key_prefix}_c7_points"
        frame_id = f"{key_prefix}_{still_time:.2f}_{frame.shape[1]}x{frame.shape[0]}"
        if st.session_state.get(f"{key_prefix}_c7_frame") != frame_id:
            st.session_state[f"{key_prefix}_c7_frame"] = frame_id
            st.session_state[points_key] = []
            st.session_state[f"{key_prefix}_c7_round"] = 0
        points: list[tuple[float, float]] = st.session_state.setdefault(points_key, [])
        round_number = int(st.session_state.get(f"{key_prefix}_c7_round", 0))
        scale = frame.shape[1] / RIM_CLICK_DISPLAY_WIDTH
        preview = frame.copy()
        for index, (x, y) in enumerate(points):
            cv2.circle(preview, (int(x), int(y)), max(4, int(6 * scale)), (239, 159, 39) if index == 0 else (107, 63, 160), -1)
        if len(points) == 2:
            cv2.line(preview, tuple(int(v) for v in points[0]), tuple(int(v) for v in points[1]), (107, 63, 160), max(2, int(2 * scale)))
        clicked = streamlit_image_coordinates(Image.fromarray(preview), width=RIM_CLICK_DISPLAY_WIDTH, key=f"{frame_id}_c7_{round_number}")
        if clicked and len(points) < 2:
            last = st.session_state.get(f"{key_prefix}_c7_last")
            if last != (clicked["x"], clicked["y"]):
                st.session_state[f"{key_prefix}_c7_last"] = (clicked["x"], clicked["y"])
                points.append((float(clicked["x"]) * scale, float(clicked["y"]) * scale))
                st.rerun()
        st.write(f"찍은 점: {len(points)}/2 ({'C7' if len(points) == 0 else '귀 앞 돌기' if len(points) == 1 else '완료'})")
        if st.button("점 다시 찍기", key=f"{key_prefix}_c7_reset"):
            st.session_state[points_key] = []
            st.session_state[f"{key_prefix}_c7_round"] = round_number + 1
            st.session_state.pop(f"{key_prefix}_c7_last", None)
            st.rerun()
        if len(points) < 2:
            return
        manual = propulsion.manual_cva(points[0], points[1])
        auto = propulsion.head_cva_at(result, float(still_time))
        columns = st.columns(3)
        columns[0].metric("논문 방식 (C7)", f"{manual:.1f}°")
        columns[1].metric("앱 자동 (어깨)", "–" if auto is None else f"{auto:.1f}°")
        columns[2].metric("차이 (앱 − 논문)", "–" if auto is None else f"{auto - manual:+.1f}°")
        if auto is not None and st.button("이 비교를 저장", key=f"{key_prefix}_c7_add"):
            try:
                database.save_cva_check(str(st.session_state["uid"]), manual_cva=manual, auto_cva=auto)
                st.success("저장했어요.")
            except database.DatabaseError as error:
                st.error(str(error))
        _render_cva_pairs()


def _render_cva_pairs() -> None:
    """저장한 (논문 방식, 앱 자동) 짝: 표, 평균 차이, 차이의 표준편차, 상관 → 자동 측정을 논문 기준으로 바꿔 쓸 수 있는지."""
    try:
        rows = database.list_cva_checks(str(st.session_state["uid"]))
    except database.DatabaseError as error:
        st.caption(str(error))
        return
    pairs = [
        {"논문 방식(C7)": round(float(row["manual_cva"]), 1), "앱 자동(어깨)": round(float(row["auto_cva"]), 1),
         "차이": round(float(row["auto_cva"]) - float(row["manual_cva"]), 1)}
        for row in rows if _safe_float(row.get("manual_cva")) is not None and _safe_float(row.get("auto_cva")) is not None
    ]
    if not pairs:
        return
    data = pd.DataFrame(pairs)
    st.dataframe(data, hide_index=True, use_container_width=True)
    st.download_button("비교 기록 내려받기 (CSV)", data.to_csv(index=False).encode("utf-8-sig"), "cva_compare.csv", "text/csv", key="cva_pairs_csv")
    if len(data) < 3:
        st.caption(f"{len(data)}명 기록됨. 3명 이상 모이면 차이가 일정한지 계산해요 (많을수록 좋아요, 목표 10명 이상).")
        return
    mean_diff, sd_diff = float(data["차이"].mean()), float(data["차이"].std(ddof=1))
    correlation = float(np.corrcoef(data["논문 방식(C7)"], data["앱 자동(어깨)"])[0, 1])
    st.markdown(f"**{len(data)}명** · 평균 차이 **{mean_diff:+.1f}°** · 차이의 흩어짐(표준편차) **{sd_diff:.1f}°** · 상관 **{correlation:.2f}**")
    if sd_diff <= CVA_PAIR_STEADY_SD:
        st.success(f"차이가 일정해요. 논문의 정상 기준에 {mean_diff:+.1f}°를 더하면 앱 자동 측정의 기준으로 쓸 수 있어요.")
    else:
        st.warning("차이가 사람마다 들쭉날쭉해요. 자동 측정으로는 논문 기준을 쓰기 어려워, 정밀 측정(C7 찍기)을 기본으로 쓰는 게 좋아요.")


def _render_result_summary(result: Any) -> None:
    """영상 하나의 핵심 수치만 카드 2~3개로 요약합니다 (자세한 내용은 아래에 접어 둠)."""
    if getattr(result, "view", "side") == "front":
        # 어깨 기울기 + = 앉은 사람의 왼쪽 어깨가 낮음
        items = [("낮은 어깨 (가만히)", _side_angle(_metric_number(result, "shoulder_tilt_static")))]
        moving = _metric_number(result, "shoulder_tilt_moving")
        if moving is not None:
            items.append(("낮은 어깨 (밀고 올 때)", _side_angle(moving)))
        st.markdown("##### 앞에서 본 결과 요약")
    else:
        change = _metric_number(result, "trunk_change")
        rate = _metric_number(result, "stroke_rate")
        pattern = str(getattr(result, "pattern", "unknown"))
        items = [
            ("밀 때 더 숙여진 정도", _metric_value(change, "°")),
            ("미는 패턴", PATTERN_NAMES.get(pattern, pattern) if pattern != "unknown" else "판정 보류"),
            ("1분에 미는 횟수", _metric_value(rate, "회")),
        ]
        st.markdown("##### 옆에서 본 결과 요약")
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        with column:
            st.metric(label, value)
    _render_reference_compare(result)


def _reference_bar(key: str, mine: float, mine_text: str) -> str:
    """"참고 범위(보라 띠) + 나(점)"를 한 줄 막대로 그린 HTML. 값은 config.REFERENCE_COMPARISONS."""
    ref = REFERENCE_COMPARISONS[key]
    axis = float(ref["axis"])  # type: ignore[arg-type]
    unit = str(ref["unit"])
    position = lambda value: max(0.0, min(100.0, value / axis * 100))  # noqa: E731
    shapes = ""
    if "band" in ref:
        low, high = ref["band"]  # type: ignore[misc]
        shapes += f"<div class='wl-ref-band' style='left:{position(low):.1f}%;width:{position(high) - position(low):.1f}%'></div>"
        ref_text = f"{low:g}~{high:g}{unit}"
    else:
        marker = float(ref["marker"])  # type: ignore[arg-type]
        shapes += f"<div class='wl-ref-tick' style='left:{position(marker):.1f}%'></div>"
        ref_text = f"약 {marker:g}{unit}"
    shapes += f"<div class='wl-ref-me' style='left:{position(mine):.1f}%'></div>"
    return (
        "<div class='wl-ref'>"
        f"<p class='wl-ref-title'>{ref['label']}</p>"
        f"<div class='wl-ref-track'>{shapes}</div>"
        "<div class='wl-ref-legend'>"
        f"<span><i class='wl-ref-key band'></i>참고 {ref_text} · {ref['who']}</span>"
        f"<span><i class='wl-ref-key me'></i><b>나 {mine_text}</b></span>"
        "</div>"
        f"<p class='wl-ref-source'>{ref['source']}</p>"
        "</div>"
    )


def _render_reference_compare(result: Any) -> None:
    """핵심 수치를 "다른 사람은 보통 이 정도 · 나는 이 정도"로 비교합니다.

    휠체어 사용자의 정상값은 문헌에 없어서, 연구 참고치나 자체 측정 범위를 "참고"로만 보여준다 (판정에 쓰지 않음).
    """
    bars: list[str] = []
    if getattr(result, "view", "side") == "front":
        tilt = _metric_number(result, "shoulder_tilt_static")
        if tilt is not None:
            bars.append(_reference_bar("shoulder_tilt_static", abs(tilt), _side_angle(tilt)))
    else:
        change = _metric_number(result, "trunk_change")
        if change is not None:
            bars.append(_reference_bar("trunk_change", change, f"{change:+.0f}°"))
        rate = _metric_number(result, "stroke_rate")
        if rate is not None:
            bars.append(_reference_bar("stroke_rate", rate, f"{rate:.0f}회"))
        pattern = str(getattr(result, "pattern", "unknown"))
        if pattern != "unknown":
            mark = "✓ 권장과 같아요" if pattern == "semicircular" else "권장과 달라요"
            bars.append(
                "<div class='wl-ref'><p class='wl-ref-title'>미는 패턴</p>"
                f"<p class='wl-ref-pattern'>권장 <b>반원형</b> · 나 <b>{PATTERN_NAMES.get(pattern, pattern)}</b> ({mark})</p>"
                "<p class='wl-ref-source'>척수손상 의학 가이드라인 (Consortium for Spinal Cord Medicine, 2005)</p></div>"
            )
    if not bars:
        return
    st.markdown("##### 다른 사람과 비교하면")
    st.markdown("".join(bars), unsafe_allow_html=True)
    _fine("휠체어 사용자의 '정상값'은 아직 연구로 정해져 있지 않아요. 아래 범위는 참고용이고, 판정은 본인의 변화로 해요.")


def _show_saved_result(key_prefix: str) -> None:
    """보관된 분석 결과를 표시하고, 경고가 없을 때만 저장 버튼을 보여줍니다."""
    result = st.session_state.get(f"{key_prefix}_result")
    if result is None:
        return
    _render_filming_checks(result)
    for note in getattr(result, "notes", None) or []:
        st.info(note)
    warnings = getattr(result, "warnings", None) or []
    for warning in warnings:
        st.warning(warning)
    if warnings:
        st.info("분석 경고가 있어 측정 결과를 저장하지 않아요. 영상을 다시 촬영해주세요.")
        if st.session_state.get("debug_mode", False):
            _render_debug(result)
        return

    _render_result_summary(result)
    photo = _posture_photo(result) if getattr(result, "view", "side") == "side" else None
    if photo is not None:
        cva = _metric_number(result, "head_cva_static")
        level_name = ("정상", "1단계 (경미)", "2단계", "3단계")
        title = (f"가만히 있을 때 목 각도 {cva:.0f}° · 논문 기준 {level_name[keywords.forward_head_level(cva)]}"
                 if cva is not None else "가만히 있을 때 자세")
        st.markdown(f"<p class='wl-section'>{title}</p>", unsafe_allow_html=True)
        st.image(photo, use_container_width=True)
        _fine(f"보라 선: 어깨→귀 · 흰 점선: 수평 · 초록 점선: 거북목 2단계 기준 {float(POSTURE_KEYWORDS['forward_head']['threshold']):.0f}° "  # type: ignore[arg-type]
              "(귀–어깨 선 기준 FNTA 25°, Diagnostics 2024 · 선 자세로 만든 기준이라 앉은 자세에서는 더 낮게 나올 수 있어요)")
    if getattr(result, "view", "side") == "side" and st.session_state.get("debug_mode", False):
        _render_c7_check(result, key_prefix)
    view_name = "앞에서" if getattr(result, "view", "side") == "front" else "옆에서"
    # 자세한 결과 안에 접는 상자가 또 있어 expander로 감쌀 수 없다(중첩 불가) → 켜고 끄는 스위치로 접는다.
    # 디버그 모드에서는 처음부터 펼친다.
    if st.toggle(f"{view_name} 본 자세한 결과 보기 (그래프·사진)", value=bool(st.session_state.get("debug_mode", False)), key=f"{key_prefix}_detail"):
        _render_analysis_result(result)
    if st.session_state.get(f"{key_prefix}_saved"):
        st.success("측정 결과를 저장했어요.")
        # 검사 → 운동으로 이어지게: 저장한 다음 단계는 결과에 맞춘 운동
        st.button("맞춤 운동 보러 가기", key=f"{key_prefix}_to_exercise", icon=":material/fitness_center:",
                  on_click=_go_to_tab, args=("운동",), use_container_width=True, type="primary")
        return
    keep_photo = photo is not None and st.checkbox(
        "자세 사진도 저장해서 다음 검사와 나란히 보기 (얼굴은 흐리게)", key=f"{key_prefix}_keep_photo"
    )
    if st.button("측정 결과 저장", key=f"{key_prefix}_save", use_container_width=True, type="primary"):
        try:
            metrics, judgments = propulsion.to_firestore(result)
            database.save_propulsion_session(
                str(st.session_state["uid"]),
                view=result.view,
                filmed_side=result.filmed_side or "",
                metrics=metrics,
                judgments=judgments,
                is_demo=False,
                answers=_current_answers(),
                posture_photo=overlay.encode_jpeg_b64(photo) if keep_photo and photo is not None else None,
            )
            st.session_state[f"{key_prefix}_saved"] = True
            st.rerun()
        except database.DatabaseError as error:
            st.error(f"{error} 위의 저장 버튼을 다시 눌러주세요.")


def _posture_photo(result: Any) -> Any:
    """가만히 있는 장면 사진에 목 각도 선을 그리고 얼굴을 흐린 자세 사진 (없으면 None). 결과마다 한 번만 만든다."""
    cached = getattr(result, "_posture_photo_cache", "unset")
    if not isinstance(cached, str):
        return cached
    frame, points = getattr(result, "still_frame", None), getattr(result, "still_points", None) or {}
    photo = None
    if frame is not None and points:
        photo = overlay.posture_photo(frame, points, str(getattr(result, "filmed_side", None) or "left"),
                                      float(POSTURE_KEYWORDS["forward_head"]["threshold"]))  # type: ignore[arg-type]
    try:
        result._posture_photo_cache = photo
    except AttributeError:
        pass
    return photo


def _render_analysis_result(result: Any) -> None:
    """영상 하나의 분석 결과를 화면에 표시합니다 ("먼저 확인해 볼 것"은 _show_exam_tab()이 맨 위에 따로 그림).

    옆: 가만히 있을 때 vs 밀 때 상체 각도 → 미는 방법 → 나머지는 접어 둠
    앞: 어깨 높이·몸통 옆 기울기 (가만히 / 밀고 올 때) → 양손 움직임 비교는 접어 둠
    """
    if getattr(result, "view", "side") == "front":
        st.markdown("### 앞에서 본 결과")
        _render_frontal_comparison(result)
        hand_items = [
            (name, metric) for name, metric in (getattr(result, "metrics", {}) or {}).items()
            if name in ("symmetry_index", "timing_diff") and _safe_float(getattr(metric, "value", None)) is not None
        ]
        if hand_items:
            with st.expander("양손 움직임 비교 (참고)"):
                _render_metric_cards(hand_items)
                _render_trajectory(result)
    else:
        st.markdown("### 옆에서 본 결과")
        _render_trunk_comparison(result)
        st.markdown("#### 미는 방법")
        _render_recommended_pattern(result)
        _render_coaching(result)
        _render_result_metrics(result)
        _render_trajectory(result)
    _render_representative_frames(result)
    _fine(LANDMARK_NOTICE)
    if st.session_state.get("debug_mode", False):
        _render_debug(result)


def _show_upload_guidance() -> None:
    """측면 영상 촬영 방법을 안내합니다. 영상을 올린 뒤에는 접어서 화면을 줄입니다."""
    with st.expander("촬영 방법"):
        st.image(overlay.draw_camera_guide_example(), width=RIM_CLICK_DISPLAY_WIDTH)
        st.markdown("\n".join(f"{number}. {step}" for number, step in enumerate(CAMERA_GUIDE_STEPS, start=1)))
        st.markdown(CAMERA_GUIDE_TIP)
        # 논문은 제자리(롤러) 조건이라 우리 촬영 방식과 다르다. "출처"가 아니라 "참고"로 적는다.
        st.markdown(
            f"<p class='wl-fine'>참고한 연구의 촬영 조건 (제자리 롤러 위에서 찍은 연구라 이 앱의 방식과 달라요): {CAMERA_GUIDE_DETAIL} · {SOURCE_CV_PATTERN}</p>",
            unsafe_allow_html=True,
        )


def _profile_rim_diameter() -> float:
    """사이드바 프로필의 림 지름(인치). 화면마다 다시 묻지 않고 프로필 값을 쓴다."""
    return _safe_float(st.session_state.get("profile_rim")) or 24.0


def _show_questions() -> None:
    """1단계: 질문 5개. 카메라로 볼 수 없는 정보(통증, 경험, 휠체어 점검 등)를 물어봅니다.

    지난번에 저장한 답변이 있으면 그대로 채워 두고, 모두 답했으면 접어서 보여준다.
    """
    saved = dict((_load_user_document().get("profile") or {}).get("screening_answers") or {})
    # 예전 통증 답(없음/가끔/밀 때마다): "없음"만 0점으로 옮기고, 나머지는 새 0~10 질문으로 다시 묻는다.
    if saved.get("pain") == "none":
        saved["pain"] = "0"
    for question in SCREENING_QUESTIONS:
        key = f"answer_{question['id']}"
        options = question["options"]
        # 선택지에 없는 값이 남아 있으면 st.radio가 멈추므로 지운다 (문항을 바꿨을 때 예전 답).
        if key in st.session_state and st.session_state[key] is not None and st.session_state[key] not in options:  # type: ignore[operator]
            del st.session_state[key]
        if key not in st.session_state and saved.get(question["id"]) in options:  # type: ignore[operator]
            st.session_state[key] = saved[question["id"]]
    remaining = len(screening.missing_questions(_current_answers()))
    title = "질문 5개 · 답변 완료" if remaining == 0 else f"질문 5개 · 남은 질문 {remaining}개"
    with st.expander(title, expanded=remaining > 0):
        for number, question in enumerate(SCREENING_QUESTIONS, start=1):
            options: dict[str, str] = question["options"]  # type: ignore[assignment]
            st.radio(
                f"{number}. {question['text']}",
                list(options),
                format_func=options.get,
                index=None,
                key=f"answer_{question['id']}",
            )


def _save_answers_to_profile() -> None:
    """답변을 프로필에 저장해 다음에 다시 묻지 않게 합니다 (실패해도 분석은 계속)."""
    try:
        database.update_user_profile(str(st.session_state["uid"]), {"screening_answers": _current_answers()})
    except database.DatabaseError:
        pass


VIDEO_SOURCES = ("지금 찍기", "찍어둔 영상 올리기")


def _record_and_analyze(*, view: str, filmed_side: str | None, key_prefix: str) -> None:
    """사이트 안 촬영 화면. "이 영상으로 분석하기"를 누르면 바로 분석하고, 녹화 중 수평·흔들림을 촬영 점검에 더합니다."""
    recording = camera_recorder.camera_recorder(key=f"{key_prefix}_camera")
    if recording is None or recording["id"] == st.session_state.get(f"{key_prefix}_recording_id"):
        return
    # 같은 녹화를 두 번 분석하지 않도록 id를 기억한다 (컴포넌트는 다시 그릴 때마다 마지막 값을 돌려준다).
    st.session_state[f"{key_prefix}_recording_id"] = recording["id"]
    if screening.missing_questions(_current_answers()):
        st.info("질문에 답하지 않은 항목이 있어요. 영상만으로 먼저 분석할게요.")
    _save_answers_to_profile()
    _analyze_uploaded_video(
        io.BytesIO(recording["video"]), view=view, filmed_side=filmed_side, rim_points=None,
        rim_diameter=_profile_rim_diameter(), key_prefix=key_prefix,
    )
    result = st.session_state.get(f"{key_prefix}_result")
    if result is None:
        return
    checks = camera_recorder.recording_checks(recording)
    # 정면: 휴대폰이 기울어 있던 만큼 어깨·몸통 각도를 되돌린다 (휴대폰 기울기가 정면 각도의 가장 큰 오차 원인).
    roll = recording.get("roll_median")
    if view == "front" and roll is not None and not getattr(result, "warnings", None):
        correction = propulsion.correct_for_camera_roll(result, float(roll), recording.get("facing", "environment"))
        if abs(correction) >= 0.1:
            checks.append(("ok", "기울기 보정", f"휴대폰이 {abs(correction):.1f}° 기울어 있던 만큼 어깨·몸통 각도를 보정했어요."))
    result.filming_checks = checks + list(getattr(result, "filming_checks", []) or [])


VIEW_PURPOSE = {"side": "목 각도 · 밀 때 상체 · 미는 방법을 봐요", "front": "어깨 높이 · 몸통 좌우 기울기를 봐요"}


def _show_video_input(view: str) -> None:
    """영상 넣기: 지금 찍기 / 영상 올리기. 올리면 바로 분석한다 (찍는 쪽 몸은 자동으로 찾는다)."""
    source = st.radio("영상", VIDEO_SOURCES, horizontal=True, key=f"{view}_source", label_visibility="collapsed")
    if view == "side":
        _show_upload_guidance()
    else:
        with st.expander("촬영 방법"):
            st.markdown("\n".join(f"{number}. {step}" for number, step in enumerate(FRONT_GUIDE_STEPS, start=1)))
            st.markdown(FRONT_GUIDE_TIP)
    if source == VIDEO_SOURCES[0]:
        _record_and_analyze(view=view, filmed_side=None, key_prefix=view)
        return
    uploaded = st.file_uploader("영상 파일", type=["mp4", "mov", "webm"], key=f"{view}_video", label_visibility="collapsed")
    st.caption("올리면 바로 분석하고, 거리·밝기·찍은 방향이 잘못됐으면 알려드려요.")
    if uploaded is None:
        return
    # 같은 파일을 다시 분석하지 않는다 (화면이 다시 그려질 때마다 업로더가 같은 파일을 돌려준다)
    file_id = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get(f"{view}_analyzed_file") == file_id:
        return
    st.session_state[f"{view}_analyzed_file"] = file_id
    if screening.missing_questions(_current_answers()):
        st.info("질문에 답하지 않은 항목이 있어요. 영상만으로 먼저 분석할게요.")
    _save_answers_to_profile()
    _analyze_uploaded_video(uploaded, view=view, filmed_side=None, rim_points=None, rim_diameter=_profile_rim_diameter(), key_prefix=view)


def _render_steps(steps: list[tuple[str, bool]]) -> None:
    """진행 막대: 끝난 단계는 보라, 남은 단계는 회색. 아래에 "다음: ○○"을 적습니다."""
    bars = "".join(f"<div class='{'done' if done else ''}'></div>" for _, done in steps)
    finished = sum(done for _, done in steps)
    upcoming = next((name for name, done in steps if not done), None)
    label = f"다음: {upcoming}" if upcoming else "모두 끝났어요"
    st.markdown(f"<div class='wl-steps'>{bars}</div><p class='wl-steps-label'>{label}</p>", unsafe_allow_html=True)


def _usable_result(key_prefix: str) -> Any:
    """경고 없이 분석된 결과만 돌려줍니다 (없거나 경고가 있으면 None)."""
    result = st.session_state.get(f"{key_prefix}_result")
    if result is None or (getattr(result, "warnings", None) or []):
        return None
    return result


EXAM_VIEWS = {"side": "옆에서 찍기", "front": "앞에서 찍기"}


def _show_exam_tab() -> None:
    """검사하기: ① 질문 → ② 영상(옆/앞 중 고르기) → ③ 결과를 한 화면에서 차례로."""
    _render_steps(
        [
            ("질문", not screening.missing_questions(_current_answers())),
            ("영상", _usable_result("side") is not None or _usable_result("front") is not None),
            ("저장", bool(st.session_state.get("side_saved") or st.session_state.get("front_saved"))),
        ]
    )
    st.markdown("<p class='wl-section'>① 질문</p>", unsafe_allow_html=True)
    _show_questions()
    st.markdown("<p class='wl-section'>② 영상</p>", unsafe_allow_html=True)
    done = {view: "✓ " if _usable_result(view) is not None else "" for view in EXAM_VIEWS}
    view = st.radio(
        "찍는 방향", list(EXAM_VIEWS), format_func=lambda key: done[key] + EXAM_VIEWS[key], horizontal=True, key="exam_view",
        label_visibility="collapsed",
    )
    st.caption(VIEW_PURPOSE[view] + " · 하나만 해도 돼요")
    _show_video_input(view)

    if st.session_state.get("side_result") is None and st.session_state.get("front_result") is None:
        return
    st.markdown("<p class='wl-section'>③ 결과</p>", unsafe_allow_html=True)
    side_result, front_result = _usable_result("side"), _usable_result("front")
    if side_result is not None or front_result is not None:
        _render_screening(side_result, front_result)
        _render_keywords(_current_keywords())
    _show_saved_result("side")
    _show_saved_result("front")


def _as_datetime(value: Any) -> datetime | None:
    """Firestore 날짜나 문자열을 UTC datetime으로 변환합니다."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


PAGES = {"홈": "home", "검사하기": "exam", "운동": "exercise", "기록": "records"}
PAGE_TITLES = {page: label for label, page in PAGES.items()}


def _go_to_tab(label: str) -> None:
    """버튼 콜백: 이 이름의 화면으로 옮겨 갑니다 (홈에서 각 화면으로 가는 구조)."""
    st.session_state["page"] = PAGES.get(label, "home")
    st.session_state["_scroll_top"] = True


def _scroll_top_if_moved() -> None:
    """화면을 옮겼으면 맨 위로 올려 줍니다 (Streamlit은 스크롤 위치를 그대로 둔다)."""
    if not st.session_state.pop("_scroll_top", False):
        return
    import streamlit.components.v1 as components

    components.html(
        "<script>const main = window.parent.document.querySelector('section.main, [data-testid=\"stMain\"]');"
        "if (main) main.scrollTo(0, 0); window.parent.scrollTo(0, 0);</script>",
        height=0,
    )


def _program_status() -> dict[str, int | None]:
    """운동 기간 상태: 마지막 실제 검사 후 며칠, 그 뒤 운동한 날 수, 이번 주(월~일) 운동한 날 수.

    운동 기록은 운동 탭의 '완료하기'로 쌓인다. 같은 날 여러 운동을 해도 하루로 센다.
    """
    status: dict[str, int | None] = {"days_since_exam": _last_real_session_days(), "days_after_exam": 0, "week_days": 0}
    try:
        logs = database.list_stretch_logs(str(st.session_state["uid"]))
    except database.DatabaseError:
        return status
    now = datetime.now(timezone.utc)
    days = {created.date() for created in (_as_datetime(log.get("created_at")) for log in logs) if created is not None}
    monday = (now - timedelta(days=now.weekday())).date()
    status["week_days"] = sum(day >= monday for day in days)
    if status["days_since_exam"] is not None:
        exam_day = (now - timedelta(days=int(status["days_since_exam"]))).date()
        status["days_after_exam"] = sum(day >= exam_day for day in days)
    return status


def _render_program_status() -> None:
    """운동 탭 위: 이번 주 운동 n/3일 + 다시 검사까지 남은 날 (28일이 지나면 다시 검사하기 버튼)."""
    status = _program_status()
    goal = EXERCISE_PROGRAM["days_per_week"]
    week = int(status["week_days"] or 0)
    since = status["days_since_exam"]
    stats = f"<div class='wl-stat'><span>이번 주 운동</span><b>{week}<small>/{goal}일</small></b></div>"
    if since is not None:
        stats += f"<div class='wl-stat'><span>마지막 검사 후</span><b>{since}<small>일째</small></b></div>"
        stats += f"<div class='wl-stat'><span>검사 후 운동</span><b>{status['days_after_exam']}<small>일</small></b></div>"
    st.markdown(f"<div class='wl-stats'>{stats}</div>", unsafe_allow_html=True)


def _last_real_session_days() -> int | None:
    """가장 최근 실제(데모 아님) 검사가 며칠 전인지. 검사 기록이 없으면 None."""
    try:
        sessions = database.list_propulsion_sessions(str(st.session_state["uid"]))
    except database.DatabaseError:
        return None
    dates = [_as_datetime(session.get("created_at")) for session in sessions if not session.get("is_demo")]
    dates = [date for date in dates if date is not None]
    if not dates:
        return None
    return max(0, (datetime.now(timezone.utc) - max(dates)).days)


def _show_home_tab() -> None:
    """홈: 처음 온 사람에게는 '검사 시작'을 가장 크게 보여주고, 다른 화면으로 가는 버튼을 둡니다."""
    days = _last_real_session_days()
    with st.container(key="home_hero"):
        if days is None:
            st.markdown(
                "<p class='hero-small'>아직 검사 기록이 없어요</p>"
                "<p class='hero-title'>첫 자세 검사를<br>시작해 볼까요?</p>"
                "<p class='hero-small'>질문 5개 + 영상 2개 (옆·앞) · 3초 가만히 있다가 평소처럼 밀면 돼요</p>",
                unsafe_allow_html=True,
            )
            button_label = "검사 시작하기"
        else:
            when = "오늘" if days == 0 else f"{days}일 전"
            st.markdown(
                f"<p class='hero-small'>마지막 검사: {when}</p>"
                "<p class='hero-title'>운동 후 다시 재서<br>변화를 확인해 보세요</p>"
                "<p class='hero-small'>같은 방법으로 찍으면 지난번과 비교해 드려요 · 3초 가만히 있다가 평소처럼 밀기</p>",
                unsafe_allow_html=True,
            )
            button_label = "다시 검사하기"
        st.button(button_label, key="home_start", type="primary", on_click=_go_to_tab, args=("검사하기",), use_container_width=True)

    with st.container(key="home_nav"):
        exercise_column, record_column = st.columns(2)
        with exercise_column:
            st.button("운동", key="home_to_exercise", icon=":material/fitness_center:", on_click=_go_to_tab, args=("운동",), use_container_width=True)
        with record_column:
            st.button("기록", key="home_to_records", icon=":material/show_chart:", on_click=_go_to_tab, args=("기록",), use_container_width=True)

    _render_program_status()

    st.markdown("<p class='wl-section'>휠런스는 이런 앱이에요</p>", unsafe_allow_html=True)
    st.markdown(
        "**가만히 앉은 자세**와 **바퀴를 미는 동작**을 함께 보고, "
        "몸·미는 기술·휠체어 세팅 가운데 **무엇을 먼저 확인하면 좋을지** 알려드려요.\n\n"
        "1. **검사하기** — 질문 5개에 답하고, 옆과 앞에서 찍은 영상을 올려요.\n"
        "2. **결과 확인** — 가만히 있을 때와 밀 때의 차이, 먼저 확인해 볼 것을 봐요.\n"
        "3. **운동** — 결과에 맞는 운동을 따라 해요.\n"
        "4. **기록** — 지난번과 비교해 변화를 확인해요."
    )
    _fine(SUPPORTED_USERS_NOTICE)


def _show_exercise_tab() -> None:
    """운동 목록을 읽고 최근 결과와 관련된 운동을 먼저 표시합니다."""
    import json
    from pathlib import Path

    st.warning("운동 전 휠체어 브레이크를 반드시 잠그세요. 통증이 있으면 즉시 중단하세요.")
    try:
        exercises = json.loads((Path(__file__).parent / "data" / "exercises.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        st.error("운동 목록을 불러오지 못했어요.")
        return

    # 최근 기록의 주의(1점)·경고(2점) 판정을 문제별 점수로 모아, 점수가 높은 문제의 운동을 먼저 보여준다.
    problem_scores: dict[str, int] = {}
    metric_problems: dict[str, str] = {
        "contact_angle": "어깨 부담", "stroke_rate": "어깨 부담", "pattern": "어깨 부담", "elbow_angle": "어깨 부담",
        "trunk_lean_mean": "상체 숙임", "trunk_lean_max": "상체 숙임", "trunk_flexion": "상체 숙임",
        "symmetry_index": "좌우 비대칭", "timing_diff": "좌우 비대칭", "shoulder_tilt": "좌우 비대칭",
        "lateral_lean": "좌우 비대칭", "lateral_lean_max": "좌우 비대칭", "pelvic_tilt": "좌우 비대칭",
        "modified_cva": "거북목",
    }
    try:
        uid = str(st.session_state["uid"])
        recent = database.list_propulsion_sessions(uid)[:5]
        for session in recent:
            for metric_name, judgment in (session.get("judgments") or {}).items():
                weight = {"caution": 1, "warning": 2}.get(judgment, 0)
                problem = metric_problems.get(metric_name)
                if weight and problem:
                    problem_scores[problem] = problem_scores.get(problem, 0) + weight
    except database.DatabaseError as error:
        st.caption(str(error))
        recent = []

    # 키워드(자세·움직임)로 고른 운동을 가장 먼저: 키워드 하나에 10점, 자세+움직임이 함께면 15점.
    found = _current_keywords(recent)
    keyword_scores = {keyword.key: keyword.weight + (5 if keyword.together else 0) for keyword in found}

    def exercise_score(item: dict[str, Any]) -> int:
        legacy = max((problem_scores.get(problem, 0) for problem in item.get("target_problems", [])), default=0)
        return sum(keyword_scores.get(key, 0) for key in item.get("keywords", [])) + legacy

    if found:
        exercises = [
            item for item in exercises
            if any(keyword.key in item.get("keywords", []) for keyword in found)
        ]
    exercises = sorted(exercises, key=lambda item: -exercise_score(item))
    if found:
        lines = "".join(f"<li><b>{keyword.label}</b>: {keyword.finding}</li>" for keyword in found)
        st.markdown(f"<div class='wl-compare'><p class='label'>내 검사 결과로 고른 운동이에요</p><ul class='note'>{lines}</ul></div>", unsafe_allow_html=True)
        missing = [keyword.label for keyword in found if not any(keyword.key in item.get("keywords", []) for item in exercises)]
        if missing:
            st.caption(f"{', '.join(missing)}: 맞춤 운동을 준비 중이에요.")
        if any(keyword.key in ("shoulder_drop", "push_side_drop") for keyword in found):
            _fine("어깨 높이 차이는 운동보다 휠체어 좌석·등받이 점검을 먼저 권해요. 휠체어 사용자 대상 좌우 교정 운동 연구는 아직 찾지 못했어요.")
    pain = keywords.pain_notice(_current_answers())
    if pain:
        st.warning(pain)
    found_keys = {keyword.key: keyword.label for keyword in found}

    # 오늘 완료한 운동 (기록 화면의 주간 횟수와 같은 저장소를 쓴다)
    done_today: set[str] = set()
    try:
        today = datetime.now(timezone.utc).date()
        for log in database.list_stretch_logs(str(st.session_state["uid"])):
            created = _as_datetime(log.get("created_at"))
            if created is not None and created.date() == today:
                done_today.add(str(log.get("exercise_id")))
    except database.DatabaseError:
        pass

    # 판별 결과가 있으면 관련 운동만, 결과가 없으면 전체 운동을 보여준다.
    finished = sum(str(item["id"]) in done_today for item in exercises)
    st.markdown("#### 운동")
    bars = "".join(f"<div class='{'done' if str(item['id']) in done_today else ''}'></div>" for item in exercises)
    st.markdown(f"<div class='wl-steps'>{bars}</div><p class='wl-steps-label wl-count'>오늘 <b>{finished}</b> / {len(exercises)}개 완료</p>", unsafe_allow_html=True)
    for exercise in exercises:
        _render_exercise_card(exercise, found_keys, str(exercise["id"]) in done_today)


SIDE_KEYWORD_METRICS = ("trunk_change", "head_cva_static")
FRONT_KEYWORD_METRICS = ("shoulder_tilt_static", "shoulder_tilt_moving", "lateral_lean_static", "head_tilt_static")


def _current_keywords(recent: list[dict[str, Any]] | None = None) -> list[keywords.Keyword]:
    """자세·움직임 키워드를 판별합니다 (keywords.detect).

    지금 화면에 있는 분석 결과를 먼저 쓰고, 없으면 가장 최근에 저장한 실제(데모 아님) 기록과 그때의 답을 쓴다.
    """
    real = [row for row in (recent or []) if not row.get("is_demo")]
    side_saved = next((row for row in real if (row.get("view") or "side") == "side"), None)
    front_saved = next((row for row in real if row.get("view") == "front"), None)
    side_now, front_now = _usable_result("side"), _usable_result("front")

    if side_now is not None:
        side = {name: _metric_number(side_now, name) for name in SIDE_KEYWORD_METRICS}
        side["pattern"] = str(getattr(side_now, "pattern", "unknown"))
    else:
        metrics = (side_saved or {}).get("metrics") or {}
        side = {name: _safe_float(metrics.get(name)) for name in SIDE_KEYWORD_METRICS}
        side["pattern"] = str(metrics.get("pattern") or "unknown")
    if front_now is not None:
        front = {name: _metric_number(front_now, name) for name in FRONT_KEYWORD_METRICS}
    else:
        metrics = (front_saved or {}).get("metrics") or {}
        front = {name: _safe_float(metrics.get(name)) for name in FRONT_KEYWORD_METRICS}
    answers = _current_answers() if (side_now is not None or front_now is not None) else dict((side_saved or front_saved or {}).get("answers") or {})
    return keywords.detect(answers, side, front)


def _render_keywords(found: list[keywords.Keyword]) -> None:
    """찾은 키워드를 카드로 보여줍니다: 자세/움직임 표시, 측정 문장, 질문으로 붙은 안내, '함께 나타남' 표시."""
    st.markdown("##### 내 몸에서 찾은 키워드")
    if not found:
        st.markdown("<p class='wl-muted'>기준을 넘은 항목이 없어요. 정기적으로 다시 재서 변화를 살펴보세요.</p>", unsafe_allow_html=True)
        return
    for keyword in found:
        badge = "<span class='wl-chip'>자세와 움직임에 함께 나타났어요</span>" if keyword.together else ""
        notes = "".join(f"<p class='note'>ℹ️ {note}</p>" for note in keyword.notes)
        st.markdown(
            f"<div class='wl-compare'><p class='label'>{'🧍 자세' if keyword.kind == '자세' else '🧑‍🦽 움직임'} · 기준 근거: {keyword.grade}</p>"
            f"<p class='nums'><b style='font-size:20px'>{keyword.label}</b> {badge}</p>"
            f"<p class='note'>{keyword.finding}</p>{notes}</div>",
            unsafe_allow_html=True,
        )
    _fine("'함께 나타났어요'는 가만히 있을 때의 자세 문제가 밀 때도 같은 방향으로 보인다는 뜻이에요. 원인을 확정하는 건 아니에요.")


def _render_exercise_card(exercise: dict[str, Any], found_keys: dict[str, str], done: bool) -> None:
    """운동 카드: 이름 → 키워드 칩 → 간단 설명(대상·늘이기·강화·원리) → 관절 마네킹 → 작은 글씨 근거 → 완료 버튼."""
    with st.container(border=True):
        st.subheader(exercise["name"])
        if exercise.get("formal_name"):
            st.caption(str(exercise["formal_name"]))
        # 칩 = 이 운동이 겨냥하는 키워드. 내 결과에서 찾은 키워드는 "· 내 결과"를 붙여 앞에 둔다.
        tags = [str(POSTURE_KEYWORDS[key]["label"]) for key in exercise.get("keywords", []) if key in POSTURE_KEYWORDS]
        mine = [found_keys[key] for key in exercise.get("keywords", []) if key in found_keys]
        chips = "".join(f"<span class='wl-chip'>{label} · 내 결과</span>" for label in mine) + "".join(
            f"<span class='wl-chip'>{label}</span>" for label in tags if label not in mine
        )
        if chips:
            st.markdown(chips, unsafe_allow_html=True)
        rows = [
            ("↔ 늘이는 근육", exercise.get("relax")),
            ("💪 강화하는 근육", exercise.get("strengthen")),
            ("💡 원리", exercise.get("principle")),
        ]
        lines = "".join(f"<div><b>{label}</b> {text}</div>" for label, text in rows if text)
        if lines:
            st.markdown(f"<div class='wl-principle'>{lines}</div>", unsafe_allow_html=True)
        if mannequin.has_motion(str(exercise["id"])):
            mannequin.show_motion(str(exercise["id"]))
        if exercise.get("form_cues"):
            items = "".join(f"<li>{html.escape(str(cue))}</li>" for cue in exercise["form_cues"])
            st.markdown(f"<div class='wl-cues'><b>✋ 이것만 주의해요</b><ul>{items}</ul></div>", unsafe_allow_html=True)
        if exercise.get("evidence"):
            st.markdown(f"<p class='wl-fine'>출처 · {html.escape(str(exercise['evidence']))}</p>", unsafe_allow_html=True)
        if done:
            st.markdown("<p class='wl-muted'>✓ 오늘 완료했어요</p>", unsafe_allow_html=True)
        elif st.button("완료하기", key=f"stretch_{exercise['id']}", icon=":material/check:", use_container_width=True):
            try:
                database.save_stretch_log(str(st.session_state["uid"]), exercise["id"])
                st.rerun()
            except database.DatabaseError as error:
                st.error(str(error))


def _render_last_comparison(sessions: list[dict[str, Any]]) -> None:
    """가장 최근 검사와 그 전 검사를 비교해 "달라졌나?"를 카드로 답합니다.

    변화가 trunk_change_min_detectable_deg(대조 영상에서 잰 흔들림 범위)보다 작으면 변화로 보지 않는다고 적는다.
    실제 기록이 2개 미만이면 발표용 예시 데이터로 보여준다 (그렇다고 표시).
    """
    def ordered(rows: list[dict[str, Any]], view: str, key: str) -> list[dict[str, Any]]:
        picked = [row for row in rows if (row.get("view") or "side") == view and _safe_float((row.get("metrics") or {}).get(key)) is not None]
        return sorted(picked, key=lambda row: _as_datetime(row.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    demo_used: list[bool] = []

    def pick(view: str, key: str) -> list[dict[str, Any]]:
        """그 항목의 실제 기록이 2개 미만이면 그 항목만 예시 데이터로 채운다."""
        rows = ordered([row for row in sessions if not row.get("is_demo")], view, key)
        if len(rows) < 2:
            rows = ordered(sessions, view, key)
            if len(rows) >= 2:
                demo_used.append(True)
        return rows

    side = pick("side", "trunk_change")
    neck_rows = pick("side", "head_cva_static")
    front = pick("front", "shoulder_tilt_static")
    used_demo = bool(demo_used)
    if len(side) < 2 and len(front) < 2 and len(neck_rows) < 2:
        st.info("검사를 두 번 이상 하면 여기에서 지난번과 비교해 드려요.")
        return

    limit = float(SCREENING_PARAMS["trunk_change_min_detectable_deg"])

    def change_note(diff: float, bigger: str, smaller: str) -> str:
        if abs(diff) < limit:
            return f"{abs(diff):.0f}° 차이 · 측정 흔들림 범위 안이라 변화로 보지 않아요"
        return f"{abs(diff):.0f}° " + (bigger if diff > 0 else smaller)

    # ── 자세 (가만히 있을 때): 거북목 → 어깨 높이 ──
    heads = [_safe_float((row.get("metrics") or {}).get("head_cva_static")) for row in neck_rows[:2]] if len(neck_rows) >= 2 else [None, None]
    if heads[0] is not None and heads[1] is not None:
        head_note = change_note(heads[0] - heads[1], "커졌어요 · 머리가 덜 나와요", "작아졌어요 · 머리가 더 나와요")
        reference = float(POSTURE_KEYWORDS["forward_head"]["threshold"])  # type: ignore[arg-type]
        photos = [row.get("posture_photo") for row in neck_rows[:2]]
        if photos[0] and photos[1]:
            _photo_compare_card("🧍 거북목 · 목 각도", photos[1], photos[0], f"{heads[1]:.0f}°", f"{heads[0]:.0f}°",
                                head_note, reference, [_as_datetime(row.get("created_at")) for row in neck_rows[:2]])
        else:
            _compare_card("🧍 거북목 · 목 각도 (클수록 바른 자세)", posture_figure.neck_angle(heads[1], heads[0], reference),
                          posture_figure.legend(f"거북목 기준 {reference:.0f}°"), "", head_note)
    if len(front) >= 2:
        before = float(front[1]["metrics"]["shoulder_tilt_static"])
        after = float(front[0]["metrics"]["shoulder_tilt_static"])
        _compare_card("🧍 어깨 높이 차이 (0°가 양쪽 같음)", posture_figure.shoulder_tilt(before, after, _side_angle),
                      posture_figure.legend("수평"), "", change_note(abs(after) - abs(before), "커졌어요", "줄었어요"))

    # ── 움직임 (밀 때) ──
    if len(side) >= 2:
        before = float(side[1]["metrics"]["trunk_change"])
        after = float(side[0]["metrics"]["trunk_change"])
        band_high = float(REFERENCE_COMPARISONS["trunk_change"]["band"][1])  # type: ignore[index]
        _compare_card("🧑‍🦽 밀 때 상체가 더 숙여지는 정도", posture_figure.trunk_lean(before, after, band_high),
                      posture_figure.legend(f"오래 탄 사람 ~{band_high:.0f}°"), "", change_note(after - before, "늘었어요", "줄었어요"))
        patterns = [str((row.get("metrics") or {}).get("pattern") or "") for row in side[:2]]
        if all(patterns) and "unknown" not in patterns:
            st.markdown(
                "<div class='wl-compare'><p class='label'>🧑‍🦽 미는 패턴 (권장: 반원형)</p>"
                f"<p class='nums'>{PATTERN_NAMES.get(patterns[1], patterns[1])} → <b style='font-size:20px'>{PATTERN_NAMES.get(patterns[0], patterns[0])}</b></p></div>",
                unsafe_allow_html=True,
            )
    if used_demo:
        _fine("실제 기록이 2개 미만인 항목은 발표용 예시 데이터로 보여드려요.")


def _photo_compare_card(label: str, before_b64: str, after_b64: str, before_text: str, after_text: str, note: str,
                        reference: float, dates: list[datetime | None]) -> None:
    """기록 비교 카드 (사진): 지난번 | 이번 자세 사진을 나란히, 위에 날짜와 각도를 크게."""
    def panel(b64: str, title: str, angle: str, color: str, when: datetime | None) -> str:
        day = when.astimezone().strftime("%m.%d") if when else ""
        return (f"<div class='wl-photo'><p style='color:{color}'>{title} <b>{angle}</b><span>{day}</span></p>"
                f"<img src='data:image/jpeg;base64,{b64}' alt='{title} 자세 사진'></div>")
    st.markdown(
        f"<div class='wl-compare'><p class='label'>{label}</p><div class='wl-photos'>"
        + panel(before_b64, "지난번", before_text, posture_figure.BEFORE, dates[1])
        + panel(after_b64, "이번", after_text, posture_figure.AFTER, dates[0])
        + f"</div><div class='wl-fig-legend'><span><i style='border-top:3px solid {posture_figure.AFTER}'></i>어깨→귀</span>"
        f"<span><i style='border-top:2px dashed {posture_figure.REF}'></i>거북목 기준 {reference:.0f}°</span></div><p class='note'>{note}</p></div>",
        unsafe_allow_html=True,
    )


def _compare_card(label: str, figure: str, legend: str, numbers: str, note: str) -> None:
    """기록 비교 카드: 제목 → 지난번|이번 선 그림(각도는 그림 위) → 범례 → 한 줄 설명. numbers는 그림에 이미 있어 쓰지 않음."""
    del numbers
    st.markdown(
        f"<div class='wl-compare'><p class='label'>{label}</p>{figure}{legend}<p class='note'>{note}</p></div>",
        unsafe_allow_html=True,
    )


def _show_statistics_tab() -> None:
    """기록: 지난 검사와 비교 → 전체 추이(상체 변화·추진 지표, 패턴 변화, 운동 횟수)."""
    try:
        sessions = database.list_propulsion_sessions(str(st.session_state["uid"]))
        stretches = database.list_stretch_logs(str(st.session_state["uid"]))
    except database.DatabaseError as error:
        st.warning(str(error))
        sessions, stretches = [], []

    st.markdown("#### 지난 검사와 비교")
    _render_last_comparison(sessions)
    st.markdown("#### 전체 추이")
    period = st.radio("조회 기간", ["전체", "월간", "주간"], horizontal=True, key="statistics_period", label_visibility="collapsed")

    now =datetime.now(timezone.utc)
    since = now - timedelta(days={"주간": 7, "월간": 30}.get(period, 36500))  # 전체 = 사실상 제한 없음
    filtered = [session for session in sessions if (_as_datetime(session.get("created_at")) or now) >= since]
    if not filtered and not stretches:
        st.info("아직 측정 기록이 없어요.")
    else:
        rows: list[dict[str, Any]] = []
        for session in filtered:
            created = _as_datetime(session.get("created_at")) or now
            row = {"created_at": created, "is_demo": session.get("is_demo", False)}
            row.update(session.get("metrics", {}))
            rows.append(row)
        data = pd.DataFrame(rows)
        if not data.empty:
            data = data.sort_values("created_at")
            if bool(data["is_demo"].any()):
                _fine("회색 선은 발표용 예시 데이터예요.")
            for group_title, group_columns in TREND_GROUPS:
                chart_columns = [column for column in group_columns if column in data and pd.to_numeric(data[column], errors="coerce").notna().any()]
                if not chart_columns:
                    continue
                st.markdown(f"##### {group_title}")
                for column in chart_columns:
                    figure = go.Figure()
                    for is_demo, label, color in ((False, "실제", CHART_PURPLE), (True, "데모", "#8B95A1")):
                        subset = data[data["is_demo"] == is_demo]
                        if column in subset and not subset.empty:
                            figure.add_trace(go.Scatter(x=subset["created_at"], y=pd.to_numeric(subset[column], errors="coerce"), mode="lines+markers", name=label, line={"color": color}))
                    figure.update_layout(height=220, title=METRIC_LABELS.get(column, (column, ""))[0], margin={"l": 10, "r": 10, "t": 40, "b": 10}, showlegend=False)
                    st.plotly_chart(figure, use_container_width=True)

            pattern_rows = [(row.get("created_at"), PATTERN_NAMES.get(str(row.get("pattern")), row.get("pattern"))) for row in rows if row.get("pattern")]
            if pattern_rows:
                st.subheader("손 궤적 패턴 변화")
                st.dataframe(pd.DataFrame(pattern_rows, columns=["측정일", "패턴"]), use_container_width=True, hide_index=True)

        # 주 단위로 묶어서 스트레칭 완료 횟수를 보여준다.
        def _week_label(value: Any) -> str:
            moment = _as_datetime(value) or now
            monday = (moment - timedelta(days=moment.weekday())).date()
            return monday.isoformat()  # 정렬용 (2026-09-28). 화면에는 "9/28주"로 표시

        weekly: dict[str, int] = {}
        for log in stretches:
            if (_as_datetime(log.get("created_at")) or now) >= since:
                label = _week_label(log.get("created_at"))
                weekly[label] = weekly.get(label, 0) + 1
        if weekly:
            st.subheader("주간 운동 완료 횟수")
            order = sorted(weekly)
            labels = [f"{int(key[5:7])}/{int(key[8:10])}주" for key in order]
            bars = go.Figure(go.Bar(x=labels, y=[weekly[key] for key in order], marker_color=CHART_PURPLE))
            bars.update_layout(height=240, margin={"l": 10, "r": 10, "t": 20, "b": 10})
            st.plotly_chart(bars, use_container_width=True)


def _load_demo_data() -> None:
    """발표용 예시 기록을 1주 간격으로 4번 저장합니다 (옆·앞 각각, 모두 is_demo=True). 예전 예시는 지웁니다."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    demo_metrics = [
        {"trunk_static": 4.0, "trunk_peak": 19.0, "trunk_change": 15.0, "contact_angle": 72.0, "stroke_rate": 34.0, "pattern": "arc", "head_cva_static": 37.0},
        {"trunk_static": 4.0, "trunk_peak": 16.0, "trunk_change": 12.0, "contact_angle": 88.0, "stroke_rate": 29.0, "pattern": "semicircular", "head_cva_static": 40.0},
        {"trunk_static": 3.0, "trunk_peak": 12.0, "trunk_change": 9.0, "contact_angle": 96.0, "stroke_rate": 25.0, "pattern": "semicircular", "head_cva_static": 42.0},
        {"trunk_static": 3.0, "trunk_peak": 10.0, "trunk_change": 7.0, "contact_angle": 101.0, "stroke_rate": 23.0, "pattern": "semicircular", "head_cva_static": 46.0},
    ]
    database.delete_demo_data(str(st.session_state["uid"]))  # 다시 불러오면 예전 예시를 지우고 새로 넣는다
    front_metrics = [
        {"shoulder_tilt_static": 6.8, "shoulder_tilt_moving": 8.9, "lateral_lean_static": -4.0, "head_tilt_static": 5.2},
        {"shoulder_tilt_static": 5.6, "shoulder_tilt_moving": 7.4, "lateral_lean_static": -3.1, "head_tilt_static": 4.1},
        {"shoulder_tilt_static": 4.1, "shoulder_tilt_moving": 5.5, "lateral_lean_static": -2.4, "head_tilt_static": 3.0},
        {"shoulder_tilt_static": 2.6, "shoulder_tilt_moving": 3.4, "lateral_lean_static": -1.5, "head_tilt_static": 2.2},
    ]
    for index, metrics in enumerate(front_metrics):
        database.save_propulsion_session(
            str(st.session_state["uid"]), view="front", filmed_side="", metrics=metrics,
            judgments={name: "pending" for name in metrics}, is_demo=True,
            created_at=now - timedelta(days=(3 - index) * 7, minutes=-5),
        )
    for index, metrics in enumerate(demo_metrics):
        judgments = {
            "contact_angle": propulsion.judge_minimum("contact_angle", metrics["contact_angle"]),
            "stroke_rate": propulsion.judge_maximum("stroke_rate", metrics["stroke_rate"]),
            "pattern": propulsion.judge_pattern(str(metrics["pattern"])).judgment,
        }
        database.save_propulsion_session(
            str(st.session_state["uid"]),
            view="side",
            filmed_side="right",
            metrics=metrics,
            judgments=judgments,
            is_demo=True,
            created_at=now - timedelta(days=(3 - index) * 7),
        )


def _show_sidebar() -> None:
    """로그인 정보와 프로필·데모 삭제·전체 삭제·디버그 설정을 표시합니다."""
    with st.sidebar:
        st.markdown(f"### {APP_NAME}")
        st.caption(str(st.session_state.get("email", "")))
        if st.button("로그아웃", use_container_width=True):
            auth.logout()
            st.rerun()
        st.divider()
        # 일반 사용자에게 필요 없는 도구는 접어 둔다.
        with st.expander("발표·개발용 도구"):
            st.session_state["debug_mode"] = st.toggle("디버그 모드", value=st.session_state.get("debug_mode", False))
            st.caption("예시 기록(옆·앞 4번씩)을 넣어 기록 화면을 미리 볼 수 있어요.")
            if st.button("데모 데이터 불러오기", key="load_demo_data", use_container_width=True):
                try:
                    _load_demo_data()
                    st.rerun()
                except database.DatabaseError as error:
                    st.error(str(error))
            if st.button("데모 데이터만 삭제", key="sidebar_delete_demo", use_container_width=True):
                try:
                    database.delete_demo_data(str(st.session_state["uid"]))
                    st.rerun()
                except database.DatabaseError as error:
                    st.error(str(error))
        st.divider()
        st.subheader("데이터 관리")
        delete_confirm = st.checkbox("내 데이터를 모두 삭제하는 것에 동의합니다.", key="delete_all_confirm")
        if st.button("내 데이터 전체 삭제", disabled=not delete_confirm, use_container_width=True):
            try:
                deleted = database.delete_all_user_data(str(st.session_state["uid"]))
                st.success(f"{deleted}개 데이터를 삭제했어요.")
            except database.DatabaseError as error:
                st.error(str(error))


def _show_authenticated_app() -> None:
    """동의 완료 사용자의 전체 앱을 표시합니다."""
    _show_sidebar()
    _render_header()
    page = st.session_state.setdefault("page", "home")
    {"home": _show_home_tab, "exam": _show_exam_tab, "exercise": _show_exercise_tab, "records": _show_statistics_tab}.get(page, _show_home_tab)()
    _show_disclaimer()
    _scroll_top_if_moved()


def main() -> None:
    """Firebase 로그인 상태에 따라 앱 흐름을 렌더링합니다."""
    if not auth.is_authenticated():
        _show_login_page()
        return
    try:
        user_document = _load_user_document()
        if _needs_consent(user_document):
            _show_consent_page()
            return
        _show_authenticated_app()
    except auth.AuthError as error:
        auth.logout()
        _show_error(str(error))
        _show_login_page()


if __name__ == "__main__":
    main()
