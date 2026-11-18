"""Wheelance의 모바일 우선 스타일을 관리합니다.

다크모드: .streamlit/config.toml에서 테마 색을 고정하지 않아 Streamlit이 휴대폰 설정(밝게/어둡게)을 따른다.
여기서는 배경·글자색을 직접 정하지 않고, 어느 바탕에서도 보이는 반투명 색과 보라 강조색만 쓴다.
"""

from __future__ import annotations

import streamlit as st


PRETENDARD_CDN: str = "https://cdn.jsdelivr.net/gh/orioncactus/pretendard@1.3.9/dist/web/static/pretendard.css"

# 강조색 (판넬과 같은 보라). 흰 글씨를 얹어도 밝기 대비가 충분한 값.
PURPLE: str = "#6B3FA0"
PURPLE_DEEP: str = "#4A2463"


def inject_custom_css() -> None:
    """모바일 화면·접근성·다크모드를 위한 전역 CSS를 Streamlit에 주입합니다.

    st.markdown은 빈 줄을 만나면 HTML 블록이 끝난 것으로 보고 나머지 CSS를 글자로 찍는다 → 빈 줄을 지우고 넣는다.
    """
    css = f"""
        <link rel="stylesheet" href="{PRETENDARD_CDN}">
        <style>
        :root {{
            --wl-purple: {PURPLE};
            --wl-purple-deep: {PURPLE_DEEP};
            --wl-tint: rgba(107, 63, 160, 0.10);      /* 연한 보라 바탕 (밝은·어두운 화면 공통) */
            --wl-tint-strong: rgba(107, 63, 160, 0.22);
            --wl-line: rgba(128, 128, 128, 0.28);      /* 카드 테두리 */
            --wl-accent-text: #5B2F92;                 /* 연한 보라 바탕 위 글자 */
            --wl-muted: rgba(128, 128, 128, 1);
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{ --wl-accent-text: #C9B3EC; --wl-tint: rgba(155, 115, 210, 0.16); --wl-tint-strong: rgba(155, 115, 210, 0.30); }}
        }}
        html, body, [class*="css"] {{
            font-family: Pretendard, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        .block-container {{ max-width: 430px; padding: 3.5rem 1rem 5rem; }}
        p, li, label, .stMarkdown {{ font-size: 17px; line-height: 1.65; }}
        h1, h2, h3 {{ letter-spacing: -0.02em; }}
        h1 {{ font-size: 28px; }}
        h2 {{ font-size: 23px; margin-top: 1rem; }}
        h3 {{ font-size: 19px; }}
        @media (min-width: 700px) {{ .block-container {{ max-width: 760px; }} }}

        /* ---------- 버튼 ---------- */
        .stButton > button, .stDownloadButton > button, .stLinkButton > a {{
            min-height: 52px;
            border-radius: 14px;
            font-size: 17px;
            font-weight: 700;
            padding: 0.65rem 1rem;
            width: 100%;
        }}
        button[data-testid="stBaseButton-primary"] {{
            background: var(--wl-purple); border-color: var(--wl-purple); color: #FFFFFF;
        }}
        button[data-testid="stBaseButton-primary"]:hover {{ background: var(--wl-purple-deep); border-color: var(--wl-purple-deep); color: #FFFFFF; }}
        button[data-testid="stBaseButton-primary"] p {{ color: #FFFFFF; }}
        button[data-testid="stBaseButton-secondary"]:hover {{ border-color: var(--wl-purple); color: var(--wl-purple); }}

        /* ---------- 탭 ---------- */
        [data-baseweb="tab-highlight"] {{ background-color: var(--wl-purple); }}
        button[role="tab"][aria-selected="true"] p {{ color: var(--wl-accent-text); font-weight: 700; }}

        /* ---------- 고르기(라디오)를 동그란 버튼으로 ---------- */
        [data-testid="stRadio"] [role="radiogroup"] {{ flex-direction: row; flex-wrap: wrap; gap: 0.45rem; }}
        [data-testid="stRadio"] [role="radiogroup"] > label {{
            border: 1px solid var(--wl-line); border-radius: 999px; padding: 0.35rem 0.9rem; margin: 0;
        }}
        [data-testid="stRadio"] [role="radiogroup"] > label > div:first-child {{ display: none; }}
        [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {{
            background: var(--wl-tint-strong); border-color: var(--wl-purple);
        }}
        [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p {{ color: var(--wl-accent-text); font-weight: 700; }}
        [data-testid="stRadio"] [role="radiogroup"] p {{ font-size: 15px; }}

        /* ---------- 체크박스 ---------- */
        label[data-baseweb="checkbox"]:has(input:checked) > span:first-child {{
            background-color: var(--wl-purple); border-color: var(--wl-purple);
        }}

        /* ---------- 입력 ---------- */
        .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {{ min-height: 48px; font-size: 17px; }}

        /* ---------- 수치 카드 ---------- */
        [data-testid="stMetric"] {{ background: var(--wl-tint); border-radius: 14px; padding: 0.8rem; min-height: 96px; }}

        /* ---------- 공통 카드·글자 ---------- */
        .wheelance-disclaimer {{ color: var(--wl-muted); font-size: 14px; margin-top: 2rem; }}
        .wl-chip {{
            display: inline-block; background: var(--wl-tint-strong); color: var(--wl-accent-text);
            border-radius: 8px; padding: 1px 8px; font-size: 13px; font-weight: 600; margin: 2px 4px 2px 0;
        }}
        .wl-principle {{
            background: var(--wl-tint); border-radius: 10px; padding: 10px 12px; margin: 6px 0 10px;
            font-size: 14px; line-height: 1.7;
        }}
        [data-testid="stMarkdownContainer"] p.wl-fine {{ font-size: 11px !important; line-height: 1.5; color: var(--wl-muted); opacity: 0.6; margin: 2px 0 6px; }}
        .wl-cues {{ border-left: 3px solid #E8A33D; padding: 6px 10px; margin: 8px 0; font-size: 14px; }}
        .wl-cues ul {{ margin: 4px 0 0; padding-left: 18px; }}
        .wl-cues li {{ margin: 2px 0; }}
        .wl-stats {{ display: flex; gap: 8px; margin: 4px 0 12px; }}
        .wl-stat {{ flex: 1; background: var(--wl-tint); border-radius: 12px; padding: 10px 12px; }}
        .wl-stat span {{ display: block; font-size: 13px; color: var(--wl-muted); }}
        .wl-stat b {{ font-size: 30px; line-height: 1.2; color: var(--wl-accent-text); }}
        .wl-stat small {{ font-size: 15px; font-weight: 600; margin-left: 2px; }}
        p.wl-steps-label.wl-count {{ font-size: 17px; color: var(--wl-accent-text); font-weight: 600; }}
        p.wl-count b {{ font-size: 24px; }}
        [data-testid="stMarkdownContainer"] p.wl-brand {{ font-size: 28px; font-weight: 800; color: var(--wl-accent-text); margin: 0 0 0.6rem; letter-spacing: -0.5px; }}
        [data-testid="stMarkdownContainer"] p.wl-login-subtitle {{ color: var(--wl-accent-text); font-size: 17px; font-weight: 700; line-height: 1.65; opacity: 1; margin: -0.35rem 0 0.7rem; text-align: left; white-space: nowrap; }}
        [data-testid="stMarkdownContainer"] p.wl-page-title {{ font-size: 22px; font-weight: 700; margin: 0; }}
        .st-key-page_bar [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap; gap: 0.6rem; align-items: center; }}
        .st-key-page_bar [data-testid="stColumn"]:first-child {{ flex: 0 0 auto; width: auto !important; min-width: 0; }}
        .st-key-page_bar [data-testid="stColumn"]:last-child {{ flex: 1 1 auto; min-width: 0; }}
        .st-key-page_bar .stButton > button {{ width: auto; padding: 0.3rem 0.9rem; min-height: 0; }}
        [data-testid="stMarkdownContainer"] p.wl-section {{ font-size: 18px; font-weight: 700; margin: 1.2rem 0 0.3rem; }}
        .wl-fig-legend {{ display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; font-size: 12px; color: var(--wl-muted); margin: 2px 0 6px; }}
        .wl-fig-legend i {{ display: inline-block; width: 18px; vertical-align: middle; margin-right: 4px; }}
        .wl-photos {{ display: flex; gap: 8px; margin: 6px 0; }}
        .wl-photo {{ flex: 1; min-width: 0; }}
        .wl-photo p {{ margin: 0 0 4px; font-size: 14px; font-weight: 700; display: flex; justify-content: space-between; align-items: baseline; }}
        .wl-photo p b {{ font-size: 20px; }}
        .wl-photo p span {{ font-size: 11px; font-weight: 500; color: var(--wl-muted); }}
        .wl-photo img {{ width: 100%; border-radius: 10px; display: block; }}
        .wl-principle b {{ color: var(--wl-accent-text); margin-right: 4px; }}
        .wl-muted {{ color: var(--wl-muted); font-size: 14px; }}
        .wl-compare {{ background: var(--wl-tint); border: 1px solid var(--wl-tint-strong); border-radius: 16px; padding: 0.9rem 1rem; margin: 0.4rem 0 0.8rem; }}
        .wl-compare .label {{ color: var(--wl-accent-text); font-size: 14px; margin: 0; }}
        .wl-compare .nums {{ font-size: 15px; margin: 0.15rem 0; color: var(--wl-accent-text); }}
        .wl-compare .nums b {{ font-size: 26px; }}
        .wl-compare .note {{ color: var(--wl-accent-text); font-size: 13px; margin: 0; }}

        /* ---------- 참고 범위 vs 나 막대 ---------- */
        .wl-ref {{ border: 1px solid var(--wl-line); border-radius: 14px; padding: 0.7rem 0.9rem; margin: 0 0 0.6rem; }}
        p.wl-ref-title {{ font-size: 15px; font-weight: 700; margin: 0 0 0.5rem; }}
        .wl-ref-track {{ position: relative; height: 12px; border-radius: 6px; background: var(--wl-line); margin: 0.2rem 0 0.5rem; }}
        .wl-ref-band {{ position: absolute; top: 0; height: 12px; border-radius: 6px; background: var(--wl-tint-strong); border: 1px solid var(--wl-purple); box-sizing: border-box; }}
        .wl-ref-tick {{ position: absolute; top: -3px; width: 3px; height: 18px; margin-left: -1.5px; background: var(--wl-purple); border-radius: 2px; }}
        .wl-ref-me {{ position: absolute; top: -4px; width: 20px; height: 20px; margin-left: -10px; border-radius: 50%; background: #EF9F27; border: 2px solid #FFFFFF; box-sizing: border-box; }}
        .wl-ref-legend {{ display: flex; flex-wrap: wrap; gap: 0.3rem 1rem; font-size: 14px; }}
        .wl-ref-key {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }}
        .wl-ref-key.band {{ background: var(--wl-tint-strong); border: 1px solid var(--wl-purple); border-radius: 3px; }}
        .wl-ref-key.me {{ background: #EF9F27; }}
        p.wl-ref-source {{ color: var(--wl-muted); font-size: 12px; margin: 0.3rem 0 0; }}
        p.wl-ref-pattern {{ font-size: 15px; margin: 0; }}

        /* ---------- 진행 막대 (검사·운동) ---------- */
        .wl-steps {{ display: flex; gap: 6px; margin: 0.3rem 0 0.2rem; }}
        .wl-steps div {{ flex: 1; height: 7px; border-radius: 4px; background: var(--wl-line); }}
        .wl-steps div.done {{ background: var(--wl-purple); }}
        .wl-steps-label {{ color: var(--wl-muted); font-size: 14px; margin: 0 0 0.6rem; }}

        /* ---------- 보라 큰 카드 (홈 시작 카드, 검사 결과 결론) ---------- */
        .st-key-home_hero, .st-key-result_hero {{
            background: var(--wl-purple);
            border-radius: 20px;
            padding: 1.4rem 1.2rem 1.2rem;
            margin: 0.5rem 0 1rem;
            box-sizing: border-box;
        }}
        /* Streamlit이 안쪽 요소에 부모 폭(px)을 직접 넣어서, 카드 안쪽 여백만큼 삐져나가지 않게 100%로 되돌린다 */
        .st-key-home_hero > div, .st-key-home_hero .stMarkdown, .st-key-home_hero .stButton,
        .st-key-result_hero > div, .st-key-result_hero .stMarkdown {{ width: 100% !important; max-width: 100%; }}
        .st-key-home_hero p, .st-key-home_hero li, .st-key-result_hero p, .st-key-result_hero li {{ color: #FFFFFF; }}
        .st-key-home_hero p {{ margin: 0; }}
        .st-key-result_hero li, .st-key-result_hero p {{ font-size: 15px; }}
        .st-key-home_hero p.hero-small, .st-key-result_hero p.hero-small {{ font-size: 14px; opacity: 0.88; line-height: 1.5; margin: 0; }}
        .st-key-home_hero p.hero-title, .st-key-result_hero p.hero-title {{ font-size: 26px; font-weight: 700; line-height: 1.35; margin: 0.4rem 0 0.6rem; }}
        .st-key-home_hero .stButton > button {{ background: #FFFFFF; color: var(--wl-purple-deep); border: none; margin-top: 0.6rem; }}
        .st-key-home_hero .stButton > button:hover {{ background: #F0EAF8; color: var(--wl-purple-deep); }}
        .st-key-home_hero .stButton > button p {{ color: var(--wl-purple-deep); font-size: 18px; }}

        /* 홈의 운동·기록 버튼은 휴대폰에서도 한 줄에 나란히 */
        .st-key-home_nav [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap; gap: 0.6rem; }}
        .st-key-home_nav [data-testid="stColumn"] {{ min-width: 0; width: 50% !important; flex: 1 1 0; }}
        </style>
        """
    st.markdown("\n".join(line for line in css.splitlines() if line.strip()), unsafe_allow_html=True)
