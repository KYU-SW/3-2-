"""자세·움직임 키워드 판별 → 질문으로 거르기 → 키워드별 운동 연결.

흐름
1. 측정값으로 키워드를 붙인다 (config.POSTURE_KEYWORDS의 기준값).
   - 자세 키워드: 영상 처음 '가만히 있는 3초'에서 잰 값 (거북목, 한쪽 어깨 처짐)
   - 움직임 키워드: 밀 때 잰 값 (상체 무너짐, 미는 패턴, 밀 때 한쪽 처짐)
2. 질문 답으로 체형 문제가 아닐 수 있는 경우를 거른다.
   - 상체를 손으로 짚어야만 일으킴 → '밀 때 상체 무너짐' 제외 (미는 데 필요한 움직임일 수 있음)
   - 휠체어 조정 안 받음 → 좌우 키워드에 "의자 때문일 수도" 안내
   - 미는 법 안 배움 / 금방 지침 → 움직임 키워드에 "기술·체력 때문일 수도" 안내
3. 짝(같은 방향의 자세 + 움직임)이 둘 다 있으면 "자세와 움직임에 함께 나타났어요"로 묶고 먼저 보여준다.
   ⚠ "자세 때문에 움직임이 나빠진다"는 연구로 확인된 것이 아니다 → '때문'이 아니라 '함께'로 표현한다.
4. 키워드마다 data/exercises.json의 운동(target_problems에 키워드 이름)을 띄운다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from config import POSTURE_KEYWORDS, SCREENING_PARAMS


@dataclass
class Keyword:
    """판별된 키워드 하나."""

    key: str
    label: str
    kind: str                      # "자세" 또는 "움직임"
    finding: str                   # 측정 결과 문장 ("가만히 있을 때 왼쪽 어깨가 7° 낮아요")
    grade: str                     # 기준값 근거 등급 (문헌 / 자체 측정 / 가설)
    notes: list[str] = field(default_factory=list)   # 질문으로 붙는 안내
    together: bool = False         # 짝 키워드와 함께 나타남
    weight: int = 10               # 운동 순서 점수 (거북목은 단계에 따라 2단계 5 · 3단계 20)


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _side_word(value: float) -> str:
    """+ = 앉은 사람의 왼쪽."""
    return "왼쪽" if value > 0 else "오른쪽"


def forward_head_level(head_angle: float) -> int:
    """앱 목 각도(어깨→귀, 수평 기준) → FNTA 단계 0(정상)~3 (Diagnostics 2024 기준을 90°−FNTA로 바꾼 값)."""
    return sum(head_angle < float(limit) for limit in POSTURE_KEYWORDS["forward_head"]["levels"])  # type: ignore[union-attr]


def detect(answers: Mapping[str, Any], side: Mapping[str, Any] | None, front: Mapping[str, Any] | None) -> list[Keyword]:
    """측정값(side/front: 지표 이름 → 숫자, side에는 "pattern" 포함)과 질문 답으로 키워드를 판별합니다."""
    side = side or {}
    front = front or {}
    found: dict[str, Keyword] = {}

    def add(key: str, finding: str) -> None:
        spec = POSTURE_KEYWORDS[key]
        found[key] = Keyword(key, str(spec["label"]), str(spec["kind"]), finding, str(spec["grade"]))

    # ---------- 자세 (가만히 있을 때) ----------
    head = _number(side.get("head_cva_static"))
    head_limit = POSTURE_KEYWORDS["forward_head"]["threshold"]
    if head is not None and head_limit is not None and head < float(head_limit):  # type: ignore[arg-type]
        # 단계마다 안내 강도를 다르게: 2단계 '경향·권장'(다른 운동 뒤로), 3단계 '거북목·먼저 추천'(맨 앞으로). 1단계는 키워드 없음.
        if forward_head_level(head) >= 3:
            add("forward_head", f"머리가 앞으로 많이 나와 있어요 (목 각도 {head:.0f}° · 논문 기준 3단계). 목 운동을 먼저 해 보세요.")
            found["forward_head"].weight = 20
        else:
            add("forward_head", f"거북목 경향이 있어요 (목 각도 {head:.0f}° · 논문 기준 2단계). 목 운동을 권해요.")
            found["forward_head"].weight = 5

    tilt = _number(front.get("shoulder_tilt_static"))
    if tilt is not None and abs(tilt) >= float(POSTURE_KEYWORDS["shoulder_drop"]["threshold"]):  # type: ignore[arg-type]
        add("shoulder_drop", f"가만히 있을 때 {_side_word(tilt)} 어깨가 {abs(tilt):.0f}° 낮아요.")

    lean = _number(front.get("lateral_lean_static"))
    if lean is not None and abs(lean) >= float(POSTURE_KEYWORDS["trunk_side_lean"]["threshold"]):  # type: ignore[arg-type]
        add("trunk_side_lean", f"가만히 있을 때 몸통이 {_side_word(lean)}으로 {abs(lean):.0f}° 기울어 있어요.")

    head_tilt = _number(front.get("head_tilt_static"))
    if head_tilt is not None and abs(head_tilt) >= float(POSTURE_KEYWORDS["head_side_tilt"]["threshold"]):  # type: ignore[arg-type]
        add("head_side_tilt", f"가만히 있을 때 고개가 {_side_word(head_tilt)}으로 {abs(head_tilt):.0f}° 기울어 있어요.")

    # ---------- 움직임 (밀 때) ----------
    # 밀 때 상체 무너짐: 많이 숙여짐(10.6° 이상) + 피로(5분 밀기 힘듦)가 함께일 때만.
    # 지치면 팔 근육 피로를 상체를 숙여 메우고, 이 방식은 팔·어깨 부상 위험을 높인다 (Rodgers 2000 JRRD; Rodgers 1994).
    # 숙임 각도 하나만으로는 의미 근거가 약하고 측정 흔들림(5~6°)도 커서, 피로가 없으면 결과 숫자로만 보여준다.
    change = _number(side.get("trunk_change"))
    tired = answers.get("fatigue") in ("tired", "little")
    if change is not None and tired and change >= float(POSTURE_KEYWORDS["trunk_collapse"]["threshold"]):  # type: ignore[arg-type]
        add("trunk_collapse", f"지쳐서 상체로 밀고 있어요. 밀 때 상체가 가만히 있을 때보다 {change:.0f}° 더 숙여지고, "
                              "5분 밀기가 힘들다고 하셨어요. 이렇게 밀면 어깨 부담이 커질 수 있어요.")

    pattern = str(side.get("pattern") or "unknown")
    if pattern not in ("semicircular", "unknown", ""):
        names = {"arc": "호형", "single_loop": "루프형", "double_loop": "이중 루프형"}
        add("push_pattern", f"미는 패턴이 {names.get(pattern, pattern)}이에요. 권장은 반원형이에요.")

    moving_tilt = _number(front.get("shoulder_tilt_moving"))
    drop_limit = float(POSTURE_KEYWORDS["push_side_drop"]["threshold"])  # type: ignore[arg-type]
    if tilt is not None and moving_tilt is not None:
        extra = abs(moving_tilt) - abs(tilt)
        same_side = moving_tilt * tilt >= 0 or abs(tilt) < 1.0
        if same_side and extra >= drop_limit:
            add("push_side_drop", f"밀고 올 때 {_side_word(moving_tilt)} 어깨가 가만히 있을 때보다 {extra:.0f}° 더 처져요.")

    # ---------- 질문으로 거르기 ----------
    if answers.get("trunk_control") == "no":
        found.pop("trunk_collapse", None)
    if answers.get("fit") in ("never", "old"):
        for key in ("shoulder_drop", "trunk_side_lean", "push_side_drop"):
            if key in found:
                found[key].notes.append("휠체어를 몸에 맞게 조정한 지 오래됐거나 받은 적이 없어요. 의자 때문일 수도 있어 세팅 점검을 먼저 권해요.")
    if answers.get("training") == "no":
        for key in ("trunk_collapse", "push_pattern", "push_side_drop"):
            if key in found:
                found[key].notes.append("미는 법을 배운 적이 없다고 하셨어요. 체형보다 미는 기술 문제일 수도 있어요.")

    # ---------- 짝: 자세 + 움직임이 함께 ----------
    for posture_key, movement_key in (
        ("forward_head", "trunk_collapse"), ("shoulder_drop", "push_side_drop"), ("trunk_side_lean", "push_side_drop"),
    ):
        if posture_key in found and movement_key in found:
            found[posture_key].together = True
            found[movement_key].together = True

    order = list(POSTURE_KEYWORDS)
    return sorted(found.values(), key=lambda keyword: (not keyword.together, keyword.kind != "자세", order.index(keyword.key)))


def pain_notice(answers: Mapping[str, Any]) -> str | None:
    """통증 4점 이상이면 운동보다 상담을 먼저 권하는 문장."""
    from screening import pain_score

    score = pain_score(answers.get("pain"))
    if score is not None and score >= SCREENING_PARAMS["pain_moderate_min"]:
        return f"통증이 {score:.0f}점이라 운동보다 전문가 상담을 먼저 받아보세요. 아래 운동은 상담 후 해주세요."
    return None
