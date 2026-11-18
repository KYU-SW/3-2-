"""질문 답변과 영상 측정값을 합쳐 "먼저 확인해 볼 것"의 순서를 정합니다.

원인을 진단하지 않습니다. 몸 / 미는 기술 / 휠체어 세팅 가운데 어느 쪽을 먼저 살펴볼지
점수로 줄 세우고, 그 근거(영상에서 본 것, 답변한 것)를 함께 돌려줍니다.

이 규칙은 임상에서 쓰는 사고방식(문진 + 정지 자세 + 동작 관찰)을 옮긴 **가설**이며,
논문으로 검증된 분류법이 아닙니다. 점수와 기준값은 config.SCREENING_PARAMS에서 바꿉니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from config import SCREENING_LEVERS, SCREENING_PARAMS, SCREENING_QUESTIONS

LEVER_ORDER: tuple[str, ...] = ("technique", "body", "setup")  # 점수가 같을 때의 순서


@dataclass
class Lever:
    """확인해 볼 것 한 가지와 그 근거."""

    key: str
    label: str
    guide: str
    score: int = 0
    video_reasons: list[str] = field(default_factory=list)
    answer_reasons: list[str] = field(default_factory=list)


@dataclass
class ScreeningResult:
    """선별 결과: 점수순으로 정렬한 확인 항목과 따로 알려줄 안내."""

    levers: list[Lever]
    notices: list[str] = field(default_factory=list)
    trunk_level: str = "unknown"  # "small" | "moderate" | "notable" | "unknown"

    @property
    def top(self) -> Lever | None:
        """점수가 가장 높은 항목. 근거가 하나도 없으면 None."""
        return self.levers[0] if self.levers and self.levers[0].score > 0 else None


def missing_questions(answers: Mapping[str, Any]) -> list[str]:
    """아직 답하지 않은 질문 id 목록을 반환합니다."""
    missing: list[str] = []
    for question in SCREENING_QUESTIONS:
        options = question["options"]
        if answers.get(str(question["id"])) not in options:  # type: ignore[operator]
            missing.append(str(question["id"]))
    return missing


# 예전 통증 답(있음/가끔/밀 때마다)으로 저장된 기록도 읽을 수 있게 대략의 점수로 바꾼다.
_LEGACY_PAIN: dict[str, float] = {"none": 0.0, "sometimes": 2.0, "pushing": 4.0}


def pain_score(answer: Any) -> float | None:
    """통증 답을 0~10 점수로 바꿉니다 (예전 3단계 답도 지원). 없거나 잘못된 값이면 None."""
    if answer is None:
        return None
    if str(answer) in _LEGACY_PAIN:
        return _LEGACY_PAIN[str(answer)]
    try:
        score = float(answer)
    except (TypeError, ValueError):
        return None
    return score if 0 <= score <= 10 else None


def _number(value: Any) -> float | None:
    """유한한 숫자만 float로 돌려줍니다 (NaN, None, 문자열은 None)."""
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def trunk_change_level(trunk_change: float | None) -> str:
    """정지 대비 밀 때 상체가 더 숙여진 정도를 3단계로 나눕니다.

    change < min_detectable          → "small"    (측정 흔들림과 구분하지 않음)
    min_detectable ≤ change < notable → "moderate"
    change ≥ notable                 → "notable"  (많이 숙여짐)
    값이 없으면 "unknown".
    """
    if trunk_change is None:
        return "unknown"
    if trunk_change < SCREENING_PARAMS["trunk_change_min_detectable_deg"]:
        return "small"
    if trunk_change < SCREENING_PARAMS["trunk_change_notable_deg"]:
        return "moderate"
    return "notable"


def frontal_sentence(key: str, value: float) -> str:
    """정면 정지 자세의 기울기를 한 문장으로 씁니다 (부호: + = 앉은 사람의 왼쪽)."""
    side_word = "왼쪽" if value > 0 else "오른쪽"
    if key == "shoulder_tilt_static":
        return f"가만히 있을 때 {side_word} 어깨가 {abs(value):.0f}° 낮아요."
    return f"가만히 있을 때 몸통이 {side_word}으로 {abs(value):.0f}° 기울어 있어요."


def recommend(answers: Mapping[str, Any], metrics: Mapping[str, Any]) -> ScreeningResult:
    """답변과 측정값으로 확인 순서를 정합니다.

    metrics: {"trunk_change": 정지 대비 밀 때 상체 숙임 증가(°), "pattern": 손 궤적 패턴 이름,
              "shoulder_tilt_static": 정면 정지 어깨 기울기(°, + 왼쪽 낮음),
              "lateral_lean_static": 정면 정지 몸통 옆 기울기(°, + 왼쪽으로)}
             값이 없는 항목은 건너뛴다.

    점수 규칙 (가설):
    [영상]
    - 정면에서 가만히 있을 때 어깨선이 frontal_notable_deg 이상 기울어 있음 → 세팅 +2
      (몸통 옆 기울기는 휴대폰 기울기에 민감해 점수에 넣지 않고 화면에 참고로만 보여준다)
    - 밀 때 상체가 많이 숙여짐(notable)   → 기술 +2, 몸 +1
      단, "상체를 스스로 못 일으킴"이면 필요한 움직임일 수 있어 점수를 주지 않고 안내만 한다.
    - 손 궤적이 권장 패턴(반원형)이 아님    → 기술 +2
    [답변]
    - 미는 방법을 배운 적 없음             → 기술 +2
    - 금방 지침 → 몸 +2 / 조금 힘듦 → 몸 +1
    - 상체 일으키기가 어렵지만 됨           → 몸 +1
    - 휠체어 점검 받은 적 없음 → 세팅 +2 / 받은 지 오래됨 → 세팅 +1
    [따로 안내]
    - 통증이 있으면 순서와 상관없이 전문가 상담을 먼저 권한다.
    """
    levers = {
        key: Lever(key, SCREENING_LEVERS[key]["label"], SCREENING_LEVERS[key]["guide"]) for key in LEVER_ORDER
    }
    notices: list[str] = []

    trunk_change = _number(metrics.get("trunk_change"))
    level = trunk_change_level(trunk_change)
    trunk_control = answers.get("trunk_control")

    if level == "notable" and trunk_change is not None:
        if trunk_control == "no":
            notices.append(
                f"밀 때 상체가 가만히 있을 때보다 {trunk_change:.0f}° 더 숙여져요. "
                "상체를 스스로 일으키기 어렵다고 답하셔서, 미는 데 필요한 움직임일 수 있다고 보고 문제로 세지 않았어요."
            )
        else:
            reason = f"밀 때 상체가 가만히 있을 때보다 {trunk_change:.0f}° 더 숙여져요."
            levers["technique"].score += 2
            levers["technique"].video_reasons.append(reason)
            levers["body"].score += 1
            levers["body"].video_reasons.append(reason)

    # 정면: 가만히 있을 때 어깨선·몸통이 뚜렷하게 기울어 있으면 세팅 점검을 먼저 권한다.
    # (휠체어 사용자의 좌우 기울기는 운동보다 좌석·지지대 근거가 있음. 앱은 유연한 틀어짐과 고정된 틀어짐을 구분 못 함)
    frontal_limit = SCREENING_PARAMS["frontal_notable_deg"]
    # 몸통 옆 기울기(lateral_lean_static)는 점수에 넣지 않는다: 실제 정면 영상에서 똑바로 민 사람 10명 중
    # 2명이 −5° 안팎으로 나왔고(휴대폰 기울기에 그대로 치우침), 일부러 비대칭으로 한 영상은 구분하지 못했다.
    shoulder_tilt = _number(metrics.get("shoulder_tilt_static"))
    if shoulder_tilt is not None and abs(shoulder_tilt) >= frontal_limit:
        levers["setup"].score += 2
        levers["setup"].video_reasons.append(frontal_sentence("shoulder_tilt_static", shoulder_tilt))

    pattern = metrics.get("pattern")
    if isinstance(pattern, str) and pattern not in ("semicircular", "unknown", ""):
        levers["technique"].score += 2
        levers["technique"].video_reasons.append("손이 돌아오는 길이 권장 패턴(반원형)이 아니에요.")

    if answers.get("training") == "no":
        levers["technique"].score += 2
        levers["technique"].answer_reasons.append("미는 방법을 배운 적이 없어요.")

    fatigue = answers.get("fatigue")
    if fatigue == "tired":
        levers["body"].score += 2
        levers["body"].answer_reasons.append("평지에서 5분이 되기 전에 쉬어야 해요.")
    elif fatigue == "little":
        levers["body"].score += 1
        levers["body"].answer_reasons.append("5분 밀고 나면 숨이 차거나 팔이 무거워요.")
    if trunk_control == "hard":
        levers["body"].score += 1
        levers["body"].answer_reasons.append("상체를 일으킬 때 몸이 흔들리거나 여러 번 시도해야 해요.")

    fit = answers.get("fit")
    if fit == "never":
        levers["setup"].score += 2
        levers["setup"].answer_reasons.append("휠체어를 몸에 맞게 조정받은 적이 없거나 모르겠다고 답하셨어요.")
    elif fit == "old":
        levers["setup"].score += 1
        levers["setup"].answer_reasons.append("휠체어 조정을 받은 지 1년이 넘었어요.")

    # 통증(NRS 0~10)은 순서와 상관없이 맨 위에 안내한다. 4점 이상(중간·심함)이면 운동보다 상담을 먼저 권한다.
    pain = pain_score(answers.get("pain"))
    if pain is not None and pain >= SCREENING_PARAMS["pain_moderate_min"]:
        notices.insert(0, f"최근 1주일 동안 통증이 {pain:.0f}점(중간 이상)이었다고 답하셨어요. 운동이나 연습보다 먼저 전문가 상담을 받아보세요.")
    elif pain is not None and pain >= SCREENING_PARAMS["pain_mild_min"]:
        notices.insert(0, f"최근 1주일 동안 가벼운 통증이 있었다고 답하셨어요 ({pain:.0f}점). 통증이 계속되거나 심해지면 전문가 상담을 받아보세요.")

    ordered = sorted(levers.values(), key=lambda lever: (-lever.score, LEVER_ORDER.index(lever.key)))
    return ScreeningResult(levers=ordered, notices=notices, trunk_level=level)
