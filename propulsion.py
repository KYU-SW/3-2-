"""휠체어 추진 영상의 측정값과 규칙 기반 코칭을 계산합니다.

전체 흐름
1) 영상을 15fps, 긴 변 720px로 줄여 한 장씩 읽는다.
2) 프레임마다 MediaPipe로 관절을 찾고, 픽셀 좌표로 바꾼다.
3) 손목 좌표를 hip 기준 상대 좌표로 바꾼다 (카메라 흔들림 보정).
4) 사용자가 첫 화면에서 찍은 림 3점도 같은 hip 기준 좌표로 바꾼다.
   → 휠체어와 몸은 같이 움직이므로, 림도 hip에 붙어서 따라간다고 본다.
5) 손목 궤적을 스무딩한 뒤 밀기(push) / 복귀(recovery)로 나눈다.
6) 스트로크 단위로 접촉 각도, 추진 빈도, 손 궤적 패턴 등을 계산한다.

좌표계: 이미지 픽셀, y축은 아래로 갈수록 커진다 (위쪽 = y가 작음).
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from config import (
    ALGORITHM_PARAMS,
    COACHING_MESSAGES,
    JUDGMENT_PENDING,
    MIN_VALID_FRAME_RATIO,
    THRESHOLDS,
    VIDEO_LONG_SIDE_PX,
    VIDEO_LONG_SIDE_PX_FRONT,
    VIDEO_MAX_SECONDS,
    VIDEO_MIN_SECONDS,
    VIDEO_SAMPLE_FPS,
    WRIST_SMOOTHING_WINDOW,
)
from camera_check import CameraCheck
from overlay import compose_overlay
from pose_utils import (
    LandmarkPoint,
    circle_from_three_points,
    create_pose,
    detect_landmarks,
    has_required_visibility,
    iter_sampled_frames,
    joint_angle,
    landmarks_to_pixels,
    line_horizontal_deviation,
    line_vertical_deviation,
    mean_visibility,
    moving_average,
    open_video,
    read_upright,
    resize_long_side,
    video_info,
)

ProgressCallback = Callable[[float], None]

PHASE_PUSH = "push"
PHASE_RECOVERY = "recovery"

PATTERN_COACHING_KEYS: dict[str, str] = {
    "arc": "pattern_arc",
    "single_looping_over": "pattern_single_looping",
    "double_looping_over": "pattern_double_looping",
}


# ---------------------------------------------------------------------------
# 결과 자료구조
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """측정값, 판정, 코칭 키를 묶은 결과입니다.

    coaching_key는 판정이 주의/경고일 때만 채워지며 config.COACHING_MESSAGES의 키다.
    """

    value: float | str | None
    judgment: str = JUDGMENT_PENDING
    coaching_key: str | None = None


@dataclass
class FrameObservation:
    """한 프레임에서 뽑은 픽셀 좌표와 visibility 정보입니다."""

    timestamp_seconds: float
    landmarks: dict[str, LandmarkPoint]
    pixels: dict[str, np.ndarray]
    anchor: np.ndarray | None
    valid: bool
    visibility: float


@dataclass
class PropulsionAnalysisResult:
    """추진 분석 결과와 화면 표시·디버그용 데이터를 포함합니다."""

    view: str
    filmed_side: str | None
    duration_seconds: float
    valid_frame_ratio: float
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    pattern: str = "unknown"
    stroke_count: int = 0
    trajectory: list[tuple[float, float, str]] = field(default_factory=list)
    rim: tuple[float, float, float] | None = None
    representative_frames: list[np.ndarray] = field(default_factory=list)
    debug: dict[str, list[Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # 촬영 점검: (상태 "ok"/"warn", 항목, 안내 문구)
    filming_checks: list[tuple[str, str, str]] = field(default_factory=list)
    # 가만히 있는 구간 가운데의 사진 한 장 (RGB) + 그 장면의 시각 — C7 탭 비교(정밀 측정 검증)에 쓴다
    still_frame: np.ndarray | None = None
    still_time: float | None = None
    # 그 장면의 관절 픽셀 좌표 (귀·어깨·골반·코·눈) — 자세 사진에 각도 선을 그리고 얼굴을 흐리는 데 쓴다
    still_points: dict[str, tuple[float, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 판정
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    """판정 가능한 유한한 숫자인지 확인합니다."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _threshold_value(metric: str, key: str) -> float | None:
    """설정 딕셔너리에서 숫자 기준을 읽고 미설정 값은 None으로 반환합니다."""
    value = THRESHOLDS.get(metric, {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def judge_minimum(metric: str, value: float | None) -> str:
    """클수록 좋은 항목 판정 (예: 접촉 각도).

    value ≥ normal_min → 정상, value ≥ caution_min → 주의, 그 밖 → 경고.
    값이 없거나(NaN) 기준값이 미설정이면 판정 보류.
    """
    normal_min = _threshold_value(metric, "normal_min")
    caution_min = _threshold_value(metric, "caution_min")
    if not _is_number(value) or normal_min is None or caution_min is None:
        return JUDGMENT_PENDING
    if value >= normal_min:
        return "normal"
    if value >= caution_min:
        return "caution"
    return "warning"


def judge_maximum(metric: str, value: float | None) -> str:
    """작을수록 좋은 항목 판정 (예: 추진 빈도, 상체 숙임).

    value ≤ normal_max → 정상, value ≤ caution_max → 주의, 그 밖 → 경고.
    값이 없거나(NaN) 기준값이 미설정이면 판정 보류.
    """
    normal_max = _threshold_value(metric, "normal_max")
    caution_max = _threshold_value(metric, "caution_max")
    if not _is_number(value) or normal_max is None or caution_max is None:
        return JUDGMENT_PENDING
    if value <= normal_max:
        return "normal"
    if value <= caution_max:
        return "caution"
    return "warning"


def judge_range(metric: str, value: float | None) -> str:
    """권장 범위 안이면 정상, 벗어나면 주의 (예: 팔꿈치 각도). 기준 미설정이면 보류."""
    low = _threshold_value(metric, "recommended_min")
    high = _threshold_value(metric, "recommended_max")
    if not _is_number(value) or low is None or high is None:
        return JUDGMENT_PENDING
    return "normal" if low <= value <= high else "caution"


def _metric(value: float | str | None, judgment: str, coaching_key: str | None = None) -> MetricResult:
    """주의/경고일 때만 코칭 키를 붙인 MetricResult를 만듭니다."""
    show_coaching = judgment in ("caution", "warning") and coaching_key in COACHING_MESSAGES
    return MetricResult(value, judgment, coaching_key if show_coaching else None)


def judge_pattern(pattern: str) -> MetricResult:
    """손 궤적 패턴 판정.

    근거: Consortium for Spinal Cord Medicine (2005) 가이드라인은 semicircular 패턴을 권장한다.
    semicircular → 정상, 나머지 3가지 → 주의, 판정 불가 → 보류.
    """
    if pattern == "semicircular":
        return MetricResult(pattern, "normal", None)
    if pattern in PATTERN_COACHING_KEYS:
        return MetricResult(pattern, "caution", PATTERN_COACHING_KEYS[pattern])
    return MetricResult(pattern, JUDGMENT_PENDING, None)


# ---------------------------------------------------------------------------
# 측면: 밀기/복귀 구분과 스트로크 분할
# ---------------------------------------------------------------------------


def detect_facing_sign(observations: Sequence[FrameObservation], side: str) -> float:
    """사람이 화면에서 바라보는 방향을 +1(오른쪽) / -1(왼쪽)으로 추정합니다.

    코(nose)는 어깨보다 앞에 있으므로, 여러 프레임에서 (코 x - 어깨 x)의 중앙값 부호를 쓴다.
    밀기 동작은 손이 "앞쪽"으로 움직이는 것이라 이 방향이 필요하다.
    """
    shoulder = f"{side}_shoulder"
    differences = [
        float(obs.pixels["nose"][0] - obs.pixels[shoulder][0])
        for obs in observations
        if obs.valid and "nose" in obs.pixels and shoulder in obs.pixels
    ]
    if not differences:
        return 1.0
    return 1.0 if float(np.median(differences)) >= 0 else -1.0


def classify_phases(
    wrists: np.ndarray,
    rim_center: np.ndarray,
    rim_radius: float,
    facing_sign: float,
) -> list[str]:
    """프레임마다 밀기(push)/복귀(recovery)를 판정합니다.

    밀기 조건 (둘 다 만족):
    1) 손목이 림 원 둘레 근처: | |손목-중심| - 반지름 | ≤ 반지름 × 허용비율
    2) 손목이 앞쪽으로 이동 중: (x_t - x_(t-1)) × 바라보는 방향 > 0
    그 뒤 짧은 끊김은 이어 붙이고, 너무 짧은 밀기는 복귀로 돌린다.
    """
    count = len(wrists)
    if count == 0:
        return []
    tolerance = rim_radius * float(ALGORITHM_PARAMS["rim_contact_tolerance_ratio"])
    distances = np.linalg.norm(wrists - rim_center[None, :], axis=1)
    near_rim = np.abs(distances - rim_radius) <= tolerance
    velocity_x = np.diff(wrists[:, 0], prepend=wrists[0, 0]) * facing_sign
    is_push = near_rim & (velocity_x > 0)

    # 1) 짧은 끊김 메우기: push 사이에 낀 gap 이하의 recovery는 push로 본다.
    gap = int(ALGORITHM_PARAMS["push_gap_tolerance_frames"])
    filled = is_push.copy()
    index = 0
    while index < count:
        if not filled[index]:
            start = index
            while index < count and not filled[index]:
                index += 1
            if 0 < start and index < count and (index - start) <= gap:
                filled[start:index] = True
        else:
            index += 1

    # 2) 너무 짧은 push는 잡음으로 보고 제거한다.
    minimum = int(ALGORITHM_PARAMS["min_push_frames"])
    index = 0
    while index < count:
        if filled[index]:
            start = index
            while index < count and filled[index]:
                index += 1
            if index - start < minimum:
                filled[start:index] = False
        else:
            index += 1

    return [PHASE_PUSH if flag else PHASE_RECOVERY for flag in filled]


def split_strokes(phases: Sequence[str]) -> list[tuple[int, int, int]]:
    """phase 목록을 스트로크 단위로 나눕니다.

    반환: (밀기 시작, 밀기 끝(포함 안 함), 다음 밀기 시작) 인덱스 튜플 목록.
    한 스트로크 = 밀기 구간 + 그 뒤 복귀 구간. 마지막 스트로크의 복귀는 영상 끝까지.
    """
    starts_ends: list[tuple[int, int]] = []
    index = 0
    while index < len(phases):
        if phases[index] == PHASE_PUSH:
            start = index
            while index < len(phases) and phases[index] == PHASE_PUSH:
                index += 1
            starts_ends.append((start, index))
        else:
            index += 1
    strokes: list[tuple[int, int, int]] = []
    for order, (start, end) in enumerate(starts_ends):
        next_start = starts_ends[order + 1][0] if order + 1 < len(starts_ends) else len(phases)
        strokes.append((start, end, next_start))
    return strokes


def split_strokes_mean_crossing(wrist_x: Sequence[float]) -> list[tuple[int, int]]:
    """대안 스트로크 분할: 손목 x좌표가 평균선을 지나는 횟수로 주기를 나눕니다.

    근거: Nourbakhsh et al. (2026) — 손목 가로 위치가 평균값과 3번 교차할 때마다 1주기.
    - 교차점 = (x - 평균)의 부호가 바뀌는 프레임 (0은 직전 부호를 유지)
    - 한 번 왕복(밀기+복귀)은 평균선을 2번 지나므로, 교차 3번이 한 주기의 시작·중간·끝이다.
      다음 주기는 이전 주기의 마지막 교차에서 시작한다 (교차 0-2, 2-4, 4-6 ...).
    반환: (주기 시작 프레임, 주기 끝 프레임) 목록. 교차가 3번 미만이면 빈 목록.
    """
    values = np.asarray(wrist_x, dtype=np.float64)
    if len(values) < 3:
        return []
    signs = np.sign(values - float(np.mean(values)))
    for index in range(1, len(signs)):  # 평균과 정확히 같은 값은 직전 부호로 본다
        if signs[index] == 0:
            signs[index] = signs[index - 1]
    crossings = [index for index in range(1, len(signs)) if signs[index] != 0 and signs[index - 1] != 0 and signs[index] != signs[index - 1]]
    step = int(ALGORITHM_PARAMS["mean_crossings_per_cycle"]) - 1
    return [(crossings[order], crossings[order + step]) for order in range(0, len(crossings) - step, step)]


def calculate_contact_angle(points: np.ndarray, center: np.ndarray) -> float:
    """한 번 밀 때 손이 림 위에서 이동한 호의 각도(°)를 계산합니다.

    프레임마다 림 중심 기준 각도 θ = atan2(y, x)를 구하고,
    연속한 프레임 사이 변화량을 -180~180°로 풀어서(unwrap) 모두 더한 절댓값이다.
    시작점과 끝점만 비교하면 180°를 넘는 경우를 잘못 재므로 누적 방식을 쓴다.
    """
    if len(points) < 2:
        return float("nan")
    relative = np.asarray(points, dtype=np.float64) - center[None, :]
    angles = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
    return float(abs(math.degrees(angles[-1] - angles[0])))


def calculate_stroke_rate(push_start_times: Sequence[float], duration_seconds: float) -> float:
    """분당 추진 횟수(strokes/min)를 계산합니다.

    스트로크가 2번 이상이면: (횟수-1) ÷ (마지막 시작 - 첫 시작) × 60
      → 영상 앞뒤의 멈춘 시간에 영향을 덜 받는다.
    1번뿐이면: 1 ÷ 영상 길이 × 60
    """
    if len(push_start_times) >= 2:
        span = float(push_start_times[-1] - push_start_times[0])
        if span > 0:
            return (len(push_start_times) - 1) / span * 60.0
    if push_start_times and duration_seconds > 0:
        return len(push_start_times) / duration_seconds * 60.0
    return float("nan")


def classify_stroke_pattern(
    push_points: np.ndarray,
    recovery_points: np.ndarray,
    rim_center: np.ndarray,
    rim_radius: float,
) -> tuple[str, str]:
    """한 스트로크의 손 궤적 패턴을 추정합니다. (패턴, 판단 근거 문장) 반환.

    기준: 복귀 중 손목이 림 중심에서 얼마나 떨어져 있는지를, 밀 때의 거리와 비교한다.
    - 기준 거리 r_push = 밀기 구간 손목-림중심 거리의 중앙값
      (MediaPipe는 손이 아니라 손목을 찍으므로 림 반지름 대신 실제 밀 때 거리를 기준으로 쓴다)
    - d = 복귀 점의 거리 - r_push
      d > margin  → 림 바깥(위)으로 돌아감 (over)
      d < -margin → 림 안쪽(아래)으로 떨어짐 (under)
      그 사이      → 밀 때와 같은 경로
    - margin = 림 반지름 × loop_crossing_min_distance_ratio
    - 복귀 시작·끝 15%는 손이 림을 놓고/잡는 전환 구간이라 제외한다.
    규칙 (Consortium for Spinal Cord Medicine 2005의 4가지 패턴 정의를 따름):
    - over와 under가 모두 있고 over → under 순서로 바뀜 → double_looping_over
    - under 비율 ≥ below_path_ratio                       → semicircular (림 아래로 복귀)
    - over 비율 ≥ 0.5                                      → single_looping_over
    - 나머지 (밀기 경로를 거의 그대로 되돌아감)            → arc
    """
    if len(push_points) < 2 or len(recovery_points) < 3:
        return "unknown", "밀기 또는 복귀 프레임이 부족해 판정 보류"

    push_distance = float(np.median(np.linalg.norm(push_points - rim_center[None, :], axis=1)))
    trim = int(len(recovery_points) * 0.15)
    middle = recovery_points[trim : len(recovery_points) - trim] if len(recovery_points) - 2 * trim >= 3 else recovery_points
    offset = np.linalg.norm(middle - rim_center[None, :], axis=1) - push_distance
    margin = rim_radius * float(ALGORITHM_PARAMS["loop_crossing_min_distance_ratio"])

    states = np.where(offset > margin, 1, np.where(offset < -margin, -1, 0))  # 1=over, -1=under
    over_ratio = float(np.mean(states == 1))
    under_ratio = float(np.mean(states == -1))
    signs = [int(state) for state in states if state != 0]
    over_to_under = any(first == 1 and second == -1 for first, second in zip(signs, signs[1:]))

    reason = f"복귀 중 림 바깥 {over_ratio:.0%}, 안쪽 {under_ratio:.0%}, 바깥→안쪽 전환 {'있음' if over_to_under else '없음'}"
    if over_ratio > 0 and under_ratio > 0 and over_to_under:
        return "double_looping_over", reason
    if under_ratio >= float(ALGORITHM_PARAMS["below_path_ratio"]):
        return "semicircular", reason
    if over_ratio >= 0.5:
        return "single_looping_over", reason
    return "arc", reason


def vote_pattern(patterns: Sequence[str]) -> str:
    """스트로크별 패턴 중 가장 많이 나온 것을 대표 패턴으로 고릅니다 (unknown 제외)."""
    known = [pattern for pattern in patterns if pattern != "unknown"]
    if not known:
        return "unknown"
    values, counts = np.unique(known, return_counts=True)
    return str(values[int(np.argmax(counts))])


# ---------------------------------------------------------------------------
# 측면 / 정면 지표 계산
# ---------------------------------------------------------------------------


def _trunk_forward_lean(shoulder: np.ndarray, hip: np.ndarray) -> float:
    """상체 숙임 각도(°): hip→shoulder 선이 수직선에서 기운 정도.

    공식: θ = arctan( |shoulder_x - hip_x| / |shoulder_y - hip_y| )
    0°면 똑바로 앉음, 클수록 앞(또는 뒤)으로 기울어짐.
    """
    return line_vertical_deviation(hip, shoulder)


def find_baseline_end(wrists: np.ndarray, times: Sequence[float], torso_length: float) -> int:
    """영상 앞부분에서 "가만히 앉아 있는" 구간이 끝나는 프레임 번호를 찾습니다.

    - 기준 위치 = 처음 0.5초 동안 손목(hip 기준) 위치의 중앙값
    - 손목이 기준 위치에서 (몸통 길이 × baseline_still_ratio) 넘게 벗어난 프레임이 2장 연속 나오면
      그 첫 프레임부터를 "움직임 시작"으로 본다.
    - 정지 구간이 baseline_min_seconds보다 짧으면 0을 돌려준다 (기준 없음).
    - baseline_max_seconds를 넘는 부분은 정지 구간으로 쓰지 않는다.
    반환값 n: wrists[:n]이 정지 구간.
    """
    count = len(wrists)
    if count < 3 or torso_length <= 0:
        return 0
    start_time = float(times[0])
    head = [index for index in range(count) if times[index] - start_time <= 0.5]
    reference = np.median(wrists[head], axis=0)
    tolerance = torso_length * float(ALGORITHM_PARAMS["baseline_still_ratio"])
    distances = np.linalg.norm(wrists - reference[None, :], axis=1)
    max_seconds = float(ALGORITHM_PARAMS["baseline_max_seconds"])

    end = count
    for index in range(count):
        if times[index] - start_time > max_seconds:
            end = index
            break
        moved_twice = distances[index] > tolerance and (index + 1 >= count or distances[index + 1] > tolerance)
        if moved_twice:
            end = index
            break
    if end == 0 or times[end - 1] - start_time < float(ALGORITHM_PARAMS["baseline_min_seconds"]):
        return 0
    return end


def calculate_trunk_comparison(
    forward_lean: np.ndarray,
    baseline_end: int,
) -> dict[str, float]:
    """가만히 있을 때와 밀 때의 상체 각도를 비교합니다.

    forward_lean: 프레임별 상체 각도(°). 0 = 똑바로 섬, + = 앞으로 숙임, − = 뒤로 기댐.
    - trunk_static  = 정지 구간의 중앙값
    - trunk_peak    = 움직임 구간의 95백분위수 (가장 많이 숙였을 때. 한두 프레임 튐은 제외)
    - trunk_change  = trunk_peak − trunk_static (밀 때 더 숙여진 정도)
    - trunk_sweep   = 움직임 구간의 95백분위수 − 5백분위수 (밀 때 상체가 앞뒤로 오간 폭)
    정지 구간이 없으면(baseline_end == 0) static·change는 NaN.
    """
    nan = float("nan")
    values = np.asarray(forward_lean, dtype=np.float64)
    moving = values[baseline_end:]
    moving = moving[np.isfinite(moving)]
    still = values[:baseline_end]
    still = still[np.isfinite(still)]
    percentile = float(ALGORITHM_PARAMS["dynamic_peak_percentile"])
    static = float(np.median(still)) if len(still) else nan
    peak = float(np.percentile(moving, percentile)) if len(moving) else nan
    low = float(np.percentile(moving, 100.0 - percentile)) if len(moving) else nan
    return {
        "trunk_static": static,
        "trunk_peak": peak,
        "trunk_change": peak - static if math.isfinite(static) and math.isfinite(peak) else nan,
        "trunk_sweep": peak - low if math.isfinite(peak) else nan,
    }


def classify_pattern_by_return_height(
    wrists: np.ndarray,
    torso_length: float,
    facing_sign: float,
) -> tuple[str, str, float]:
    """림 없이 손 궤적 패턴을 추정합니다. (패턴, 판단 근거 문장, 높이 차이 비율) 반환.

    생각: 밀 때는 손이 앞으로, 돌아올 때는 뒤로 움직인다. 돌아오는 길이 미는 길보다
    낮으면 반원형, 같으면 호형, 높으면 루프형이다 (Consortium 2005의 패턴 정의와 같은 구분).
    - 앞으로 움직이는 프레임: (x_t − x_(t−1)) × 바라보는 방향 > 0
    - 뒤로 움직이는 프레임: 그 반대. 거의 안 움직인 프레임은 뺀다.
    - d = (뒤로 갈 때 손목 y의 중앙값 − 앞으로 갈 때 손목 y의 중앙값) ÷ 몸통 길이
      (y는 아래로 갈수록 크므로 d > 0 이면 돌아올 때 더 낮다)
    - d > pattern_return_lower_ratio   → semicircular
      d < −pattern_return_higher_ratio → single_looping_over
      그 사이                           → arc
    이중 루프(위로 갔다가 아래로 교차)는 이 값 하나로는 구분하지 못한다.
    기준값은 실제 영상 16개에서 정했다 (config.ALGORITHM_PARAMS 주석 참고).
    """
    nan = float("nan")
    if len(wrists) < 3 or torso_length <= 0:
        return "unknown", "움직인 프레임이 부족해 판정 보류", nan
    velocity_x = np.diff(wrists[:, 0], prepend=wrists[0, 0]) * facing_sign
    moving = np.abs(velocity_x) > torso_length * float(ALGORITHM_PARAMS["pattern_min_speed_ratio"])
    forward = moving & (velocity_x > 0)
    backward = moving & (velocity_x < 0)
    minimum = int(ALGORITHM_PARAMS["pattern_min_frames"])
    if int(forward.sum()) < minimum or int(backward.sum()) < minimum:
        return "unknown", "앞뒤로 움직인 프레임이 부족해 판정 보류", nan
    difference = float(np.median(wrists[backward, 1]) - np.median(wrists[forward, 1])) / torso_length
    reason = f"돌아올 때 손 높이가 밀 때보다 몸통 길이의 {abs(difference):.0%}만큼 {'낮음' if difference > 0 else '높음'}"
    if difference > float(ALGORITHM_PARAMS["pattern_return_lower_ratio"]):
        return "semicircular", reason, difference
    if difference < -float(ALGORITHM_PARAMS["pattern_return_higher_ratio"]):
        return "single_looping_over", reason, difference
    return "arc", reason, difference


def calculate_side_metrics(
    observations: Sequence[FrameObservation],
    side: str,
    rim_offset: np.ndarray | None,
    rim_radius: float | None,
    duration_seconds: float,
    result: PropulsionAnalysisResult,
    frame_width: float = 0.0,
) -> None:
    """측면 영상에서 추진 지표를 계산해 result에 채웁니다.

    - 화면 가운데 구간(side_center_zone)에 있는 프레임만 쓴다 (frame_width가 0이면 전체).
    - 손 궤적 패턴은 림 없이 classify_pattern_by_return_height()로 정한다.
    - 림(rim_offset, rim_radius)이 있으면 접촉 각도·팔꿈치 각도를 추가로 계산한다.
      없으면 추진 빈도는 손목 x 평균선 교차 방식으로 센다.
    """
    wrist_name = f"{side}_wrist"
    shoulder_name = f"{side}_shoulder"
    elbow_name = f"{side}_elbow"
    hip_name = f"{side}_hip"

    tracked = [
        obs for obs in observations
        if obs.valid and obs.anchor is not None
        and has_required_visibility(obs.landmarks, (wrist_name,))
    ]
    facing = detect_facing_sign(tracked, side)
    # 나가는 쪽(바라보는 방향의 화면 끝)은 뒤에서 비스듬히 보여 각도가 달라지므로 더 많이 뺀다.
    exit_cut = float(ALGORITHM_PARAMS["side_exit_cut_ratio"])
    entry_cut = float(ALGORITHM_PARAMS["side_entry_cut_ratio"])
    zone_low, zone_high = (entry_cut, 1.0 - exit_cut) if facing > 0 else (exit_cut, 1.0 - entry_cut)

    def in_usable_zone(obs: FrameObservation) -> bool:
        """골반이 분석에 쓰는 화면 구간 안에 있는지."""
        if frame_width <= 0 or obs.anchor is None:
            return True
        return zone_low <= float(obs.anchor[0]) / frame_width <= zone_high

    usable = [obs for obs in tracked if in_usable_zone(obs)]
    if len(usable) < 2:
        result.warnings.append(
            "손목이 거의 보이지 않아요. 촬영한 쪽 팔이 화면에 잘 보이도록 다시 촬영해주세요."
            if len(tracked) < 2
            else "사람이 화면 가운데를 지나는 장면이 거의 없어요. 화면 가운데를 지나가도록 다시 촬영해주세요."
        )
        return

    # 손목 좌표를 hip 기준 상대 좌표로 바꾸고, 떨림을 줄인다.
    raw_wrists = np.array([obs.pixels[wrist_name] - obs.anchor for obs in usable])
    wrists = np.array(moving_average(raw_wrists, WRIST_SMOOTHING_WINDOW))
    times = [obs.timestamp_seconds for obs in usable]

    # 상체 각도(부호 있음): 수직 위쪽 기준, 바라보는 방향으로 기울면 +(앞으로 숙임), 반대면 −(뒤로 기댐)
    #   θ = atan2( (shoulder_x − hip_x) × 바라보는 방향, hip_y − shoulder_y )
    forward_lean = np.full(len(usable), np.nan)
    torso_lengths: list[float] = []
    for index, obs in enumerate(usable):
        if has_required_visibility(obs.landmarks, (shoulder_name, hip_name)):
            delta = obs.pixels[shoulder_name] - obs.pixels[hip_name]
            forward_lean[index] = math.degrees(math.atan2(float(delta[0]) * facing, -float(delta[1])))
            torso_lengths.append(float(np.linalg.norm(delta)))
    torso_length = float(np.median(torso_lengths)) if torso_lengths else 0.0
    baseline_end = find_baseline_end(wrists, times, torso_length)
    comparison = calculate_trunk_comparison(forward_lean, baseline_end)

    # 머리 앞으로 나옴 (거북목, modified CVA): 어깨 → 귀 선이 수평선과 이루는 각도.
    #   CVA = atan2(어깨_y − 귀_y, (귀_x − 어깨_x) × 바라보는 방향)
    #   90° = 귀가 어깨 바로 위, 작을수록 머리가 앞으로 나옴. 원래 CVA는 C7을 쓰지만 MediaPipe에 없어 어깨로 대신한다
    #   → 문헌의 CVA 정상값과 직접 비교할 수 없다 (자체 기준·본인 변화로 쓴다).
    ear_name = f"{side}_ear"
    head_cva = np.full(len(usable), np.nan)
    for index, obs in enumerate(usable):
        if ear_name in obs.pixels and has_required_visibility(obs.landmarks, (ear_name, shoulder_name)):
            delta = obs.pixels[ear_name] - obs.pixels[shoulder_name]
            head_cva[index] = math.degrees(math.atan2(-float(delta[1]), float(delta[0]) * facing))
    still_head, moving_head = head_cva[:baseline_end], head_cva[baseline_end:]
    still_head, moving_head = still_head[np.isfinite(still_head)], moving_head[np.isfinite(moving_head)]
    head_static = float(np.median(still_head)) if len(still_head) >= 3 else float("nan")
    head_moving = float(np.median(moving_head)) if len(moving_head) >= 3 else float("nan")
    if baseline_end == 0:
        result.notes.append(
            "영상 처음에 가만히 앉아 있는 구간을 찾지 못해 '가만히 있을 때' 기준을 잡지 못했어요. "
            "다음엔 화면 안에서 3초 동안 손을 무릎이나 바퀴 위에 두고 가만히 있다가 밀어주세요."
        )

    has_rim = rim_offset is not None and rim_radius is not None
    if has_rim:
        phases = classify_phases(wrists, rim_offset, float(rim_radius), facing)
        # 가만히 있는 동안 손을 림에 얹고 있으면 미세한 떨림이 밀기로 잡힌다 → 정지 구간은 밀기에서 뺀다.
        phases[:baseline_end] = ["still"] * baseline_end
        strokes = split_strokes(phases)
    else:
        # 림이 없으면 손이 림에 닿았는지 알 수 없다. 움직임 구간의 손목 x가 평균선을 지나는 횟수로 주기를 세고,
        # 그림에 쓸 구간 이름은 손이 움직이는 방향으로 붙인다 (앞으로 = push, 뒤로 = recovery).
        velocity_x = np.diff(wrists[:, 0], prepend=wrists[0, 0]) * facing
        phases = [PHASE_PUSH if value > 0 else PHASE_RECOVERY for value in velocity_x]
        phases[:baseline_end] = ["still"] * baseline_end
        moving_x = wrists[baseline_end:, 0]
        # 손이 앞뒤로 충분히 움직였을 때만 미는 동작으로 센다. 가만히 있는 사람의 미세한 떨림도
        # 평균선을 계속 넘나들어 "밀기"로 세어지기 때문 (실측: 가만히 앉은 사람에게서 4회로 잘못 셈).
        # 실제 추진 영상 16개의 손목 앞뒤 범위는 몸통 길이의 0.59~1.16배였다.
        swing = float(np.percentile(moving_x, 95) - np.percentile(moving_x, 5)) if len(moving_x) else 0.0
        cycles = (
            split_strokes_mean_crossing(moving_x)
            if swing >= torso_length * float(ALGORITHM_PARAMS["stroke_min_swing_ratio"])
            else []
        )
        strokes = [(baseline_end + start, baseline_end + start, baseline_end + end) for start, end in cycles]

    # 손 궤적 패턴: 림 없이, 움직임 구간에서 돌아오는 높이와 미는 높이를 비교한다. 미는 동작이 없으면 판정하지 않는다.
    if strokes:
        pattern, pattern_reason, pattern_difference = classify_pattern_by_return_height(wrists[baseline_end:], torso_length, facing)
    else:
        pattern, pattern_reason, pattern_difference = "unknown", "미는 동작을 찾지 못해 판정 보류", float("nan")

    contact_angles: list[float] = []
    elbow_angles: list[float] = []
    for push_start, push_end, _next_start in (strokes if has_rim else []):
        push_points = wrists[push_start:push_end]
        contact_angles.append(calculate_contact_angle(push_points, rim_offset))

        # 팔꿈치 각도: 밀기 구간 중 손목이 가장 높을 때(= 림 최상단에 가장 가까울 때)
        top_index = push_start + int(np.argmin(push_points[:, 1]))
        top_obs = usable[top_index]
        if has_required_visibility(top_obs.landmarks, (shoulder_name, elbow_name, wrist_name)):
            elbow_angles.append(
                joint_angle(
                    top_obs.pixels[shoulder_name],
                    top_obs.pixels[elbow_name],
                    top_obs.pixels[wrist_name],
                )
            )

    trunk_angles = [
        _trunk_forward_lean(obs.pixels[shoulder_name], obs.pixels[hip_name])
        for obs in usable
        if has_required_visibility(obs.landmarks, (shoulder_name, hip_name))
    ]
    trunk_angles = [angle for angle in trunk_angles if math.isfinite(angle)]

    push_start_times = [times[start] for start, _, _ in strokes]
    contact_angle = float(np.nanmean(contact_angles)) if contact_angles else float("nan")
    if has_rim:
        stroke_rate = calculate_stroke_rate(push_start_times, duration_seconds)
    elif strokes:
        # 주기 n개가 차지한 시간으로 나눈다: n ÷ (마지막 주기 끝 − 첫 주기 시작) × 60
        span = float(times[min(strokes[-1][2], len(times) - 1)] - times[strokes[0][0]])
        stroke_rate = len(strokes) / span * 60.0 if span > 0 else float("nan")
    else:
        stroke_rate = float("nan")
    trunk_mean = float(np.mean(trunk_angles)) if trunk_angles else float("nan")
    trunk_max = float(np.max(trunk_angles)) if trunk_angles else float("nan")
    trunk_range = float(np.max(trunk_angles) - np.min(trunk_angles)) if trunk_angles else float("nan")
    elbow_mean = float(np.nanmean(elbow_angles)) if elbow_angles else float("nan")

    result.stroke_count = len(strokes)
    result.pattern = pattern
    result.metrics = {
        "contact_angle": _metric(contact_angle, judge_minimum("contact_angle", contact_angle), "contact_angle_low"),
        "stroke_rate": _metric(stroke_rate, judge_maximum("stroke_rate", stroke_rate), "stroke_rate_high"),
        "pattern": judge_pattern(result.pattern),
        "trunk_lean_mean": _metric(trunk_mean, judge_maximum("trunk_lean", trunk_mean), "trunk_lean_high"),
        "trunk_lean_max": _metric(trunk_max, judge_maximum("trunk_lean", trunk_max), "trunk_lean_high"),
        "trunk_lean_range": MetricResult(trunk_range, JUDGMENT_PENDING, None),
        "elbow_angle": MetricResult(elbow_mean, judge_range("elbow_angle", elbow_mean), None),
    }
    # 정지 vs 동작 비교값. 판정 기준이 없으므로 판정은 보류하고, 해석은 screening.py가 맡는다.
    for name, value in comparison.items():
        result.metrics[name] = MetricResult(value, JUDGMENT_PENDING, None)
    result.metrics["head_cva_static"] = MetricResult(head_static, JUDGMENT_PENDING, None)
    result.metrics["head_cva_moving"] = MetricResult(head_moving, JUDGMENT_PENDING, None)
    result.debug["head_cva"] = [float(value) for value in head_cva]
    result.debug["head_cva_time"] = [float(value) for value in times]
    if not strokes:
        result.warnings.append(
            "밀기 동작을 찾지 못했어요. 림 3점을 바퀴 손잡이 테 위에 정확히 찍었는지, "
            "평소 속도로 5회 이상 밀었는지 확인해주세요."
            if has_rim
            else "미는 동작을 찾지 못했어요. 평소 속도로 5회 이상 밀었는지 확인해주세요."
        )

    result.trajectory = [
        (float(point[0]), float(point[1]), phase) for point, phase in zip(wrists, phases)
    ]
    if has_rim:
        result.rim = (float(rim_offset[0]), float(rim_offset[1]), float(rim_radius))
    result.debug.update(
        {
            "time": times,
            "forward_lean": [float(value) for value in forward_lean],
            "baseline_seconds": [float(times[baseline_end - 1] - times[0])] if baseline_end else [0.0],
            "visibility": [obs.visibility for obs in usable],
            "wrist_x": [float(point[0]) for point in wrists],
            "wrist_y": [float(point[1]) for point in wrists],
            "phase": phases,
            "pattern_votes": [f"{pattern}: {pattern_reason}"],
            "pattern_return_height": [pattern_difference],
            # 기존 방식(림 접촉+이동 방향)과 대안 방식(평균선 교차) 스트로크 수 비교용
            "stroke_count_compare": [len(strokes), len(split_strokes_mean_crossing(wrists[:, 0]))],
        }
    )


def _local_top_times(times: np.ndarray, heights_y: np.ndarray) -> list[float]:
    """손이 가장 높이 올라간 순간(= 밀기 시작 무렵)들의 시각을 찾습니다.

    y는 아래로 갈수록 크므로 y의 극솟값을 찾는다.
    - 양옆보다 작거나 같고
    - 전체 움직임 범위 × front_peak_prominence_ratio 이상 주변보다 높고
    - 직전 극솟값과 front_peak_min_separation_sec 이상 떨어져 있어야 한다.
    """
    if len(heights_y) < 3:
        return []
    value_range = float(np.max(heights_y) - np.min(heights_y))
    if value_range <= 0:
        return []
    prominence = value_range * float(ALGORITHM_PARAMS["front_peak_prominence_ratio"])
    separation = float(ALGORITHM_PARAMS["front_peak_min_separation_sec"])
    # 극솟값 주변 ±separation초를 살펴본다 (손이 한 번 올라갔다 내려오는 폭을 충분히 담기 위함).
    window = max(1, int(round(separation * VIDEO_SAMPLE_FPS)))
    tops: list[float] = []
    for index in range(1, len(heights_y) - 1):
        start = max(0, index - window)
        end = min(len(heights_y), index + window + 1)
        neighborhood = heights_y[start:end]
        if heights_y[index] > np.min(neighborhood):
            continue
        if float(np.max(neighborhood) - heights_y[index]) < prominence:
            continue
        if tops and times[index] - tops[-1] < separation:
            continue
        tops.append(float(times[index]))
    return tops


def find_front_baseline_end(widths: np.ndarray, times: Sequence[float]) -> int:
    """정면 영상 앞부분에서 "가만히 앉아 있는" 구간이 끝나는 프레임 번호를 찾습니다.

    카메라 쪽으로 다가오면 어깨 너비(픽셀)가 커진다.
    - 기준 너비 = 처음 0.5초 동안의 중앙값
    - 너비가 기준 × front_approach_scale_ratio를 넘은 상태가 5프레임 이어지면 그 첫 프레임을 출발로 본다.
    - 정지 구간이 baseline_min_seconds보다 짧으면 0 (기준 없음), baseline_max_seconds까지만 쓴다.
    """
    count = len(widths)
    if count < 3:
        return 0
    start_time = float(times[0])
    head = [index for index in range(count) if times[index] - start_time <= 0.5]
    reference = float(np.median(widths[head]))
    limit = reference * float(ALGORITHM_PARAMS["front_approach_scale_ratio"])
    max_seconds = float(ALGORITHM_PARAMS["baseline_max_seconds"])
    end = count
    for index in range(count):
        if times[index] - start_time > max_seconds:
            end = index
            break
        if widths[index] > limit and bool(np.all(widths[index : index + 5] > limit)):
            end = index
            break
    if end == 0 or times[end - 1] - start_time < float(ALGORITHM_PARAMS["baseline_min_seconds"]):
        return 0
    return end


def calculate_frontal_comparison(
    observations: Sequence[FrameObservation],
    result: PropulsionAnalysisResult,
) -> None:
    """정면 영상에서 어깨 기울기와 몸통 옆 기울기를 "가만히 있을 때 / 밀고 올 때"로 나눠 result에 채웁니다.

    부호는 앉은 사람 기준이다.
    - 어깨 기울기 = atan2(왼쪽 어깨 y − 오른쪽 어깨 y, 두 어깨의 가로 거리)
        + = 왼쪽 어깨가 낮음, − = 오른쪽 어깨가 낮음
    - 몸통 옆 기울기 = 골반 중점 → 어깨 중점 선이 수직에서 기운 각도
        + = 왼쪽으로 기움, − = 오른쪽으로 기움
    static = 정지 구간 중앙값, moving = 이동 구간 중앙값, sway = 이동 구간의 95−5백분위수(좌우로 흔들린 폭).
    휴대폰이 좌우로 기울어 있으면 그 각도만큼 값이 치우친다 (실측: 일자 영상 10개가 −2~−3° 치우침).
    """
    frames = [obs for obs in observations if obs.valid]
    nan = float("nan")
    names = ("shoulder_tilt_static", "shoulder_tilt_moving", "shoulder_tilt_sway", "lateral_lean_static", "lateral_lean_moving", "lateral_lean_sway")
    if len(frames) < 3:
        for name in names:
            result.metrics[name] = MetricResult(nan, JUDGMENT_PENDING, None)
        return
    times = [obs.timestamp_seconds for obs in frames]
    tilts, leans, widths, heads = [], [], [], []
    for obs in frames:
        left_shoulder, right_shoulder = obs.pixels["left_shoulder"], obs.pixels["right_shoulder"]
        # 머리 옆 기울기 = 양쪽 눈을 잇는 선의 기울기 (+ = 앉은 사람의 왼쪽 눈이 낮음 = 머리가 왼쪽으로 기움)
        left_eye, right_eye = obs.pixels.get("left_eye"), obs.pixels.get("right_eye")
        eye_gap = abs(float(left_eye[0] - right_eye[0])) if left_eye is not None and right_eye is not None else 0.0
        heads.append(math.degrees(math.atan2(float(left_eye[1] - right_eye[1]), eye_gap)) if eye_gap > 1.0 else float("nan"))
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_hip = (obs.pixels["left_hip"] + obs.pixels["right_hip"]) / 2
        # 화면에서 사람의 왼쪽이 어느 방향인지 (카메라를 보고 있으면 화면 오른쪽 = +1)
        left_direction = 1.0 if left_shoulder[0] >= right_shoulder[0] else -1.0
        tilts.append(math.degrees(math.atan2(float(left_shoulder[1] - right_shoulder[1]), abs(float(left_shoulder[0] - right_shoulder[0])))))
        leans.append(math.degrees(math.atan2(float(mid_shoulder[0] - mid_hip[0]) * left_direction, float(mid_hip[1] - mid_shoulder[1]))))
        widths.append(float(np.linalg.norm(left_shoulder - right_shoulder)))
    tilt_array, lean_array = np.array(tilts), np.array(leans)
    baseline_end = find_front_baseline_end(np.array(widths), times)
    percentile = float(ALGORITHM_PARAMS["dynamic_peak_percentile"])

    def summarize(values: np.ndarray) -> tuple[float, float, float]:
        """(정지 중앙값, 이동 중앙값, 이동 중 흔들린 폭)."""
        still, moving = values[:baseline_end], values[baseline_end:]
        static = float(np.median(still)) if len(still) else nan
        if len(moving) < 3:
            return static, nan, nan
        return static, float(np.median(moving)), float(np.percentile(moving, percentile) - np.percentile(moving, 100.0 - percentile))

    tilt_static, tilt_moving, tilt_sway = summarize(tilt_array)
    lean_static, lean_moving, lean_sway = summarize(lean_array)
    for name, value in zip(names, (tilt_static, tilt_moving, tilt_sway, lean_static, lean_moving, lean_sway)):
        result.metrics[name] = MetricResult(value, JUDGMENT_PENDING, None)
    head_array = np.array(heads)
    still_heads = head_array[:baseline_end][np.isfinite(head_array[:baseline_end])]
    moving_heads = head_array[baseline_end:][np.isfinite(head_array[baseline_end:])]
    result.metrics["head_tilt_static"] = MetricResult(float(np.median(still_heads)) if len(still_heads) >= 3 else nan, JUDGMENT_PENDING, None)
    result.metrics["head_tilt_moving"] = MetricResult(float(np.median(moving_heads)) if len(moving_heads) >= 3 else nan, JUDGMENT_PENDING, None)
    result.debug["head_tilt_series"] = [float(value) for value in head_array]
    if baseline_end == 0:
        result.notes.append(
            "영상 처음에 가만히 앉아 있는 구간을 찾지 못해 '가만히 있을 때' 값을 잡지 못했어요. "
            "다음엔 출발하기 전에 3초 동안 가만히 있어주세요."
        )
    result.debug.update(
        {
            "tilt_time": [float(value) for value in times],
            "shoulder_tilt_series": [float(value) for value in tilt_array],
            "lateral_lean_series": [float(value) for value in lean_array],
            "shoulder_width_px": widths,
            "baseline_seconds": [float(times[baseline_end - 1] - times[0])] if baseline_end else [0.0],
        }
    )


CAMERA_ROLL_METRICS: tuple[str, ...] = (
    "shoulder_tilt_static", "shoulder_tilt_moving", "lateral_lean_static", "lateral_lean_moving",
    "head_tilt_static", "head_tilt_moving",
)


def correct_for_camera_roll(result: PropulsionAnalysisResult, camera_roll_deg: float, facing: str) -> float:
    """사이트 안 촬영 화면이 잰 휴대폰 기울기만큼 정면 각도를 되돌립니다. 적용한 보정값(°)을 돌려줍니다.

    camera_roll_deg: 녹화 중 휴대폰 기울기의 중앙값. + = 화면을 보는 쪽에서 휴대폰이 시계 방향으로 돌아감.
    - 뒷면 카메라: 휴대폰이 시계 방향으로 θ 돌면 영상 속 장면은 반시계로 θ 돈다.
      사람이 카메라를 보면 사람의 왼쪽이 화면 오른쪽이므로, 왼쪽 어깨가 올라가 보여 어깨 기울기가 −θ,
      몸통 위쪽이 사람의 오른쪽으로 가 보여 몸통 옆 기울기도 −θ 치우친다 → +θ를 더해 되돌린다.
    - 앞면 카메라(facing="user"): 녹화 영상은 좌우가 뒤집히지 않은 원본이라 장면이 시계 방향으로 돈다 → −θ.
    흔들린 폭(sway)은 차이값이라 보정하지 않는다.
    ⚠ 부호는 실제 휴대폰으로 확인 필요 (휴대폰을 일부러 5° 기울여 똑바로 앉은 사람을 찍고, 보정 후 0°에 가까운지 볼 것).
    """
    correction = -float(camera_roll_deg) if facing == "user" else float(camera_roll_deg)
    for name in CAMERA_ROLL_METRICS:
        metric = result.metrics.get(name)
        if metric is not None and _is_number(metric.value):
            result.metrics[name] = MetricResult(float(metric.value) + correction, metric.judgment, metric.coaching_key)
    for series in ("shoulder_tilt_series", "lateral_lean_series", "head_tilt_series"):
        if series in result.debug:
            result.debug[series] = [float(value) + correction for value in result.debug[series]]
    result.debug["camera_roll_correction"] = [correction]
    return correction


def calculate_front_metrics(
    observations: Sequence[FrameObservation],
    result: PropulsionAnalysisResult,
) -> None:
    """정면 영상에서 좌우 대칭, 타이밍, 어깨 기울기, 측방 기울기를 계산합니다."""
    required = ("left_wrist", "right_wrist", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
    usable = [obs for obs in observations if obs.valid and has_required_visibility(obs.landmarks, required)]
    if len(usable) < 2:
        # 손이 안 보여도 어깨·몸통 기울기는 잴 수 있으므로 경고로 막지 않고 안내만 남긴다.
        result.notes.append("양손이 잘 보이지 않아 좌우 손 움직임 비교는 하지 못했어요.")
        return

    times = np.array([obs.timestamp_seconds for obs in usable])
    left_y =np.array(moving_average(np.array([obs.pixels["left_wrist"][1] - obs.anchor[1] for obs in usable])))
    right_y = np.array(moving_average(np.array([obs.pixels["right_wrist"][1] - obs.anchor[1] for obs in usable])))

    # 1) 대칭 지수 = 작은 쪽 세로 이동 범위 ÷ 큰 쪽 세로 이동 범위 (1에 가까울수록 대칭)
    left_range = float(np.max(left_y) - np.min(left_y))
    right_range = float(np.max(right_y) - np.min(right_y))
    larger = max(left_range, right_range)
    symmetry = min(left_range, right_range) / larger if larger > 0 else float("nan")

    # 2) 타이밍 차이 = 왼손 최고점 시각과 가장 가까운 오른손 최고점 시각 차이의 평균(초)
    left_tops = _local_top_times(times, left_y)
    right_tops = _local_top_times(times, right_y)
    differences = [min(abs(left - right) for right in right_tops) for left in left_tops] if right_tops else []
    timing = float(np.mean(differences)) if differences else float("nan")

    # 3) 어깨 기울기 = 양쪽 어깨를 잇는 선과 수평선 사이 각도의 평균(°)
    shoulder_tilts = [line_horizontal_deviation(obs.pixels["left_shoulder"], obs.pixels["right_shoulder"]) for obs in usable]
    # 4) 체간 측방 기울기 = 골반 중점→어깨 중점 선이 수직선에서 벗어난 각도(°)
    lateral = [
        line_vertical_deviation(
            (obs.pixels["left_hip"] + obs.pixels["right_hip"]) / 2,
            (obs.pixels["left_shoulder"] + obs.pixels["right_shoulder"]) / 2,
        )
        for obs in usable
    ]
    shoulder = float(np.nanmean(shoulder_tilts))
    lateral_mean = float(np.nanmean(lateral))
    lateral_max = float(np.nanmax(lateral))

    result.metrics = {
        "symmetry_index": _metric(symmetry, judge_minimum("symmetry_index", symmetry), "asymmetry_high"),
        "timing_diff": _metric(timing, judge_maximum("timing_diff", timing), "asymmetry_high"),
        "shoulder_tilt": _metric(shoulder, judge_maximum("shoulder_tilt", shoulder), None),
        "lateral_lean": _metric(lateral_mean, judge_maximum("lateral_lean", lateral_mean), None),
        "lateral_lean_max": _metric(lateral_max, judge_maximum("lateral_lean", lateral_max), None),
    }
    result.trajectory = [(float(t), float(y), "left") for t, y in zip(times, left_y)] + [
        (float(t), float(y), "right") for t, y in zip(times, right_y)
    ]
    result.debug.update(
        {
            "time": [float(t) for t in times],
            "visibility": [obs.visibility for obs in usable],
            "wrist_x": [float(obs.pixels["left_wrist"][0]) for obs in usable],
            "wrist_y": [float(y) for y in left_y],
            "right_wrist_y": [float(y) for y in right_y],
            "phase": ["front"] * len(usable),
            "left_tops": left_tops,
            "right_tops": right_tops,
        }
    )


# ---------------------------------------------------------------------------
# 영상 처리 진입점
# ---------------------------------------------------------------------------


def extract_first_frame(video_bytes: bytes) -> np.ndarray | None:
    """림 3점 클릭용 첫 화면을 분석과 같은 크기(BGR)로 반환합니다."""
    path = _write_temp_video(video_bytes)
    try:
        capture, rotate_code = open_video(path)
        try:
            success, frame = read_upright(capture, rotate_code)
        finally:
            capture.release()
        return resize_long_side(frame) if success else None
    finally:
        _remove_file(path)


def _write_temp_video(video_bytes: bytes) -> str:
    """업로드 바이트를 분석용 임시 파일로만 저장하고 경로를 반환합니다."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temporary_file:
        temporary_file.write(video_bytes)
        return temporary_file.name


def _remove_file(path: str) -> None:
    """임시 파일을 지웁니다 (원본 영상을 남기지 않기 위함)."""
    if path and os.path.exists(path):
        os.unlink(path)


def _required_landmarks(view: str, side: str | None) -> tuple[str, ...]:
    """프레임을 유효하다고 볼 최소 랜드마크 목록."""
    if view == "side" and side in ("left", "right"):
        return (f"{side}_shoulder", f"{side}_hip")
    return ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def _anchor(pixels: dict[str, np.ndarray], view: str, side: str | None) -> np.ndarray:
    """hip 기준점(픽셀). 측면은 촬영한 쪽 hip, 정면은 양쪽 hip 중점."""
    if view == "side" and side in ("left", "right"):
        return pixels[f"{side}_hip"]
    return (pixels["left_hip"] + pixels["right_hip"]) / 2


def _body_scale(pixels: dict[str, np.ndarray], view: str, side: str | None) -> float:
    """몸 크기의 기준 길이(픽셀). 측면은 어깨–골반 길이, 정면은 양쪽 어깨 사이 너비."""
    if view == "side" and side in ("left", "right"):
        return float(np.linalg.norm(pixels[f"{side}_shoulder"] - pixels[f"{side}_hip"]))
    return float(np.linalg.norm(pixels["left_shoulder"] - pixels["right_shoulder"]))


def keep_tracked_subject(
    observations: Sequence[FrameObservation],
    view: str,
    side: str | None,
    frame_width: float,
) -> None:
    """처음 잡힌 사람만 끝까지 따라가고, 나머지 프레임은 무효로 바꿉니다 (observations를 직접 고침).

    실제 야외 영상에서 생긴 문제를 막는다:
    1) 화면 끝에서 몸이 잘린 프레임 → 어깨·골반이 가로 끝(frame_edge_margin_ratio) 안쪽에 없으면 무효
    2) 대상이 나간 뒤 지나가는 다른 사람을 잡는 문제
       → 직전에 인정한 프레임과 비교해 기준점이 (몸 크기 × track_max_jump_ratio × 지난 프레임 수)보다
         멀리 뛰었거나, 몸 크기가 track_scale_range를 벗어나면 무효
       → track_lost_seconds 넘게 대상을 놓치면 나간 것으로 보고 그 뒤는 모두 무효
    """
    margin = frame_width * float(ALGORITHM_PARAMS["frame_edge_margin_ratio"])
    max_jump = float(ALGORITHM_PARAMS["track_max_jump_ratio"])
    scale_low, scale_high = ALGORITHM_PARAMS["track_scale_range"]  # type: ignore[misc]
    lost_seconds = float(ALGORITHM_PARAMS["track_lost_seconds"])
    required = _required_landmarks(view, side)

    last_anchor: np.ndarray | None = None
    last_scale = 0.0
    last_index = 0
    last_time = 0.0
    gone = False
    for index, obs in enumerate(observations):
        if not obs.valid or obs.anchor is None:
            continue
        accepted = not gone
        if accepted and frame_width > 0:
            accepted = all(margin <= float(obs.pixels[name][0]) <= frame_width - margin for name in required)
        scale = _body_scale(obs.pixels, view, side)
        if accepted and view == "side" and side in ("left", "right"):
            # 앉은 사람만 대상으로 삼는다: 무릎이 보이면 골반과 비슷한 높이여야 한다 (서 있는 사람 제외).
            knee_name = f"{side}_knee"
            if has_required_visibility(obs.landmarks, (knee_name,)):
                drop = abs(float(obs.pixels[knee_name][1] - obs.pixels[f"{side}_hip"][1]))
                accepted = drop < scale * float(ALGORITHM_PARAMS["seated_knee_hip_ratio"])
        if accepted and last_anchor is not None:
            if obs.timestamp_seconds - last_time > lost_seconds:
                gone = True
                accepted = False
            else:
                # 놓친 프레임이 많아도 허용 거리는 track_max_gap_frames장 분량까지만 늘린다
                # (오래 놓친 뒤에는 멀리 있는 다른 사람도 허용 거리 안에 들어오기 때문).
                gap = min(index - last_index, int(ALGORITHM_PARAMS["track_max_gap_frames"]))
                jump = float(np.linalg.norm(obs.anchor - last_anchor))
                accepted = (
                    jump <= last_scale * max_jump * gap
                    and float(scale_low) * last_scale <= scale <= float(scale_high) * last_scale
                )
        if accepted and scale > 0:
            last_anchor, last_scale, last_index, last_time = obs.anchor, scale, index, obs.timestamp_seconds
        else:
            obs.valid = False
            obs.anchor = None


# ---------------------------------------------------------------------------
# 촬영 점검 (찍은 영상에 대한 피드백)
# ---------------------------------------------------------------------------

_BODY_BOX_NAMES: tuple[str, ...] = (
    "left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee",
)


def frame_brightness(frame: np.ndarray, pixels: dict[str, np.ndarray]) -> tuple[float, float]:
    """(화면 전체 평균 밝기, 몸 부분 평균 밝기). 몸 부분 = 어깨·골반·무릎을 감싸는 상자. 0~255."""
    whole = float(frame.mean())
    points = np.array([pixels[name] for name in _BODY_BOX_NAMES if name in pixels], dtype=np.float64)
    if len(points) < 2:
        return whole, float("nan")
    height, width = frame.shape[:2]
    x0, y0 = np.clip(points.min(axis=0), 0, [width, height]).astype(int)
    x1, y1 = np.clip(points.max(axis=0), 0, [width, height]).astype(int)
    roi = frame[y0:y1, x0:x1]
    return whole, (float(roi.mean()) if roi.size else float("nan"))


def body_size_ratios(
    observations: Sequence[FrameObservation], view: str, side: str | None, short_side: float
) -> list[float]:
    """유효 프레임마다 몸 크기 ÷ 화면 짧은 변. 측면 = 어깨–골반 길이, 정면 = 양쪽 어깨 가로 거리."""
    if short_side <= 0:
        return []
    ratios: list[float] = []
    for obs in observations:
        if not obs.valid:
            continue
        if view == "side" and side in ("left", "right"):
            size = float(np.linalg.norm(obs.pixels[f"{side}_shoulder"] - obs.pixels[f"{side}_hip"]))
        else:
            size = abs(float(obs.pixels["left_shoulder"][0] - obs.pixels["right_shoulder"][0]))
        ratios.append(size / short_side)
    return ratios


def assess_camera_motion(camera: CameraCheck) -> list[tuple[str, str, str]]:
    """올린 영상의 휴대폰 기울기(배경 세로선)와 흔들림(배경 움직임) 점검. 판단할 단서가 없으면 아무것도 말하지 않는다."""
    checks: list[tuple[str, str, str]] = []
    shake = camera.shake_ratio(float(ALGORITHM_PARAMS["film_shake_shift"]))  # type: ignore[arg-type]
    shaking = shake is not None and shake >= float(ALGORITHM_PARAMS["film_shake_ratio"])  # type: ignore[arg-type]
    # 흔들리는 영상은 번진 선 때문에 기울기를 잘못 읽어서(시험: 실제 0° → 6°) 기울기 판단을 건너뛴다.
    tilt = None if shaking else camera.tilt_deg()
    if tilt is not None:
        if abs(tilt) >= float(ALGORITHM_PARAMS["film_tilt_warn_deg"]):  # type: ignore[arg-type]
            checks.append(("warn", "휴대폰 기울기", f"휴대폰이 약 {abs(tilt):.0f}° 기울어져 찍힌 것 같아요. 수평을 맞춰 세워 두고 찍어주세요. 기울면 각도가 그만큼 틀어져요."))
        else:
            checks.append(("ok", "휴대폰 기울기", "배경 세로선으로 보면 수평이 맞아요."))
    if shake is not None:
        if shaking:
            checks.append(("warn", "흔들림", f"영상의 {shake:.0%} 동안 화면이 흔들리거나 움직였어요. 휴대폰을 세워 고정하고 찍어주세요."))
        else:
            checks.append(("ok", "흔들림", "휴대폰이 잘 고정돼 있었어요."))
    return checks


def assess_camera_angle(observations: Sequence[FrameObservation], view: str) -> list[tuple[str, str, str]]:
    """찍은 방향 점검: 옆 영상인데 비스듬하거나, 앞 영상인데 옆으로 돌아 찍혔는지 (어깨 너비 ÷ 몸통 길이)."""
    ratios = []
    for obs in observations:
        if not obs.valid or not all(name in obs.pixels for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")):
            continue
        width = abs(float(obs.pixels["left_shoulder"][0] - obs.pixels["right_shoulder"][0]))
        mid_shoulder = (obs.pixels["left_shoulder"][1] + obs.pixels["right_shoulder"][1]) / 2
        mid_hip = (obs.pixels["left_hip"][1] + obs.pixels["right_hip"][1]) / 2
        torso = abs(float(mid_hip - mid_shoulder))
        if torso > 1:
            ratios.append(width / torso)
    if len(ratios) < 5:
        return []
    ratio = float(np.median(ratios))
    if view == "side":
        if ratio > float(ALGORITHM_PARAMS["film_side_max_shoulder_ratio"]):  # type: ignore[arg-type]
            return [("warn", "찍은 방향", "몸이 비스듬히 찍혔어요. 카메라를 몸 옆에서 정확히 옆모습이 보이게 두세요. 각도가 틀어지면 목·상체 각도가 다르게 나와요.")]
        return [("ok", "찍은 방향", "옆모습으로 잘 찍혔어요.")]
    if ratio < float(ALGORITHM_PARAMS["film_front_min_shoulder_ratio"]):  # type: ignore[arg-type]
        return [("warn", "찍은 방향", "몸이 옆으로 돌아 찍혔어요. 카메라를 정면에 두고 똑바로 다가오세요.")]
    return [("ok", "찍은 방향", "정면으로 잘 찍혔어요.")]


def assess_filming_basic(
    size_ratios: Sequence[float],
    brightness: Sequence[tuple[float, float]],
    valid_frame_ratio: float,
    view: str,
) -> list[tuple[str, str, str]]:
    """거리(몸 크기)·밝기·관절 검출을 점검합니다. 기준은 config의 film_* (실제 영상 기반 어림값)."""
    checks: list[tuple[str, str, str]] = []
    params = ALGORITHM_PARAMS

    if size_ratios:
        if view == "side":
            size = float(np.median(size_ratios))
            low, high = params["film_side_size_range"]  # type: ignore[misc]
            good_distance = "약 3m"
        else:
            size = float(np.median(size_ratios[:5]))  # 출발할 때(멀리 있을 때) 크기
            low, high = params["film_front_size_range"]  # type: ignore[misc]
            good_distance = "약 4m"
        if size < float(low):
            checks.append(("warn", "거리", f"사람이 너무 작게 찍혔어요. 카메라를 더 가까이 두세요 ({good_distance})."))
        elif size > float(high):
            checks.append(("warn", "거리", f"사람이 너무 크게 찍혔어요. 카메라를 더 멀리 두세요 ({good_distance})."))
        else:
            checks.append(("ok", "거리", "사람 크기가 적당해요."))
        if view == "front" and len(size_ratios) >= 5:
            grown = float(np.percentile(size_ratios, 90)) / max(size, 1e-9)
            if grown < float(params["film_front_approach_ratio"]):  # type: ignore[arg-type]
                checks.append(("warn", "다가오기", "카메라 쪽으로 다가오는 장면이 짧아요. 멀리서 출발해 카메라 앞까지 밀고 오세요."))

    body_values = [body for _, body in brightness if np.isfinite(body)]
    if body_values:
        body = float(np.median(body_values))
        whole = float(np.median([whole for whole, b in brightness if np.isfinite(b)]))
        if whole < float(params["film_whole_min_brightness"]) or body < float(params["film_body_min_brightness"]):  # type: ignore[arg-type]
            checks.append(("warn", "밝기", "몸이 어둡게 찍혔어요. 더 밝은 곳에서 찍어주세요."))
        elif whole > 0 and body / whole < float(params["film_backlight_ratio"]):  # type: ignore[arg-type]
            checks.append(("warn", "밝기", "역광이에요. 해나 창문을 등지지 말고, 빛이 사람 쪽을 비추게 찍어주세요."))
        else:
            checks.append(("ok", "밝기", "밝기가 적당해요."))

    if valid_frame_ratio < float(params["film_valid_ratio_good"]):  # type: ignore[arg-type]
        checks.append(("warn", "관절 보임", f"관절이 잘 보인 장면이 {valid_frame_ratio:.0%}예요. 화면에 한 사람만 나오고, 몸이 가려지지 않게 찍어주세요."))
    else:
        checks.append(("ok", "관절 보임", f"관절이 잘 보였어요 ({valid_frame_ratio:.0%})."))
    return checks


def assess_filming_motion(result: "PropulsionAnalysisResult") -> list[tuple[str, str, str]]:
    """분석 후에만 알 수 있는 것: 처음 가만히 있는 구간, (측면) 미는 횟수."""
    checks: list[tuple[str, str, str]] = []
    baseline = (result.debug.get("baseline_seconds") or [None])[0]
    if baseline is not None:
        if baseline < float(ALGORITHM_PARAMS["baseline_min_seconds"]):  # type: ignore[arg-type]
            checks.append(("warn", "가만히 3초", "처음에 가만히 있는 구간이 짧아요. 출발 전에 3초 동안 움직이지 마세요."))
        else:
            checks.append(("ok", "가만히 3초", f"처음에 {baseline:.1f}초 가만히 있었어요."))
    if result.view == "side":
        if result.stroke_count < int(ALGORITHM_PARAMS["film_min_strokes"]):  # type: ignore[arg-type]
            checks.append(("warn", "미는 동작", "화면 안에서 미는 동작이 적게 잡혔어요. 카메라 앞을 지나가며 평소처럼 여러 번 밀어주세요."))
        else:
            checks.append(("ok", "미는 동작", f"미는 동작 {result.stroke_count}회를 찾았어요."))
    return checks


STILL_CANDIDATE_SECONDS = 10.0  # 가만히 있는 구간 사진 후보를 모으는 시간 (정지 구간 최대 길이와 같게)


STILL_POINT_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_hip", "right_hip")


def _pick_still_frame(
    result: PropulsionAnalysisResult,
    candidates: Sequence[tuple[float, np.ndarray]],
    observations: Sequence[FrameObservation] = (),
) -> None:
    """정지 구간 가운데 시각에 가장 가까운 후보 사진을 result.still_frame(RGB)로 남깁니다. 정지 구간이 없으면 남기지 않음."""
    times = result.debug.get("head_cva_time") or result.debug.get("time") or []
    baseline = float((result.debug.get("baseline_seconds") or [0.0])[0])
    if not candidates or not times or baseline <= 0:
        return
    target = float(times[0]) + baseline / 2
    timestamp, frame = min(candidates, key=lambda item: abs(item[0] - target))
    result.still_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result.still_time = float(timestamp)
    match = next((obs for obs in observations if abs(obs.timestamp_seconds - float(timestamp)) < 1e-6), None)
    if match is not None:
        result.still_points = {
            name: (float(match.pixels[name][0]), float(match.pixels[name][1]))
            for name in STILL_POINT_NAMES
            if name in match.pixels and match.landmarks.get(name) is not None and match.landmarks[name].visibility >= 0.3
        }


def head_cva_at(result: PropulsionAnalysisResult, timestamp: float) -> float | None:
    """그 시각에 가장 가까운 프레임의 앱 자동 거북목 각도(어깨–귀). 없으면 None."""
    times = result.debug.get("head_cva_time") or []
    values = result.debug.get("head_cva") or []
    pairs = [(abs(float(t) - timestamp), float(v)) for t, v in zip(times, values) if _is_number(v)]
    if not pairs:
        return None
    gap, value = min(pairs)
    return value if gap <= 0.5 else None


def manual_cva(c7: Sequence[float], tragus: Sequence[float]) -> float:
    """논문 방식 CVA: C7을 지나는 수평선과 C7→귀 앞 돌기(tragus) 선 사이 각도(°). 작을수록 머리가 앞으로 나옴.

    사진 좌표(y는 아래로 증가)에서 atan2(C7_y − tragus_y, |tragus_x − C7_x|).
    """
    return math.degrees(math.atan2(float(c7[1]) - float(tragus[1]), abs(float(tragus[0]) - float(c7[0]))))


SIDE_DETECT_POINTS = ("shoulder", "elbow", "wrist", "hip", "ear")


def detect_filmed_side(video_path: str, frames: int = 20) -> str:
    """옆 영상에서 카메라 쪽 몸(왼쪽/오른쪽)을 자동으로 고릅니다.

    앞 몇 초의 프레임에서 왼쪽·오른쪽 관절의 깊이(z, 작을수록 카메라에 가까움)와 보이는 정도(visibility)를 비교한다.
    카메라 쪽 관절이 더 가깝고 더 잘 보인다. 판단이 안 되면 "left".
    """
    score = 0.0
    pose = create_pose()
    try:
        for count, sampled in enumerate(iter_sampled_frames(video_path, max_seconds=4.0, long_side=VIDEO_LONG_SIDE_PX)):
            if count >= frames:
                break
            landmarks = detect_landmarks(pose, sampled.frame)
            for part in SIDE_DETECT_POINTS:
                left, right = landmarks.get(f"left_{part}"), landmarks.get(f"right_{part}")
                if left is None or right is None:
                    continue
                score += (right.z - left.z) + (left.visibility - right.visibility)  # + 이면 왼쪽이 카메라 쪽
    finally:
        pose.close()  # type: ignore[attr-defined]
    return "left" if score >= 0 else "right"


def _analyze_video_path(
    video_path: str,
    *,
    view: str,
    filmed_side: str | None,
    rim_points: Sequence[Sequence[float]] | None,
    progress_callback: ProgressCallback | None,
) -> PropulsionAnalysisResult:
    """임시 영상 경로를 MediaPipe로 처리해 결과를 만듭니다."""
    if view == "side" and filmed_side not in ("left", "right"):
        filmed_side = detect_filmed_side(video_path)
    side = filmed_side if view == "side" else None
    result = PropulsionAnalysisResult(view, filmed_side, 0.0, 0.0)

    # 림 3점은 선택 사항이다. 없으면 상체 각도 비교만 하고, 손 궤적 패턴·접촉 각도는 건너뛴다.
    circle: tuple[np.ndarray, float] | None = None
    if view == "side":
        if side not in ("left", "right"):
            result.warnings.append("측면 영상은 촬영한 쪽(왼쪽/오른쪽)을 선택해야 해요.")
            return result
        if rim_points:
            if len(rim_points) != 3:
                result.warnings.append("첫 화면에서 바퀴 손잡이 테(림) 위의 점 3개를 찍어주세요.")
                return result
            circle = circle_from_three_points(*rim_points)
            if circle is None or circle[1] <= 0:
                result.warnings.append("찍은 3점으로 원을 만들 수 없어요. 림 둘레에서 서로 멀리 떨어진 3점을 찍어주세요.")
                return result

    source_fps, frame_count = video_info(video_path)
    source_duration = frame_count / source_fps if source_fps > 0 else 0.0
    if 0 < source_duration < VIDEO_MIN_SECONDS:
        result.warnings.append(f"영상이 너무 짧아요. {VIDEO_MIN_SECONDS}초 이상 촬영해주세요.")
        return result
    if source_duration > VIDEO_MAX_SECONDS:
        result.notes.append(f"영상이 {VIDEO_MAX_SECONDS}초보다 길어서 앞의 {VIDEO_MAX_SECONDS}초만 분석했어요.")
    analyzed_duration = min(source_duration, float(VIDEO_MAX_SECONDS)) if source_duration > 0 else float(VIDEO_MAX_SECONDS)
    expected_samples = max(1, int(analyzed_duration * min(VIDEO_SAMPLE_FPS, source_fps or VIDEO_SAMPLE_FPS)))

    positions = tuple(ALGORITHM_PARAMS["representative_positions"])  # type: ignore[arg-type]
    target_samples = [int(position * expected_samples) for position in positions]
    kept_frames: dict[int, tuple[np.ndarray, int]] = {}  # 샘플 번호 → (프레임, observations 인덱스)

    required = _required_landmarks(view, side)
    observations: list[FrameObservation] = []
    pose = create_pose()
    sample_number = 0
    frame_width = 0.0
    short_side = 0.0
    brightness_by_index: dict[int, tuple[float, float]] = {}  # observations 인덱스 → (전체, 몸) 밝기
    still_candidates: list[tuple[float, np.ndarray]] = []    # (시각, 프레임) — 가만히 있는 구간 사진 후보
    camera = CameraCheck()  # 올린 영상의 휴대폰 기울기·흔들림을 배경으로 어림
    try:
        long_side = VIDEO_LONG_SIDE_PX_FRONT if view == "front" else VIDEO_LONG_SIDE_PX
        for sampled in iter_sampled_frames(video_path, max_seconds=float(VIDEO_MAX_SECONDS), long_side=long_side):
            frame_width = float(sampled.frame.shape[1])
            short_side = float(min(sampled.frame.shape[:2]))
            landmarks = detect_landmarks(pose, sampled.frame)
            pixels = landmarks_to_pixels(landmarks, sampled.frame.shape)
            camera.update(sampled.frame, pixels, sampled.timestamp_seconds)
            valid = bool(landmarks) and has_required_visibility(landmarks, required)
            observations.append(
                FrameObservation(
                    timestamp_seconds=sampled.timestamp_seconds,
                    landmarks=landmarks,
                    pixels=pixels,
                    anchor=_anchor(pixels, view, side) if valid else None,
                    valid=valid,
                    visibility=mean_visibility(landmarks, required),
                )
            )
            if valid and sample_number % 5 == 0:  # 밝기는 5장에 1장만 잰다
                brightness_by_index[len(observations) - 1] = frame_brightness(sampled.frame, pixels)
            if sample_number in target_samples and valid:
                kept_frames[sample_number] = (sampled.frame.copy(), len(observations) - 1)
            # 가만히 있는 구간 사진 후보: 앞 10초 동안 0.5초마다 한 장 (나중에 정지 구간 가운데 장면을 고른다)
            if view == "side" and valid and sampled.timestamp_seconds <= STILL_CANDIDATE_SECONDS and (
                not still_candidates or sampled.timestamp_seconds - still_candidates[-1][0] >= 0.5
            ):
                still_candidates.append((sampled.timestamp_seconds, sampled.frame.copy()))
            sample_number += 1
            if progress_callback is not None:
                progress_callback(min(1.0, sample_number / expected_samples))
    finally:
        pose.close()  # type: ignore[attr-defined]

    if progress_callback is not None:
        progress_callback(1.0)

    keep_tracked_subject(observations, view, side, frame_width)
    kept_frames = {sample: kept for sample, kept in kept_frames.items() if observations[kept[1]].valid}

    total = len(observations)
    valid_indexes = [index for index, obs in enumerate(observations) if obs.valid]
    valid_count = len(valid_indexes)
    result.duration_seconds = observations[-1].timestamp_seconds if observations else 0.0
    # 유효 비율은 대상이 처음 잡힌 때부터 마지막으로 잡힌 때까지의 프레임으로만 계산한다.
    # (대상이 화면 밖으로 나간 뒤의 빈 화면 때문에 비율이 낮아져 분석이 막히던 문제를 막는다)
    span = valid_indexes[-1] - valid_indexes[0] + 1 if valid_indexes else 0
    result.valid_frame_ratio = valid_count / span if span else 0.0
    result.debug = {
        "time": [obs.timestamp_seconds for obs in observations],
        "visibility": [obs.visibility for obs in observations],
    }

    if total == 0:
        result.warnings.append("영상을 읽을 수 없어요. mp4 또는 mov 파일인지 확인해주세요.")
        return result
    result.debug["camera_tilt"] = [camera.tilt_deg()]
    result.debug["camera_shifts"] = list(camera.shifts)
    camera_checks = assess_camera_motion(camera)
    if valid_count == 0:
        result.warnings.append(
            "사람이 감지되지 않았어요. 사람이 너무 작게(멀리) 찍혔거나 어둡지 않은지 확인하고, "
            "상체와 골반이 모두 보이게 다시 찍어주세요."
        )
        result.filming_checks = camera_checks
        return result
    # 촬영 점검은 분석이 막히는 경우에도 보여준다 (무엇을 고쳐 찍을지 알려주기 위해).
    result.filming_checks = assess_filming_basic(
        body_size_ratios(observations, view, side, short_side),
        [value for index, value in brightness_by_index.items() if observations[index].valid],
        result.valid_frame_ratio,
        view,
    ) + assess_camera_angle(observations, view) + camera_checks
    if result.valid_frame_ratio < MIN_VALID_FRAME_RATIO:
        result.warnings.append(
            f"관절이 잘 보이는 프레임이 {result.valid_frame_ratio:.0%}뿐이에요 "
            f"(최소 {MIN_VALID_FRAME_RATIO:.0%}). 흔들림을 줄이고, 몸이 화면 밖으로 나가지 않게 다시 촬영해주세요."
        )
        return result

    if view == "side":
        rim_offset: np.ndarray | None = None
        rim_radius: float | None = None
        if circle is not None:
            # 림 원 중심을 "첫 화면의 hip" 기준 상대 좌표로 바꾼다.
            # 림은 첫 화면에서 찍었으므로 첫 유효 프레임의 hip을 기준으로 삼는다.
            first_anchor = next(obs.anchor for obs in observations if obs.valid and obs.anchor is not None)
            rim_offset = np.asarray(circle[0], dtype=np.float64) - first_anchor
            rim_radius = float(circle[1])
        calculate_side_metrics(observations, side or "left", rim_offset, rim_radius, result.duration_seconds, result, frame_width)
        _pick_still_frame(result, still_candidates, observations)
    else:
        calculate_front_metrics(observations, result)
        calculate_frontal_comparison(observations, result)
    result.filming_checks.extend(assess_filming_motion(result))
    tilt_warning = next((message for status, label, message in result.filming_checks if status != "ok" and label == "휴대폰 기울기"), None)
    if tilt_warning and view == "front":
        # 정면은 휴대폰 기울기만큼 어깨·몸통 기울기가 그대로 틀어진다 (시험: 바른 자세 영상을 5° 돌리니 어깨 6° 낮음으로 나옴)
        result.notes.append("휴대폰이 기울어 찍혀서 아래 어깨·몸통 기울기 값이 그만큼 틀어졌을 수 있어요. 수평을 맞춰 다시 찍는 걸 권해요.")
    # 전체 프레임 기준 visibility (분석에서 빠진 프레임 포함)
    result.debug["frame_time"] = [obs.timestamp_seconds for obs in observations]
    result.debug["frame_visibility"] = [obs.visibility for obs in observations]

    # 대표 프레임 3장에 스켈레톤·궤적·림을 합성한다 (RGB로 반환).
    for sample in sorted(kept_frames):
        frame, obs_index = kept_frames[sample]
        obs = observations[obs_index]
        trajectory_px: list[tuple[float, float, str]] = []
        rim_px: tuple[float, float, float] | None = None
        if view == "side" and obs.anchor is not None:
            trajectory_px = [(x + obs.anchor[0], y + obs.anchor[1], phase) for x, y, phase in result.trajectory]
            if result.rim is not None:
                rim_px = (result.rim[0] + obs.anchor[0], result.rim[1] + obs.anchor[1], result.rim[2])
        composed = compose_overlay(frame, obs.landmarks, trajectory_px, rim_px)
        result.representative_frames.append(cv2.cvtColor(composed, cv2.COLOR_BGR2RGB))

    return result


def analyze_video_bytes(
    video_bytes: bytes,
    *,
    view: str,
    filmed_side: str | None = None,
    rim_points: Sequence[Sequence[float]] | None = None,
    rim_diameter_inch: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PropulsionAnalysisResult:
    """업로드 영상 바이트를 분석합니다. 원본은 임시 파일로만 쓰고 바로 지웁니다.

    rim_points: extract_first_frame()이 준 프레임의 픽셀 좌표 3개 (측면만 필요).
    rim_diameter_inch: 현재 지표가 모두 각도·비율이라 계산에는 쓰지 않으며,
                       cm 환산 기능을 추가할 때를 위해 인터페이스만 유지한다.
    """
    _ = rim_diameter_inch
    path = _write_temp_video(video_bytes)
    try:
        return _analyze_video_path(
            path,
            view=view,
            filmed_side=filmed_side,
            rim_points=rim_points,
            progress_callback=progress_callback,
        )
    finally:
        _remove_file(path)


def to_firestore(result: PropulsionAnalysisResult) -> tuple[dict[str, Any], dict[str, str]]:
    """Firestore 저장용 (metrics, judgments) 사전을 만듭니다.

    숫자는 소수 둘째 자리로 반올림하고, 계산할 수 없던 값(NaN)은 None으로 저장한다.
    """
    metrics: dict[str, Any] = {}
    judgments: dict[str, str] = {}
    for name, metric in result.metrics.items():
        value = metric.value
        if isinstance(value, float):
            value = round(value, 2) if math.isfinite(value) else None
        metrics[name] = value
        judgments[name] = metric.judgment
    if result.view == "side":
        metrics["stroke_count"] = result.stroke_count
    return metrics, judgments
