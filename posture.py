"""정지 자세 사진의 랜드마크 기반 측정과 판정을 담당합니다.

핵심 아이디어: 휠체어를 "자"로 쓴다.
- 정면 사진: 사용자가 좌석(또는 양쪽 바퀴 축) 위 2점을 찍으면 그 선을 수평 기준으로 삼는다.
- 측면 사진: 사용자가 등받이 프레임 위 2점을 찍으면 그 선을 수직 기준으로 삼는다.
이렇게 하면 폰이 조금 기울어진 채 찍혀도, 휠체어에 대한 몸의 기울기를 잴 수 있다.
기준선을 찍지 않으면 사진의 가로·세로 축을 대신 쓰고 그 사실을 안내한다.

좌표계: 분석용으로 긴 변 720px로 줄인 사진의 픽셀 좌표 (y는 아래로 증가).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

import propulsion
from config import JUDGMENT_PENDING
from overlay import draw_pose_skeleton
from pose_utils import (
    create_pose,
    detect_landmarks,
    has_required_visibility,
    landmarks_to_pixels,
    resize_long_side,
)

COLOR_REFERENCE: tuple[int, int, int] = (221, 138, 55)   # 기준선 (BGR, 블루)
COLOR_BODY_LINE: tuple[int, int, int] = (48, 90, 216)    # 측정선 (BGR, 코랄)


@dataclass
class PostureMetric:
    """자세 측정값과 판정입니다."""

    value: float | None
    judgment: str = JUDGMENT_PENDING


@dataclass
class PostureAnalysisResult:
    """자세 사진 분석 결과입니다."""

    view: str
    metrics: dict[str, PostureMetric] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    annotated_frame: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


def prepare_image(image_bytes: bytes) -> np.ndarray | None:
    """업로드 사진을 분석과 같은 크기(긴 변 720px)의 BGR 프레임으로 변환합니다.

    기준선 클릭 화면도 이 프레임으로 보여줘야 클릭 좌표와 분석 좌표가 일치한다.
    """
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return None if frame is None else resize_long_side(frame)


def acute_line_angle(first_direction: np.ndarray, second_direction: np.ndarray) -> float:
    """방향이 없는 두 직선 사이의 예각(0~90°)을 계산합니다.

    공식: θ = arccos( |a·b| / (|a||b|) )
    선에는 앞뒤 방향이 없으므로 내적의 절댓값을 써서 항상 0~90°가 나온다.
    """
    first = np.asarray(first_direction, dtype=np.float64)[:2]
    second = np.asarray(second_direction, dtype=np.float64)[:2]
    norms = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norms == 0.0:
        return float("nan")
    cosine = abs(float(np.dot(first, second))) / norms
    return math.degrees(math.acos(min(1.0, cosine)))


def _reference_direction(points: Sequence[Sequence[float]] | None, fallback: np.ndarray) -> tuple[np.ndarray, bool]:
    """클릭한 2점으로 기준선 방향을 만듭니다. 없거나 같은 점이면 fallback을 씁니다.

    반환: (방향 벡터, 사용자 기준선을 썼는지 여부)
    """
    if points and len(points) >= 2:
        direction = np.asarray(points[-1], dtype=np.float64)[:2] - np.asarray(points[-2], dtype=np.float64)[:2]
        if float(np.linalg.norm(direction)) > 5.0:
            return direction, True
    return fallback, False


def _perpendicular(direction: np.ndarray) -> np.ndarray:
    """2D 벡터를 90° 돌린 벡터 (수평 기준선 → 수직 기준선)."""
    return np.array([-direction[1], direction[0]], dtype=np.float64)


def _draw_line(frame: np.ndarray, first: np.ndarray, last: np.ndarray, color: tuple[int, int, int], thickness: int = 3) -> None:
    """두 점을 잇는 선을 프레임에 그립니다 (제자리 수정)."""
    cv2.line(frame, (int(first[0]), int(first[1])), (int(last[0]), int(last[1])), color, thickness, cv2.LINE_AA)


def _draw_reference(frame: np.ndarray, points: Sequence[Sequence[float]] | None) -> None:
    """사용자가 찍은 기준선을 사진 끝까지 연장해 그립니다 (제자리 수정)."""
    if not points or len(points) < 2:
        return
    first = np.asarray(points[-2], dtype=np.float64)[:2]
    last = np.asarray(points[-1], dtype=np.float64)[:2]
    direction = last - first
    length = float(np.linalg.norm(direction))
    if length <= 0:
        return
    unit = direction / length
    span = float(max(frame.shape[:2]))
    _draw_line(frame, first - unit * span, first + unit * span, COLOR_REFERENCE, 2)
    for point in (first, last):
        cv2.circle(frame, (int(point[0]), int(point[1])), 7, COLOR_REFERENCE, -1, cv2.LINE_AA)


def analyze_photo_bytes(
    image_bytes: bytes,
    *,
    view: str,
    seat_points: Sequence[Sequence[float]] | None = None,
    backrest_points: Sequence[Sequence[float]] | None = None,
    filmed_side: str = "left",
) -> PostureAnalysisResult:
    """정면 또는 측면 사진을 분석합니다. 원본 사진은 메모리에서만 쓰고 저장하지 않습니다.

    seat_points / backrest_points: prepare_image() 프레임의 픽셀 좌표 2개.
    filmed_side: 측면 사진에서 카메라 쪽에 있는 몸의 방향 ("left"/"right").
    """
    result = PostureAnalysisResult(view=view)
    frame = prepare_image(image_bytes)
    if frame is None:
        result.warnings.append("사진을 읽지 못했어요. JPG 또는 PNG 사진을 선택해주세요.")
        return result

    pose = create_pose(static_image_mode=True)
    try:
        landmarks = detect_landmarks(pose, frame)
    finally:
        pose.close()  # type: ignore[attr-defined]

    side = filmed_side if filmed_side in ("left", "right") else "left"
    required = (
        ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if view == "front"
        else (f"{side}_ear", f"{side}_shoulder", f"{side}_hip")
    )
    if not landmarks:
        result.warnings.append("사람이 감지되지 않았어요. 밝은 곳에서 상체와 골반이 모두 보이게 찍어주세요.")
        return result
    if not has_required_visibility(landmarks, required):
        hint = "양쪽 어깨와 골반" if view == "front" else "카메라 쪽 귀·어깨·골반"
        result.warnings.append(f"{hint}이 잘 보이지 않아요. 팔걸이나 옷에 가리지 않게 다시 찍어주세요.")
        return result

    pixels = landmarks_to_pixels(landmarks, frame.shape)
    annotated = draw_pose_skeleton(frame, landmarks)
    result.notes.append("추진 습관의 결과로 나타난 현재 자세 확인용 측정이에요.")

    if view == "front":
        # 수평 기준 = 좌석선. 수직 기준 = 좌석선에 수직인 선.
        horizontal, used_seat = _reference_direction(seat_points, np.array([1.0, 0.0]))
        vertical = _perpendicular(horizontal)
        if not used_seat:
            result.notes.append("좌석 기준선을 찍지 않아 사진의 가로축을 기준으로 계산했어요. 폰이 기울면 오차가 생겨요.")
        shoulder_mid = (pixels["left_shoulder"] + pixels["right_shoulder"]) / 2
        hip_mid = (pixels["left_hip"] + pixels["right_hip"]) / 2

        # 어깨 비대칭: 양쪽 어깨선과 좌석선 사이 각도
        shoulder_tilt = acute_line_angle(pixels["right_shoulder"] - pixels["left_shoulder"], horizontal)
        # 골반 경사(pelvic obliquity): 양쪽 hip 선과 좌석선 사이 각도
        pelvic_tilt = acute_line_angle(pixels["right_hip"] - pixels["left_hip"], horizontal)
        # 체간 측방 기울기: 골반 중점→어깨 중점 선이 좌석선의 수직선에서 벗어난 각도
        lateral_lean = acute_line_angle(shoulder_mid - hip_mid, vertical)

        result.metrics = {
            "shoulder_tilt": PostureMetric(shoulder_tilt, propulsion.judge_maximum("shoulder_tilt", shoulder_tilt)),
            "pelvic_tilt": PostureMetric(pelvic_tilt, propulsion.judge_maximum("pelvic_tilt", pelvic_tilt)),
            "lateral_lean": PostureMetric(lateral_lean, propulsion.judge_maximum("lateral_lean", lateral_lean)),
        }
        _draw_reference(annotated, seat_points if used_seat else None)
        _draw_line(annotated, pixels["left_shoulder"], pixels["right_shoulder"], COLOR_BODY_LINE)
        _draw_line(annotated, pixels["left_hip"], pixels["right_hip"], COLOR_BODY_LINE)
        _draw_line(annotated, hip_mid, shoulder_mid, COLOR_BODY_LINE)
    else:
        ear = pixels[f"{side}_ear"]
        shoulder = pixels[f"{side}_shoulder"]
        hip = pixels[f"{side}_hip"]
        backrest, used_backrest = _reference_direction(backrest_points, np.array([0.0, 1.0]))
        if not used_backrest:
            result.notes.append("등받이 기준선을 찍지 않아 사진의 세로축을 기준으로 계산했어요.")

        # 체간 굴곡: hip→shoulder 선과 등받이 선 사이 각도 (등받이에서 몸이 얼마나 떨어져 숙였는지)
        trunk_flexion = acute_line_angle(shoulder - hip, backrest)
        # modified CVA: ear–shoulder 선과 수평선 사이 각도.
        # 원래 CVA는 C7(목 아래 뼈)을 쓰지만 MediaPipe에 없어서 shoulder로 대신한다.
        # 작을수록 머리가 앞으로 나온 자세(거북목). 수평은 중력 기준이라 사진 가로축을 쓴다.
        modified_cva = acute_line_angle(ear - shoulder, np.array([1.0, 0.0]))
        result.notes.append("거북목 각도(modified CVA)는 C7 대신 어깨 점을 쓴 근사치이고, 폰을 수평으로 들고 찍어야 정확해요.")

        result.metrics = {
            "modified_cva": PostureMetric(modified_cva, propulsion.judge_minimum("modified_cva", modified_cva)),
            "trunk_flexion": PostureMetric(trunk_flexion, propulsion.judge_maximum("trunk_lean", trunk_flexion)),
        }
        _draw_reference(annotated, backrest_points if used_backrest else None)
        _draw_line(annotated, hip, shoulder, COLOR_BODY_LINE)
        _draw_line(annotated, shoulder, ear, COLOR_BODY_LINE)

    result.annotated_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return result
