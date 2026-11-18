"""분석 대표 프레임에 포즈 스켈레톤, 손목 궤적, 림 원을 합성합니다.

입력 프레임은 OpenCV 기본 형식인 BGR이며, 색상 상수도 (B, G, R) 순서다.
화면(Streamlit)에 보여줄 때는 호출하는 쪽에서 RGB로 바꾼다.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from config import MIN_LANDMARK_VISIBILITY
from pose_utils import LANDMARK_NAMES, POSE_CONNECTIONS, LandmarkPoint, normalized_to_pixel

# (B, G, R)
COLOR_PUSH: tuple[int, int, int] = (117, 158, 29)       # 틸 그린 #1D9E75 - 밀기
COLOR_RECOVERY: tuple[int, int, int] = (48, 90, 216)    # 코랄 #D85A30 - 복귀
COLOR_SKELETON: tuple[int, int, int] = (235, 235, 235)
COLOR_JOINT: tuple[int, int, int] = (60, 60, 60)
COLOR_RIM: tuple[int, int, int] = (221, 138, 55)        # 블루 #378ADD - 림 원

TrajectoryPoint = tuple[float, float, str]


def draw_pose_skeleton(
    frame: np.ndarray,
    landmarks: dict[str, LandmarkPoint],
    *,
    minimum_visibility: float = MIN_LANDMARK_VISIBILITY,
) -> np.ndarray:
    """정규화된 MediaPipe 포즈 랜드마크와 연결선을 프레임에 그립니다.

    얼굴 점(0~10번)은 화면을 복잡하게 만들어서 연결선만 그리고 점은 생략한다.
    """
    output = frame.copy()
    for first_index, last_index in POSE_CONNECTIONS:
        first_name = LANDMARK_NAMES[first_index]
        last_name = LANDMARK_NAMES[last_index]
        if first_name not in landmarks or last_name not in landmarks:
            continue
        first = landmarks[first_name]
        last = landmarks[last_name]
        if min(first.visibility, last.visibility) < minimum_visibility:
            continue
        cv2.line(
            output,
            normalized_to_pixel(first.as_array(), output),
            normalized_to_pixel(last.as_array(), output),
            COLOR_SKELETON,
            3,
            cv2.LINE_AA,
        )
    for index, name in enumerate(LANDMARK_NAMES):
        point = landmarks.get(name)
        if point is None or index <= 10 or point.visibility < minimum_visibility:
            continue
        center = normalized_to_pixel(point.as_array(), output)
        cv2.circle(output, center, 5, COLOR_SKELETON, -1, cv2.LINE_AA)
        cv2.circle(output, center, 5, COLOR_JOINT, 1, cv2.LINE_AA)
    return output


def draw_wrist_trajectory(
    frame: np.ndarray,
    trajectory: Sequence[TrajectoryPoint],
) -> np.ndarray:
    """손목 궤적(픽셀 좌표)을 밀기·복귀 구간별 색으로 그립니다."""
    output = frame.copy()
    points = list(trajectory)
    for index in range(1, len(points)):
        x0, y0, _ = points[index - 1]
        x1, y1, phase = points[index]
        color = COLOR_PUSH if phase == "push" else COLOR_RECOVERY
        cv2.line(output, (int(x0), int(y0)), (int(x1), int(y1)), color, 3, cv2.LINE_AA)
    return output


def draw_rim_circle(
    frame: np.ndarray,
    center: Sequence[float],
    radius: float,
) -> np.ndarray:
    """사용자가 지정한 림 원(픽셀 좌표)을 점선 느낌의 원으로 표시합니다."""
    output = frame.copy()
    center_pixel = (int(center[0]), int(center[1]))
    radius_pixel = max(1, int(radius))
    for start_angle in range(0, 360, 20):
        cv2.ellipse(output, center_pixel, (radius_pixel, radius_pixel), 0, start_angle, start_angle + 12, COLOR_RIM, 2, cv2.LINE_AA)
    cv2.drawMarker(output, center_pixel, COLOR_RIM, cv2.MARKER_CROSS, 14, 2)
    return output


COLOR_EXAMPLE_POINT: tuple[int, int, int] = (40, 90, 230)   # 예시 그림의 클릭 위치 (BGR, 주황빨강)
COLOR_EXAMPLE_CHAIR: tuple[int, int, int] = (70, 70, 70)
COLOR_EXAMPLE_PERSON: tuple[int, int, int] = (200, 200, 200)


def _dashed_line(image: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int], dash: int = 8) -> None:
    """start→end 점선을 그립니다 (OpenCV에 점선 함수가 없어 짧은 선분을 번갈아 그림)."""
    length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    count = max(1, int(length // dash))
    for index in range(0, count, 2):
        a = index / count
        b = min(1.0, (index + 1) / count)
        p = (int(start[0] + (end[0] - start[0]) * a), int(start[1] + (end[1] - start[1]) * a))
        q = (int(start[0] + (end[0] - start[0]) * b), int(start[1] + (end[1] - start[1]) * b))
        cv2.line(image, p, q, color, 2, cv2.LINE_AA)


def _example_point(image: np.ndarray, center: tuple[int, int], number: int, text_offset: tuple[int, int]) -> None:
    """예시 그림에 번호 붙은 클릭 위치를 그립니다."""
    cv2.circle(image, center, 8, COLOR_EXAMPLE_POINT, -1, cv2.LINE_AA)
    cv2.circle(image, center, 8, (255, 255, 255), 2, cv2.LINE_AA)
    position = (center[0] + text_offset[0], center[1] + text_offset[1])
    cv2.putText(image, str(number), position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_EXAMPLE_POINT, 2, cv2.LINE_AA)


def draw_reference_example(view: str) -> np.ndarray:
    """자세 사진 기준선 클릭 위치를 보여주는 예시 그림 (RGB)을 그립니다.

    - side: 옆에서 본 휠체어(사람은 화면 왼쪽을 봄). 등받이 프레임 위(1)·아래(2) → 수직 기준선
    - front: 앞에서 본 휠체어. 좌석 양쪽 끝(1, 2) → 수평 기준선
    사람은 연한 회색, 휠체어는 진한 회색, 찍을 점은 주황빨강으로 구분한다.
    """
    image = np.full((300, 340, 3), 255, dtype=np.uint8)
    chair, person = COLOR_EXAMPLE_CHAIR, COLOR_EXAMPLE_PERSON
    if view == "side":
        # 사람 (앉은 자세, 왼쪽을 봄)
        cv2.circle(image, (200, 42), 16, person, -1, cv2.LINE_AA)
        cv2.line(image, (205, 132), (200, 62), person, 12, cv2.LINE_AA)   # 몸통
        cv2.line(image, (205, 132), (130, 128), person, 11, cv2.LINE_AA)  # 허벅지
        cv2.line(image, (130, 128), (122, 205), person, 10, cv2.LINE_AA)  # 정강이
        cv2.line(image, (200, 72), (185, 120), person, 8, cv2.LINE_AA)    # 팔
        # 휠체어
        cv2.circle(image, (200, 205), 80, chair, 4, cv2.LINE_AA)          # 큰 바퀴
        cv2.circle(image, (200, 205), 70, chair, 1, cv2.LINE_AA)          # 림
        cv2.circle(image, (200, 205), 5, chair, -1, cv2.LINE_AA)
        cv2.line(image, (105, 145), (238, 145), chair, 6, cv2.LINE_AA)    # 좌석
        cv2.line(image, (238, 145), (250, 40), chair, 6, cv2.LINE_AA)     # 등받이 프레임
        cv2.line(image, (250, 40), (285, 42), chair, 6, cv2.LINE_AA)      # 손잡이
        cv2.line(image, (110, 145), (100, 255), chair, 4, cv2.LINE_AA)    # 앞 프레임
        cv2.line(image, (100, 215), (125, 215), chair, 4, cv2.LINE_AA)    # 발판
        cv2.circle(image, (100, 268), 15, chair, 3, cv2.LINE_AA)          # 앞바퀴
        # 기준선과 클릭 위치
        _dashed_line(image, (252, 25), (236, 160), COLOR_EXAMPLE_POINT)
        _example_point(image, (249, 52), 1, (-28, 8))
        _example_point(image, (239, 135), 2, (14, 6))
    else:
        # 사람 (정면)
        cv2.circle(image, (170, 45), 17, person, -1, cv2.LINE_AA)
        cv2.line(image, (170, 65), (170, 150), person, 30, cv2.LINE_AA)   # 몸통
        cv2.line(image, (150, 160), (140, 250), person, 12, cv2.LINE_AA)  # 다리
        cv2.line(image, (190, 160), (200, 250), person, 12, cv2.LINE_AA)
        # 휠체어
        cv2.line(image, (100, 70), (240, 70), chair, 5, cv2.LINE_AA)      # 등받이 윗단
        cv2.line(image, (100, 70), (100, 170), chair, 5, cv2.LINE_AA)     # 등받이 기둥
        cv2.line(image, (240, 70), (240, 170), chair, 5, cv2.LINE_AA)
        cv2.line(image, (95, 170), (245, 170), chair, 8, cv2.LINE_AA)     # 좌석
        for x in (62, 278):                                               # 양쪽 큰 바퀴 (정면에서는 세로 타원)
            cv2.ellipse(image, (x, 195), (16, 85), 0, 0, 360, chair, 4, cv2.LINE_AA)
        # 기준선과 클릭 위치
        _dashed_line(image, (40, 170), (300, 170), COLOR_EXAMPLE_POINT)
        _example_point(image, (100, 170), 1, (-8, -16))
        _example_point(image, (240, 170), 2, (-8, -16))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_camera_guide_example() -> np.ndarray:
    """측면 추진 영상 촬영 위치를 보여주는 예시 그림 (RGB)을 그립니다.

    왼쪽: 위에서 내려다본 모습 (휠체어 옆 약 1.3m에 카메라, 바퀴와 평행)
    오른쪽: 카메라 화면에서 확인하는 법 (바퀴가 원 = O, 타원 = X)
    """
    image = np.full((230, 340, 3), 255, dtype=np.uint8)
    chair, person = COLOR_EXAMPLE_CHAIR, COLOR_EXAMPLE_PERSON
    good, bad = (117, 158, 29), (48, 60, 216)  # BGR: 초록, 빨강
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- 왼쪽: 위에서 본 모습 (사람은 위쪽을 향해 간다) ---
    cv2.rectangle(image, (42, 70), (98, 150), chair, 3, cv2.LINE_AA)        # 좌석
    cv2.rectangle(image, (28, 62), (38, 158), chair, -1, cv2.LINE_AA)       # 왼쪽 바퀴
    cv2.rectangle(image, (102, 62), (112, 158), chair, -1, cv2.LINE_AA)     # 오른쪽 바퀴
    cv2.circle(image, (70, 108), 16, person, -1, cv2.LINE_AA)               # 머리
    cv2.arrowedLine(image, (70, 56), (70, 26), person, 3, cv2.LINE_AA, tipLength=0.4)  # 가는 방향
    # 카메라 (바퀴 옆, 렌즈가 바퀴를 향함)
    cv2.rectangle(image, (182, 94), (200, 122), COLOR_EXAMPLE_POINT, -1, cv2.LINE_AA)
    cv2.rectangle(image, (174, 102), (182, 114), COLOR_EXAMPLE_POINT, -1, cv2.LINE_AA)
    # 거리 표시
    cv2.arrowedLine(image, (118, 108), (170, 108), COLOR_EXAMPLE_POINT, 2, cv2.LINE_AA, tipLength=0.18)
    cv2.arrowedLine(image, (170, 108), (118, 108), COLOR_EXAMPLE_POINT, 2, cv2.LINE_AA, tipLength=0.18)
    cv2.putText(image, "3 m", (128, 96), font, 0.5, COLOR_EXAMPLE_POINT, 2, cv2.LINE_AA)
    # 직각 표시 (카메라가 바퀴를 정면으로 봄)
    cv2.line(image, (118, 120), (130, 120), chair, 1, cv2.LINE_AA)
    cv2.line(image, (130, 120), (130, 108), chair, 1, cv2.LINE_AA)

    cv2.line(image, (222, 10), (222, 220), (220, 220, 220), 1, cv2.LINE_AA)  # 구분선

    # --- 오른쪽: 카메라 화면에서 확인 ---
    cv2.circle(image, (275, 62), 36, chair, 3, cv2.LINE_AA)                  # 원 = 평행하게 잘 찍힘
    cv2.circle(image, (275, 62), 4, chair, -1, cv2.LINE_AA)
    cv2.putText(image, "O", (316, 72), font, 0.8, good, 2, cv2.LINE_AA)
    cv2.ellipse(image, (275, 168), (19, 36), 0, 0, 360, chair, 3, cv2.LINE_AA)  # 타원 = 비스듬함
    cv2.circle(image, (275, 168), 4, chair, -1, cv2.LINE_AA)
    cv2.putText(image, "X", (316, 178), font, 0.8, bad, 2, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_rim_click_points(frame: np.ndarray, points: Sequence[Sequence[float]]) -> np.ndarray:
    """림 3점 클릭 단계에서 사용자가 찍은 점을 번호와 함께 표시합니다."""
    output = frame.copy()
    for number, point in enumerate(points, start=1):
        center = (int(point[0]), int(point[1]))
        cv2.circle(output, center, 7, COLOR_RIM, -1, cv2.LINE_AA)
        cv2.putText(output, str(number), (center[0] + 9, center[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RIM, 2, cv2.LINE_AA)
    return output


def compose_overlay(
    frame: np.ndarray,
    landmarks: dict[str, LandmarkPoint],
    trajectory: Sequence[TrajectoryPoint] = (),
    rim: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """림 원 → 손목 궤적 → 스켈레톤 순서로 한 프레임에 합성합니다 (BGR 반환)."""
    output = frame.copy()
    if rim is not None:
        output = draw_rim_circle(output, rim[:2], rim[2])
    if len(trajectory) > 1:
        output = draw_wrist_trajectory(output, trajectory)
    return draw_pose_skeleton(output, landmarks)


def representative_overlays(
    frames: Iterable[np.ndarray],
    landmark_sets: Iterable[dict[str, LandmarkPoint]],
) -> list[np.ndarray]:
    """여러 프레임에 스켈레톤만 합성합니다 (정지 사진 분석 등에서 사용)."""
    return [draw_pose_skeleton(frame, landmarks) for frame, landmarks in zip(frames, landmark_sets)]


# ---------------------------------------------------------------------------
# 자세 사진 (기록 비교용): 가만히 있는 장면에 거북목 각도 선을 그리고 얼굴을 흐린다.
POSTURE_PHOTO_WIDTH = 360


def _blur_face(image: np.ndarray, points: dict[str, tuple[float, float]]) -> None:
    """코·눈 주변을 둥글게 강하게 흐립니다 (귀는 각도 기준점이라 남긴다)."""
    face = [points[name] for name in ("nose", "left_eye", "right_eye") if name in points]
    if not face:
        return
    center = np.mean(np.asarray(face), axis=0)
    ear = next((points[name] for name in ("left_ear", "right_ear") if name in points), None)
    radius = 0.75 * float(np.hypot(*(np.asarray(ear) - center))) if ear is not None else image.shape[1] * 0.06
    radius = max(radius, image.shape[1] * 0.04)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, -1)
    kernel = int(radius) // 2 * 2 + 31
    blurred = cv2.GaussianBlur(image, (kernel, kernel), 0)
    image[mask > 0] = blurred[mask > 0]


def posture_photo(
    frame_rgb: np.ndarray,
    points: dict[str, tuple[float, float]],
    side: str,
    reference_deg: float | None = None,
) -> np.ndarray | None:
    """옆모습 자세 사진: 어깨→귀 선(보라), 어깨 수평선(흰 점선), 기준 각도선(초록 점선), 얼굴 흐림, 상체만 잘라 냄.

    side = 찍힌 쪽 몸("left"/"right"). 귀·어깨·골반이 없으면 None.
    """
    ear, shoulder, hip = (points.get(f"{side}_{name}") for name in ("ear", "shoulder", "hip"))
    if ear is None or shoulder is None or hip is None:
        return None
    image = frame_rgb.copy()
    _blur_face(image, points)
    ex, ey = ear
    sx, sy = shoulder
    facing = 1.0 if ex >= sx else -1.0
    torso = max(abs(hip[1] - sy), 1.0)
    width = image.shape[1]
    thick = max(2, int(width / 220))
    length = torso * 0.9

    # 어깨 수평선 (얼굴 쪽으로)
    _dashed_line(image, (int(sx), int(sy)), (int(sx + facing * length), int(sy)), (255, 255, 255), dash=max(6, thick * 4))
    # 기준 각도선
    if reference_deg is not None:
        rad = np.radians(reference_deg)
        end = (int(sx + facing * length * 0.85 * np.cos(rad)), int(sy - length * 0.85 * np.sin(rad)))
        _dashed_line(image, (int(sx), int(sy)), end, (63, 174, 122), dash=max(6, thick * 4))
    # 어깨 → 귀
    cv2.line(image, (int(sx), int(sy)), (int(ex), int(ey)), (139, 92, 246), thick + 2, cv2.LINE_AA)
    for x, y in (ear, shoulder):
        cv2.circle(image, (int(x), int(y)), thick * 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (int(x), int(y)), thick * 3, (139, 92, 246), thick, cv2.LINE_AA)

    # 머리~가슴만 잘라 냄: 귀 위쪽 ~ 어깨 아래 가슴, 어깨 중심 좌우
    y0 = int(max(0, ey - torso * 0.5))
    y1 = int(min(image.shape[0], sy + torso * 0.55))
    half = torso * 0.6
    x0 = int(max(0, sx - half))
    x1 = int(min(width, sx + half))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    scale = POSTURE_PHOTO_WIDTH / crop.shape[1]
    return cv2.resize(crop, (POSTURE_PHOTO_WIDTH, max(1, int(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def encode_jpeg_b64(image_rgb: np.ndarray, quality: int = 72) -> str:
    """RGB 이미지를 JPEG base64 문자열로 (Firestore 문서에 넣을 수 있게 작게)."""
    import base64

    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer.tobytes()).decode("ascii") if ok else ""
