"""MediaPipe Pose 공통 처리, visibility 필터, 좌표 변환과 기하 계산.

좌표계 규칙 (중요)
- MediaPipe는 x를 영상 너비로, y를 영상 높이로 나눈 0~1 정규화 좌표를 준다.
  가로·세로 비율이 달라서 이 좌표로 각도나 원을 계산하면 모양이 찌그러진다.
- 그래서 모든 기하 계산은 "분석용으로 축소한 프레임의 픽셀 좌표"에서 한다.
- 이미지 좌표는 y축이 아래로 갈수록 커진다. (위쪽 = y가 작음)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import cv2
import mediapipe as mp
import numpy as np

from config import (
    MIN_LANDMARK_VISIBILITY,
    VIDEO_LONG_SIDE_PX,
    VIDEO_SAMPLE_FPS,
    WRIST_SMOOTHING_WINDOW,
)

POSE_LANDMARK = mp.solutions.pose.PoseLandmark
# MediaPipe의 POSE_CONNECTIONS는 (정수 인덱스, 정수 인덱스) 튜플 집합이다.
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = tuple(
    (int(first), int(last)) for first, last in mp.solutions.pose.POSE_CONNECTIONS
)
LANDMARK_NAMES: tuple[str, ...] = tuple(landmark.name.lower() for landmark in POSE_LANDMARK)


@dataclass(frozen=True)
class LandmarkPoint:
    """정규화 좌표계의 포즈 랜드마크 한 점입니다."""

    x: float
    y: float
    z: float
    visibility: float

    def as_array(self) -> np.ndarray:
        """좌표 세 요소를 NumPy 배열로 반환합니다."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)


@dataclass(frozen=True)
class SampledFrame:
    """분석에 사용하도록 샘플링·축소된 영상 프레임입니다."""

    index: int
    timestamp_seconds: float
    frame: np.ndarray


def create_pose(static_image_mode: bool = False) -> object:
    """MediaPipe Pose 추정기를 만듭니다. 사용 후 반드시 close()를 호출합니다."""
    return mp.solutions.pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def pose_landmarks_to_dict(results: object) -> dict[str, LandmarkPoint]:
    """MediaPipe 결과를 이름 기반 랜드마크 사전으로 변환합니다."""
    output: dict[str, LandmarkPoint] = {}
    landmark_list = getattr(getattr(results, "pose_landmarks", None), "landmark", None)
    if not landmark_list:
        return output
    for landmark_name in POSE_LANDMARK:
        landmark = landmark_list[landmark_name.value]
        output[landmark_name.name.lower()] = LandmarkPoint(
            x=float(landmark.x),
            y=float(landmark.y),
            z=float(landmark.z),
            visibility=float(landmark.visibility),
        )
    return output


def detect_landmarks(pose: object, frame_bgr: np.ndarray) -> dict[str, LandmarkPoint]:
    """BGR 프레임 한 장에서 포즈 랜드마크를 추출합니다."""
    results = pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))  # type: ignore[attr-defined]
    return pose_landmarks_to_dict(results)


def has_required_visibility(
    landmarks: dict[str, LandmarkPoint],
    required_names: Iterable[str],
    minimum: float = MIN_LANDMARK_VISIBILITY,
) -> bool:
    """필수 랜드마크가 모두 visibility 기준을 넘는지 확인합니다."""
    return all(
        name in landmarks and landmarks[name].visibility >= minimum
        for name in required_names
    )


def mean_visibility(landmarks: dict[str, LandmarkPoint], names: Iterable[str]) -> float:
    """지정한 랜드마크들의 평균 visibility를 계산합니다 (디버그용)."""
    values = [landmarks[name].visibility for name in names if name in landmarks]
    return float(np.mean(values)) if values else 0.0


def filter_visible_landmarks(
    landmarks: dict[str, LandmarkPoint],
    minimum: float = MIN_LANDMARK_VISIBILITY,
) -> dict[str, LandmarkPoint]:
    """visibility가 기준 미만인 랜드마크를 제외합니다."""
    return {name: point for name, point in landmarks.items() if point.visibility >= minimum}


def landmarks_to_pixels(
    landmarks: dict[str, LandmarkPoint],
    frame_shape: Sequence[int],
) -> dict[str, np.ndarray]:
    """정규화 좌표를 프레임 픽셀 좌표 [x, y]로 변환합니다.

    공식: x_px = x * 너비, y_px = y * 높이
    """
    height, width = int(frame_shape[0]), int(frame_shape[1])
    return {
        name: np.array([point.x * width, point.y * height], dtype=np.float64)
        for name, point in landmarks.items()
    }


def hip_center(
    landmarks: dict[str, LandmarkPoint],
    side: str | None = None,
) -> np.ndarray | None:
    """hip 기준점을 계산합니다 (정규화 좌표).

    - side가 "left"/"right"면 그쪽 hip 하나를 쓴다.
      측면 영상에서는 반대쪽 hip이 가려져 visibility가 낮은 경우가 많기 때문이다.
    - side가 None이면 양쪽 hip의 중점을 쓴다 (정면 영상).
    """
    if side in ("left", "right"):
        name = f"{side}_hip"
        if not has_required_visibility(landmarks, (name,)):
            return None
        return landmarks[name].as_array()
    if not has_required_visibility(landmarks, ("left_hip", "right_hip")):
        return None
    return (landmarks["left_hip"].as_array() + landmarks["right_hip"].as_array()) / 2.0


def to_hip_relative_coordinates(
    landmarks: dict[str, LandmarkPoint],
    side: str | None = None,
) -> dict[str, np.ndarray]:
    """hip 기준점을 원점으로 하는 상대 좌표를 계산합니다 (정규화 좌표).

    화면 좌표 대신 몸 기준 좌표를 쓰면, 보호자가 들고 찍어 화면이 움직여도
    궤적 모양이 유지된다. 각도 계산이 필요하면 pixel 버전을 사용한다.
    """
    center = hip_center(landmarks, side)
    if center is None:
        return {}
    return {name: point.as_array() - center for name, point in landmarks.items()}


def moving_average(
    values: Sequence[np.ndarray] | np.ndarray,
    window: int = WRIST_SMOOTHING_WINDOW,
) -> list[np.ndarray]:
    """좌표 시계열에 가운데 정렬 이동평균을 적용해 떨림을 줄입니다.

    앞쪽 값만 평균내면 궤적이 뒤로 밀리는 지연이 생기므로,
    각 시점 앞뒤 window//2 개를 함께 평균낸다. 양 끝은 가능한 범위만 쓴다.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return []
    half = max(0, int(window) // 2)
    smoothed: list[np.ndarray] = []
    for index in range(len(array)):
        start = max(0, index - half)
        end = min(len(array), index + half + 1)
        smoothed.append(np.mean(array[start:end], axis=0))
    return smoothed


def angle_between_vectors(first: np.ndarray, second: np.ndarray) -> float:
    """두 벡터 사이의 각도(0~180°)를 계산합니다.

    공식: θ = arccos( (a·b) / (|a||b|) )
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0.0 or second_norm == 0.0:
        return float("nan")
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def joint_angle(first: np.ndarray, vertex: np.ndarray, last: np.ndarray) -> float:
    """첫 점-꼭짓점-마지막 점으로 이루어진 관절 각도를 계산합니다.

    예: 팔꿈치 각도 = shoulder-elbow-wrist, 180°면 팔이 쭉 펴진 상태.
    """
    return angle_between_vectors(np.asarray(first) - np.asarray(vertex), np.asarray(last) - np.asarray(vertex))


def line_vertical_deviation(first: np.ndarray, last: np.ndarray) -> float:
    """두 점을 잇는 선이 수직선에서 벗어난 각도(0~90°)를 계산합니다.

    방향(위→아래, 아래→위)과 무관하게 기울어진 정도만 본다.
    공식: θ = arctan( |Δx| / |Δy| )
    """
    delta = np.asarray(last, dtype=np.float64)[:2] - np.asarray(first, dtype=np.float64)[:2]
    if not np.any(delta):
        return float("nan")
    return math.degrees(math.atan2(abs(delta[0]), abs(delta[1])))


def line_horizontal_deviation(first: np.ndarray, last: np.ndarray) -> float:
    """두 점을 잇는 선이 수평선에서 벗어난 각도(0~90°)를 계산합니다.

    공식: θ = arctan( |Δy| / |Δx| )  (예: 양쪽 어깨선의 기울기)
    """
    delta = np.asarray(last, dtype=np.float64)[:2] - np.asarray(first, dtype=np.float64)[:2]
    if not np.any(delta):
        return float("nan")
    return math.degrees(math.atan2(abs(delta[1]), abs(delta[0])))


def circle_from_three_points(
    first: Sequence[float], second: Sequence[float], third: Sequence[float]
) -> tuple[np.ndarray, float] | None:
    """세 점을 지나는 원의 중심과 반지름을 계산합니다.

    원의 중심 c는 세 점까지 거리가 같으므로
    2(b-a)·c = |b|²-|a|², 2(c3-a)·c = |c3|²-|a|² 연립방정식을 푼다.
    세 점이 거의 한 직선 위에 있으면 원을 만들 수 없어 None을 반환한다.
    """
    point_a = np.asarray(first, dtype=np.float64)[:2]
    point_b = np.asarray(second, dtype=np.float64)[:2]
    point_c = np.asarray(third, dtype=np.float64)[:2]
    matrix = 2.0 * np.array([point_b - point_a, point_c - point_a])
    values = np.array(
        [
            np.dot(point_b, point_b) - np.dot(point_a, point_a),
            np.dot(point_c, point_c) - np.dot(point_a, point_a),
        ]
    )
    scale = max(1.0, float(np.max(np.abs(matrix))))
    if abs(float(np.linalg.det(matrix))) < 1e-6 * scale * scale:
        return None
    center = np.linalg.solve(matrix, values)
    radius = float(np.linalg.norm(center - point_a))
    return center, radius


def resize_long_side(frame: np.ndarray, long_side: int = VIDEO_LONG_SIDE_PX) -> np.ndarray:
    """영상의 긴 변을 제한하고 종횡비를 유지합니다."""
    height, width = frame.shape[:2]
    current_long_side = max(height, width)
    if current_long_side <= long_side:
        return frame
    scale = long_side / current_long_side
    return cv2.resize(
        frame,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


_ROTATE_CODES: dict[int, int] = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def open_video(video_path: str) -> tuple[cv2.VideoCapture, int | None]:
    """영상을 열고, 세워서 찍은 영상을 바로 세우기 위한 회전 코드를 함께 반환합니다.

    휴대폰을 세로로 들고 찍으면 영상 자체는 누운 채 저장되고 "90° 돌려서 보여주라"는 정보만 붙는다.
    OpenCV는 환경에 따라 이 정보를 자동 적용하기도 하고 안 하기도 해서,
    자동 회전을 끄고 직접 돌린다 (누운 프레임으로는 각도 계산이 전부 틀어진다).
    """
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    degrees = int(round(capture.get(cv2.CAP_PROP_ORIENTATION_META) or 0)) % 360
    return capture, _ROTATE_CODES.get(degrees)


def read_upright(capture: cv2.VideoCapture, rotate_code: int | None) -> tuple[bool, np.ndarray | None]:
    """프레임 한 장을 읽어 바로 세운 방향으로 반환합니다."""
    success, frame = capture.read()
    if success and rotate_code is not None:
        frame = cv2.rotate(frame, rotate_code)
    return success, frame


def video_info(video_path: str) -> tuple[float, int]:
    """영상의 원본 fps와 전체 프레임 수를 반환합니다 (읽을 수 없으면 0)."""
    capture = cv2.VideoCapture(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    return fps, frame_count


def iter_sampled_frames(
    video_path: str,
    sample_fps: int = VIDEO_SAMPLE_FPS,
    max_seconds: float | None = None,
    long_side: int = VIDEO_LONG_SIDE_PX,
) -> Iterator[SampledFrame]:
    """영상을 sample_fps 간격으로 샘플링하고 축소한 프레임을 차례로 반환합니다.

    원본 fps가 30이고 sample_fps가 15면 2프레임마다 1장을 쓴다.
    max_seconds가 있으면 그 시간까지만 읽는다.
    long_side: 축소 후 긴 변의 픽셀 수 (멀리 있는 사람을 봐야 하는 정면 영상은 더 크게 쓴다).
    """
    capture, rotate_code = open_video(video_path)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or float(sample_fps)
    stride = max(1, round(source_fps / max(1, sample_fps)))
    frame_index = 0
    try:
        while True:
            success, frame = read_upright(capture, rotate_code)
            if not success:
                break
            timestamp = frame_index / source_fps
            if max_seconds is not None and timestamp > max_seconds:
                break
            if frame_index % stride == 0:
                yield SampledFrame(
                    index=frame_index,
                    timestamp_seconds=timestamp,
                    frame=resize_long_side(frame, long_side),
                )
            frame_index += 1
    finally:
        capture.release()


def normalized_to_pixel(point: np.ndarray, frame: np.ndarray) -> tuple[int, int]:
    """정규화 좌표를 영상 픽셀 좌표로 변환합니다."""
    height, width = frame.shape[:2]
    return int(point[0] * width), int(point[1] * height)
