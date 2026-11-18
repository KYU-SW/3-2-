"""올린 영상(센서 없음)에서 휴대폰 기울기와 흔들림을 영상만 보고 어림합니다.

- 기울기: 배경의 세로선(벽 모서리·문틀·창틀 등)이 화면 세로에서 얼마나 기울었는지 (길이 가중 중앙값).
  휴대폰이 기울면 세로선이 모두 같은 쪽으로 기운다. 위아래로 숙여 찍으면 세로선이 좌우 반대로 모여 평균이 0에 가깝다.
- 흔들림: 배경 점들이 프레임마다 얼마나 움직였는지 (사람·휠체어 영역은 빼고 광류로 추적).
사람·휠체어 영역은 관절 위치로 상자를 만들어 빼고, 바퀴가 있는 아래쪽까지 넉넉히 뺀다.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import cv2
import numpy as np

WORK_WIDTH = 360            # 계산용으로 줄인 화면 너비 (빠르게)
TILT_EVERY_SECONDS = 1.0    # 기울기는 1초에 한 장만 본다
TILT_MAX_FRAMES = 10
VERTICAL_WITHIN_DEG = 20.0  # 세로에서 이 각도 안의 선만 '세로선'으로 본다


class CameraCheck:
    """분석 루프에서 프레임마다 update()를 부르고, 끝나면 tilt_deg()·shake_ratio()를 읽는다."""

    def __init__(self) -> None:
        self.prev: np.ndarray | None = None
        self.shifts: list[float] = []       # 프레임 간 배경 이동량 (화면 너비 대비 비율)
        self.tilt_lines: list[tuple[float, float]] = []  # (세로에서 기운 각도, 선 길이)
        self.tilt_frames = 0
        self.last_tilt_time = -1e9

    def update(self, frame_bgr: np.ndarray, pixels: Mapping[str, np.ndarray], timestamp: float) -> None:
        height, width = frame_bgr.shape[:2]
        scale = WORK_WIDTH / width
        gray = cv2.cvtColor(cv2.resize(frame_bgr, (WORK_WIDTH, int(height * scale))), cv2.COLOR_BGR2GRAY)
        mask = self._background_mask(gray.shape, pixels, scale)

        if self.prev is not None:
            self._track_shift(self.prev, gray, mask)
        self.prev = gray

        if timestamp - self.last_tilt_time >= TILT_EVERY_SECONDS and self.tilt_frames < TILT_MAX_FRAMES:
            self.last_tilt_time = timestamp
            self.tilt_frames += 1
            self._collect_vertical_lines(gray, mask)

    @staticmethod
    def _background_mask(shape: tuple[int, int], pixels: Mapping[str, np.ndarray], scale: float) -> np.ndarray:
        """사람·휠체어를 뺀 배경만 255."""
        mask = np.full(shape, 255, dtype=np.uint8)
        if pixels:
            points = np.array([np.asarray(p, dtype=float) for p in pixels.values()]) * scale
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            body_w, body_h = max(x1 - x0, 10.0), max(y1 - y0, 10.0)
            # 휠체어 바퀴는 몸보다 앞뒤로 넓고 바닥까지 내려간다 → 좌우 60%, 위 30%, 아래는 화면 끝까지
            left, right = int(x0 - body_w * 0.6), int(x1 + body_w * 0.6)
            top = int(y0 - body_h * 0.3)
            cv2.rectangle(mask, (max(left, 0), max(top, 0)), (min(right, shape[1] - 1), shape[0] - 1), 0, -1)
        return mask

    def _track_shift(self, prev: np.ndarray, gray: np.ndarray, mask: np.ndarray) -> None:
        corners = cv2.goodFeaturesToTrack(prev, maxCorners=150, qualityLevel=0.01, minDistance=8, mask=mask)
        if corners is None or len(corners) < 15:
            return
        moved, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, corners, None, winSize=(15, 15), maxLevel=2)
        good = status.reshape(-1) == 1
        if good.sum() < 15:
            return
        delta = (moved[good] - corners[good]).reshape(-1, 2)
        self.shifts.append(float(np.median(np.hypot(delta[:, 0], delta[:, 1]))) / WORK_WIDTH)

    def _collect_vertical_lines(self, gray: np.ndarray, mask: np.ndarray) -> None:
        edges = cv2.Canny(gray, 60, 160)
        edges[mask == 0] = 0
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=int(gray.shape[0] * 0.15), maxLineGap=4)
        if lines is None:
            return
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx, dy = float(x2 - x1), float(y2 - y1)
            if dy < 0:
                dx, dy = -dx, -dy
            angle = math.degrees(math.atan2(dx, dy))  # 0 = 똑바른 세로선
            if abs(angle) <= VERTICAL_WITHIN_DEG:
                self.tilt_lines.append((angle, math.hypot(dx, dy)))

    def tilt_deg(self, min_total_length: float = 300.0) -> float | None:
        """배경 세로선 기울기(길이 가중 중앙값). 세로선이 부족하면 None (판단 안 함)."""
        if not self.tilt_lines:
            return None
        angles = np.array([a for a, _ in self.tilt_lines])
        lengths = np.array([l for _, l in self.tilt_lines])
        if lengths.sum() < min_total_length:
            return None
        order = np.argsort(angles)
        cumulative = np.cumsum(lengths[order])
        return float(angles[order][np.searchsorted(cumulative, cumulative[-1] / 2)])

    def shake_ratio(self, threshold: float) -> float | None:
        """배경이 화면 너비의 threshold 비율 이상 움직인 프레임 쌍의 비율. 추적할 점이 부족하면 None."""
        if len(self.shifts) < 10:
            return None
        return float(np.mean(np.asarray(self.shifts) > threshold))
