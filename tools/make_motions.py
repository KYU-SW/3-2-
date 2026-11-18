"""운동 마네킹 동작을 직접 만든다 (영상에서 뽑으면 동작이 작고 흔들리거나, 뒤에서 찍어 깊이를 못 뽑는 경우).

각 동작은 참고 영상의 순서를 따라 '단계(steps)'로 나누고, 플레이어가 단계 설명을 화면에 띄운다.
좌표: 골반 중심 원점, 미터, x = 앉은 사람의 왼쪽(+), y = 위, z = 앞. 형식은 tools/extract_motion.py와 같다.
사용법 (wheelance 폴더): python tools/make_motions.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FPS = 15.0
POINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip",
)
UPPER_ARM, FOREARM = 0.28, 0.26
NECK_PIVOT = np.array([0.0, 0.50, -0.01])
HEAD = {
    "nose": (0.0, 0.59, 0.10), "left_eye": (0.035, 0.625, 0.085), "right_eye": (-0.035, 0.625, 0.085),
    "left_ear": (0.075, 0.60, 0.0), "right_ear": (-0.075, 0.60, 0.0),
}
SHOULDER = {1: np.array([0.15, 0.44, 0.0]), -1: np.array([-0.15, 0.44, 0.0])}  # 1 = 왼쪽, -1 = 오른쪽
Arm = tuple[np.ndarray, np.ndarray]


# ---------- 공통 ----------
def ease(t: float) -> float:
    """0~1을 부드럽게 (smoothstep)."""
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def mix(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * ease(t)


def mix_arm(a: Arm, b: Arm, t: float) -> Arm:
    return mix(a[0], b[0], t), mix(a[1], b[1], t)


def ramp(t: float, start: float, end: float) -> float:
    """start~end 사이에서 0→1."""
    return ease((t - start) / (end - start))


def rest_arm(side: int) -> Arm:
    """팔을 내려 허벅지 위에 둔 자세 (팔꿈치, 손목)."""
    return np.array([side * 0.17, 0.17, 0.03]), np.array([side * 0.17, -0.04, 0.16])


def chest_arm(side: int, press: float) -> Arm:
    """양손을 가슴에 X자로 포갬: 왼손은 오른쪽 쇄골 아래, 오른손은 왼쪽 쇄골 아래. press(0~1)만큼 아래로 눌러 고정."""
    down = 0.03 * press
    elbow = np.array([side * 0.20, 0.24 - down, 0.12])
    wrist = np.array([-side * 0.06, 0.37 - down, 0.13 + (0.01 if side > 0 else 0.0)])
    return elbow, wrist


def circle_arm(side: int, theta: float) -> Arm:
    """곧게 편 팔로 크게 원 그리기: θ=0 아래 → 90° 앞 → 180° 위 → 뒤."""
    direction = np.array([side * 0.30, -math.cos(theta), math.sin(theta)])
    direction /= np.linalg.norm(direction)
    shoulder = SHOULDER[side]
    return shoulder + direction * UPPER_ARM, shoulder + direction * (UPPER_ARM + FOREARM)


def w_arm(side: int, squeeze: float) -> Arm:
    """W 자세: 팔꿈치는 어깨보다 조금 낮게 옆·뒤로, 손은 귀 높이로 세우고 뒤로. squeeze만큼 날개뼈를 조여 더 뒤로."""
    back = -0.05 - 0.04 * squeeze
    return np.array([side * 0.36, 0.33, back]), np.array([side * 0.34, 0.58, back - 0.03])


def head_points(extension_deg: float, turn_deg: float = 0.0, side_tilt_deg: float = 0.0) -> dict[str, np.ndarray]:
    """고개: 위로 젖힘(+ = 얼굴이 위·뒤로, − = 숙임) → 옆으로 기울임(+ = 왼쪽 귀가 왼쪽 어깨 쪽으로) → 좌우로 돌림(+ = 왼쪽). 목 기준점이 축."""
    ext, turn, tilt = math.radians(extension_deg), math.radians(turn_deg), math.radians(side_tilt_deg)
    out = {}
    for name, point in HEAD.items():
        v = np.array(point) - NECK_PIVOT
        v = np.array([v[0], v[1] * math.cos(ext) + v[2] * math.sin(ext), -v[1] * math.sin(ext) + v[2] * math.cos(ext)])
        v = np.array([v[0] * math.cos(tilt) + v[1] * math.sin(tilt), -v[0] * math.sin(tilt) + v[1] * math.cos(tilt), v[2]])
        v = np.array([v[0] * math.cos(turn) + v[2] * math.sin(turn), v[1], -v[0] * math.sin(turn) + v[2] * math.cos(turn)])
        out[name] = NECK_PIVOT + v
    return out


SPINE_PIVOT = np.array([0.0, 0.08, 0.0])  # 몸통을 기울일 때 도는 점 (허리 아래쪽)


def bend(point: np.ndarray, side_deg: float, forward_deg: float) -> np.ndarray:
    """상체 기울이기: 옆(+ = 앉은 사람의 왼쪽으로) → 앞(+ = 앞으로 숙임). 허리 아래쪽이 축."""
    side, forward = math.radians(side_deg), math.radians(forward_deg)
    v = point - SPINE_PIVOT
    v = np.array([v[0] * math.cos(side) + v[1] * math.sin(side), -v[0] * math.sin(side) + v[1] * math.cos(side), v[2]])
    v = np.array([v[0], -v[2] * math.sin(forward) + v[1] * math.cos(forward), v[2] * math.cos(forward) + v[1] * math.sin(forward)])
    return SPINE_PIVOT + v


def solve_elbow(shoulder: np.ndarray, wrist: np.ndarray, pole: np.ndarray) -> np.ndarray:
    """어깨·손목 위치로 팔꿈치를 계산 (두 마디 IK). pole = 팔꿈치가 향할 쪽."""
    reach = wrist - shoulder
    distance = float(np.linalg.norm(reach))
    direction = reach / max(distance, 1e-6)
    distance = min(distance, UPPER_ARM + FOREARM - 1e-4)
    along = (UPPER_ARM ** 2 - FOREARM ** 2 + distance ** 2) / (2 * distance)
    height = math.sqrt(max(UPPER_ARM ** 2 - along ** 2, 0.0))
    bend_dir = pole - np.dot(pole, direction) * direction
    bend_dir = bend_dir / max(float(np.linalg.norm(bend_dir)), 1e-6)
    return shoulder + direction * along + bend_dir * height


def assemble(
    head: dict[str, np.ndarray], arms: dict[int, Arm], shoulder_shift: np.ndarray | None = None,
    side_bend: float = 0.0, forward_bend: float = 0.0, shoulder_drop: dict[int, float] | None = None,
) -> np.ndarray:
    """자세 조립. 팔 좌표는 몸을 기울이기 전 기준으로 주면, 몸통 기울기(side_bend·forward_bend)에 맞춰 함께 돈다."""
    shift = shoulder_shift if shoulder_shift is not None else np.zeros(3)
    drops = shoulder_drop or {}
    points = dict(head)
    for side, name in ((1, "left"), (-1, "right")):
        points[f"{name}_shoulder"] = SHOULDER[side] + shift + np.array([0.0, -drops.get(side, 0.0), 0.0])
        points[f"{name}_elbow"], points[f"{name}_wrist"] = arms[side]
    for name in list(points):
        points[name] = bend(np.asarray(points[name], dtype=float), side_bend, forward_bend)
    # 팔 길이 고정: 손목은 닿는 범위 안으로, 팔꿈치는 어깨·손목으로 다시 계산 (자세 사이를 섞을 때 팔이 줄었다 늘었다 하지 않게)
    for name in ("left", "right"):
        shoulder, wrist, hint = points[f"{name}_shoulder"], points[f"{name}_wrist"], points[f"{name}_elbow"]
        reach = wrist - shoulder
        limit = UPPER_ARM + FOREARM - 0.005
        if float(np.linalg.norm(reach)) > limit:
            wrist = shoulder + reach / np.linalg.norm(reach) * limit
            points[f"{name}_wrist"] = wrist
        points[f"{name}_elbow"] = solve_elbow(shoulder, wrist, hint - shoulder)
    points["left_hip"], points["right_hip"] = np.array([0.10, 0.0, 0.0]), np.array([-0.10, 0.0, 0.0])
    return np.array([points[name] for name in POINTS])


# ---------- 1. 목 앞쪽 스트레칭 (사각근·흉쇄유돌근) ----------
FRONT_STEPS = [
    (0.0, 1.0, "편하게 앉아요"),
    (1.0, 2.5, "양손을 가슴에 X자로 포개요"),
    (2.5, 3.5, "가슴을 아래로 눌러 고정해요"),
    (3.5, 5.5, "턱을 들며 고개를 뒤로 젖혀요"),
    (5.5, 7.0, "고개를 오른쪽으로 돌려요"),
    (7.0, 11.0, "목 앞이 당기는 느낌으로 유지 (10~20초)"),
    (11.0, 12.5, "가운데로 돌아와요"),
    (12.5, 14.0, "반대쪽(왼쪽)으로 돌려요"),
    (14.0, 18.0, "유지 (10~20초)"),
    (18.0, 20.5, "천천히 제자리로"),
]


def front_stretch(t: float) -> np.ndarray:
    press = ramp(t, 2.5, 3.5) if t < 18.0 else 1 - ramp(t, 18.0, 19.5)
    ext = 35 * ramp(t, 3.5, 5.5) if t < 18.0 else 35 * (1 - ramp(t, 18.0, 19.5))
    if t < 5.5:
        turn = 0.0
    elif t < 11.0:
        turn = -35 * ramp(t, 5.5, 7.0)  # 오른쪽 = 앉은 사람 기준 −
    elif t < 12.5:
        turn = -35 * (1 - ramp(t, 11.0, 12.5))
    elif t < 18.0:
        turn = 35 * ramp(t, 12.5, 14.0)
    else:
        turn = 35 * (1 - ramp(t, 18.0, 19.5))
    arms = {}
    for side in (1, -1):
        on_chest = chest_arm(side, press)
        if t < 2.5:
            arms[side] = mix_arm(rest_arm(side), on_chest, (t - 1.0) / 1.5) if t >= 1.0 else rest_arm(side)
        elif t < 18.0:
            arms[side] = on_chest
        else:
            arms[side] = mix_arm(on_chest, rest_arm(side), (t - 18.5) / 2.0)
    return assemble(head_points(ext, turn), arms, np.array([0.0, -0.015 * press, 0.0]))


# ---------- 2. 10초 거북목 교정 ----------
QUICK_STEPS = [
    (0.0, 1.0, "편하게 앉아요"),
    (1.0, 2.0, "양손을 가슴에 붙여요"),
    (2.0, 3.0, "가슴을 아래로 내려요"),
    (3.0, 5.0, "고개를 위로 크게 올려요"),
    (5.0, 8.0, "목 앞이 시원하게 늘어나게 유지"),
    (8.0, 10.0, "천천히 제자리로"),
]


def quick_correction(t: float) -> np.ndarray:
    press = ramp(t, 2.0, 3.0) if t < 8.0 else 1 - ramp(t, 8.0, 9.5)
    ext = 50 * ramp(t, 3.0, 5.0) if t < 8.0 else 50 * (1 - ramp(t, 8.0, 9.5))
    arms = {}
    for side in (1, -1):
        on_chest = chest_arm(side, press)
        if t < 2.0:
            arms[side] = mix_arm(rest_arm(side), on_chest, (t - 1.0) / 1.0) if t >= 1.0 else rest_arm(side)
        elif t < 8.0:
            arms[side] = on_chest
        else:
            arms[side] = mix_arm(on_chest, rest_arm(side), (t - 8.3) / 1.6)
    return assemble(head_points(ext), arms, np.array([0.0, -0.02 * press, 0.0]))


# ---------- 3. 팔 W 모양 스트레칭 ----------
W_STEPS = [
    (0.0, 1.0, "편하게 앉아요"),
    (1.0, 3.6, "두 팔을 크게 뒤로 돌려요"),
    (3.6, 5.0, "팔을 W 모양으로 (날개뼈 조이기)"),
    (5.0, 7.0, "고개를 천천히 뒤로 젖혀요"),
    (7.0, 12.0, "어깨가 올라가지 않게 유지 (10~20초)"),
    (12.0, 14.0, "천천히 제자리로"),
]


def w_stretch(t: float) -> np.ndarray:
    arms = {}
    for side in (1, -1):
        rest = rest_arm(side)
        if t < 1.0:
            arm = rest
        elif t < 3.6:
            arm = circle_arm(side, math.radians(215) * ease((t - 1.0) / 2.6))
            if t < 1.4:
                arm = mix_arm(rest, arm, (t - 1.0) / 0.4)
        elif t < 5.0:
            arm = mix_arm(circle_arm(side, math.radians(215)), w_arm(side, 0.0), (t - 3.6) / 1.4)
        elif t < 12.0:
            arm = w_arm(side, ramp(t, 5.0, 6.5))
        else:
            arm = mix_arm(w_arm(side, 1.0), rest, (t - 12.0) / 2.0)
        arms[side] = arm
    if 5.0 <= t < 12.0:
        back = 0.03 * ramp(t, 5.0, 6.5)
        ext = 40 * ramp(t, 5.0, 7.0) + (1.5 * math.sin((t - 7.0) * 1.2) if t >= 7.0 else 0.0)
    elif t >= 12.0:
        back = 0.03 * (1 - ramp(t, 12.0, 14.0))
        ext = 40 * (1 - ramp(t, 12.0, 13.6))
    else:
        back, ext = 0.0, 0.0
    return assemble(head_points(ext), arms, np.array([0.0, 0.0, -back]))


# ---------- 4. 한쪽 처진 어깨 (낮은 쪽 광배근·옆구리 늘이기) — 예시는 오른쪽 어깨가 낮은 경우 ----------
SHOULDER_DROP_STEPS = [
    (0.0, 1.0, "편하게 앉아요"),
    (1.0, 2.5, "거울로 낮은 쪽 어깨를 확인해요 (예: 오른쪽)"),
    (2.5, 4.0, "낮은 쪽 팔을 머리 위로 쭉 올려요"),
    (4.0, 5.5, "팔꿈치를 굽혀 머리 뒤로 넘기고, 반대 손으로 팔꿈치를 잡아요"),
    (5.5, 7.5, "몸을 반대쪽으로 기울이며 옆구리를 늘여요"),
    (7.5, 12.5, "등과 옆구리가 늘어나는 느낌으로 유지"),
    (12.5, 14.5, "천천히 제자리로"),
]


def shoulder_drop_stretch(t: float) -> np.ndarray:
    right_rest, left_rest = rest_arm(-1), rest_arm(1)
    right_up = (SHOULDER[-1] + np.array([0.02, UPPER_ARM, 0.0]), SHOULDER[-1] + np.array([0.04, UPPER_ARM + FOREARM, 0.0]))
    right_behind = (np.array([-0.09, 0.80, -0.03]), np.array([0.06, 0.60, -0.10]))       # 팔꿈치 머리 위, 손은 머리 뒤
    left_hold_wrist = right_behind[0] + np.array([0.04, -0.02, 0.04])                     # 왼손이 오른 팔꿈치를 잡음
    left_hold = (solve_elbow(SHOULDER[1], left_hold_wrist, np.array([1.0, 0.0, 0.3])), left_hold_wrist)
    if t < 2.5:
        right, left = right_rest, left_rest
    elif t < 4.0:
        right, left = mix_arm(right_rest, right_up, (t - 2.5) / 1.5), left_rest
    elif t < 5.5:
        right, left = mix_arm(right_up, right_behind, (t - 4.0) / 1.5), mix_arm(left_rest, left_hold, (t - 4.0) / 1.5)
    elif t < 12.5:
        right, left = right_behind, left_hold
    else:
        right, left = mix_arm(right_behind, right_rest, (t - 12.5) / 2.0), mix_arm(left_hold, left_rest, (t - 12.5) / 2.0)
    lean = 20 * ramp(t, 5.5, 7.5) if t < 12.5 else 20 * (1 - ramp(t, 12.5, 14.0))   # 왼쪽(높은 쪽)으로 기울임
    look = -8 * ramp(t, 1.0, 1.8) * (1 - ramp(t, 2.0, 2.5))                           # 거울 보며 오른쪽 어깨 확인
    return assemble(head_points(-5, look, 0), {1: left, -1: right}, side_bend=lean)


# ---------- 5. 승모근·견갑거근 스트레칭 (어깨 끌어내리고 고개를 대각선 아래로) — 예시는 오른쪽을 늘이는 경우 ----------
TRAP_STEPS = [
    (0.0, 1.0, "편하게 앉아요"),
    (1.0, 2.5, "오른팔을 등 뒤로 걸어 어깨를 아래로 끌어내려요"),
    (2.5, 4.0, "왼손으로 머리 뒤쪽을 감싸 잡아요"),
    (4.0, 6.0, "고개를 왼쪽 겨드랑이 쪽(대각선 아래)으로 지그시 당겨요"),
    (6.0, 11.0, "목 뒤·날개뼈 위쪽이 늘어나게 유지 (20~30초)"),
    (11.0, 13.0, "천천히 제자리로 (반대쪽도 같은 방법)"),
]


def trap_release(t: float) -> np.ndarray:
    right_rest, left_rest = rest_arm(-1), rest_arm(1)
    right_back = (np.array([-0.20, 0.22, -0.12]), np.array([-0.04, 0.10, -0.16]))   # 손을 등 뒤 허리로
    pull = ramp(t, 4.0, 6.0) if t < 11.0 else 1 - ramp(t, 11.0, 12.6)
    head = head_points(-30 * pull, 30 * pull, 20 * pull)                               # 숙이고, 왼쪽으로 돌리고, 왼쪽으로 기울임
    hand_target = head["left_ear"] * 0.4 + head["right_ear"] * 0.6 + np.array([0.0, 0.05, -0.09])  # 머리 뒤 오른쪽 위
    left_hold = (solve_elbow(SHOULDER[1], hand_target, np.array([1.0, 0.4, 0.4])), hand_target)
    if t < 1.0:
        right, left = right_rest, left_rest
    elif t < 2.5:
        right, left = mix_arm(right_rest, right_back, (t - 1.0) / 1.5), left_rest
    elif t < 11.0:
        right = right_back
        left = mix_arm(left_rest, left_hold, (t - 2.5) / 1.5) if t < 4.0 else left_hold
    else:
        right, left = mix_arm(right_back, right_rest, (t - 11.0) / 2.0), mix_arm(left_hold, left_rest, (t - 11.0) / 2.0)
    drop = 0.03 * (ramp(t, 1.0, 2.5) if t < 11.0 else 1 - ramp(t, 11.0, 12.6))
    return assemble(head, {1: left, -1: right}, shoulder_drop={-1: drop})


# ---------- 6. 앉아서 옆구리 늘이기 (척추측만 교정 운동) ----------
SIDE_STEPS = [
    (0.0, 1.0, "의자에 바르게 앉아요"),
    (1.0, 2.5, "오른팔을 머리 위로 올려요"),
    (2.5, 4.0, "몸을 왼쪽으로 기울이며 팔을 멀리 뻗어요"),
    (4.0, 8.0, "옆구리가 늘어나게 열까지 세며 유지"),
    (8.0, 9.5, "가운데로 돌아와요"),
    (9.5, 11.0, "왼팔을 올려 오른쪽으로 기울여요"),
    (11.0, 15.0, "열까지 세며 유지"),
    (15.0, 17.0, "천천히 제자리로"),
]


def overhead_arm(side: int) -> Arm:
    """머리 위로 뻗은 팔 (살짝 굽힘)."""
    elbow = SHOULDER[side] + np.array([-side * 0.03, UPPER_ARM, 0.02])
    return elbow, elbow + np.array([-side * 0.06, FOREARM - 0.01, 0.0])


def side_stretch(t: float) -> np.ndarray:
    right_rest, left_rest = rest_arm(-1), rest_arm(1)
    hand_on_thigh = {1: (np.array([0.20, 0.18, 0.08]), np.array([0.16, 0.0, 0.25])), -1: (np.array([-0.20, 0.18, 0.08]), np.array([-0.16, 0.0, 0.25]))}
    if t < 1.0:
        arms, lean = {1: left_rest, -1: right_rest}, 0.0
    elif t < 8.0:
        arms = {-1: mix_arm(right_rest, overhead_arm(-1), (t - 1.0) / 1.5), 1: mix_arm(left_rest, hand_on_thigh[1], (t - 1.0) / 1.5)}
        lean = 22 * ramp(t, 2.5, 4.0) + (1.5 * math.sin((t - 4.0) * 1.5) if t >= 4.0 else 0.0)
    elif t < 9.5:
        arms = {-1: mix_arm(overhead_arm(-1), right_rest, (t - 8.0) / 1.5), 1: mix_arm(hand_on_thigh[1], left_rest, (t - 8.0) / 1.5)}
        lean = 22 * (1 - ramp(t, 8.0, 9.3))
    elif t < 15.0:
        arms = {1: mix_arm(left_rest, overhead_arm(1), (t - 9.5) / 1.5), -1: mix_arm(right_rest, hand_on_thigh[-1], (t - 9.5) / 1.5)}
        lean = -22 * ramp(t, 9.5, 11.0) - (1.5 * math.sin((t - 11.0) * 1.5) if t >= 11.0 else 0.0)
    else:
        arms = {1: mix_arm(overhead_arm(1), left_rest, (t - 15.0) / 2.0), -1: mix_arm(hand_on_thigh[-1], right_rest, (t - 15.0) / 2.0)}
        lean = -22 * (1 - ramp(t, 15.0, 16.5))
    return assemble(head_points(0, 0, 0), arms, side_bend=lean)


# ---------- 7. 반원형으로 휠체어 미는 법 ----------
# 바퀴 손잡이 테(림): 몸 옆 바깥, 중심 높이 RIM_CENTER_Y, 반지름 RIM_RADIUS. 각도 φ: 90° = 테의 맨 위, 0° = 앞쪽.
# 골반 기준 좌표: 바퀴 축은 골반보다 약 22cm 아래 (바퀴 반지름 약 30cm, 축이 바닥에서 30cm, 골반 높이 52cm)
RIM_X, RIM_CENTER_Y, RIM_CENTER_Z, RIM_RADIUS = 0.32, -0.18, -0.02, 0.27
PUSH_START_DEG, PUSH_END_DEG = 120.0, 45.0  # 밀기 시작(테 뒤쪽 위) → 놓는 곳(앞쪽 위)
PUSH_SLOW, PUSH_PERIOD = 4.0, 1.4  # 첫 번은 천천히(단계 설명), 그다음은 평소 속도로
PUSH_STEPS = [
    (0.0, 1.0, "손을 바퀴 테 뒤쪽 위(시계 10~11시)에 얹어요"),
    (1.0, 2.2, "앞으로 길고 부드럽게 밀어요"),
    (2.2, 2.8, "앞에서 손을 놓아요"),
    (2.8, PUSH_SLOW, "손이 테보다 아래로 내려가 반원을 그리며 뒤로 돌아와요"),
    (PUSH_SLOW, PUSH_SLOW + 4 * PUSH_PERIOD, "길게 밀고, 손은 테 아래로 (반원형) · 짧게 자주 밀지 않기"),
]


def rim_point(side: int, phi_deg: float) -> np.ndarray:
    phi = math.radians(phi_deg)
    return np.array([side * RIM_X, RIM_CENTER_Y + RIM_RADIUS * math.sin(phi), RIM_CENTER_Z + RIM_RADIUS * math.cos(phi)])


def hand_cycle(side: int, u: float) -> tuple[np.ndarray, float]:
    """한 번 미는 동안 손 위치 (u: 0~1) + 상체 숙임(°). 0~0.55 밀기(테를 따라 앞으로), 0.55~1 돌아오기(테 아래 반원)."""
    if u < 0.55:
        s = ease(u / 0.55)
        return rim_point(side, PUSH_START_DEG + (PUSH_END_DEG - PUSH_START_DEG) * s), 4 + 6 * s  # 테를 따라 앞으로, 밀면서 살짝 숙임
    s = (u - 0.55) / 0.45
    release, grab = rim_point(side, PUSH_END_DEG), rim_point(side, PUSH_START_DEG)
    # 반원형: 놓은 손이 미는 길(테)보다 아래로 내려갔다가 뒤로 돌아온다
    low = (release + grab) / 2 + np.array([-side * 0.03, -0.09, 0.0])
    a = release + (low - release) * ease(min(1.0, s / 0.5))
    b = low + (grab - low) * ease(max(0.0, (s - 0.5) / 0.5))
    point = a if s < 0.5 else b
    # 테 바깥이 아닌 몸 쪽으로 살짝: 손은 테 안쪽 아래로 지나감
    point = point + np.array([-side * 0.02 * math.sin(math.pi * s), 0.0, 0.0])
    return point, 10 - 6 * ease(s)


def wheelchair_push(t: float) -> np.ndarray:
    if t < PUSH_SLOW:
        u = min(0.999, t / PUSH_SLOW)
    else:
        u = ((t - PUSH_SLOW) % PUSH_PERIOD) / PUSH_PERIOD
    arms, lean = {}, 0.0
    for side in (1, -1):
        wrist, lean = hand_cycle(side, u)
        arms[side] = (solve_elbow(SHOULDER[side], wrist, np.array([side * 0.6, 0.0, -1.0])), wrist)
    # 팔은 몸을 숙인 뒤 자리를 기준으로 만들었으므로, 숙임은 머리·어깨에만 반영되도록 손목은 그대로 둔다
    frame = assemble(head_points(-5), arms, forward_bend=lean)
    for side, name in ((1, "left"), (-1, "right")):
        shoulder = frame[POINTS.index(f"{name}_shoulder")]
        wrist = arms[side][1]
        reach = wrist - shoulder
        limit = UPPER_ARM + FOREARM - 0.005
        if np.linalg.norm(reach) > limit:  # 팔이 닿지 않는 곳이면 닿는 데까지만
            wrist = shoulder + reach / np.linalg.norm(reach) * limit
        frame[POINTS.index(f"{name}_wrist")] = wrist
        frame[POINTS.index(f"{name}_elbow")] = solve_elbow(shoulder, wrist, np.array([side * 0.6, 0.0, -1.0]))
    return frame


MOTIONS: dict[str, tuple[Callable[[float], np.ndarray], list[tuple[float, float, str]], str, dict[str, object]]] = {
    "neck_front_stretch": (front_stretch, FRONT_STEPS, "https://www.youtube.com/shorts/Lqcg1PWi424", {}),
    "neck_quick_correction": (quick_correction, QUICK_STEPS, "https://www.youtube.com/shorts/z-FI2mni_Nk", {}),
    "neck_w_stretch": (w_stretch, W_STEPS, "https://www.youtube.com/shorts/ARRJEBm7eDk", {}),
    "shoulder_drop_stretch": (shoulder_drop_stretch, SHOULDER_DROP_STEPS, "유튜브 '한쪽 쳐진 어깨 1분만에 교정하는법' (서서 하는 동작을 앉은 자세로 바꿈)", {}),
    "trap_release": (trap_release, TRAP_STEPS, "유튜브 '팔 걸고 끌어내리는 승모근 해방 스트레칭'", {}),
    "side_stretch": (side_stretch, SIDE_STEPS, "유튜브 '1분 척추측만증 교정운동'", {}),
    "push_technique": (wheelchair_push, PUSH_STEPS, "척수손상 의학 가이드라인 2005 (반원형 권장)", {"wheelchair": {"x": RIM_X, "y": RIM_CENTER_Y, "z": RIM_CENTER_Z, "rim": RIM_RADIUS}}),
}


if __name__ == "__main__":
    for exercise_id, (make, steps, url, props) in MOTIONS.items():
        duration = steps[-1][1]
        frames = [make(i / FPS) for i in range(int(duration * FPS))]
        motion = {
            "fps": FPS,
            "points": list(POINTS),
            "source": f"직접 만든 동작 (참고: {url})",
            "steps": [{"from": a, "to": b, "text": text} for a, b, text in steps],
            "props": props,
            "frames": np.round(np.stack(frames), 4).tolist(),
        }
        out = ROOT / "data" / "motions" / f"{exercise_id}.json"
        out.write_text(json.dumps(motion, ensure_ascii=False), encoding="utf-8")
        print(out.name, "frames", len(frames), "seconds", duration)
