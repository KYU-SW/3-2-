"""운동 영상 → 관절 마네킹 동작 데이터(JSON).

사용법 (wheelance 폴더에서):
    python tools/extract_motion.py 영상.mp4 운동id
→ data/motions/운동id.json 생성. 운동 탭의 마네킹이 이 파일을 재생한다.

처리
1. 영상을 15fps로 읽고, 몸이 같은 방향을 보는 가장 긴 구간만 쓴다 (옆모습 인트로 제외).
2. MediaPipe Pose의 world 좌표(골반 중심 원점, 미터)로 상체·머리 점을 뽑는다. 다리는 쓰지 않는다
   (영상에 안 보이는 경우가 많아 플레이어가 의자에 앉은 고정 자세로 그린다).
3. 뒤에서 찍은 영상이면(코가 카메라 반대쪽) 180° 돌려 마네킹이 앞을 보게 한다.
4. 떨림을 줄이려고 5프레임 이동평균을 한다.
원본 영상은 저장하지 않고 좌표만 저장한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pose_utils import open_video, read_upright  # noqa: E402

FPS = 15.0
SMOOTH = 5
POINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip",
)
INDEX = {landmark.name.lower(): landmark.value for landmark in mp.solutions.pose.PoseLandmark}


def read_frames(path: str) -> list[np.ndarray]:
    capture, rotate = open_video(path)
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(source_fps / FPS))
    frames, index = [], 0
    while True:
        ok, frame = read_upright(capture, rotate)
        if not ok or frame is None:
            break
        if index % step == 0:
            frames.append(frame)
        index += 1
    capture.release()
    return frames


def facing_run(rows: list[np.ndarray | None]) -> tuple[int, int]:
    """몸이 대부분 시간 보는 방향(어깨선의 좌우 회전)과 35° 이내인 프레임이 가장 길게 이어진 구간.

    옆모습 인트로처럼 몸 방향이 다른 장면을 빼기 위함. 자막이 바뀌는 건 영향을 주지 않는다. 4프레임 이하 끊김은 이어 붙인다.
    """
    left, right = POINTS.index("left_shoulder"), POINTS.index("right_shoulder")
    yaws = [float(np.degrees(np.arctan2(row[left, 2] - row[right, 2], row[left, 0] - row[right, 0]))) if row is not None else None for row in rows]
    known = [yaw for yaw in yaws if yaw is not None]
    center = float(np.median(known))
    good = [yaw is not None and abs((yaw - center + 180) % 360 - 180) <= 35 for yaw in yaws]
    best, start, gap = (0, 0), None, 0
    for index, ok in enumerate(good + [False] * 5):
        if ok:
            start = index if start is None else start
            gap = 0
            last_ok = index
        elif start is not None:
            gap += 1
            if gap > 4:
                if last_ok + 1 - start > best[1] - best[0]:
                    best = (start, last_ok + 1)
                start, gap = None, 0
    return best


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """벡터 a를 b 방향으로 돌리는 회전 행렬 (로드리게스 공식)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1 / (1 + c))


def straighten(data: np.ndarray) -> np.ndarray:
    """휴대폰 각도 보정: 영상 전체의 '평소' 몸통(골반 중심→어깨 중심, 중앙값)을 수직으로 세우고,
    어깨선이 좌우(x)로 놓이게 돌려 마네킹이 정면(+z)을 보게 한다. 동작 자체(숙임·회전의 변화)는 그대로 남는다."""
    # 평소(중앙값) 자세로 몸 기준 좌표축을 만든다: x = 어깨선(오른→왼), y = 몸통(골반→어깨, x에 수직), z = 앞.
    # 이 축이 화면의 x·y·z가 되게 돌리면 몸통은 수직, 어깨선은 수평, 얼굴은 정면이 된다.
    # (몸통만 세우면 어깨선이 9~15° 기운 채 남았다 — 영상 각도 때문, 대칭 동작인 W 스트레칭도 같았음)
    index = {name: i for i, name in enumerate(POINTS)}
    hip = (data[:, index["left_hip"]] + data[:, index["right_hip"]]) / 2
    shoulder = (data[:, index["left_shoulder"]] + data[:, index["right_shoulder"]]) / 2
    x_axis = np.median(data[:, index["left_shoulder"]] - data[:, index["right_shoulder"]], axis=0)
    x_axis /= np.linalg.norm(x_axis)
    up = np.median(shoulder - hip, axis=0)
    y_axis = up - np.dot(up, x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    basis = np.stack([x_axis, y_axis, z_axis])  # 각 행 = 새 축
    data = data @ basis.T
    # 앉은 영상은 골반이 가려져 MediaPipe 골반 위치가 부정확하다 (어깨선을 맞추면 몸통이 옆으로 9~15° 기운 채 남음).
    # → 평소 어깨 중심이 마네킹 골반 바로 위에 오도록 앞뒤·좌우만 옮긴다 (높이와 동작 변화는 그대로).
    shoulder = (data[:, index["left_shoulder"]] + data[:, index["right_shoulder"]]) / 2
    offset = np.median(shoulder, axis=0)
    data[:, :, 0] -= offset[0]
    data[:, :, 2] -= offset[2]
    return data


def extract(path: str) -> dict[str, object]:
    frames = read_frames(path)
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True)
    rows: list[np.ndarray | None] = []
    for frame in frames:
        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        world = result.pose_world_landmarks
        if world is None:
            rows.append(None)
            continue
        points = world.landmark
        if min(points[INDEX["left_shoulder"]].visibility, points[INDEX["right_shoulder"]].visibility) < 0.5:
            rows.append(None)
            continue
        # MediaPipe world: x 오른쪽, y 아래, z 카메라 쪽이 − → 플레이어 좌표(y 위, z 앞)로
        rows.append(np.array([[points[INDEX[name]].x, -points[INDEX[name]].y, -points[INDEX[name]].z] for name in POINTS]))
    pose.close()
    start, end = facing_run(rows)
    rows = rows[start:end]
    valid = [row for row in rows if row is not None]
    if len(valid) < 10:
        raise SystemExit(f"사람을 거의 못 찾았어요 ({len(valid)}프레임)")
    # 빈 프레임은 직전 값으로 채움
    filled, last = [], valid[0]
    for row in rows:
        last = row if row is not None else last
        filled.append(last)
    data = np.stack(filled)  # (프레임, 점, xyz)
    # 뒤에서 찍었으면(코가 어깨 중심보다 뒤쪽, z<0) y축으로 180° 돌린다
    nose_z = data[:, POINTS.index("nose"), 2] - (data[:, POINTS.index("left_shoulder"), 2] + data[:, POINTS.index("right_shoulder"), 2]) / 2
    flipped = bool(np.median(nose_z) < 0)
    if flipped:
        data[:, :, 0] *= -1
        data[:, :, 2] *= -1
    data = straighten(data)
    # 이동평균
    kernel = np.ones(SMOOTH) / SMOOTH
    padded = np.concatenate([np.repeat(data[:1], SMOOTH // 2, 0), data, np.repeat(data[-1:], SMOOTH // 2, 0)])
    smooth = np.stack([np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), 0, padded[:, p, :]) for p in range(len(POINTS))], 1)
    return {
        "fps": FPS,
        "points": list(POINTS),
        "flipped_from_back": flipped,
        "scene_seconds": [round(start / FPS, 2), round(end / FPS, 2)],
        "frames": np.round(smooth, 4).tolist(),
    }


if __name__ == "__main__":
    video, exercise_id = sys.argv[1], sys.argv[2]
    motion = extract(video)
    out = ROOT / "data" / "motions" / f"{exercise_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(motion, ensure_ascii=False), encoding="utf-8")
    print(out.name, "frames", len(motion["frames"]), "scene", motion["scene_seconds"], "flipped", motion["flipped_from_back"])
