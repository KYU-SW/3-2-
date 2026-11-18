"""실제 영상 파일로 전체 파이프라인을 검증한다.
- 사람 없는 영상: 진짜 MediaPipe로 돌려서 경고가 나오는지
- 합성 관절 영상: MediaPipe 결과만 가짜로 넣고(노이즈 + 카메라 이동) 나머지는 전부 실제 코드로
"""
import math, sys, tempfile
import cv2, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import propulsion as P
from pose_utils import LandmarkPoint, LANDMARK_NAMES
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from test_algo import stroke

W, H, FPS = 1280, 720, 30          # 원본 영상 (분석 시 720px로 축소됨 → 720x405)
SCALE = 720 / 1280

def write_video(seconds, fps=FPS):
    path = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(int(seconds * fps)):
        frame = np.full((H, W, 3), 200, np.uint8)
        cv2.putText(frame, str(i), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        out.write(frame)
    out.release()
    return open(path, "rb").read()

ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("OK  " if cond else "FAIL"), name, extra)

# 1) 사람 없는 영상 → 진짜 MediaPipe
r = P.analyze_video_bytes(write_video(6), view="side", filmed_side="left", rim_points=[(300, 150), (360, 210), (300, 270)])
check("no-person warning", r.warnings and "사람이 감지되지" in r.warnings[0], r.warnings)
# 2) 너무 짧은 영상
r = P.analyze_video_bytes(write_video(2), view="front")
check("too-short warning", r.warnings and "짧아요" in r.warnings[0], r.warnings)
# 3) 림 점 없음 / 일직선 3점
r = P.analyze_video_bytes(b"", view="side", filmed_side="left", rim_points=[(0, 0), (1, 1)])
check("incomplete rim warning", r.warnings and "3개" in r.warnings[0])
r = P.analyze_video_bytes(b"", view="side", filmed_side="left", rim_points=[(0, 0), (1, 1), (2, 2)])
check("collinear rim warning", r.warnings and "원을 만들 수 없어요" in r.warnings[0])
# 4) 첫 화면 크기 = 분석 프레임 크기
first = P.extract_first_frame(write_video(1))
check("first frame resized", first is not None and first.shape[:2] == (405, 720), first.shape)

# 5) 합성 관절로 전체 파이프라인 (측면, 왼쪽에서 촬영, 사람은 화면 오른쪽을 봄)
rng = np.random.default_rng(0)
R_px = 60.0
def make_side_sequence(kind, n_strokes=8):
    pts = []
    for _ in range(n_strokes):
        # 루프형은 실제 영상에서 본 크기(몸통 길이의 약 23% 위로)에 맞춰 높게 만든다
        p, rec = stroke(kind, facing=1.0, loop_height=1.6)
        pts += list(p * (R_px / 100)) + list(rec * (R_px / 100))
    return np.array(pts)

def install_fake(sequence, facing=1.0, pan_px_per_frame=0.6, lean_deg=12):
    counter = {"i": 0}
    rim_rel = np.array([10.0, -35.0])       # hip 기준 림 중심 위치(픽셀)
    def fake(pose, frame):
        i = counter["i"]; counter["i"] += 1
        h, w = frame.shape[:2]
        hip = np.array([300.0 + pan_px_per_frame * i, 260.0])   # 카메라가 따라가며 흔들림/이동
        wrist = hip + rim_rel + sequence[i % len(sequence)] + rng.normal(0, 1.5, 2)
        lean = math.radians(lean_deg + 4 * math.sin(i / 10))
        shoulder = hip + np.array([math.sin(lean) * 110 * facing, -math.cos(lean) * 110])
        elbow = (shoulder + wrist) / 2 + np.array([-15 * facing, 0])
        nose = shoulder + np.array([35 * facing, -45])
        pts = {"left_hip": hip, "left_shoulder": shoulder, "left_elbow": elbow, "left_wrist": wrist, "nose": nose,
               "right_hip": hip + [5, 0], "right_shoulder": shoulder + [5, 0]}
        out = {}
        for name in LANDMARK_NAMES:
            p = pts.get(name, hip)
            vis = 0.95 if name in pts else 0.1
            out[name] = LandmarkPoint(float(p[0] / w), float(p[1] / h), 0.0, vis)
        return out
    P.detect_landmarks = fake
    return rim_rel

original = P.detect_landmarks
for kind, want in [("semicircular", "semicircular"), ("arc", "arc"), ("single", "single_looping_over")]:
    seq = make_side_sequence(kind)
    rim_rel = install_fake(seq)
    hip0 = np.array([300.0, 260.0])
    c = hip0 + rim_rel
    rim_points = [(c[0] + R_px * math.cos(a), c[1] + R_px * math.sin(a)) for a in (0.3, 2.0, 4.0)]
    progress = []
    r = P.analyze_video_bytes(write_video(12), view="side", filmed_side="left", rim_points=rim_points, progress_callback=progress.append)
    m, j = P.to_firestore(r)
    check(f"side {kind}", not r.warnings and r.pattern == want and r.stroke_count >= 6,
          f"pattern={r.pattern} strokes={r.stroke_count} contact={m['contact_angle']} rate={m['stroke_rate']} lean_mean={m['trunk_lean_mean']} elbow={m['elbow_angle']} warn={r.warnings}")
    check("  frames/progress/lengths", len(r.representative_frames) == 3 and progress[-1] == 1.0
          and len({len(r.debug[k]) for k in ("time", "visibility", "wrist_x", "wrist_y", "phase")}) == 1,
          f"frames={len(r.representative_frames)} shape={r.representative_frames[0].shape if r.representative_frames else None}")
    check("  lean ≈ 12°", 8 < m["trunk_lean_mean"] < 16)

# 6) 영상 30초 → 앞 20초만 분석 + 안내
install_fake(make_side_sequence("semicircular"))
r = P.analyze_video_bytes(write_video(25), view="side", filmed_side="left", rim_points=rim_points)
check("long video truncated note", r.notes and r.duration_seconds <= 20.1, (r.notes, r.duration_seconds))

# 6-1) 정지 3초 → 밀기: 가만히 있을 때 상체 3°, 밀 때 3~19°를 오감. 림 있음 / 림 없음 둘 다
def install_fake_with_still(sequence, still_frames=45, facing=1.0):
    counter = {"i": 0}
    rim_rel = np.array([10.0, -35.0])
    def fake(pose, frame):
        i = counter["i"]; counter["i"] += 1
        h, w = frame.shape[:2]
        hip = np.array([300.0, 260.0])
        if i < still_frames:      # 손을 림 위에 얹고 가만히 있음
            wrist = hip + rim_rel + sequence[0] + rng.normal(0, 0.8, 2)
            lean_deg = 3.0
        else:
            k = i - still_frames
            wrist = hip + rim_rel + sequence[k % len(sequence)] + rng.normal(0, 1.5, 2)
            lean_deg = 11.0 + 8.0 * math.sin(k / 4)
        lean = math.radians(lean_deg)
        shoulder = hip + np.array([math.sin(lean) * 110 * facing, -math.cos(lean) * 110])
        elbow = (shoulder + wrist) / 2 + np.array([-15 * facing, 0])
        nose = shoulder + np.array([35 * facing, -45])
        pts = {"left_hip": hip, "left_shoulder": shoulder, "left_elbow": elbow, "left_wrist": wrist, "nose": nose,
               "right_hip": hip + [5, 0], "right_shoulder": shoulder + [5, 0]}
        return {n: LandmarkPoint(float(pts.get(n, hip)[0] / w), float(pts.get(n, hip)[1] / h), 0.0, 0.95 if n in pts else 0.1) for n in LANDMARK_NAMES}
    P.detect_landmarks = fake

for label, points in (("with rim", rim_points), ("no rim", None)):
    install_fake_with_still(make_side_sequence("semicircular"))
    r = P.analyze_video_bytes(write_video(14), view="side", filmed_side="left", rim_points=points)
    m, j = P.to_firestore(r)
    check(f"still->push ({label})", not r.warnings and 2 < m["trunk_static"] < 4 and 17 < m["trunk_peak"] < 20.5 and 13.5 < m["trunk_change"] < 17.5
          and 2.5 <= r.debug["baseline_seconds"][0] <= 3.3 and 5 <= r.stroke_count <= 8,
          f"static={m['trunk_static']} peak={m['trunk_peak']} change={m['trunk_change']} sweep={m['trunk_sweep']} baseline={r.debug['baseline_seconds']} strokes={r.stroke_count} rate={m['stroke_rate']} pattern={r.pattern} warn={r.warnings}")
    if points is None:
        check("  no rim: pattern from return height, rate from mean-crossing", r.pattern == "semicircular" and r.rim is None and m["contact_angle"] is None and 30 < m["stroke_rate"] < 45
              and {"push", "recovery", "still"} >= set(r.debug["phase"]), (r.pattern, m["contact_angle"], m["stroke_rate"], r.debug["pattern_votes"]))
    else:
        check("  with rim: pattern still classified", r.pattern == "semicircular", r.pattern)

# 사람이 화면 왼쪽을 볼 때도 앞으로 숙임이 +로 나오는지 (부호 확인)
install_fake_with_still(make_side_sequence("semicircular") * np.array([-1.0, 1.0]), facing=-1.0)
r = P.analyze_video_bytes(write_video(14), view="side", filmed_side="left")
m, j = P.to_firestore(r)
check("facing left: forward lean is still positive", 2 < m["trunk_static"] < 4 and 13.5 < m["trunk_change"] < 17.5, (m["trunk_static"], m["trunk_change"]))

# 6-2) 가만히 앉아만 있는 사람(손목이 조금 떨릴 뿐) → 미는 동작 0회, 패턴 보류, 경고
def install_fake_still_only():
    def fake(pose, frame):
        h, w = frame.shape[:2]
        hip = np.array([300.0, 260.0])
        wrist = hip + np.array([10.0, -35.0]) + rng.normal(0, 1.5, 2)
        shoulder = hip + np.array([math.sin(math.radians(3)) * 110, -math.cos(math.radians(3)) * 110])
        pts = {"left_hip": hip, "left_shoulder": shoulder, "left_elbow": (shoulder + wrist) / 2, "left_wrist": wrist, "nose": shoulder + [35, -45],
               "right_hip": hip + [5, 0], "right_shoulder": shoulder + [5, 0]}
        return {n: LandmarkPoint(float(pts.get(n, hip)[0] / w), float(pts.get(n, hip)[1] / h), 0.0, 0.95 if n in pts else 0.1) for n in LANDMARK_NAMES}
    P.detect_landmarks = fake

install_fake_still_only()
r = P.analyze_video_bytes(write_video(8), view="side", filmed_side="left")
check("sitting still: no strokes counted from jitter", r.stroke_count == 0 and r.pattern == "unknown" and r.warnings and "미는 동작을 찾지" in r.warnings[0], (r.stroke_count, r.pattern, r.warnings))

# 7) 정면: 왼손이 오른손보다 덜 움직이고 0.2초 늦음
def fake_front(pose, frame):
    i = fake_front.i; fake_front.i += 1
    h, w = frame.shape[:2]; t = i / 15
    hipL, hipR = np.array([330.0, 300]), np.array([390.0, 300])
    shL, shR = np.array([320.0, 180]), np.array([400.0, 186])   # 오른쪽 어깨가 살짝 낮음
    wL = np.array([290.0, 250 + 30 * math.sin(2 * math.pi * (t - 0.2) / 1.6)])
    wR = np.array([430.0, 250 + 45 * math.sin(2 * math.pi * t / 1.6)])
    pts = {"left_hip": hipL, "right_hip": hipR, "left_shoulder": shL, "right_shoulder": shR, "left_wrist": wL, "right_wrist": wR}
    return {n: LandmarkPoint(float(pts.get(n, hipL)[0] / w), float(pts.get(n, hipL)[1] / h), 0.0, 0.95 if n in pts else 0.1) for n in LANDMARK_NAMES}
fake_front.i = 0
P.detect_landmarks = fake_front
r = P.analyze_video_bytes(write_video(10), view="front")
m, j = P.to_firestore(r)
check("front metrics", not r.warnings and 0.6 < m["symmetry_index"] < 0.75 and 0.1 < m["timing_diff"] < 0.3 and 3 < m["shoulder_tilt"] < 6,
      f"{m}")
check("front: signed static shoulder tilt (right lower = negative)", -5.5 < m["shoulder_tilt_static"] < -3.5 and abs(m["lateral_lean_static"]) < 1, (m["shoulder_tilt_static"], m["lateral_lean_static"]))

# 8) 정면: 3초 가만히 → 카메라 쪽으로 다가옴(어깨 너비 60→150px). 왼쪽 어깨가 7° 낮고, 밀 때 몸통이 좌우로 흔들림
def fake_front_approach(pose, frame):
    i = fake_front_approach.i; fake_front_approach.i += 1
    h, w = frame.shape[:2]; t = i / 15
    grow = 1.0 if t < 3 else 1.0 + 0.3 * (t - 3)
    half = 30.0 * grow
    center = np.array([w / 2, h * 0.45])
    tilt = math.radians(7.0)
    sway = 0.0 if t < 3 else 6.0 * grow * math.sin(2 * math.pi * (t - 3) / 1.5)
    shL = center + np.array([half * math.cos(tilt) + sway, half * math.sin(tilt)])     # 사람의 왼쪽 = 화면 오른쪽, 더 낮음(y 큼)
    shR = center + np.array([-half * math.cos(tilt) + sway, -half * math.sin(tilt)])
    hipL, hipR = center + np.array([half * 0.8, half * 3.0]), center + np.array([-half * 0.8, half * 3.0])
    pts = {"left_hip": hipL, "right_hip": hipR, "left_shoulder": shL, "right_shoulder": shR}
    return {n: LandmarkPoint(float(pts.get(n, hipL)[0] / w), float(pts.get(n, hipL)[1] / h), 0.0, 0.95 if n in pts else 0.1) for n in LANDMARK_NAMES}
fake_front_approach.i = 0
P.detect_landmarks = fake_front_approach
r = P.analyze_video_bytes(write_video(8), view="front")
m, j = P.to_firestore(r)
check("front approach: still 3s then moving", not r.warnings and 6 < m["shoulder_tilt_static"] < 8 and 6 < m["shoulder_tilt_moving"] < 8
      and 2.5 <= r.debug["baseline_seconds"][0] <= 3.6 and m["lateral_lean_sway"] > 3 and abs(m["lateral_lean_static"]) < 1,
      f"{ {k: m.get(k) for k in ('shoulder_tilt_static', 'shoulder_tilt_moving', 'shoulder_tilt_sway', 'lateral_lean_static', 'lateral_lean_moving', 'lateral_lean_sway')} } base={r.debug['baseline_seconds']} warn={r.warnings} notes={r.notes}")
P.detect_landmarks = original
print("ALL PASS" if ok else "SOME FAILED")
