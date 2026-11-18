"""자세 사진(기준선 보정)과 실시간 코칭 프레임 처리를 검증한다."""
import math, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import posture, live_coach
from pose_utils import LandmarkPoint, LANDMARK_NAMES

ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(("OK  " if cond else "FAIL"), name, extra)

W, H = 720, 540
def rot(p, deg, c=(360, 270)):
    a = math.radians(deg); x, y = p[0] - c[0], p[1] - c[1]
    return (c[0] + x * math.cos(a) - y * math.sin(a), c[1] + x * math.sin(a) + y * math.cos(a))

def fake_for(points):
    def fake(pose, frame):
        h, w = frame.shape[:2]
        return {n: LandmarkPoint(points.get(n, (0, 0))[0] / w, points.get(n, (0, 0))[1] / h, 0, 0.95 if n in points else 0.1) for n in LANDMARK_NAMES}
    return fake

photo = cv2.imencode(".jpg", np.full((H, W, 3), 180, np.uint8))[1].tobytes()

# 정면: 실제 어깨 기울기 5°, 골반 3°, 폰을 10° 기울여 찍음
TILT = 10
body = {"left_shoulder": (300, 200), "right_shoulder": (300 + 120 * math.cos(math.radians(5)), 200 + 120 * math.sin(math.radians(5))),
        "left_hip": (310, 360), "right_hip": (310 + 100 * math.cos(math.radians(3)), 360 + 100 * math.sin(math.radians(3)))}
seat = [(250, 400), (470, 400)]
rotated = {k: rot(v, TILT) for k, v in body.items()}
posture.detect_landmarks = fake_for(rotated)
r = posture.analyze_photo_bytes(photo, view="front", seat_points=[rot(p, TILT) for p in seat])
check("front with seat line: shoulder 5°", abs(r.metrics["shoulder_tilt"].value - 5) < 0.5, r.metrics["shoulder_tilt"].value)
check("front with seat line: pelvis 3°", abs(r.metrics["pelvic_tilt"].value - 3) < 0.5, r.metrics["pelvic_tilt"].value)
r2 = posture.analyze_photo_bytes(photo, view="front", seat_points=None)
check("without seat line the tilt error shows (≈15°)", abs(r2.metrics["shoulder_tilt"].value - 15) < 0.5 and any("가로축" in n for n in r2.notes), r2.metrics["shoulder_tilt"].value)
check("annotated image RGB", r.annotated_frame is not None and r.annotated_frame.shape == (H, W, 3))

# 측면(오른쪽): 몸이 등받이 대비 20° 숙임, 등받이는 사진에서 8° 기울어 보임
back = [(200, 150), (200 + 250 * math.sin(math.radians(8)), 400)]
hip = (240, 380)
lean = math.radians(8 + 20)
shoulder = (hip[0] + 180 * math.sin(lean), hip[1] - 180 * math.cos(lean))
ear = (shoulder[0] + 40, shoulder[1] - 40)          # CVA 45°
posture.detect_landmarks = fake_for({"right_hip": hip, "right_shoulder": shoulder, "right_ear": ear})
# 등받이 선: 위(200,150) → 아래로 8° 기울어짐 (x가 아래로 갈수록 증가). 몸도 같은 방향 기준 +20°
back = [(200, 150), (200 - 250 * math.tan(math.radians(8)), 400)]
r = posture.analyze_photo_bytes(photo, view="side", backrest_points=back, filmed_side="right")
check("side analysis ok", r.metrics and not r.warnings, (r.warnings, {k: round(v.value, 1) for k, v in r.metrics.items()}))
tf = r.metrics["trunk_flexion"].value
check("side CVA 45°", abs(r.metrics["modified_cva"].value - 45) < 0.5, r.metrics["modified_cva"].value)
# 등받이 위쪽이 +x로 8° 기울고, 몸은 +x로 28° 기움 → 등받이 대비 숙임 20° (사진 세로축 기준이면 28°)
check("trunk flexion measured against backrest (20°)", abs(tf - 20) < 0.5, tf)
r_no = posture.analyze_photo_bytes(photo, view="side", backrest_points=None, filmed_side="right")
check("without backrest line → 28° vs image vertical", abs(r_no.metrics["trunk_flexion"].value - 28) < 0.5, r_no.metrics["trunk_flexion"].value)
r = posture.analyze_photo_bytes(photo, view="side", backrest_points=back, filmed_side="left")
check("side: wrong side selected → visibility warning", r.warnings and "잘 보이지" in r.warnings[0], r.warnings)

# 실시간 코칭: 가짜 관절 + 실제 그리기
live_coach.detect_landmarks = fake_for({"left_wrist": (400, 300), "left_shoulder": (300, 150), "left_hip": (280, 350)})
state = live_coach.LiveCoachState("left")
frame = np.zeros((480, 640, 3), np.uint8)
for i in range(40):
    out = live_coach.process_frame(frame, state, now=i * 0.1)
check("live: trail keeps only last 2s", len(state.trail) <= 21, len(state.trail))
check("live: lean computed", state.last_lean is not None and 5 < state.last_lean < 6.5, state.last_lean)
check("live: output frame", out.shape == frame.shape and out.any())
state.close()
# 실제 MediaPipe로 빈 화면 (사람 없음) → 에러 없이 통과
import pose_utils; live_coach.detect_landmarks = pose_utils.detect_landmarks
state = live_coach.LiveCoachState("right"); out = live_coach.process_frame(frame, state); state.close()
check("live: real mediapipe on empty frame", out.shape == frame.shape and state.last_lean is None)
check("cloud detection off locally", live_coach.is_cloud_environment() is False)
print("ALL PASS" if ok else "SOME FAILED")
