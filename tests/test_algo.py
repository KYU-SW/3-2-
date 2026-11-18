"""합성 손목 궤적으로 추진 분석 알고리즘을 검증한다."""
import math, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import propulsion as P

FPS = 15
R = 100.0            # 림 반지름(px)
C = np.array([0.0, 0.0])  # 림 중심 (hip 기준 상대 좌표라고 가정)

def stroke(kind, facing=1.0, push_frames=12, rec_frames=12, loop_height=0.8):
    """한 스트로크: 림 위쪽 호를 따라 뒤(-) → 앞(+)으로 밀고, kind 방식으로 복귀."""
    # 밀기: 각도 -150° → -30° (이미지 좌표에서 y가 음수 = 림 위쪽). 접촉 각도 120°.
    angles = np.linspace(math.radians(-150), math.radians(-30), push_frames)
    push = np.stack([np.cos(angles) * R * facing, np.sin(angles) * R], axis=1)
    start, end = push[0], push[-1]
    t = np.linspace(0, 1, rec_frames + 2)[1:-1]
    base = end[None, :] + (start - end)[None, :] * t[:, None]
    if kind == "semicircular":   # 림 아래로 크게 내려오며 복귀
        rec = base + np.stack([np.zeros_like(t), np.sin(t * math.pi) * 1.4 * R], axis=1)
    elif kind == "arc":          # 밀기 경로를 거의 그대로 되돌아감
        back = angles[::-1][1:-1]
        idx = np.linspace(0, len(back) - 1, rec_frames).astype(int)
        rec = np.stack([np.cos(back[idx]) * (R + 8) * facing, np.sin(back[idx]) * (R + 8)], axis=1)  # 림에서 떨어져(허용치 밖) 같은 호로 복귀
    elif kind == "single":       # 밀기 경로보다 위로 크게 돌아감
        rec = base + np.stack([np.zeros_like(t), -np.sin(t * math.pi) * loop_height * R], axis=1)
    elif kind == "double":       # 위로 갔다가 아래로 교차
        rec = base + np.stack([np.zeros_like(t), -np.sin(t * 2 * math.pi) * 0.6 * R], axis=1)
    return push, rec

def run(kind, n=6, facing=1.0):
    pts = []
    for _ in range(n):
        p, r = stroke(kind, facing)
        pts += list(p) + list(r)
    w = np.array(pts)
    phases = P.classify_phases(w, C, R, facing)
    strokes = P.split_strokes(phases)
    pats, angles = [], []
    for s, e, nx in strokes:
        pat, _ = P.classify_stroke_pattern(w[s:e], w[e:nx], C, R)
        pats.append(pat); angles.append(P.calculate_contact_angle(w[s:e], C))
    times = [i / FPS for i in range(len(w))]
    rate = P.calculate_stroke_rate([times[s] for s, _, _ in strokes], times[-1])
    return len(strokes), P.vote_pattern(pats), float(np.mean(angles)), rate

ok = True
expect_rate = 60 * FPS / 24  # 24프레임마다 1스트로크 → 37.5회/분
for kind, want in [("semicircular", "semicircular"), ("arc", "arc"), ("single", "single_looping_over"), ("double", "double_looping_over")]:
    for facing in (1.0, -1.0):
        n, pat, ang, rate = run(kind, facing=facing)
        good = n == 6 and pat == want and 90 < ang < 130 and abs(rate - expect_rate) < 3
        ok &= good
        print(f"{'OK ' if good else 'FAIL'} {kind:12s} facing={facing:+.0f} strokes={n} pattern={pat:22s} contact={ang:6.1f}° rate={rate:5.1f}/min")

# 판정: 기준값 None → 보류, NaN → 보류, 기준값 있으면 판정
P.THRESHOLDS["contact_angle"].update(normal_min=90, caution_min=70)
checks = [
    (P.judge_minimum("contact_angle", 100.0), "normal"),
    (P.judge_minimum("contact_angle", 80.0), "caution"),
    (P.judge_minimum("contact_angle", 50.0), "warning"),
    (P.judge_minimum("contact_angle", float("nan")), "pending"),
    (P.judge_maximum("stroke_rate", 99.0), "pending"),
    (P.judge_pattern("arc").coaching_key, "pattern_arc"),
    (P.judge_pattern("semicircular").judgment, "normal"),
]
for got, want in checks:
    ok &= got == want
    print(("OK  " if got == want else "FAIL"), "judge", got, "==", want)

# 대안 스트로크 분할: 손목 x가 평균선을 3번 지나면 1회 (Nourbakhsh et al., 2026)
for kind in ("semicircular", "arc", "single", "double"):
    pts = []
    for _ in range(6):
        p, r = stroke(kind)
        pts += list(p) + list(r)
    cycles = P.split_strokes_mean_crossing(np.array(pts)[:, 0])
    good = len(cycles) in (5, 6)  # 영상 앞뒤가 잘려 첫/마지막 주기는 빠질 수 있다
    ok &= good
    print(f"{'OK ' if good else 'FAIL'} mean-crossing {kind:12s} cycles={len(cycles)} (합성 6회)")
x = np.sin(np.linspace(0, 2 * math.pi * 4, 200))  # 사인파 4주기 → 교차 7번 → 3주기
got = len(P.split_strokes_mean_crossing(x))
ok &= got == 3
print(("OK  " if got == 3 else "FAIL"), "mean-crossing sine cycles", got, "== 3")
ok &= P.split_strokes_mean_crossing([1.0, 2.0]) == []

# 정지 구간 찾기: 3초(45프레임) 가만히 → 이후 손목이 크게 움직임. 몸통 길이 100px → 허용 15px
rng = np.random.default_rng(1)
still = np.array([20.0, -30.0]) + rng.normal(0, 1.0, (45, 2))
moving = np.array([[20 + 60 * math.sin(i / 4), -30 + 40 * math.cos(i / 4)] for i in range(90)])
wr = np.vstack([still, moving])
tt = [i / FPS for i in range(len(wr))]
end = P.find_baseline_end(wr, tt, 100.0)
good = 43 <= end <= 50
ok &= good; print(("OK  " if good else "FAIL"), "baseline end frame", end, "(정지 45프레임)")
end0 = P.find_baseline_end(moving, tt[:90], 100.0)
ok &= end0 == 0; print(("OK  " if end0 == 0 else "FAIL"), "no baseline when moving from start", end0)

# 정지 vs 동작 비교: 정지 2°, 밀 때 2~17° 사이를 오감 → peak≈17, change≈15, sweep≈15
lean = np.concatenate([np.full(45, 2.0), 9.5 + 7.5 * np.sin(np.arange(150) / 3)])
c = P.calculate_trunk_comparison(lean, 45)
good = abs(c["trunk_static"] - 2) < 0.1 and 15.5 < c["trunk_peak"] <= 17 and 13.5 < c["trunk_change"] <= 15 and 13 < c["trunk_sweep"] <= 15
ok &= good; print(("OK  " if good else "FAIL"), "trunk comparison", {k: round(v, 1) for k, v in c.items()})
c0 = P.calculate_trunk_comparison(lean[45:], 0)
good = math.isnan(c0["trunk_static"]) and math.isnan(c0["trunk_change"]) and c0["trunk_peak"] > 15
ok &= good; print(("OK  " if good else "FAIL"), "no baseline -> change is NaN")

# 림 없는 패턴 분류: 돌아올 때 높이 − 밀 때 높이 (몸통 길이 100 기준). 실제 영상에서 본 값 크기로 만든 궤적
def back_and_forth(return_offset, cycles=5, frames=12, width=60.0):
    """앞으로(x 증가) 갔다가 return_offset만큼 높이를 바꿔 뒤로 돌아오는 손목 궤적."""
    pts = []
    for _ in range(cycles):
        pts += [(-width / 2 + width * i / (frames - 1), 0.0) for i in range(frames)]
        pts += [(width / 2 - width * i / (frames - 1), return_offset * math.sin(math.pi * i / (frames - 1)) * 1.5) for i in range(frames)]
    return np.array(pts)

for offset, want in [(20.0, "semicircular"), (0.0, "arc"), (-30.0, "single_looping_over")]:
    got, reason, diff = P.classify_pattern_by_return_height(back_and_forth(offset), 100.0, 1.0)
    ok &= got == want
    print(("OK  " if got == want else "FAIL"), f"rim-free pattern offset={offset:+.0f} -> {got} (d={diff:+.2f}, {reason})")
got, _, _ = P.classify_pattern_by_return_height(back_and_forth(20.0)[:, ::1] * np.array([-1.0, 1.0]), 100.0, -1.0)
ok &= got == "semicircular"; print(("OK  " if got == "semicircular" else "FAIL"), "rim-free pattern facing left ->", got)
got, _, diff = P.classify_pattern_by_return_height(np.zeros((30, 2)), 100.0, 1.0)
ok &= got == "unknown" and math.isnan(diff); print(("OK  " if got == "unknown" else "FAIL"), "rim-free pattern no movement ->", got)

# 대상 추적: 화면 끝에서 잘린 프레임, 멀리 뛴 다른 사람, 오래 놓친 뒤의 프레임은 무효
def obs_at(t, hip_x, torso=60.0):
    hip = np.array([hip_x, 300.0]); sh = hip + np.array([0.0, -torso])
    return P.FrameObservation(t, {}, {"left_hip": hip, "left_shoulder": sh}, hip.copy(), True, 0.9)
seq = [obs_at(i / 15, 600 - 5 * i) for i in range(30)]            # 대상: 왼쪽으로 이동
seq.append(obs_at(30 / 15, 15.0))                                  # 화면 끝(720의 3% 안쪽) → 잘림
seq += [obs_at((31 + i) / 15, 150.0 + i) for i in range(10)]      # 갑자기 멀리 나타난 다른 사람 (계속 보임)
seq += [obs_at(5.0 + i / 15, 440.0) for i in range(5)]            # 1초 넘게 놓친 뒤 → 나간 것으로 봄
P.keep_tracked_subject(seq, "side", "left", 720.0)
flags = [o.valid for o in seq]
good = all(flags[:30]) and not any(flags[30:])
ok &= good; print(("OK  " if good else "FAIL"), "subject tracking keeps first person only", sum(flags), "/", len(flags))

# 선별 규칙
import screening as S
base = {"trunk_control": "yes", "pain": "0", "training": "yes", "fit": "recent", "fatigue": "fine"}
cases = [
    ("nothing notable -> no top", S.recommend(base, {"trunk_change": 2.0, "pattern": "semicircular"}).top, None),
    ("big trunk change -> technique", S.recommend(base, {"trunk_change": 14.0, "pattern": "semicircular"}).top.key, "technique"),
    ("no trunk control -> not counted", S.recommend({**base, "trunk_control": "no"}, {"trunk_change": 14.0, "pattern": "unknown"}).top, None),
    ("never fitted -> setup", S.recommend({**base, "fit": "never"}, {"trunk_change": 2.0}).top.key, "setup"),
    ("tired -> body", S.recommend({**base, "fatigue": "tired"}, {"trunk_change": None}).top.key, "body"),
    ("non-SC pattern + untrained -> technique", S.recommend({**base, "training": "no", "fit": "never"}, {"pattern": "arc"}).top.key, "technique"),
    ("front: left shoulder 7 deg lower -> setup", S.recommend(base, {"shoulder_tilt_static": 7.0}).top.key, "setup"),
    ("front: 3 deg is within error -> no top", S.recommend(base, {"shoulder_tilt_static": -3.0, "lateral_lean_static": 2.0}).top, None),
    ("front sentence", S.recommend(base, {"shoulder_tilt_static": -6.4}).top.video_reasons, ["가만히 있을 때 오른쪽 어깨가 6° 낮아요."]),
    ("front: lateral lean is not scored", S.recommend(base, {"lateral_lean_static": -6.4}).top, None),
    ("levels", [S.trunk_change_level(v) for v in (None, 4.0, 8.0, 12.0)], ["unknown", "small", "moderate", "notable"]),
    ("missing questions", S.missing_questions({"pain": "0", "training": "bogus"}), ["trunk_control", "training", "fit", "fatigue"]),
]
for name, got, want in cases:
    ok &= got == want
    print(("OK  " if got == want else "FAIL"), "screening:", name, "->", got)
r = S.recommend({**base, "trunk_control": "no", "pain": "7"}, {"trunk_change": 14.0})
good = len(r.notices) == 2 and "전문가 상담" in r.notices[0] and "필요한 움직임" in r.notices[1]
ok &= good; print(("OK  " if good else "FAIL"), "screening: pain notice first, trunk-control notice second", r.notices)
# 자세·움직임 키워드: 판별 → 질문으로 거르기 → 짝
import keywords as K
both = K.detect({"trunk_control": "yes", "fit": "never", "training": "yes", "fatigue": "little"},
                {"head_cva_static": 40.0, "trunk_change": 14.0, "pattern": "arc"},
                {"shoulder_tilt_static": 6.0, "shoulder_tilt_moving": 10.0})
keys = [k.key for k in both]
for name, got, want in [
    ("keywords found", sorted(keys), sorted(["forward_head", "trunk_collapse", "push_pattern", "shoulder_drop", "push_side_drop"])),
    ("pairs marked together and shown first", [k.together for k in both][:4], [True, True, True, True]),
    ("fit=never adds setup note to shoulder keywords", any("세팅 점검" in n for k in both if k.key == "shoulder_drop" for n in k.notes), True),
    ("trunk_control=no removes trunk collapse", "trunk_collapse" in [k.key for k in K.detect({"trunk_control": "no"}, {"trunk_change": 14.0}, None)], False),
    ("opposite-side moving tilt is not a side drop", "push_side_drop" in [k.key for k in K.detect({}, None, {"shoulder_tilt_static": 6.0, "shoulder_tilt_moving": -10.0})], False),
    ("trunk side lean + head tilt (front, 7°·7.5°)", sorted(k.key for k in K.detect({}, None, {"lateral_lean_static": -7.0, "head_tilt_static": 7.5})), ["head_side_tilt", "trunk_side_lean"]),
    ("straight-sitting range (5.4°, 4.4°) not flagged", K.detect({}, None, {"lateral_lean_static": -5.4, "head_tilt_static": 4.4}), []),
    ("below thresholds: nothing", K.detect({}, {"head_cva_static": 70.0, "trunk_change": 5.0, "pattern": "semicircular"}, {"shoulder_tilt_static": 2.0, "shoulder_tilt_moving": 3.0}), []),
]:
    ok &= got == want
    print(("OK  " if got == want else "FAIL"), "keywords:", name, "->", got)

# 휴대폰 기울기 보정: 뒷면 카메라가 시계 방향으로 2.5° 돌면 어깨·몸통이 −2.5° 치우쳐 보임 → +2.5° 되돌림
for facing, roll, want in (("environment", 2.5, 0.0), ("user", -2.5, 0.0), ("environment", -3.0, 5.0)):
    roll_result = P.PropulsionAnalysisResult("front", None, 10.0, 1.0)
    measured = {"environment": -2.5, "user": -2.5}[facing] if want == 0.0 else 8.0
    roll_result.metrics = {"shoulder_tilt_static": P.MetricResult(measured), "lateral_lean_static": P.MetricResult(measured), "shoulder_tilt_sway": P.MetricResult(4.0)}
    roll_result.debug = {"shoulder_tilt_series": [measured, measured]}
    P.correct_for_camera_roll(roll_result, roll, facing)
    got = (round(roll_result.metrics["shoulder_tilt_static"].value, 3), round(roll_result.metrics["lateral_lean_static"].value, 3), roll_result.metrics["shoulder_tilt_sway"].value, round(roll_result.debug["shoulder_tilt_series"][0], 3))
    good = got == (want, want, 4.0, want)
    ok &= good
    print(("OK  " if good else "FAIL"), f"camera roll correction {facing} roll={roll}", got)

# 촬영 점검: 거리·밝기·관절 보임
def film_labels(checks: list) -> list:
    return [(status, label) for status, label, _ in checks]
film_good = P.assess_filming_basic([0.16] * 20, [(160.0, 90.0)] * 5, 1.0, "side")
film_bad = P.assess_filming_basic([0.05] * 20, [(160.0, 10.0)] * 5, 0.6, "side")
film_front_still = P.assess_filming_basic([0.07] * 20, [(160.0, 90.0)] * 5, 1.0, "front")
for name, got, want in [
    ("film good", film_labels(film_good), [("ok", "거리"), ("ok", "밝기"), ("ok", "관절 보임")]),
    ("film far/dark/hidden", film_labels(film_bad), [("warn", "거리"), ("warn", "밝기"), ("warn", "관절 보임")]),
    ("film front not approaching", ("warn", "다가오기") in film_labels(film_front_still), True),
]:
    ok &= got == want
    print(("OK  " if got == want else "FAIL"), name, "->", got)

# 올린 영상 점검: 배경 세로선 기울기 · 흔들림 (camera_check)
import cv2 as _cv2
import camera_check
def _lines_frame(deg, shift=(0, 0)):
    img = np.full((640, 360, 3), 200, np.uint8)
    dots = np.random.default_rng(1)  # 추적할 점(배경 무늬)
    for _ in range(300):
        x, y = int(dots.integers(0, 360)), int(dots.integers(0, 640))
        _cv2.circle(img, (x, y), 2, (90, 90, 90), -1)
    for x in range(30, 360, 60):
        _cv2.line(img, (x, 20), (x, 620), (40, 40, 40), 3)
    M = _cv2.getRotationMatrix2D((180, 320), deg, 1.0)
    M[:, 2] += shift
    return _cv2.warpAffine(img, M, (360, 640), borderValue=(200, 200, 200))
for want_deg in (0.0, 6.0):
    cam = camera_check.CameraCheck()
    for i in range(12):
        cam.update(_lines_frame(want_deg), {}, float(i))
    got_deg = cam.tilt_deg()
    good = got_deg is not None and abs(abs(got_deg) - want_deg) < 1.5
    ok &= good
    print(("OK  " if good else "FAIL"), f"camera tilt {want_deg}°", "->", got_deg)
cam = camera_check.CameraCheck()
rng = np.random.default_rng(0)
for i in range(15):
    cam.update(_lines_frame(0.0, tuple(rng.normal(0, 6, 2))), {}, i / 15)
shaky = cam.shake_ratio(0.006)
ok &= shaky is not None and shaky > 0.5
print(("OK  " if shaky and shaky > 0.5 else "FAIL"), "camera shake detected ->", shaky)
levels = [K.forward_head_level(a) for a in (80.0, 70.0, 55.0, 45.0)]
ok &= levels == [0, 1, 2, 3]
print(("OK  " if levels == [0, 1, 2, 3] else "FAIL"), "forward head FNTA levels (app = 90 - FNTA)", levels)
fh2 = K.detect({}, {"head_cva_static": 55.0}, None)[0]
fh3 = K.detect({}, {"head_cva_static": 45.0}, None)[0]
good = fh2.weight < 10 < fh3.weight and "경향" in fh2.finding and "3단계" in fh3.finding
ok &= good
print(("OK  " if good else "FAIL"), "forward head: level 2 = 권장(뒤로), level 3 = 먼저", fh2.weight, fh3.weight)
no_tired = [k.key for k in K.detect({"fatigue": "fine"}, {"trunk_change": 14.0}, None)]
tired = [k.key for k in K.detect({"fatigue": "tired"}, {"trunk_change": 14.0}, None)]
good = "trunk_collapse" not in no_tired and "trunk_collapse" in tired
ok &= good
print(("OK  " if good else "FAIL"), "trunk collapse needs fatigue (Rodgers 2000)", no_tired, tired)
print("ALL PASS" if ok else "SOME FAILED")
