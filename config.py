"""Wheelance의 공통 상수와 판정 기준 설정."""

from __future__ import annotations

from typing import Final

APP_NAME: Final[str] = "Wheelance"
APP_TAGLINE: Final[str] = "앉은 자세와 미는 동작을 함께 보는 휠체어 자세 선별"
SUPPORTED_USERS_NOTICE: Final[str] = (
    "양손으로 추진하는 수동 휠체어 사용자를 위한 프로토타입입니다. "
    "편마비 한손·한발 추진자와 전동 휠체어 사용자는 이번 범위에서 제외합니다."
)
MEDICAL_DISCLAIMER: Final[str] = (
    "본 결과는 의료 진단이 아니라 참고용 선별 결과입니다. 정확한 평가와 교정은 재활의학과 전문의 또는 "
    "작업/물리치료사와 상담하세요."
)
LANDMARK_NOTICE: Final[str] = "MediaPipe 랜드마크는 해부학적 지점의 근사치입니다."

# 영상 전처리 상수: 무료 서버 메모리 제약을 고려한 제품 요구사항입니다.
VIDEO_MAX_SECONDS: Final[int] = 20
VIDEO_MIN_SECONDS: Final[int] = 5
VIDEO_SAMPLE_FPS: Final[int] = 15
VIDEO_LONG_SIDE_PX: Final[int] = 720
# 정면 영상은 멀리(4~5m)서 시작해 사람이 작게 찍힌다. 720으로 줄이면 어깨 너비가 20~30px뿐이라
# 1px 오차가 2~3°가 된다. 그래서 정면만 더 큰 크기로 분석한다.
VIDEO_LONG_SIDE_PX_FRONT: Final[int] = 1280
MIN_LANDMARK_VISIBILITY: Final[float] = 0.5
MIN_VALID_FRAME_RATIO: Final[float] = 0.6
WRIST_SMOOTHING_WINDOW: Final[int] = 5
DEFAULT_RIM_DIAMETER_INCH: Final[float] = 24.0
CM_PER_INCH: Final[float] = 2.54

# 알고리즘 초기값: 의학적 기준이 아니며 실제 영상으로 조정이 필요합니다.
ALGORITHM_PARAMS: Final[dict[str, object]] = {
    # 손목과 림 원 둘레 사이 거리 허용치(림 반지름 대비).
    # 손은 림을 잡지만 MediaPipe는 손목을 찍으므로 손 길이만큼 여유를 둡니다.
    "rim_contact_tolerance_ratio": 0.30,
    # 이 프레임 수 이상 연속으로 밀어야 한 번의 스트로크로 인정합니다.
    "min_push_frames": 3,
    # 밀기 구간 사이의 짧은 끊김(프레임)은 같은 스트로크로 이어 붙입니다.
    "push_gap_tolerance_frames": 1,
    # 복귀 중 손목이 밀 때보다 림 바깥/안쪽으로 벗어났다고 볼 최소 거리(림 반지름 대비).
    "loop_crossing_min_distance_ratio": 0.15,
    # 복귀 구간 중 이 비율 이상이 림 안쪽이면 semicircular로 판정.
    "below_path_ratio": 0.5,
    # 정면 영상에서 밀기 시작(손 최고점)으로 볼 최소 간격(초)과 높이 변화 비율.
    "front_peak_min_separation_sec": 0.4,
    "front_peak_prominence_ratio": 0.2,
    # 대표 프레임을 뽑을 영상 위치(0~1).
    "representative_positions": (0.2, 0.5, 0.8),
    # 대안 스트로크 분할: 손목 x좌표가 평균선을 이 횟수만큼 지나면 1회 (Nourbakhsh et al., 2026).
    "mean_crossings_per_cycle": 3,
    # 정지 자세 기준(영상 앞부분): 손목이 처음 위치에서 (몸통 길이 × 이 비율) 안에 머무는 동안을 "가만히 있음"으로 본다.
    "baseline_still_ratio": 0.15,
    # 정지 구간이 이 시간(초)보다 짧으면 기준으로 쓰지 않는다 (안내는 "3초 가만히").
    "baseline_min_seconds": 1.5,
    # 정지 구간으로 쓸 최대 시간(초).
    "baseline_max_seconds": 10.0,
    # 밀 때 "가장 숙였을 때/폈을 때"를 한두 프레임의 튐에 흔들리지 않게 백분위수로 잡는다.
    "dynamic_peak_percentile": 95,
    # --- 손 궤적 패턴 (림 없이): 손목이 뒤로 돌아올 때의 높이 − 앞으로 밀 때의 높이, 몸통 길이 대비 ---
    # 2026-10-02 실제 휠체어 영상 16개(반원형 6, 호형 6, 루프 4)에서 정한 값. 16개 중 13개를 맞힘.
    # 같은 영상으로 정한 값이라 낙관적이다. 다른 영상으로 다시 확인할 것.
    # 이 값보다 더 낮게 돌아오면 반원형 (실측: 반원형 평균 +0.115, 호형 −0.025)
    "pattern_return_lower_ratio": 0.08,
    # 이 값보다 더 높게 돌아오면 루프형 (실측: 루프 평균 −0.234)
    "pattern_return_higher_ratio": 0.12,
    # 한 프레임에 손목이 (몸통 길이 × 이 비율)보다 덜 움직이면 "멈춤"으로 보고 패턴 계산에서 뺀다.
    "pattern_min_speed_ratio": 0.01,
    # 패턴을 판단하려면 앞으로/뒤로 움직인 프레임이 각각 이만큼은 있어야 한다.
    "pattern_min_frames": 5,
    # 손목이 앞뒤로 (몸통 길이 × 이 비율) 이상 움직여야 미는 동작으로 센다 (실제 추진은 0.59~1.16배).
    "stroke_min_swing_ratio": 0.25,
    # --- 측면 영상에서 분석에서 빼는 화면 가로 구간 (화면 너비 대비 비율) ---
    # 고정 카메라 앞을 지나가면 화면 끝에서는 사람을 비스듬히 보게 된다.
    # 2026-10-02 의자 영상(가만히 앉은 사람을 밀어서 지나가기, 2개)에서 잰 결과:
    #   나가는 쪽(뒤에서 비스듬히 보이는 쪽) 끝 20%에서만 상체 각도가 6~9° 더 뒤로 기운 것처럼 나왔고,
    #   들어오는 쪽 끝과 가운데는 5~6° 안에서 같았다 (표준편차 약 1.7°).
    # 그래서 나가는 쪽 20%만 빼고, 들어오는 쪽은 몸이 잘리지 않는 한 쓴다.
    "side_exit_cut_ratio": 0.2,
    "side_entry_cut_ratio": 0.03,
    # --- 정면 영상의 정지 구간 ---
    # 카메라 쪽으로 다가오면 어깨 너비(픽셀)가 커진다. 처음 값의 이 배수를 넘어 계속 커지면 "출발"로 본다.
    "front_approach_scale_ratio": 1.10,
    # --- 앉은 사람만 대상으로 삼기 (측면) ---
    # 휠체어 사용자는 항상 앉아 있으므로, 무릎이 골반과 비슷한 높이일 때만 대상으로 본다.
    # |무릎 y − 골반 y| < 몸통 길이 × 이 값. 뒤에서 미는 사람이나 지나가는 사람(서 있음)을 걸러낸다.
    "seated_knee_hip_ratio": 0.45,
    # --- 대상 추적: 지나가는 다른 사람이나 화면 끝에서 잘린 몸을 걸러낸다 ---
    # 직전 프레임 대비 기준점(hip)이 (몸 크기 × 이 값 × 지난 프레임 수)보다 멀리 뛰면 다른 사람으로 본다.
    "track_max_jump_ratio": 0.8,
    "track_max_gap_frames": 3,
    # 직전 프레임 대비 몸 크기가 이 범위를 벗어나면 다른 사람이나 잘못된 검출로 본다.
    "track_scale_range": (0.6, 1.6),
    # 대상을 이 시간(초) 넘게 놓치면 화면 밖으로 나간 것으로 보고 그 뒤는 분석하지 않는다.
    "track_lost_seconds": 1.0,
    # 어깨·골반이 화면 가로 끝에서 이 비율 안쪽에 있어야 몸이 잘리지 않은 것으로 본다.
    "frame_edge_margin_ratio": 0.03,
    # --- 촬영 점검 (찍은 영상에 대한 피드백) ---
    # 2026-10-02 실제 영상 28개(잘 찍힌 영상)에서 잰 범위를 바탕으로 한 어림값. 잘못 찍은 영상으로는 아직 확인 안 함.
    # 몸 크기 = 측면: 몸통 길이 ÷ 화면 짧은 변 (실측 0.14~0.18, 약 3m)
    "film_side_size_range": (0.09, 0.30),
    # 정면: 처음 어깨 너비 ÷ 화면 짧은 변 (실측 0.058~0.081, 약 4m 출발)
    "film_front_size_range": (0.04, 0.15),
    # 정면: 다가오면서 어깨 너비가 처음의 이 배수 이상 커져야 "카메라 쪽으로 왔다"로 본다 (실측 1.7배 이상)
    "film_front_approach_ratio": 1.3,
    # 밝기(0~255). 몸 부분(어깨·골반·무릎 상자) 밝기는 옷 색에 크게 좌우된다:
    # 잘 찍힌 영상 28개에서 몸 31~102, 몸÷전체 0.20~0.65 (검은 옷도 관절 검출 100%).
    # 그래서 아주 극단적인 경우만 안내한다.
    "film_whole_min_brightness": 40.0,   # 화면 전체가 이보다 어두우면 "어두움" (실측 135~166)
    "film_body_min_brightness": 20.0,    # 몸 부분이 이보다 어두우면 "어두움"
    "film_backlight_ratio": 0.15,        # 몸÷전체가 이보다 낮으면 "역광"
    # 관절이 잘 보인 프레임 비율이 이보다 낮으면 안내 (분석을 막는 기준은 MIN_VALID_FRAME_RATIO)
    "film_valid_ratio_good": 0.85,
    # 측면에서 이 횟수보다 적게 밀면 안내
    "film_min_strokes": 2,
    # 찍은 방향 점검: 어깨 너비 ÷ 어깨~골반 길이 (화면 기준, 중앙값).
    # 실제 옆 영상 2개(2026-10-01)가 0.05·0.12(90백분위 0.08·0.22), 정면이면 대략 0.8 안팎.
    # 옆 영상이 이보다 크면 "비스듬히 찍힘", 앞 영상이 이보다 작으면 "옆으로 돌아 찍힘". 어림값.
    "film_side_max_shoulder_ratio": 0.4,
    # 올린 영상의 휴대폰 기울기(배경 세로선)·흔들림(배경 이동) 기준 (2026-11-10 시험)
    #   정상 영상 28개: 기울기 0~1.8°, 흔들린 프레임 0% → 경고 0개.
    #   일부러 돌린 영상: 3°→4.0°, 5°→5.8·7.7°, 8°→8.8° (실제보다 1~3° 크게 읽음 → 안내는 '약 N°').
    #   일부러 흔든 영상: 흔들린 프레임 28~85% → 모두 경고. 실제로 잘못 찍은 영상으로는 아직 확인 전.
    "film_tilt_warn_deg": 3.0,
    "film_shake_shift": 0.006,   # 한 프레임 사이에 배경이 화면 너비의 0.6% 이상 움직이면 '흔들린 프레임'
    "film_shake_ratio": 0.2,     # 흔들린 프레임이 20% 이상이면 안내
    "film_front_min_shoulder_ratio": 0.5,
    "weight_shift_interval_min": 20,
}

# 의학적 판정 기준. 근거가 확정되지 않은 값은 TODO로 남기며 판정을 보류합니다.
THRESHOLDS: Final[dict[str, dict[str, object]]] = {
    "contact_angle": {
        "normal_min": None,
        "caution_min": None,
        "source": "TODO: 가이드라인 원문 확인 필요",
    },
    "stroke_rate": {
        "normal_max": None,
        "caution_max": None,
        "source": "TODO: 가이드라인 원문 확인 필요",
    },
    "trunk_lean": {
        "normal_max": None,
        "caution_max": None,
        "source": "TODO: 가이드라인 원문 확인 필요",
    },
    "elbow_angle": {
        "recommended_min": None,
        "recommended_max": None,
        "source": "TODO: 가이드라인 원문 확인 필요",
    },
    "symmetry_index": {
        "normal_min": None,
        "caution_min": None,
        "source": "TODO: 출처 입력",
    },
    "timing_diff": {
        "normal_max": None,
        "caution_max": None,
        "source": "TODO: 출처 입력",
    },
    "shoulder_tilt": {
        "normal_max": None,
        "caution_max": None,
        "source": "TODO: 출처 입력",
    },
    "pelvic_tilt": {
        "normal_max": None,
        "caution_max": None,
        "source": "TODO: 출처 입력",
    },
    "modified_cva": {
        "normal_min": None,
        "caution_min": None,
        "source": "TODO: 출처 입력 (서 있는 자세 연구 기준값이면 좌식 적용은 참고치로 표기)",
    },
    "lateral_lean": {
        "normal_max": None,
        "caution_max": None,
        "source": "TODO: 출처 입력",
    },
}

JUDGMENT_PENDING: Final[str] = "pending"
JUDGMENT_LABELS: Final[dict[str, str]] = {
    "normal": "✓ 정상",
    "caution": "! 주의",
    "warning": "✕ 경고",
    JUDGMENT_PENDING: "– 판정 보류(기준값 미설정)",
}

PATTERN_LABELS: Final[dict[str, str]] = {
    "arc": "arc",
    "semicircular": "semicircular",
    "single_looping_over": "single-looping over",
    "double_looping_over": "double-looping over",
    "unknown": "판정 보류",
}

COACHING_MESSAGES: Final[dict[str, str]] = {
    "contact_angle_low": "바퀴를 더 뒤쪽에서 잡고, 앞쪽까지 길게 밀어보세요.",
    "stroke_rate_high": "짧게 여러 번보다 길게 한 번 미는 것이 어깨 부담을 줄입니다.",
    "pattern_arc": "바퀴를 놓은 뒤 손을 림 아래로 자연스럽게 떨어뜨리며 되돌아오세요.",
    "pattern_single_looping": "복귀할 때 손이 림 위로 크게 돌아가요. 손을 아래로 떨어뜨리며 반원을 그리듯 되돌아오세요.",
    "pattern_double_looping": "복귀 동작이 복잡해요. 바퀴를 놓은 뒤 손을 림 아래로 내려 한 번에 반원을 그리며 돌아오세요.",
    "trunk_lean_high": "밀 때마다 몸을 숙이는 습관이 있어요. 상체를 등받이 쪽에 두고 팔로 밀어보세요.",
    "asymmetry_high": "한쪽 팔을 더 많이 쓰고 있어요. 양팔을 같은 길이로 밀어보세요.",
}

# ---------------------------------------------------------------------------
# 선별(스크리닝): 질문 + 정지 vs 동작 비교 → "먼저 확인해 볼 것"
# ---------------------------------------------------------------------------

# 질문 5개. 검증된 설문에서 가져온 문항이 아니라 이 프로토타입용으로 만든 초안이다.
# (통증 문항을 WUSPI 등 기존 설문으로 바꾸려면 원문과 사용 조건을 먼저 확인할 것)
# 문항 작성 원칙 (2026-10-29): 누가 읽어도 같은 장면을 떠올리게 한다.
#  - "가끔·오래·힘들다" 같은 느낌 말 대신 기간(최근 1주일, 1년), 행동(팔짱 끼고, 쉬지 않고 5분), 사람(치료사·보조기기센터)으로 기준을 준다.
#  - 답은 서로 겹치지 않고, 하나만 고를 수 있게 한다.
#  - 키(yes/hard/no …)는 screening.py 점수 규칙과 저장된 기록이 쓰므로 바꾸지 않는다.
#  - 기존 검증 설문(WUSPI 등)이 아니라 이 앱용으로 만든 문항이다.
SCREENING_QUESTIONS: Final[tuple[dict[str, object], ...]] = (
    {
        "id": "trunk_control",
        "text": "휠체어에 앉아 팔짱을 낀 채(손으로 아무것도 잡지 않고) 상체를 앞으로 숙였다가 다시 똑바로 일으켜 보세요. 어떤가요?",
        "options": {
            "yes": "혼자 바로 일으킬 수 있어요",
            "hard": "일으킬 수는 있지만 몸이 흔들리거나 여러 번 시도해야 해요",
            "no": "손으로 짚거나 잡아야만 일으킬 수 있어요",
        },
    },
    {
        # 통증 숫자 척도(NRS 0~10): 의료 현장 표준, 사용 허락 불필요. 2026-11-05 "있음/가끔/밀 때마다"에서 교체.
        "id": "pain",
        "text": "최근 1주일 동안 어깨·목·손목 통증이 가장 심했을 때는 몇 점이었나요? (0 = 통증 없음, 10 = 상상할 수 있는 가장 심한 통증)",
        "options": {str(score): str(score) for score in range(11)},
    },
    {
        "id": "training",
        "text": "치료사나 강사에게 휠체어 미는 방법(손 잡는 위치, 미는 길이 등)을 직접 배운 적이 있나요?",
        "options": {"yes": "배운 적 있어요", "no": "배운 적 없어요 / 기억나지 않아요"},
    },
    {
        "id": "fit",
        "text": "지금 쓰는 휠체어를 치료사나 보조기기센터에서 내 몸에 맞게 조정(좌석 높이·등받이·바퀴 위치)받은 적이 있나요?",
        "options": {
            "recent": "1년 안에 받았어요",
            "old": "받았지만 1년이 넘었어요",
            "never": "받은 적 없어요 / 모르겠어요",
        },
    },
    {
        "id": "fatigue",
        "text": "평평한 길에서 쉬지 않고 5분 동안 휠체어를 밀면 어떤가요?",
        "options": {
            "fine": "5분 내내 밀 수 있고, 끝나도 숨이 차지 않아요",
            "little": "5분은 밀 수 있지만, 끝나면 숨이 차거나 팔이 무거워요",
            "tired": "5분이 되기 전에 쉬어야 해요",
        },
    },
)

# 선별 규칙에 쓰는 수치. 의학적 판정 기준이 아니라 "먼저 볼 순서"를 정하는 가설 값이다.
SCREENING_PARAMS: Final[dict[str, float]] = {
    # 통증 NRS 구간: 척수손상 성인 6,096명 분석에서 가장 잘 맞은 구간 1~3 가벼움 / 4~6 중간 / 7~10 심함
    # (SCI Model Systems, PubMed 21353824 · 초록 확인). 어깨 통증만 따로 검증한 값은 아님.
    "pain_mild_min": 1.0,
    "pain_moderate_min": 4.0,
    # 같은 사람의 통증 점수가 이만큼 바뀌어야 의미 있는 변화 (척수손상 82명, 약 1.8점 · 2006 연구, 2차 자료 확인)
    "pain_meaningful_change": 2.0,
    # 이보다 작은 정지-동작 차이는 측정 흔들림과 구분하지 않는다.
    # 근거: 2026-10-02 의자 영상 — 자세를 바꾸지 않은 사람이 화면을 지나가는 동안
    #   상체 각도가 5~6° 범위(5~95백분위) 안에서 움직였다. 이 범위 안의 차이는 해석하지 않는다.
    #   (문헌의 2D 영상 측정 오차 ±3~4°와도 맞는 크기)
    "trunk_change_min_detectable_deg": 6.0,
    # 이 이상이면 "밀 때 상체가 많이 숙여짐"으로 본다.
    # 근거: 평지 추진 중 몸통 굴곡 변화 7.37±3.26°(경험자, N=7, IMU 측정) 평균+1SD = 10.63
    #   Symonds A et al. (2016). J Rehabil Assist Technol Eng 3:2055668316678362.
    # 주의: 측정 방식(IMU vs 영상)과 대상 수가 달라 그대로 옮길 수 없는 참고치다.
    "trunk_change_notable_deg": 10.6,
    # 정면에서 가만히 있을 때 어깨선·몸통이 이 각도 이상 기울면 "뚜렷하게 기울어 있음"으로 본다.
    # 근거: 2D 영상 측정 오차 ±3~4°(Brink 2013, Sánchez 2017)보다 큰 값으로 정함. 의학적 기준이 아니다.
    #   2026-10-02 실제 정면 영상: 일자 10개 −4.1~+0.8°, 일부러 비대칭 2개 +6.8°, +7.9°.
    #   휴대폰이 좌우로 기울면 그만큼 치우치므로(일자 영상이 −2~−3° 치우침) 경계 근처 값은 조심해서 본다.
    "frontal_notable_deg": 5.0,
}

# 정면 촬영 안내. 거리·높이는 촬영 때 안내한 어림값이며 검증된 수치가 아니다.
# (실제 정면 영상에서 5~6m는 사람이 너무 작게 찍혀 약 4m로 줄였다)
FRONT_GUIDE_STEPS: Final[tuple[str, ...]] = (
    "**앞에 세워두기** — 휴대폰을 세로로, 가슴 높이쯤(바닥에서 약 80~100cm)에 세워두세요.",
    "**수평 맞추기** — 휴대폰이 좌우로 기울지 않게 해주세요. 화면 속 바닥선이나 벽의 가로선이 수평이면 돼요.",
    "**촬영** — 약 4m 앞에서 카메라를 보고 **3초 가만히** 있다가, 평소 속도로 카메라 쪽으로 밀고 와서 2m 앞에서 멈추세요.",
)
FRONT_GUIDE_TIP: Final[str] = (
    "화면에는 **한 사람만** 나오게 해주세요. 휴대폰이 기울어 있으면 그만큼 어깨도 기운 것으로 나와요."
)

# 먼저 확인해 볼 것 3가지와 안내 문구
SCREENING_LEVERS: Final[dict[str, dict[str, str]]] = {
    "technique": {
        "label": "미는 기술",
        "guide": "길게 밀고, 손을 놓은 뒤에는 아래로 떨어뜨리며 돌아오는 방법을 연습해 보세요. 가능하면 치료사에게 미는 방법을 배워보세요.",
    },
    "body": {
        "label": "몸 (근력·유연성)",
        "guide": "운동 탭의 운동을 무리 없는 범위에서 해보세요. 금방 지치거나 뻣뻣한 곳이 있으면 치료사와 상담해 보세요.",
    },
    "setup": {
        "label": "휠체어 세팅",
        "guide": "좌석·등받이 높이와 바퀴 위치가 몸에 맞는지 담당 치료사나 보조기기센터에서 점검받아 보세요.",
    },
}

SCREENING_NOTICE: Final[str] = (
    "이 안내는 영상과 답변을 정해진 규칙으로 합친 참고용 결과예요. 원인을 진단하는 것이 아니에요."
)

SOURCE_GUIDELINE: Final[str] = (
    "Consortium for Spinal Cord Medicine (2005). Preservation of upper limb function "
    "following spinal cord injury: A clinical practice guideline for health-care professionals. "
    "Journal of Spinal Cord Medicine, 28(5), 433–470."
)

SOURCE_CV_PATTERN: Final[str] = (
    "Nourbakhsh S, Archambault PS, Martinez Rocha J, Monacelli E (2026). Computer Vision-Based "
    "Classification of Manual Wheelchair Propulsion Patterns. IEEE Trans Neural Syst Rehabil Eng, "
    "34, 2786–2794. doi:10.1109/TNSRE.2026.3697633"
)
SOURCE_CADENCE: Final[str] = (
    "Kim S, Park J, Eun SD, Kang D (2025). Effects of cadence control on upper-limb kinematics and "
    "muscle activation during manual wheelchair propulsion in individuals with spinal cord injury. "
    "Life, 15(12), 1885."
)

# 측면 촬영 안내: 휴대폰을 세워두고 그 앞을 지나가는 방식 (혼자 찍을 수 있음).
# 2026-10-02 실제 영상(휠체어 16개, 의자 대조 4개)에서 확인한 것:
#   - 1배 줌, 고정 카메라로 화면을 가로질러 지나가도 상체 각도는 나가는 쪽 끝 20%만 빼면 5~6° 안에서 안정적
#   - 멀리서 2배 줌으로 찍은 영상이 더 좋다는 근거는 없었음 (역광과 섞여 판단 불가)
#   - 역광(창문을 등짐)과 화면에 두 사람이 같이 나오는 것은 검출을 망침
# "약 3m", "무릎 높이"는 촬영 때 안내한 어림값이며 실제 거리는 재지 않았다.
# "바퀴가 원으로 보이면 평행"은 기하학적 사실(원을 비스듬히 보면 타원)이다.
# CAMERA_GUIDE_DETAIL: 참고한 논문(Nourbakhsh et al., 2026, 제자리 롤러)의 카메라 배치 — 우리 방식과 다름.
CAMERA_GUIDE_STEPS: Final[tuple[str, ...]] = (
    "**옆에 세워두기** — 휴대폰을 가로로, 지나갈 길에서 옆으로 약 3m 떨어진 곳에 세워두세요. (삼각대나 낮은 의자 위)",
    "**낮게** — 카메라를 바퀴 중심 높이에 맞추세요. (바닥에서 약 30cm, 무릎 높이쯤)",
    "**확인** — 화면 가운데를 지날 때 바퀴가 **동그란 원**으로 보이면 OK. 머리부터 바퀴 아래까지 다 나와야 해요.",
    "**촬영** — 화면 한쪽 끝 안에서 **3초 가만히 앉아 있다가**, 평소 속도로 밀어서 반대쪽 끝으로 지나가세요.",
)
CAMERA_GUIDE_TIP: Final[str] = (
    "화면에는 **한 사람만** 나오게 해주세요. 창문이나 밝은 빛을 등지고 찍으면 몸이 검게 나와 분석이 잘 안 돼요. "
    "확대(줌)는 쓰지 않아도 돼요."
)
CAMERA_GUIDE_DETAIL: Final[str] = (
    "바퀴–카메라 거리 130±30cm · 카메라 높이 30±10cm(바퀴 반지름 정도) · 바퀴 면과 평행(±5°)"
)

# 추진 빈도 참고치: 판정 기준이 아니라 표시용.
# 50회/분: Kim et al. (2025)이 "일상 추진 속도를 대표하는 중간 cadence"로 정한 메트로놈 실험 조건(50 bpm)
# (67회/분은 원출처 미상의 2차 인용이라 사용하지 않음)
STROKE_RATE_REFERENCE: Final[str] = "참고: 일상 추진 수준 약 50회/분(Kim et al., 2025 실험 조건) · 판정 기준 아님"

# 운동 기간 관리 (2026-11-08)
# 주 3회: 휠체어 사용자 어깨 운동 프로그램 STOMPS (Haubert 2021, 주 3회) · 자세 교정 운동 메타분석 (Khorramroo 2026, 6~12주·주 3회) — 초록 확인
# 재검사 시점은 고정하지 않는다 (사용자가 정함 · 고정 기간을 둘 근거 없음)
EXERCISE_PROGRAM: Final[dict[str, int]] = {
    "days_per_week": 3,
}

# 사이트 안 촬영 화면 (components/camera_recorder) 설정 (2026-11-05)
# 기울기 기준은 정면 어깨 기울기 판정(5°)보다 충분히 작게: 휴대폰 1° 기울면 측정 각도도 약 1° 치우친다.
CAMERA_RECORDER: Final[dict[str, object]] = {
    "level_ok_deg": 2.0,        # 휴대폰 좌우 기울기가 이 안이면 "수평 맞음"(초록)
    "pitch_warn_deg": 12.0,     # 카메라가 위·아래로 이보다 숙으면 안내
    "shake_accel": 0.7,         # 중력을 뺀 가속도(m/s², 평균)가 이보다 크면 "흔들려요" (어림값, 실제 휴대폰으로 조정 필요)
    "durations": [10, 15, 20],  # 촬영 시간(초) 선택지
    "default_duration": 15,
    "waits": [5, 10, 15],       # 녹화 시작 전 대기(초): 휴대폰을 두고 출발 위치로 가는 시간
    "default_wait": 10,
    "still_seconds": 3,         # 녹화 시작 후 가만히 있으라고 안내하는 시간 (정지 기준 구간)
    "bitrate": 1_500_000,       # 녹화 화질 (bps). 15초 ≈ 3MB
    "max_mb": 25,
}

# 자세·움직임 키워드 (keywords.py) — 휠체어 사용자에게 잘 나타나는 불균형 (2026-11-08)
# 키워드마다 data/exercises.json의 "keywords"에 같은 키가 있는 운동을 띄운다.
# grade = 기준값 근거: 문헌 / 자체 측정 / 가설(확인 필요)
POSTURE_KEYWORDS: Final[dict[str, dict[str, object]]] = {
    "forward_head": {
        "label": "거북목", "kind": "자세",
        # 앱 목 각도 = 어깨→귀 선과 수평선 사이 각도 (가만히 있는 구간 중앙값).
        # 문헌 FNTA(귀 tragus–어깨 acromion 선과 '수직선' 사이 각도)와 같은 두 점이라 앱 각도 = 90° − FNTA.
        # FNTA 단계 (Diagnostics 2024, 청소년 145명, 깊이 카메라): 정상 <12°, 1단계 12~25°, 2단계 25~40°, 3단계 ≥40°
        #   → 앱 각도: 정상 >78°, 1단계 65~78°, 2단계 50~65°, 3단계 <50°.
        # 운동 추천은 2단계 이상(앱 65° 미만). 1단계는 '경미'라 알려만 준다.
        # 한계: 문헌은 선 자세, 앱은 앉은 자세(휠체어). 앉은 휠체어 사용자는 대부분 거북목 범위로 나온다(Pooryamanesh 2024).
        "threshold": 65.0, "grade": "문헌 (FNTA 2단계, Diagnostics 2024)",
        "levels": (78.0, 65.0, 50.0),  # 앱 각도가 이보다 작으면 1·2·3단계
    },
    "shoulder_drop": {
        "label": "한쪽 어깨 처짐", "kind": "자세",
        # 정면 기준 공통 규칙: 건강한 성인 정상 범위 상한(평균 + 2SD) + 사진 측정의 최소 변화량 약 2°(MDC95 2.4°).
        #   정상값: 건강한 성인 100명, 선 자세 정면 사진 (Postural Evaluation in Young Healthy Adults, PMC9680464, Table 2)
        #   어깨(acromion) 높이 차이 1.3 ± 1.0° → 정상 상한 3.3° + 2° ≈ 5°. 평가자 간 신뢰도 ICC 0.91.
        "threshold": 5.0, "grade": "문헌 (건강한 성인 정상 상한 + 측정 오차)",
    },
    "trunk_side_lean": {
        "label": "몸통 옆 기울기", "kind": "자세",
        # 앞 영상 가만히 구간: 골반 중점→어깨 중점 선이 수직에서 기운 각도.
        # 정면 기준 공통 규칙: 건강한 성인 정상 범위 상한(평균 + 2SD) + 사진 측정의 최소 변화량 약 2°(MDC95 2.4°).
        #   정상값: 건강한 성인 100명, 선 자세 정면 사진 (Postural Evaluation in Young Healthy Adults, PMC9680464, Table 2)
        #   몸통 기울기 1.6 ± 1.2° → 정상 상한 4.0° + 2° = 6°. 단, 평가자 간 신뢰도 ICC 0.44(낮음) → 참고용으로만 보여준다.
        # 앉은 정면 BlazePose 몸통 기울기 탐지 F1 0.92 (Pavlikov 2026, Sensors) — 측정 가능성의 근거.
        "threshold": 6.0, "grade": "문헌 (정상 상한 + 측정 오차 · 신뢰도 낮음)",
    },
    "head_side_tilt": {
        "label": "머리 옆 기울기", "kind": "자세",
        # 앞 영상 가만히 구간: 양쪽 눈을 잇는 선의 기울기.
        # 정면 기준 공통 규칙: 건강한 성인 정상 범위 상한(평균 + 2SD) + 사진 측정의 최소 변화량 약 2°(MDC95 2.4°).
        #   정상값: 건강한 성인 100명, 선 자세 정면 사진 (Postural Evaluation in Young Healthy Adults, PMC9680464, Table 2)
        #   머리 기울기 2.0 ± 1.4° → 정상 상한 4.8° + 2° ≈ 7°. 평가자 간 신뢰도 ICC 0.51(보통).
        "threshold": 7.0, "grade": "문헌 (정상 상한 + 측정 오차 · 신뢰도 보통)",
    },
    "trunk_collapse": {
        "label": "밀 때 상체 무너짐", "kind": "움직임",
        # 밀 때 더 숙여진 각도 ≥ 10.6° (휠체어 경험자 7.37 ± 3.26°의 평균 + 1SD, Symonds 2016)
        # + 피로 질문에서 '숨참·팔 무거움' 또는 '5분 전에 쉼'일 때만 키워드 (Rodgers 2000: 피로 → 상체 숙여 밀기 → 팔·어깨 부상 위험).
        "threshold": 10.6, "grade": "문헌 (Symonds 2016 · Rodgers 2000)",
    },
    "push_pattern": {
        "label": "권장과 다른 미는 패턴", "kind": "움직임",
        "threshold": None, "grade": "문헌 (척수손상 가이드라인 2005)",
    },
    "push_side_drop": {
        "label": "밀 때 한쪽 처짐", "kind": "움직임",
        # 밀고 올 때 어깨 기울기가 가만히 있을 때보다 같은 쪽으로 이만큼 더 커지면.
        # 같은 사람·같은 영상 안의 '변화'라 정상값 대신 측정 최소 변화량(사진 측정 MDC95 약 2.4°)보다 큰 3°를 쓴다.
        # 휠체어 추진 중 좌우 비대칭이 있다는 보고(SCIRE 근거 요약, 수준 4)가 있지만 각도 기준값은 문헌에 없다.
        "threshold": 3.0, "grade": "측정 오차 기준 (정상값 문헌 없음)",
    },
}

# 결과 화면의 "다른 사람은 보통 이 정도 · 나는 이 정도" 비교 (2026-11-05)
# 휠체어 사용자의 "정상값"은 문헌에 없다 → 화면에는 반드시 "참고"와 출처·인원을 함께 적는다. 판정에 쓰지 않는다.
#  - band: 참고 범위 (낮은값, 높은값), marker: 참고 한 점, axis: 막대 그림의 끝값
REFERENCE_COMPARISONS: Final[dict[str, dict[str, object]]] = {
    "trunk_change": {
        "label": "밀 때 더 숙여진 정도",
        "unit": "°",
        # Symonds et al. (2016) 평지 추진 중 몸통 굴곡 변화: 경험자 7.37±3.26° (N=7, IMU) → 평균±1SD
        "band": (4.1, 10.6),
        "axis": 30.0,
        "who": "휠체어를 오래 탄 사람 7명 (평균 ± 표준편차)",
        "source": "Symonds et al., 2016 · 센서 측정이라 영상과 방법이 달라요",
    },
    "shoulder_tilt_static": {
        "label": "가만히 있을 때 어깨 높이 차이",
        "unit": "°",
        # 건강한 성인 100명(선 자세) 어깨 높이 차이 1.3 ± 1.0° → 평균 ± 2SD 상한 3.3°
        # (참고: 우리 정면 영상 바르게 앉은 10개도 모두 3.4° 이내로 비슷했다)
        "band": (0.0, 3.3),
        "axis": 12.0,
        "who": "건강한 성인 100명 (평균 + 2SD)",
        "source": "Postural Evaluation in Young Healthy Adults (2022, 선 자세 사진)",
    },
    "stroke_rate": {
        "label": "1분에 미는 횟수",
        "unit": "회",
        # Kim et al. (2025) 메트로놈 실험 조건(일상 속도 대표). 모집단 평균이 아니다.
        "marker": 50.0,
        "axis": 100.0,
        "who": "연구에서 정한 일상 속도",
        "source": "Kim et al., 2025 실험 조건 · 평균값이 아니에요",
    },
}
