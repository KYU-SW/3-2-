"""실시간 웹캠 코칭 (로컬 실행 전용).

웹캠 화면에 스켈레톤, 최근 2초 손목 궤적, 실시간 상체 숙임 각도를 그린다.
패턴 분류 같은 정밀 분석은 업로드 영상에서만 한다.

클라우드(Streamlit Community Cloud)에서는 브라우저와 서버가 영상을 주고받으려면
TURN 서버가 필요해서, 로컬에서만 켜고 클라우드에서는 안내 문구만 보여준다.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import streamlit as st

from config import THRESHOLDS
from overlay import COLOR_RECOVERY, draw_pose_skeleton
from pose_utils import create_pose, detect_landmarks, has_required_visibility, landmarks_to_pixels, line_vertical_deviation

TRAIL_SECONDS: Final[float] = 2.0
LOCAL_ONLY_MESSAGE: Final[str] = (
    "실시간 웹캠 코칭은 로컬 실행 전용 기능이에요. 클라우드에서는 TURN 서버가 필요해요."
)


def is_cloud_environment() -> bool:
    """Streamlit Community Cloud에서 실행 중인지 추정합니다.

    Community Cloud는 앱을 /mount/src 아래에 두고 HOSTNAME을 "streamlit"으로 쓴다.
    WHEELANCE_FORCE_LOCAL=1 환경변수로 강제로 로컬 모드를 켤 수 있다.
    """
    if os.environ.get("WHEELANCE_FORCE_LOCAL") == "1":
        return False
    return Path("/mount/src").exists() or os.environ.get("HOSTNAME") == "streamlit"


def local_only_message() -> str:
    """실시간 코칭 기능의 실행 조건 안내 문구를 반환합니다."""
    return LOCAL_ONLY_MESSAGE


class LiveCoachState:
    """웹캠 프레임 사이에 유지할 상태 (MediaPipe 추정기, 손목 궤적).

    webrtc 콜백은 별도 스레드에서 돌기 때문에 Lock으로 보호한다.
    MediaPipe 추정기는 첫 프레임에서 만든다 (화면을 그릴 때 오류가 나면 다른 탭까지 멈추기 때문).
    """

    def __init__(self, side: str = "left") -> None:
        self.side = side if side in ("left", "right") else "left"
        self._pose: Any = None
        self.trail: deque[tuple[float, np.ndarray]] = deque()
        self.lock = threading.Lock()
        self.last_lean: float | None = None

    @property
    def pose(self) -> Any:
        """MediaPipe Pose 추정기 (처음 쓸 때 생성)."""
        if self._pose is None:
            self._pose = create_pose()
        return self._pose

    def close(self) -> None:
        """MediaPipe 자원을 해제합니다."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None


def _lean_limit() -> float | None:
    """상체 숙임 경고 기준 (config의 trunk_lean.caution_max). 미설정이면 None."""
    value = THRESHOLDS.get("trunk_lean", {}).get("caution_max")
    return float(value) if isinstance(value, (int, float)) else None


def process_frame(frame_bgr: np.ndarray, state: LiveCoachState, now: float | None = None) -> np.ndarray:
    """웹캠 프레임 한 장에 코칭 정보를 그려서 반환합니다 (BGR).

    1) 관절 추출 → 스켈레톤
    2) 촬영한 쪽 손목을 최근 TRAIL_SECONDS초 동안 선으로 이어 궤적 잔상 표시
    3) 상체 숙임 = hip→shoulder 선이 수직에서 기운 각도. 기준 초과 시 빨간 경고
    """
    current = time.time() if now is None else now
    with state.lock:
        landmarks = detect_landmarks(state.pose, frame_bgr)
        output = draw_pose_skeleton(frame_bgr, landmarks) if landmarks else frame_bgr.copy()
        wrist_name = f"{state.side}_wrist"
        shoulder_name = f"{state.side}_shoulder"
        hip_name = f"{state.side}_hip"
        pixels = landmarks_to_pixels(landmarks, frame_bgr.shape) if landmarks else {}

        if landmarks and has_required_visibility(landmarks, (wrist_name,)):
            state.trail.append((current, pixels[wrist_name]))
        while state.trail and current - state.trail[0][0] > TRAIL_SECONDS:
            state.trail.popleft()
        points = [point for _, point in state.trail]
        for first, last in zip(points, points[1:]):
            cv2.line(output, (int(first[0]), int(first[1])), (int(last[0]), int(last[1])), COLOR_RECOVERY, 3, cv2.LINE_AA)

        state.last_lean = None
        if landmarks and has_required_visibility(landmarks, (shoulder_name, hip_name)):
            state.last_lean = line_vertical_deviation(pixels[hip_name], pixels[shoulder_name])

    limit = _lean_limit()
    if state.last_lean is not None:
        warning = limit is not None and state.last_lean > limit
        color = (40, 40, 220) if warning else (117, 158, 29)
        text = f"Trunk lean {state.last_lean:.0f} deg" + ("  LEAN BACK" if warning else "")
        cv2.rectangle(output, (10, 10), (10 + 16 * len(text), 50), (255, 255, 255), -1)
        cv2.putText(output, text, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return output


def render_live_coach() -> None:
    """실시간 웹캠 코칭 화면을 그립니다. 클라우드에서는 안내만 표시합니다."""
    st.caption("패턴 분류와 정밀 분석은 업로드 영상에서만 해요. 여기서는 자세를 실시간으로 확인해요.")
    if is_cloud_environment():
        st.info(local_only_message())
        return

    try:
        import av
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
    except ImportError:
        st.warning("streamlit-webrtc가 설치되지 않았어요. requirements.txt로 설치해주세요.")
        return

    side = st.radio(
        "화면에 보이는 쪽 팔 (앉은 사람 기준)",
        ["left", "right"],
        # 사람의 왼쪽이 카메라를 향하면 화면 속 사람은 왼쪽을 바라본다.
        format_func=lambda value: "왼팔 (사람이 화면 왼쪽을 봐요)" if value == "left" else "오른팔 (사람이 화면 오른쪽을 봐요)",
        key="live_side",
    )
    if _lean_limit() is None:
        st.caption("상체 숙임 경고 기준값이 아직 설정되지 않아 각도만 표시해요 (config.py의 trunk_lean).")

    state_key = f"live_coach_state_{side}"
    if state_key not in st.session_state:
        st.session_state[state_key] = LiveCoachState(side)
    state: LiveCoachState = st.session_state[state_key]

    def callback(frame: Any) -> Any:
        """webrtc가 프레임마다 호출하는 함수 (별도 스레드)."""
        image = frame.to_ndarray(format="bgr24")
        try:
            image = process_frame(image, state)
        except Exception:  # MediaPipe 오류가 나도 원본 화면은 계속 보여준다.
            pass
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    try:
        webrtc_streamer(
            key=f"wheelance-live-{side}",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=callback,
            media_stream_constraints={"video": True, "audio": False},
            # 로컬 전용 기능이라 같은 컴퓨터 안에서 바로 연결한다 (외부 STUN 서버 불필요).
            rtc_configuration={"iceServers": []},
            async_processing=True,
        )
    except Exception as error:  # 웹캠 기능 오류가 앱 전체를 멈추지 않게 이 탭 안에서만 처리한다.
        st.warning(f"실시간 웹캠을 시작하지 못했어요. 업로드 분석은 그대로 쓸 수 있어요. ({type(error).__name__})")
