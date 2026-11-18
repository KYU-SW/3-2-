"""사이트 안 촬영 화면 (카메라 + 수평선 + 흔들림 안내 + 녹화).

화면은 components/camera_recorder/index.html (브라우저 JS)이 그린다.
- 수평선: 휴대폰 기울기 센서(DeviceMotion)의 중력 방향으로 화면이 몇 도 돌아갔는지 잰다.
- 흔들림: 중력을 뺀 가속도의 평균이 기준(config.CAMERA_RECORDER["shake_accel"])을 넘으면 안내한다.
- 녹화가 끝나면 영상을 base64로 돌려준다 → 기존 영상 분석(propulsion)에 그대로 넣는다.
휴대폰 카메라는 https 주소(배포 사이트)나 localhost에서만 켜진다.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

from config import CAMERA_RECORDER

_COMPONENT = components.declare_component(
    "wheelance_camera_recorder",
    path=str(Path(__file__).parent / "components" / "camera_recorder"),
)


def camera_recorder(key: str) -> dict[str, Any] | None:
    """촬영 화면을 그리고, "이 영상으로 분석하기"를 누르면 녹화 결과를 돌려줍니다.

    반환값: {"id", "video" (bytes), "mime", "seconds", "sensor", "max_tilt", "shake_ratio"} 또는 None
    """
    value = _COMPONENT(**CAMERA_RECORDER, key=key, default=None)
    if not isinstance(value, dict) or not value.get("video_b64"):
        return None
    try:
        video = base64.b64decode(str(value["video_b64"]))
    except (ValueError, TypeError):
        return None
    return {
        "id": value.get("id"),
        "video": video,
        "mime": str(value.get("mime") or ""),
        "seconds": value.get("seconds"),
        "sensor": bool(value.get("sensor")),
        "max_tilt": value.get("max_tilt"),
        "shake_ratio": value.get("shake_ratio"),
        "roll_median": value.get("roll_median"),
        "facing": str(value.get("facing") or "environment"),
    }


def recording_checks(recording: dict[str, Any]) -> list[tuple[str, str, str]]:
    """녹화 중 센서 기록을 촬영 점검 항목으로 바꿉니다 (분석 결과의 촬영 점검에 덧붙임)."""
    if not recording.get("sensor"):
        return [("ok", "휴대폰 수평", "기울기 센서가 없어 확인하지 못했어요 (컴퓨터이거나 허용 안 함).")]
    checks: list[tuple[str, str, str]] = []
    limit = float(CAMERA_RECORDER["level_ok_deg"]) * 2  # type: ignore[arg-type]
    tilt = recording.get("max_tilt")
    if tilt is not None and float(tilt) > limit:
        checks.append(("warn", "휴대폰 수평", f"녹화 중 휴대폰이 최대 {float(tilt):.0f}° 기울었어요. 수평을 맞춰 다시 찍는 걸 권해요."))
    else:
        checks.append(("ok", "휴대폰 수평", "녹화하는 동안 수평이 잘 맞았어요."))
    shake = recording.get("shake_ratio")
    if shake is not None and float(shake) >= 0.1:
        checks.append(("warn", "흔들림", f"녹화 시간의 {float(shake) * 100:.0f}% 동안 흔들렸어요. 휴대폰을 고정하고 다시 찍는 걸 권해요."))
    else:
        checks.append(("ok", "흔들림", "흔들림 없이 찍혔어요."))
    return checks
