"""운동 동작을 관절 마네킹으로 보여줍니다.

동작 데이터: data/motions/<운동 id>.json (tools/extract_motion.py가 영상에서 만듦, 좌표만 있고 원본 영상은 없음)
화면: components/mannequin_player.html (three.js)
"""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

ROOT = Path(__file__).parent
MOTION_DIR = ROOT / "data" / "motions"
PLAYER = ROOT / "components" / "mannequin_player.html"


def has_motion(exercise_id: str) -> bool:
    """이 운동의 마네킹 동작 데이터가 있는지."""
    return (MOTION_DIR / f"{exercise_id}.json").exists()


def show_motion(exercise_id: str) -> None:
    """마네킹 재생 화면을 그립니다 (데이터가 없으면 아무것도 안 함)."""
    path = MOTION_DIR / f"{exercise_id}.json"
    if not path.exists():
        return
    html = PLAYER.read_text(encoding="utf-8").replace("__MOTION__", path.read_text(encoding="utf-8"))
    components.html(html, height=390)
