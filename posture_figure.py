"""기록 화면의 '전·후 비교 그림'.

사진은 저장하지 않으므로(동의서: 측정 수치만 저장), 저장된 각도로 선 그림을 그린다.
왼쪽 '지난번', 오른쪽 '이번' 두 칸을 나란히 두고, '이번' 칸에는 지난번 선을 흐리게 겹쳐 변화가 보이게 한다.
초록 점선은 기준선.
"""
from __future__ import annotations

import math
from typing import Callable

BEFORE = "#9a93a6"
AFTER = "#8b5cf6"
REF = "#3fae7a"
INK = "currentColor"
PANEL_W, PANEL_H = 150, 165


def _line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 3, dash: str = "", opacity: float = 1.0) -> str:
    dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
    return (f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' stroke-width='{width}' "
            f"stroke-linecap='round' opacity='{opacity}'{dash_attr}/>")


def _circle(x: float, y: float, r: float, color: str, width: float = 3, fill: str = "none", opacity: float = 1.0) -> str:
    return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{fill}' stroke='{color}' stroke-width='{width}' opacity='{opacity}'/>"


def _text(x: float, y: float, text: str, color: str, size: int = 13, weight: int = 700) -> str:
    return f"<text x='{x:.1f}' y='{y:.1f}' fill='{color}' font-size='{size}' font-weight='{weight}' text-anchor='middle'>{text}</text>"


def _arc(cx: float, cy: float, r: float, start_deg: float, end_deg: float, color: str) -> str:
    """화면 각도(0°=오른쪽, 반시계 +)로 호를 그립니다."""
    (x1, y1), (x2, y2) = [(cx + r * math.cos(math.radians(a)), cy - r * math.sin(math.radians(a))) for a in (start_deg, end_deg)]
    sweep = 0 if end_deg > start_deg else 1
    return f"<path d='M{x1:.1f},{y1:.1f} A{r},{r} 0 0 {sweep} {x2:.1f},{y2:.1f}' fill='none' stroke='{color}' stroke-width='2'/>"


def _pair(draw: Callable[[float, str, float], str], before: float, after: float, before_label: str, after_label: str) -> str:
    """두 칸: 왼쪽 지난번, 오른쪽 이번(지난번을 흐리게 겹침). 칸 위에 각도를 크게 적는다."""
    left = draw(before, BEFORE, 1.0)
    right = draw(before, BEFORE, 0.35) + draw(after, AFTER, 1.0)
    body = (
        f"<g>{left}</g><g transform='translate({PANEL_W},0)'>{right}</g>"
        + _line(PANEL_W, 8, PANEL_W, PANEL_H - 4, INK, 1, "2 4", 0.4)
        + _text(PANEL_W / 2, 20, f"지난번 {before_label}", BEFORE, 14)
        + _text(PANEL_W * 1.5, 20, f"이번 {after_label}", AFTER, 16, 800)
    )
    return (f"<div style='width:100%;max-width:340px;margin:6px auto'><svg viewBox='0 0 {PANEL_W * 2} {PANEL_H}' "
            f"style='width:100%;height:auto;display:block' xmlns='http://www.w3.org/2000/svg'>{body}</svg></div>")


def neck_angle(before: float, after: float, reference: float | None) -> str:
    """옆모습: 어깨→귀 선이 수평선과 이루는 각도(거북목 각도). 클수록 머리가 덜 나옴."""
    sx, sy, length = 45.0, 135.0, 82.0

    def draw(angle: float, color: str, opacity: float) -> str:
        rad = math.radians(angle)
        ex, ey = sx + length * math.cos(rad), sy - length * math.sin(rad)
        parts = []
        if opacity == 1.0:  # 기준·보조선은 한 번만
            parts += [
                _line(sx, sy, sx - 18, sy + 28, INK, 2, "2 4", 0.6),
                _line(sx, sy, sx + 95, sy, INK, 1.2, "3 4", 0.6),
                _circle(sx, sy, 4, INK, 0, INK),
            ]
            if reference is not None:
                rr = math.radians(reference)
                parts.append(_line(sx, sy, sx + 70 * math.cos(rr), sy - 70 * math.sin(rr), REF, 2, "5 4"))
            parts.append(_arc(sx, sy, 24, 0, angle, color))
        parts += [_line(sx, sy, ex, ey, color, 4, "", opacity), _circle(ex, ey, 12, color, 3, "none", opacity)]
        return "".join(parts)

    return _pair(draw, before, after, f"{before:.0f}°", f"{after:.0f}°")


def trunk_lean(before: float, after: float, reference: float | None) -> str:
    """옆모습: 가만히 있을 때(세로 점선)보다 밀 때 상체가 더 숙여진 각도."""
    hx, hy, length = 55.0, 150.0, 105.0

    def draw(angle: float, color: str, opacity: float) -> str:
        rad = math.radians(angle)
        ex, ey = hx + length * math.sin(rad), hy - length * math.cos(rad)
        parts = []
        if opacity == 1.0:
            parts += [_line(hx, hy, hx, hy - length, INK, 1.5, "3 4", 0.6), _circle(hx, hy, 4, INK, 0, INK)]
            if reference is not None:
                rr = math.radians(reference)
                parts.append(_line(hx, hy, hx + 95 * math.sin(rr), hy - 95 * math.cos(rr), REF, 2, "5 4"))
            parts.append(_arc(hx, hy, 34, 90, 90 - angle, color))
        parts += [_line(hx, hy, ex, ey, color, 4, "", opacity), _circle(ex + 4 * math.sin(rad), ey - 14, 11, color, 3, "none", opacity)]
        return "".join(parts)

    return _pair(draw, before, after, f"{before:+.0f}°", f"{after:+.0f}°")


def shoulder_tilt(before: float, after: float, side_label: Callable[[float], str] = lambda v: f"{abs(v):.0f}°") -> str:
    """앞모습: 양 어깨를 잇는 선의 기울기. + = 앉은 사람의 왼쪽(화면 오른쪽) 어깨가 낮음."""
    cx, cy, half = 75.0, 100.0, 52.0

    def draw(angle: float, color: str, opacity: float) -> str:
        rad = math.radians(angle)
        dx, dy = half * math.cos(rad), half * math.sin(rad)
        parts = []
        if opacity == 1.0:
            parts += [
                _line(cx - half - 8, cy, cx + half + 8, cy, REF, 2, "5 4"),
                _circle(cx, cy - 42, 14, INK, 1.5, "none", 0.6),
                _line(cx, cy - 28, cx, cy + 50, INK, 1.5, "2 4", 0.6),
            ]
        parts += [
            _line(cx - dx, cy - dy, cx + dx, cy + dy, color, 4, "", opacity),
            _circle(cx - dx, cy - dy, 5, color, 0, color, opacity),
            _circle(cx + dx, cy + dy, 5, color, 0, color, opacity),
        ]
        return "".join(parts)

    return _pair(draw, before, after, side_label(before), side_label(after))


def legend(reference_label: str | None) -> str:
    items = [f"<span><i style='border-top:3px solid {BEFORE}'></i>지난번</span>", f"<span><i style='border-top:4px solid {AFTER}'></i>이번</span>"]
    if reference_label:
        items.append(f"<span><i style='border-top:2px dashed {REF}'></i>{reference_label}</span>")
    return "<div class='wl-fig-legend'>" + "".join(items) + "</div>"
