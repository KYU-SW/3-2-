"""브라우저에서 동작하는 체중 이동 알림 타이머 컴포넌트.

Streamlit은 버튼을 누를 때마다 파이썬 스크립트를 다시 실행하므로,
타이머는 파이썬이 아니라 브라우저의 JavaScript로 돌린다.
다음 알림 시각을 브라우저 저장소(localStorage)에 적어 두어서, 화면이 다시 그려져도 이어서 센다.
저장소가 막힌 브라우저(시크릿 모드 등)에서는 화면이 다시 그려질 때 처음부터 센다.
"""

from __future__ import annotations

import streamlit.components.v1 as components

from config import ALGORITHM_PARAMS


def render_weight_shift_timer(interval_minutes: int | None = None) -> None:
    """체중 이동 알림 타이머를 그립니다. 시간이 되면 문구 강조 + (지원 기기) 진동."""
    minutes = int(interval_minutes or ALGORITHM_PARAMS["weight_shift_interval_min"])  # type: ignore[arg-type]
    components.html(
        f"""
        <div style="font-family:Pretendard,-apple-system,sans-serif;background:#F4F7F6;border-radius:14px;padding:14px 16px;color:#1F2A2E;">
          <div style="font-size:15px;color:#5F6D70;">다음 체중 이동까지</div>
          <div id="wl-count" style="font-size:34px;font-weight:700;color:#1D9E75;margin:4px 0;">--:--</div>
          <button id="wl-done" style="min-height:44px;width:100%;border:none;border-radius:12px;background:#1D9E75;color:#fff;font-size:16px;font-weight:700;">체중 이동 했어요 · 다시 시작</button>
          <div style="font-size:13px;color:#5F6D70;margin-top:8px;">{minutes}분 간격 · 브라우저를 열어둔 동안만 작동해요.</div>
        </div>
        <script>
        (() => {{
          const interval = {minutes} * 60 * 1000;
          const key = "wheelance_next_weight_shift";
          const read = () => {{ try {{ return Number(localStorage.getItem(key) || 0); }} catch (e) {{ return 0; }} }};
          const write = (v) => {{ try {{ localStorage.setItem(key, String(v)); }} catch (e) {{}} }};
          let next = read();
          if (!next || next < Date.now() - interval) {{ next = Date.now() + interval; write(next); }}
          let alerted = false;
          const count = document.getElementById("wl-count");
          document.getElementById("wl-done").onclick = () => {{ next = Date.now() + interval; write(next); alerted = false; tick(); }};
          const tick = () => {{
            const remaining = next - Date.now();
            if (remaining <= 0) {{
              count.textContent = "지금 체중을 이동해요";
              count.style.color = "#B42318";
              count.style.fontSize = "24px";
              if (!alerted) {{ alerted = true; try {{ navigator.vibrate && navigator.vibrate([300, 150, 300]); }} catch (e) {{}} }}
              return;
            }}
            const m = Math.floor(remaining / 60000), s = Math.floor((remaining % 60000) / 1000);
            count.textContent = `${{m}}:${{String(s).padStart(2, "0")}}`;
            count.style.color = "#1D9E75";
            count.style.fontSize = "34px";
          }};
          tick();
          setInterval(tick, 1000);
        }})();
        </script>
        """,
        height=170,
    )
