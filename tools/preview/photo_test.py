import sys, cv2
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import propulsion, overlay
video, side, out = sys.argv[1], sys.argv[2], sys.argv[3]
r = propulsion.analyze_video_bytes(open(video, "rb").read(), view="side", filmed_side=side, rim_points=None, rim_diameter_inch=24.0)
print("warn", r.warnings, "still", r.still_time, "pts", sorted(r.still_points))
m = r.metrics.get("head_cva_static"); print("cva", m and m.value)
if r.still_frame is not None:
    img = overlay.posture_photo(r.still_frame, r.still_points, side, 43.0)
    print("photo", None if img is None else img.shape)
    if img is not None:
        cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
