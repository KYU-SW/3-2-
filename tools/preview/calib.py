import sys, os, cv2, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import propulsion
# 사용: python tools/preview/calib.py videos [bad]  (videos/validation, videos/broken_samples)
sp=sys.argv[1]
def make(src, name, fn):
    cap=cv2.VideoCapture(src); fps=cap.get(cv2.CAP_PROP_FPS) or 30
    ok,f=cap.read(); h,w=f.shape[:2]; i=0
    out=cv2.VideoWriter(f"{sp}/broken_samples/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    rng=np.random.default_rng(0)
    while ok:
        out.write(fn(f,w,h,i,rng)); ok,f=cap.read(); i+=1
    out.release()
def rot(deg): return lambda f,w,h,i,r: cv2.warpAffine(f, cv2.getRotationMatrix2D((w/2,h/2),deg,1.08),(w,h))
def shake(px): 
    def fn(f,w,h,i,r):
        dx,dy=r.normal(0,px,2); M=np.float32([[1.08,0,dx-w*0.04],[0,1.08,dy-h*0.04]]); return cv2.warpAffine(f,M,(w,h))
    return fn
base=sp+"/validation/arc_02.mp4"; fbase=sp+"/validation/flat_02.mp4"
jobs=[]
if len(sys.argv)>2 and sys.argv[2]=="bad":
    for n,fn,src in [("rot3",rot(3),base),("rot5",rot(5),base),("rot8",rot(8),base),("rot5f",rot(5),fbase),("shake_s",shake(3),base),("shake_m",shake(8),base),("shake_f",shake(8),fbase)]:
        make(src,n,fn); jobs.append((f"{sp}/broken_samples/{n}.mp4", "front" if src==fbase else "side"))
else:
    for f in sorted(os.listdir(sp+"/validation")):
        jobs.append((sp+"/validation/"+f, "front" if f.startswith(("flat","asym")) else "side"))
for path,view in jobs:
    r=propulsion.analyze_video_bytes(open(path,"rb").read(), view=view, filmed_side=None)
    s=np.array(r.debug.get("camera_shifts") or [0])
    t=r.debug.get("camera_tilt",[None])[0]
    print(os.path.basename(path), view, "tilt", None if t is None else round(t,1), "shift p50/p90", round(float(np.median(s))*1000,2), round(float(np.percentile(s,90))*1000,2), "ratio>.006", round(float(np.mean(s>0.006)),2), "| warn:", [l for st,l,m in r.filming_checks if st!="ok"], flush=True)
