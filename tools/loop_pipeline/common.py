import cv2, numpy as np, os
# All intermediate data lives in the work directory (frames/, masks/, depth/, clean/, sprites/ ...).
SCR=os.environ.get("LOOP_WORK",os.getcwd())
os.makedirs(os.path.join(SCR,"thumbs"),exist_ok=True)
W,H=1280,720
_cache={}
def load(i):
    if i not in _cache:
        _cache[i]=cv2.imread(f"{SCR}/frames/f{i:04d}.png")
    return _cache[i]
def gray(i): return cv2.cvtColor(load(i),cv2.COLOR_BGR2GRAY)
_sift=cv2.SIFT_create(6000)
_feat={}
def feats(i):
    if i not in _feat:
        _feat[i]=_sift.detectAndCompute(gray(i),None)
    return _feat[i]
def matches(a,b,ratio=0.7):
    ka,da=feats(a); kb,db=feats(b)
    fl=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=64))
    m=fl.knnMatch(da,db,k=2)
    good=[x for x,y in m if x.distance<ratio*y.distance]
    pa=np.float32([ka[x.queryIdx].pt for x in good]); pb=np.float32([kb[x.trainIdx].pt for x in good])
    return pa,pb
def trans_ransac(pa,pb,thr=1.5,iters=2000,rng=np.random.default_rng(0)):
    """robust translation estimate: returns (dx,dy), inlier mask"""
    if len(pa)<3: return None,None
    d=pb-pa
    best=None;bestn=-1
    for _ in range(min(iters,len(d)*3)):
        c=d[rng.integers(len(d))]
        n=(np.linalg.norm(d-c,axis=1)<thr).sum()
        if n>bestn: bestn=n;best=c
    inl=np.linalg.norm(d-best,axis=1)<thr
    return d[inl].mean(0),inl

import json as _json
CMID={}
def cmid(f):
    """camera offset (back-layer pixels) of frame f, from campath.json written by campath.py"""
    if not CMID:
        cp=_json.load(open(os.path.join(SCR,'campath.json')))
        CMID.update({int(k):np.array(v) for k,v in cp['mid'].items()})
    return CMID[f]
def shift_img(img,dx,dy,interp=cv2.INTER_LINEAR,border=cv2.BORDER_CONSTANT,bval=0):
    M=np.float32([[1,0,dx],[0,1,dy]])
    return cv2.warpAffine(img,M,(img.shape[1],img.shape[0]),flags=interp,borderMode=border,borderValue=bval)
def from_frame(g,f,r,img=None):
    """content of frame g re-positioned so that scene points at parallax ratio r line up with frame f"""
    d=r*(cmid(f)-cmid(g))
    src=load(g) if img is None else img
    return shift_img(src,d[0],d[1])
RATIOS=[1.0,1.6,2.0,2.18,2.3,2.45,2.6,2.8]
