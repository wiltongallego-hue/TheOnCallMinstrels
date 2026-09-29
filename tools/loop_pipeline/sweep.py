import sys; sys.path.insert(0,'.')
from common import *
KAPPA=0.83
RGRID=np.concatenate([np.arange(1.0,1.9,0.1),np.arange(1.9,3.01,0.02)])
FMIN,FMAX=136,216
yy,xx=np.mgrid[0:H,0:W]
LOGO=((xx-148.5)**2+(yy-150)**2)<=(106**2)
def hud_mask(f):
    m=np.zeros((H,W),bool)
    if f>=146: m[:115,945:]=True
    return m
def static_excl(f):
    return LOGO|hud_mask(f)
def shift_vec(g,f,r):
    """screen displacement taking a scene point at depth r from frame f position to frame g position"""
    d=cmid(g)-cmid(f)
    return np.array([r*d[0],(1+KAPPA*(r-1))*d[1]])
def warp_from(g,f,r,img,roi,interp=cv2.INTER_LINEAR,bval=0):
    """sample img (frame g space) at positions p+shift for p in roi of frame f; roi=(x0,y0,x1,y1)"""
    x0,y0,x1,y1=roi
    s=shift_vec(g,f,r)
    # output(p) = img(p + s)  -> warpAffine with M maps dst->src when WARP_INVERSE_MAP
    M=np.float32([[1,0,x0+s[0]],[0,1,y0+s[1]]])
    return cv2.warpAffine(img,M,(x1-x0,y1-y0),flags=interp|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,borderValue=bval)
def sources_for(f,mind=45,maxd=260,n=8):
    cands=[]
    for g in range(FMIN,FMAX+1):
        d=abs(cmid(g)[0]-cmid(f)[0])
        if mind<=d<=maxd: cands.append((d,g))
    # spread choices: take evenly across sorted by displacement, both sides
    before=[g for d,g in cands if g<f]; after=[g for d,g in cands if g>f]
    pick=[]
    for side in (before,after):
        if side:
            idx=np.linspace(0,len(side)-1,min(len(side),n//2)).round().astype(int)
            pick+= [side[i] for i in sorted(set(idx))]
    return pick
