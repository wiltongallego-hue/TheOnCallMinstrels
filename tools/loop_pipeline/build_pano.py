import sys; sys.path.insert(0,'.')
from common import *
FR=list(range(136,217))
PR=[1.0]   # only the back-layer median panorama is used by postfill.py
# static exclusion mask (screen space) per frame
yy,xx=np.mgrid[0:H,0:W]
LOGO=((xx-148.5)**2+(yy-150)**2)<=(104**2)
def excl(f):
    m=LOGO.copy()
    if f>=146: m|=(xx>=950)&(yy<=112)
    return m
OFFY=70
def pano_geom(r):
    cx=np.array([cmid(f)[0] for f in FR]); cy=np.array([cmid(f)[1] for f in FR])
    # U = p - r*c + (0,OFFY)
    wpan=int(np.ceil(W - r*cx.min()))+2
    hpan=int(np.ceil(H + OFFY - r*cy.min()))+2
    return wpan,hpan
res={}
for r in PR:
    wp,hp=pano_geom(r)
    stack=np.zeros((len(FR),hp,wp,3),np.uint8)
    valid=np.zeros((len(FR),hp,wp),bool)
    for i,f in enumerate(FR):
        c=cmid(f); dx,dy=-r*c[0],-r*c[1]+OFFY
        M=np.float32([[1,0,dx],[0,1,dy]])
        stack[i]=cv2.warpAffine(load(f),M,(wp,hp),flags=cv2.INTER_LINEAR)
        v=(~excl(f)).astype(np.uint8)*255
        v=cv2.erode(v,np.ones((3,3),np.uint8))
        valid[i]=cv2.warpAffine(v,M,(wp,hp),flags=cv2.INTER_NEAREST)>0
    med=np.zeros((hp,wp,3),np.float32); mad=np.full((hp,wp),255,np.float32); cnt=valid.sum(0)
    for y0 in range(0,hp,48):
        s=stack[:,y0:y0+48].astype(np.float32); v=valid[:,y0:y0+48]
        s[~v]=np.nan
        m=np.nanmedian(s,axis=0)
        dev=np.nanmedian(np.abs(s-m[None]).sum(3),axis=0) if True else None
        med[y0:y0+48]=np.nan_to_num(m); mad[y0:y0+48]=np.nan_to_num(dev,nan=255)
    mad[cnt<8]=255
    np.save(f'pano_med_{r:.2f}.npy',med.astype(np.uint8)); np.save(f'pano_mad_{r:.2f}.npy',mad.astype(np.float16)); np.save(f'pano_cnt_{r:.2f}.npy',cnt.astype(np.int16))
    print(r,'pano',wp,hp,'mad median',float(np.median(mad[cnt>=8])),flush=True)
