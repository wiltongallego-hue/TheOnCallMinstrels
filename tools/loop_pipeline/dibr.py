import sys; sys.path.insert(0,'.')
import cv2 as _cv; _cv.setNumThreads(1)
from fill import *
from depth import RSET, DROI
_D={}
FIRE152=(415,295,690,505)
def fire_zone(g):
    c=cmid(g); x0,y0,x1,y1=FIRE152
    m=np.zeros((H,W),bool)
    xa,xb=int(x0+c[0]),int(x1+c[0]); ya,yb=int(y0+c[1]),int(y1+c[1])
    m[max(0,ya):max(0,yb),max(0,xa):max(0,xb)]=True
    return m
def note_colored(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v=hsv[...,0],hsv[...,1],hsv[...,2]
    gold=(h>=22)&(h<=34)&(s>=60)&(v>=190)
    cyan=(h>=80)&(h<=105)&(s>=40)&(v>=200)
    return gold|cyan
def unstable_notes(g):
    I=load(g); nc=note_colored(I)
    h=g+6 if g+6<=220 else g-6
    J=warp_from(h,g,1.0,load(h),(0,0,W,H)).astype(np.float32)
    diff=np.abs(J-I.astype(np.float32)).sum(2)
    m=nc&(diff>40)
    return cv2.dilate(m.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0
def depth_full(g):
    """returns (depth map, dynamic mask)"""
    if g not in _D:
        z=np.load(f'depth/d_{g}.npz')
        r=z['r'].astype(np.float32); cost=z['cost'].astype(np.float32)
        r=cv2.medianBlur(r,5)
        r[r<1.06]=1.0
        full=np.ones((H,W),np.float32); cfull=np.full((H,W),999,np.float32)
        x0,y0,x1,y1=DROI; full[y0:y1,x0:x1]=r; cfull[y0:y1,x0:x1]=cost
        full=np.round(full/0.1)*0.1
        fz=fire_zone(g)
        full[fz]=1.0
        dyn=(cfull>70)&~fz
        dyn=cv2.dilate(dyn.astype(np.uint8),np.ones((3,3),np.uint8))>0
        dyn|=unstable_notes(g)
        _D[g]=(full,dyn)
    return _D[g]
def forward_warp(g,f,valid,need=None):
    """render frame g into frame f's viewpoint using g's depth map (far->near z-order).
    Only the bounding box of `need` is rendered; returns full-size canvas/coverage."""
    D,dyn=depth_full(g)
    valid=valid&~dyn
    if need is None: need=np.ones((H,W),bool)
    ys,xs=np.nonzero(need)
    X0,X1=max(0,xs.min()-2),min(W,xs.max()+3); Y0,Y1=max(0,ys.min()-2),min(H,ys.max()+3)
    s_lo=shift_vec(g,f,1.0); s_hi=shift_vec(g,f,2.8)
    SX0=int(max(0,np.floor(X0+min(s_lo[0],s_hi[0]))-3)); SX1=int(min(W,np.ceil(X1+max(s_lo[0],s_hi[0]))+3))
    SY0=int(max(0,np.floor(Y0+min(s_lo[1],s_hi[1]))-3)); SY1=int(min(H,np.ceil(Y1+max(s_lo[1],s_hi[1]))+3))
    canvas=np.zeros((H,W,3),np.float32); cov=np.zeros((H,W),bool)
    if SX1<=SX0 or SY1<=SY0: return canvas,cov
    Is=load(g)[SY0:SY1,SX0:SX1].astype(np.float32); Ds=D[SY0:SY1,SX0:SX1]; Vs=valid[SY0:SY1,SX0:SX1]
    cw,ch=X1-X0,Y1-Y0
    cvs=np.zeros((ch,cw,3),np.float32); cvv=np.zeros((ch,cw),bool)
    for r in np.unique(Ds[Vs]):
        sel=(np.abs(Ds-r)<1e-3)&Vs
        if sel.sum()<3: continue
        s=shift_vec(g,f,float(r))
        M=np.float32([[1,0,SX0-s[0]-X0],[0,1,SY0-s[1]-Y0]])
        w=cv2.warpAffine(sel.astype(np.float32),M,(cw,ch),flags=cv2.INTER_LINEAR)
        c=cv2.warpAffine(Is*sel[...,None],M,(cw,ch),flags=cv2.INTER_LINEAR)
        ok=w>0.55
        cvs[ok]=c[ok]/w[ok][:,None]; cvv|=ok
    canvas[Y0:Y1,X0:X1]=cvs; cov[Y0:Y1,X0:X1]=cvv
    return canvas,cov
def dibr_fill(f,M,nsrc=70):
    out=load(f).astype(np.float32).copy()
    remaining=M.copy()
    srcs=[g for g in range(136,221) if g!=f and abs(cmid(g)[0]-cmid(f)[0])>=10]
    srcs.sort(key=lambda g:abs(g-f))
    used=[]
    for g in srcs[:nsrc]:
        valid=~invalid_src(g)
        canvas,cov=forward_warp(g,f,valid,remaining)
        take=remaining&cov
        # accept only compact chunks from each source (avoids pixel patchwork)
        take=cv2.morphologyEx(take.astype(np.uint8),cv2.MORPH_OPEN,np.ones((7,7),np.uint8))>0
        if take.any():
            out[take]=canvas[take]; remaining&=~take; used.append((g,int(take.sum())))
        if not remaining.any(): break
    left_before_final=int(remaining.sum())
    if remaining.any():
        # leftover gaps: fill larger ones from sources (still compact-only, smaller kernel), then inpaint the rest
        for g in srcs[:nsrc]:
            canvas,cov=forward_warp(g,f,~invalid_src(g),remaining)
            take=cv2.morphologyEx((remaining&cov).astype(np.uint8),cv2.MORPH_OPEN,np.ones((4,4),np.uint8))>0
            if take.any(): out[take]=canvas[take]; remaining&=~take
            if not remaining.any(): break
    used.append(('final_left',left_before_final))
    out=np.clip(out+0.5,0,255).astype(np.uint8)
    if remaining.any():
        out=cv2.inpaint(out,cv2.dilate(remaining.astype(np.uint8)*255,np.ones((3,3),np.uint8)),5,cv2.INPAINT_TELEA)
    return out,used,int(remaining.sum())
def clean_plate(f):
    I=load(f)
    base=I.copy()
    if f>=148:   # HUD removal (back layer only, many sources)
        hm=hud_box_mask(f)
        roi,col,rr,cc=fill_region(f,hm,[1.0],nsrc=40,nmed=5,agg=5)
        base=composite(base,roi,col,hm,feather=1.5)
        holes=(cc>1e5)&hm[roi[1]:roi[3],roi[0]:roi[2]]
        if holes.any():
            hmask=np.zeros((H,W),np.uint8); hmask[roi[1]:roi[3],roi[0]:roi[2]][holes]=255
            base=cv2.inpaint(base,cv2.dilate(hmask,np.ones((3,3),np.uint8)),5,cv2.INPAINT_TELEA)
    M=wiz_generous(f)&~LOGO
    filled,used,left=dibr_fill(f,M)
    a=cv2.GaussianBlur(M.astype(np.float32),(0,0),1.5); a=np.maximum(a,M*1.0)[...,None]
    out=np.clip(base*(1-a)+filled*a+0.5,0,255).astype(np.uint8)
    return out,base,used,left
if __name__=='__main__':
    for f in range(int(sys.argv[1]),int(sys.argv[2])+1):
        out,hudonly,used,left=clean_plate(f)
        cv2.imwrite(f'clean/full_{f}.png',out); cv2.imwrite(f'clean/hud_{f}.png',hudonly)
        print(f,'sources',used,'holes',left,flush=True)
