import sys; sys.path.insert(0,'.')
from sweep import *
from scipy import ndimage as ndi
FALL=range(136,221)
def wiz(f):
    return np.unpackbits(np.load(f'masks/wiz_{f}.npy'))[:H*W].reshape(H,W).astype(bool)
_W={}
def wiz_generous(f,dil=7):
    if f not in _W:
        m=np.zeros((H,W),bool)
        for k in range(-2,3):
            g=f+k
            if 136<=g<=220: m|=wiz(g)
        m=cv2.dilate(m.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dil+1,2*dil+1)))>0
        _W[f]=m
    return _W[f]
ZERO_BOX=(1170,46,1230,108)     # x0,y0,x1,y1 of the "0" score glyph (frames 148-189)
CNT_BOX=(958,34,1234,112)       # gem + counter (frames >=190)
def hud_box_mask(f):
    m=np.zeros((H,W),bool)
    if 148<=f: x0,y0,x1,y1=ZERO_BOX; m[y0:y1,x0:x1]=True
    if 190<=f: x0,y0,x1,y1=CNT_BOX; m[y0:y1,x0:x1]=True
    return m
def invalid_src(g):
    return LOGO|hud_box_mask(g)|wiz_generous(g)
def fill_region(f,fillmask,rgrid,nsrc=12,nmed=7,agg=15,bias1=4.0,force_r1_above=None):
    ys,xs=np.nonzero(fillmask)
    x0=max(0,xs.min()-10);x1=min(W,xs.max()+11);y0=max(0,ys.min()-10);y1=min(H,ys.max()+11)
    roi=(x0,y0,x1,y1)
    cands=[g for g in range(136,221) if g!=f and abs(cmid(g)[0]-cmid(f)[0])>=12]
    cands.sort(key=lambda g:abs(g-f))
    cands=cands[:nsrc]+[g for g in cands[nsrc:] if (g-f)%5==0][:6]
    cands.sort(key=lambda g:abs(g-f))
    srcI={g:load(g) for g in cands}
    srcV={g:(~invalid_src(g)).astype(np.uint8)*255 for g in cands}
    h,w=y1-y0,x1-x0
    costs=[];cols=[]
    for r in rgrid:
        S=[];V=[]
        for g in cands:
            S.append(warp_from(g,f,r,srcI[g],roi).astype(np.float32))
            V.append(warp_from(g,f,r,srcV[g],roi,interp=cv2.INTER_NEAREST)>0)
        S=np.stack(S);V=np.stack(V)
        Sn=S.copy(); Sn[~V]=np.nan
        cnt=V.sum(0)
        med=np.nanmedian(Sn,0)
        mad=np.nanmedian(np.abs(Sn-med[None]).sum(3),0)
        cost=np.nan_to_num(mad,nan=500.0)
        cost[cnt<2]+=300; cost[cnt==0]=1e6
        cost=cv2.boxFilter(cost,-1,(agg,agg))
        if r==1.0: cost-=bias1
        if force_r1_above is not None and r!=1.0:
            yy=np.arange(y0,y1)[:,None]; cost=cost+np.where(yy<force_r1_above,1e7,0)
        order=np.cumsum(V,0)
        pick=V&(order<=nmed)
        Sp=S.copy(); Sp[~pick]=np.nan
        cols.append(np.nan_to_num(np.nanmedian(Sp,0)).astype(np.float32))
        costs.append(cost.astype(np.float32))
    costs=np.stack(costs); cols=np.stack(cols)
    lab=np.argmin(costs,0).astype(np.uint8)
    lab=cv2.medianBlur(lab,9)
    col=np.take_along_axis(cols,lab[None,:,:,None].repeat(3,3).astype(np.int64),0)[0]
    bc=np.take_along_axis(costs,lab[None].astype(np.int64),0)[0]
    return roi,col,np.array(rgrid)[lab],bc
FILL_R=[1.0,1.25,1.5]+[round(x,2) for x in np.arange(2.0,2.61,0.04)]
def composite(I,roi,col,mask,feather=2):
    x0,y0,x1,y1=roi
    out=I.astype(np.float32).copy()
    a=mask[y0:y1,x0:x1].astype(np.float32)
    if feather>0: a=cv2.GaussianBlur(a,(0,0),feather); a=np.maximum(a,mask[y0:y1,x0:x1]*1.0)
    out[y0:y1,x0:x1]=out[y0:y1,x0:x1]*(1-a[...,None])+col*a[...,None]
    return np.clip(out+0.5,0,255).astype(np.uint8)
def clean_frame(f,remove_wizard=True,remove_hud=True):
    I=load(f)
    out=I.copy()
    info={}
    if remove_hud and f>=148:
        hm=hud_box_mask(f)
        roi,col,rr,cc=fill_region(f,hm,[1.0],nsrc=40,nmed=5,agg=5)
        holes=(cc>1e6)&hm[roi[1]:roi[3],roi[0]:roi[2]]
        out=composite(out,roi,col,hm,feather=1.5)
        info['hud_holes']=int(holes.sum())
        if holes.any():
            hmask=np.zeros((H,W),np.uint8); hmask[roi[1]:roi[3],roi[0]:roi[2]][holes]=255
            out=cv2.inpaint(out,cv2.dilate(hmask,np.ones((3,3),np.uint8)),5,cv2.INPAINT_TELEA)
    if remove_wizard:
        wm=wiz_generous(f)&~LOGO
        roi,col,rr,cc=fill_region(f,wm,FILL_R)
        out=composite(out,roi,col,wm,feather=2)
        info['wiz_bad']=int(((cc>200)&wm[roi[1]:roi[3],roi[0]:roi[2]]).sum())
    return out,info
if __name__=='__main__':
    mode=sys.argv[1]
    for f in range(int(sys.argv[2]),int(sys.argv[3])+1):
        if mode=='full':
            out,info=clean_frame(f,True,True); cv2.imwrite(f'clean/full_{f}.png',out)
        else:
            out,info=clean_frame(f,False,True); cv2.imwrite(f'clean/hud_{f}.png',out)
        print(mode,f,info,flush=True)
