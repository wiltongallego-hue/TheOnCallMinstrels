import sys; sys.path.insert(0,'.')
from dibr import *
from scipy import ndimage as ndi
import json
EROI=(360,60,980,660)
def expl_full(s):
    e=np.zeros((H,W),np.float32); x0,y0,x1,y1=EROI
    e[y0:y1,x0:x1]=np.load(f'masks/expl_{s}.npy').astype(np.float32)
    return cv2.GaussianBlur(e,(3,3),0)
def extract(s):
    I=load(s).astype(np.float32); C=cv2.imread(f'clean/full_{s}.png').astype(np.float32)
    G=wiz_generous(s)
    d=cv2.GaussianBlur(np.abs(I-C).sum(2),(3,3),0); d[~G]=0
    ex=expl_full(s)
    cand=(ex>32)&(d>24)&G
    # fire behind him is dynamic too: drop fire-coloured pixels inside the fireplace zone
    fz=cv2.dilate(fire_zone(s).astype(np.uint8),np.ones((15,15),np.uint8))>0
    B,Gc,R=I[...,0],I[...,1],I[...,2]
    fire=(R>190)&(Gc>110)&(R-B>80)&fz
    cand&=~fire
    cand=cv2.morphologyEx(cand.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
    lab,n=ndi.label(cand)
    core=wiz(s)
    ov=ndi.sum(core,lab,index=np.arange(1,n+1)); sizes=ndi.sum(cand,lab,index=np.arange(1,n+1))
    k=int(np.argmax(ov))+1
    main=ndi.binary_fill_holes(lab==k)
    main=cv2.morphologyEx(main.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))>0
    lab2,n2=ndi.label(main)
    if n2>1:
        sz=ndi.sum(main,lab2,index=np.arange(1,n2+1)); main=lab2==(int(np.argmax(sz))+1)
    main=ndi.binary_fill_holes(main)
    # attached effects: bright sparkles / notes close to the body
    V=I.max(2)
    dist=ndi.distance_transform_edt(~main)
    eff=np.zeros_like(main)
    for j in range(1,n+1):
        if j==k: continue
        comp=lab==j
        if sizes[j-1]<1800 and dist[comp].min()<30 and V[comp].mean()>170: eff|=comp
    body=main|eff
    inner=cv2.erode(body.astype(np.uint8),np.ones((3,3),np.uint8))>0
    band=(cv2.dilate(body.astype(np.uint8),np.ones((5,5),np.uint8))>0)&~inner
    soft=np.clip((d-15)/30,0,1)*np.clip((ex-20)/20,0,1)
    alpha=np.where(inner,1.0,np.where(band,soft,0.0)).astype(np.float32)
    alpha=cv2.GaussianBlur(alpha,(0,0),0.5)*(cv2.dilate(body.astype(np.uint8),np.ones((5,5),np.uint8))>0)
    a3=alpha[...,None]
    F=np.where(a3>0.6,I,np.clip((I-(1-a3)*C)/np.maximum(a3,1e-3),I-60,I+60))
    F=np.clip(F,0,255)
    ys,xs=np.nonzero(alpha>0.02)
    x0,x1=max(0,xs.min()-2),min(W,xs.max()+3); y0,y1=max(0,ys.min()-2),min(H,ys.max()+3)
    rgba=np.dstack([F[y0:y1,x0:x1],alpha[y0:y1,x0:x1]*255]).astype(np.uint8)
    cv2.imwrite(f'sprites/s{s}.png',rgba)
    # ---- body-only variant (used mirrored): no notes, no detached bits
    m2=cv2.morphologyEx(main.astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
    l3,n3=ndi.label(m2)
    if n3>1:
        sz=ndi.sum(m2,l3,index=np.arange(1,n3+1)); m2=l3==(int(np.argmax(sz))+1)
    m2=(cv2.dilate(m2.astype(np.uint8),np.ones((5,5),np.uint8))>0)&main
    nc=note_colored(load(s))
    ln,nn=ndi.label(nc)
    edge=(cv2.dilate(m2.astype(np.uint8),np.ones((5,5),np.uint8))>0)&~(cv2.erode(m2.astype(np.uint8),np.ones((9,9),np.uint8))>0)
    for j in range(1,nn+1):
        comp=ln==j
        if (comp&edge).any() and (comp&m2).any(): m2&=~(cv2.dilate(comp.astype(np.uint8),np.ones((3,3),np.uint8))>0)
    # effect colours never belong to the wizard: gold notes, cyan/teal trails; fire orange near the fireplace
    hsv=cv2.cvtColor(load(s),cv2.COLOR_BGR2HSV); hh,ss,vv=hsv[...,0],hsv[...,1],hsv[...,2]
    fx=((hh>=23)&(hh<=34)&(ss>=60)&(vv>=190))|((hh>=78)&(hh<=104)&(ss>=35)&(vv>=110))
    fx|=fz&(hh<=20)&(ss>=120)&(vv>=200)
    m2&=~(cv2.dilate(fx.astype(np.uint8),np.ones((3,3),np.uint8))>0)
    # body envelope around the feet: cape to the left, lute neck to the right, hat on top
    by0,bx0=np.nonzero(main); fb=by0.max()
    lowp=bx0[by0>fb-0.35*(fb-by0.min())]; pv=float(np.median(lowp))
    env=np.zeros_like(m2); env[max(0,int(fb-300)):int(fb+6),max(0,int(pv-118)):min(W,int(pv+160))]=True
    m2&=env
    m2=cv2.morphologyEx(m2.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))>0
    l4,n4=ndi.label(m2)
    if n4>1:
        sz=ndi.sum(m2,l4,index=np.arange(1,n4+1)); m2=l4==(int(np.argmax(sz))+1)
    m2=ndi.binary_fill_holes(m2)
    inner2=cv2.erode(m2.astype(np.uint8),np.ones((3,3),np.uint8))>0
    band2=(cv2.dilate(m2.astype(np.uint8),np.ones((3,3),np.uint8))>0)&~inner2
    ab=np.where(inner2,1.0,np.where(band2,soft,0.0)).astype(np.float32)
    ab=cv2.GaussianBlur(ab,(0,0),0.5)*(cv2.dilate(m2.astype(np.uint8),np.ones((3,3),np.uint8))>0)
    rgba_b=np.dstack([F[y0:y1,x0:x1],ab[y0:y1,x0:x1]*255]).astype(np.uint8)
    cv2.imwrite(f'sprites/b{s}.png',rgba_b)
    by,bx=np.nonzero(main)
    top,bot=by.min(),by.max()
    low=by>bot-0.35*(bot-top)
    return dict(x=int(x0),y=int(y0),w=int(x1-x0),h=int(y1-y0),pivotX=float(np.median(bx[low])),feetY=float(bot),top=float(top))
if __name__=='__main__':
    meta={}
    for s in range(int(sys.argv[1]),int(sys.argv[2])+1):
        meta[s]=extract(s); print(s,meta[s],flush=True)
    json.dump(meta,open(f'sprites/meta_{sys.argv[1]}.json','w'))
