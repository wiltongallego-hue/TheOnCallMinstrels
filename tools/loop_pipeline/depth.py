import sys; sys.path.insert(0,'.')
from fill import *
RSET=np.round(np.arange(1.0,2.76,0.05),2)
DROI=(0,40,1280,680)
def neighbours(g):
    """frames with ~15px and ~32px camera displacement on each side"""
    out=[]
    for target in (15,32):
        for side in (-1,1):
            best=None;bd=1e9
            for k in range(1,60):
                h=g+side*k
                if h<136 or h>220: break
                d=abs(abs(cmid(h)[0]-cmid(g)[0])-target)
                if d<bd: bd=d;best=h
            if best is not None and abs(cmid(best)[0]-cmid(g)[0])>6: out.append(best)
    if len(out)<2:   # static-camera frames: look further ahead
        for target in (45,80):
            for k in range(1,90):
                h=g+k
                if h>220: break
                if abs(cmid(h)[0]-cmid(g)[0])>=target: out.append(h); break
    return sorted(set(out))
def depth_map(g,bias=5.0,agg=7):
    x0,y0,x1,y1=DROI
    I=load(g)[y0:y1,x0:x1].astype(np.float32)
    nb=neighbours(g)
    NI={h:load(h) for h in nb}; NV={h:(~invalid_src(h)).astype(np.uint8)*255 for h in nb}
    best=np.full(I.shape[:2],1e9,np.float32); bestr=np.ones(I.shape[:2],np.float32); second=np.full(I.shape[:2],1e9,np.float32)
    for r in RSET:
        e=np.full(I.shape[:2],1e4,np.float32)
        for h in nb:
            J=warp_from(h,g,r,NI[h],DROI).astype(np.float32)
            V=warp_from(h,g,r,NV[h],DROI,interp=cv2.INTER_NEAREST)>0
            en=np.abs(J-I).sum(2); en[~V]=1e4
            np.minimum(e,en,out=e)
        c=cv2.boxFilter(e,-1,(agg,agg))
        if r==1.0: c-=bias
        upd=c<best
        second=np.where(upd,best,np.minimum(second,c))
        bestr[upd]=r; best[upd]=c[upd]
    return bestr,best,second-best,nb
if __name__=='__main__':
    for g in range(int(sys.argv[1]),int(sys.argv[2])+1):
        r,b,conf,nb=depth_map(g)
        np.savez_compressed(f'depth/d_{g}.npz',r=r.astype(np.float16),cost=np.minimum(b,1e4).astype(np.float16),conf=np.minimum(conf,1e4).astype(np.float16))
        print(g,nb,flush=True)
