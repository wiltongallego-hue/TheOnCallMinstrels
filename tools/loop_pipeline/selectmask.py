import sys; sys.path.insert(0,'.')
from sweep import *
from scipy import ndimage as ndi
ROI=(360,60,980,660)
def select(f,prev_c,thr=38):
    x0,y0,x1,y1=ROI
    D=np.load(f'masks/expl_{f}.npy').astype(np.float32)
    B=(D>thr).astype(np.uint8)
    B=cv2.morphologyEx(B,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    B=cv2.morphologyEx(B,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    n,lab,st,cen=cv2.connectedComponentsWithStats(B)
    # core: largest component near expected centre
    best=None;bs=-1
    for k in range(1,n):
        cx,cy=cen[k][0]+x0,cen[k][1]+y0
        a=st[k,4]
        d=np.hypot(cx-prev_c[0],(cy-prev_c[1])*0.6)
        score=a-8*d
        if a>1500 and d<160 and score>bs: bs=score;best=k
    core=(lab==best)
    # attach nearby components (effects: sparkles, notes, dust) within 28px of core
    dist=ndi.distance_transform_edt(~core)
    M=core.copy()
    for k in range(1,n):
        if k==best: continue
        comp=lab==k
        if dist[comp].min()<28 and st[k,4]<6000: M|=comp
    M=ndi.binary_fill_holes(cv2.morphologyEx(M.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((9,9),np.uint8)))
    full=np.zeros((H,W),bool); full[y0:y1,x0:x1]=M
    ys,xs=np.nonzero(core)
    c=(np.median(xs)+x0,np.median(ys)+y0)
    return full,c
if __name__=='__main__':
    prev=(640,420)
    tiles=[]
    for f in range(136,221):
        m,prev=select(f,prev)
        np.save(f'masks/wiz_{f}.npy',np.packbits(m))
        if f%3==0:
            I=load(f)[60:660,360:980].copy()
            mm=m[60:660,360:980]
            ov=I.copy(); ov[mm]=(0.5*ov[mm]+0.5*np.array([0,0,255])).astype(np.uint8)
            t=cv2.resize(ov,(248,240)); cv2.putText(t,str(f),(3,14),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1)
            tiles.append(t)
    while len(tiles)%7: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(f"{SCR}/thumbs/wizsel.png",np.vstack([np.hstack(tiles[i:i+7]) for i in range(0,len(tiles),7)]))
