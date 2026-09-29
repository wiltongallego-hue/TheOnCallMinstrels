import sys; sys.path.insert(0,'.')
from dibr import *
MED1=np.load('pano_med_1.00.npy')
def pano_sample(f,OFFY=70):
    c=cmid(f); M=np.float32([[1,0,c[0]],[0,1,c[1]-OFFY]])
    return cv2.warpAffine(MED1,M,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
def counter_edge(f,M):
    """y of the counter/map back edge next to the fill region (first row of a long run of near-depth pixels)"""
    D,_=depth_full(f)
    ys,xs=np.nonzero(M); x0,x1=xs.min(),xs.max()
    cols=list(range(max(0,x0-60),x0-4,3))+list(range(x1+5,min(W,x1+61),3))
    edges=[]
    for x in cols:
        near=D[:,x]>=1.8
        for y in range(380,H-30):
            if near[y:y+30].mean()>0.9: edges.append(y); break
    return float(np.median(edges)) if edges else 500.0+2.0*cmid(f)[1]
def postfill(f):
    C=cv2.imread(f'clean/full_{f}.png').astype(np.float32)
    M=wiz_generous(f)&~LOGO
    e=counter_edge(f,M)
    yy=np.arange(H)[:,None].repeat(W,1).astype(np.float32)
    above=np.clip((e-2-yy)/4.0,0,1)                 # 1 well above the edge, feathered 4px
    fz=cv2.dilate(fire_zone(f).astype(np.uint8),np.ones((9,9),np.uint8))>0
    w=above*M*(~fz)
    w=cv2.GaussianBlur(w.astype(np.float32),(0,0),1.2)*M
    P=pano_sample(f).astype(np.float32)
    # match the median panorama to this frame's lighting using a ring just outside the fill region
    ring=(cv2.dilate(M.astype(np.uint8),np.ones((31,31),np.uint8))>0)&~(cv2.dilate(M.astype(np.uint8),np.ones((5,5),np.uint8))>0)
    ring&=(yy<e-6)&~fz&~LOGO
    if ring.sum()>500:
        I=load(f).astype(np.float32)
        for ch in range(3):
            a=P[...,ch][ring]; b=I[...,ch][ring]
            gain=np.clip((b.std()+1)/(a.std()+1),0.85,1.15); off=b.mean()-gain*a.mean()
            P[...,ch]=P[...,ch]*gain+off
    out=C*(1-w[...,None])+P*w[...,None]
    cv2.imwrite(f'clean/full_{f}.png',np.clip(out+0.5,0,255).astype(np.uint8))
    return e
if __name__=='__main__':
    for f in range(int(sys.argv[1]),int(sys.argv[2])+1):
        print(f,'edge',round(postfill(f),1),flush=True)
