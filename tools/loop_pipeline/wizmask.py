import sys; sys.path.insert(0,'.')
from sweep import *
from scipy import ndimage as ndi
ROI=(360,60,980,660)
def explain_residual(f,roi=ROI,win=5):
    x0,y0,x1,y1=roi
    I=load(f)[y0:y1,x0:x1].astype(np.float32)
    best=np.full(I.shape[:2],1e9,np.float32)
    for g in sources_for(f):
        Jg=load(g); Vg=(~static_excl(g)).astype(np.uint8)*255
        for r in RGRID:
            J=warp_from(g,f,r,Jg,roi).astype(np.float32)
            V=warp_from(g,f,r,Vg,roi,interp=cv2.INTER_NEAREST)>0
            e=np.abs(J-I).sum(2)
            e=cv2.boxFilter(e,-1,(win,win))
            e[~V]=1e9
            np.minimum(best,e,out=best)
    return np.minimum(best,1000)
if __name__=='__main__':
    fr=range(int(sys.argv[1]),int(sys.argv[2])+1)
    for f in fr:
        D=explain_residual(f)
        np.save(f'masks/expl_{f}.npy',D.astype(np.float16))
        print(f,flush=True)
