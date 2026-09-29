import sys; sys.path.insert(0,'.')
from dibr import *
mode=sys.argv[1]
for f in range(int(sys.argv[2]),int(sys.argv[3])+1):
    if mode=='hud':
        I=load(f)
        if f>=148:
            hm=hud_box_mask(f)
            roi,col,rr,cc=fill_region(f,hm,[1.0],nsrc=40,nmed=5,agg=5)
            I=composite(I,roi,col,hm,feather=1.5)
        cv2.imwrite(f'clean/hud_{f}.png',I); print('hud',f,flush=True)
    else:
        out,hudonly,used,left=clean_plate(f)
        cv2.imwrite(f'clean/full_{f}.png',out); cv2.imwrite(f'clean/hud_{f}.png',hudonly)
        print('full',f,'holes',left,flush=True)
