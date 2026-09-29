from common import *
import json
F0,F1=128,220
frames=list(range(F0,F1+1))
def sel(p,layer):
    x,y=p[:,0],p[:,1]
    notlogo=~((x<280)&(y<250)); nothud=~((x>880)&(y<120))
    notwiz=~((x>430)&(x<860)&(y>150)&(y<600))
    if layer=='mid': return notlogo&nothud&notwiz&(y>=60)&(y<545)
    if layer=='fg':  return (y>=585)
    if layer=='far': return notlogo&nothud&notwiz&(y<420)
pairs={'mid':[], 'fg':[], 'far':[]}
ks=[1,2,4,8,16]
for a in frames:
    for k in ks:
        b=a+k
        if b>F1: continue
        pa,pb=matches(a,b)
        if len(pa)<10: continue
        s=sel(pa,'mid'); t,inl=trans_ransac(pa[s],pb[s],thr=1.2)
        if t is None or inl.sum()<25: continue
        pairs['mid'].append((a,b,float(t[0]),float(t[1]),int(inl.sum())))
        tm=t
        # fg: exclude mid-consistent points
        s=sel(pa,'fg')
        if s.sum()>=6:
            d=(pb-pa)[s]; keep=np.linalg.norm(d-tm,axis=1)>3
            if keep.sum()>=4:
                t2,inl2=trans_ransac(pa[s][keep],pb[s][keep],thr=2.0)
                if t2 is not None and inl2.sum()>=4: pairs['fg'].append((a,b,float(t2[0]),float(t2[1]),int(inl2.sum())))
        s=sel(pa,'far')
        d=(pb-pa)[s]; keep=np.linalg.norm(d-tm,axis=1)>2.0
        if keep.sum()>=6 and abs(tm[0])>2:
            t3,inl3=trans_ransac(pa[s][keep],pb[s][keep],thr=1.2)
            # far layer must move in same direction but slower
            if t3 is not None and inl3.sum()>=6 and 0.2<t3[0]/tm[0]<0.9:
                pairs['far'].append((a,b,float(t3[0]),float(t3[1]),int(inl3.sum())))
def solve(pr):
    idx={f:i for i,f in enumerate(frames)}
    n=len(frames)
    rows=[];rhs=[];wts=[]
    for a,b,dx,dy,w in pr:
        r=np.zeros(n); r[idx[b]]=1; r[idx[a]]=-1
        rows.append(r); rhs.append([dx,dy]); wts.append(np.sqrt(min(w,400)))
    # anchor first frame
    r=np.zeros(n); r[0]=1; rows.append(r); rhs.append([0,0]); wts.append(100)
    A=np.array(rows)*np.array(wts)[:,None]; B=np.array(rhs)*np.array(wts)[:,None]
    sol,*_=np.linalg.lstsq(A,B,rcond=None)
    res=[]
    for a,b,dx,dy,w in pr:
        e=sol[idx[b]]-sol[idx[a]]-np.array([dx,dy]); res.append(np.linalg.norm(e))
    return sol,np.array(res)
out={}
for L in ['mid','fg','far']:
    sol,res=solve(pairs[L])
    print(L,"pairs",len(pairs[L]),"resid median %.2f p95 %.2f max %.2f"%(np.median(res),np.percentile(res,95),res.max()))
    out[L]={str(f):[float(sol[i,0]),float(sol[i,1])] for i,f in enumerate(frames)}
json.dump(out,open('campath.json','w'),indent=1)
m=out['mid'];g=out['fg'];fa=out['far']
print(" f    mid_x   mid_y  | fg_x   fg_y  | far_x  far_y | vmid")
for f in frames:
    v=m[str(f)][0]-m[str(f-1)][0] if str(f-1) in m else 0
    print(f"{f:4d} {m[str(f)][0]:8.2f} {m[str(f)][1]:6.2f} | {g[str(f)][0]:8.2f} {g[str(f)][1]:6.2f} | {fa[str(f)][0]:7.2f} {fa[str(f)][1]:6.2f} | {v:6.2f}")
