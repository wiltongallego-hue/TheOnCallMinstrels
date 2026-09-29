import sys; sys.path.insert(0,'.')
from dibr import *
import json, glob, os, shutil
REPO=os.environ.get('LOOP_REPO',os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
PUB=os.path.join(REPO,'remotion','public')
RG=2.4     # parallax ratio of the counter/map surface the wizard walks on
meta={}
for fn in glob.glob('sprites/meta_*.json'):
    meta.update({int(k):v for k,v in json.load(open(fn)).items()})
def clean_img(b): return cv2.imread(f'clean/full_{b}.png')
def hud_img(b): return cv2.imread(f'clean/hud_{b}.png')
def render_virtual(p):
    """background for a virtual camera position p (back-layer coords) between existing frames"""
    b=min(range(154,217),key=lambda g:abs(cmid(g)[0]-p[0]))
    src=clean_img(b).astype(np.float32)
    D,dyn=depth_full(b)
    D=D.copy(); D[wiz_generous(b)]=1.0
    canvas=np.zeros((H,W,3),np.float32); cov=np.zeros((H,W),bool)
    d=np.array(p)-cmid(b)
    for r in np.unique(D):
        sel=np.abs(D-r)<1e-3
        sx=r*d[0]; sy=(1+KAPPA*(r-1))*d[1]
        M=np.float32([[1,0,sx],[0,1,sy]])
        w=cv2.warpAffine(sel.astype(np.float32),M,(W,H),flags=cv2.INTER_LINEAR)
        c=cv2.warpAffine(src*sel[...,None],M,(W,H),flags=cv2.INTER_LINEAR)
        ok=w>0.55; canvas[ok]=c[ok]/w[ok][:,None]; cov|=ok
    out=np.where(cov[...,None],canvas,src)      # tiny disocclusions: keep source pixel
    return np.clip(out+0.5,0,255).astype(np.uint8), b
def sprite_entry(s,targetX,feetY,mirror):
    m=meta[s]
    return dict(src=(f'sprites/b{s}.webp' if mirror else f'sprites/s{s}.webp'),x=m['x'],y=m['y'],w=m['w'],h=m['h'],pivotX=round(m['pivotX'],2),
                targetX=round(float(targetX),2),dy=round(float(feetY-m['feetY']),2),mirror=bool(mirror))
def ease_out(t): return 1-(1-t)**2
def ease_in(t): return t*t
frames=[]; virtual={}
# F: forward walk, original frames (HUD removed)
for s in range(142,213): frames.append(dict(bg=f'bg/hud_{s}.webp',sprite=None))
# T1a: camera eases to a stop (212 -> 216); wizard plays his own stopping frames 213..220
c0=cmid(212); c1=cmid(216); ND=8
stop_sprites=[213,214,215,216,217,218,219,220]
for k in range(1,ND+1):
    p=c0+(c1-c0)*ease_out(k/ND)
    img,b=render_virtual(p); name=f'bg/v_decel_{k}.webp'; virtual[name]=img
    s=stop_sprites[k-1]
    tx=meta[s]['pivotX']+RG*(p[0]-cmid(s)[0])
    fy=meta[s]['feetY']+(1+KAPPA*(RG-1))*(p[1]-cmid(s)[1])
    frames.append(dict(bg=name,sprite=sprite_entry(s,tx,fy,False)))
last=frames[-1]['sprite']; hold_tx=last['targetX']; hold_fy=meta[220]['feetY']+last['dy']
# T1b: hold, turn around
for k in range(6):
    frames.append(dict(bg=frames[-1]['bg'],sprite=sprite_entry(220,hold_tx,hold_fy,k>=3)))
# T1c: camera accelerates back (216 -> 212); wizard starts walking left
walk_map=list(range(210,217))
for k in range(1,ND+1):
    p=c1+(c0-c1)*ease_in(k/ND)
    img,b=render_virtual(p); name=f'bg/v_accel_{k}.webp'; virtual[name]=img
    s=walk_map[(k-1)%len(walk_map)]
    t=k/ND
    tx=(1-t)*hold_tx+t*meta[211]['pivotX']; fy=(1-t)*hold_fy+t*meta[211]['feetY']
    frames.append(dict(bg=name,sprite=sprite_entry(s,tx,fy,True)))
# R: camera retraces the walk (211 -> 154); wizard walks back, jumping over the same spots
JUMP2=(193,209); JUMP1=(165,186)
def ground(b):
    """y of the surface the wizard stands on in frame b (main counter or the map extension)"""
    if b>=210: return 590.0
    return 525.0+2.0*cmid(b)[1]
map_cycle=list(range(210,217)); mid_cycle=list(range(187,193)); fire_cycle=list(range(158,165))
wi=0
for b in range(211,153,-1):
    if JUMP2[0]<=b<=JUMP2[1] or JUMP1[0]<=b<=JUMP1[1]:
        e=sprite_entry(b,meta[b]['pivotX'],meta[b]['feetY'],True)
        if JUMP2[0]<=b<=JUMP2[1]: e['src']=f'sprites/s{b}.webp'   # golden burst: keep the full sprite with its sparkles
        frames.append(dict(bg=f'bg/full_{b}.webp',sprite=e))
        continue
    if b>=210:   s=map_cycle[wi%len(map_cycle)]; wi+=1
    elif b>=187: s=mid_cycle[192-b]
    elif b>=158: s=fire_cycle[164-b]
    else:        s=max(b,155)   # last steps: start-walking frames played backwards = coming to a stop (154 carries a stray note)
    tx=meta[b]['pivotX']
    fy=meta[s]['feetY']+(ground(b)-ground(s))
    frames.append(dict(bg=f'bg/full_{b}.webp',sprite=sprite_entry(s,tx,fy,True)))
# T2: back at the fireplace, turn to face right and idle (original frames reversed, HUD removed)
for s in range(153,142,-1): frames.append(dict(bg=f'bg/hud_{s}.webp',sprite=None))
# ---- write assets
for d in ['bg','sprites']:
    shutil.rmtree(f'{PUB}/{d}',ignore_errors=True); os.makedirs(f'{PUB}/{d}')
Q=[cv2.IMWRITE_WEBP_QUALITY,92]
need=set(e['bg'] for e in frames)
for name in need:
    if name in virtual: img=virtual[name]
    else:
        kind,num=os.path.basename(name)[:-5].split('_'); img=cv2.imread(f'clean/{kind}_{num}.png')
    cv2.imwrite(f'{PUB}/{name}',img,Q)
for s in sorted(set(e['sprite']['src'] for e in frames if e['sprite'])):
    base=os.path.basename(s)[:-5]
    cv2.imwrite(f'{PUB}/{s}',cv2.imread(f'sprites/{base}.png',cv2.IMREAD_UNCHANGED),[cv2.IMWRITE_WEBP_QUALITY,100])
json.dump(dict(fps=24,width=1280,height=720,frames=frames),open(f'{PUB}/timeline.json','w'),indent=0)
print('timeline frames',len(frames),'=',len(frames)/24,'s; bg images',len(need))
