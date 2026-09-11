"""Slower exhaustive coarse appearance fallback; no source detector dependency."""
import cv2
import numpy as np

def dense_candidates(image,patch,per_pose=3):
    # Downsample only scene search. Candidate poses are reverified at full resolution.
    small=cv2.resize(image.astype(np.float32),None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    small=small-cv2.GaussianBlur(small,(0,0),2.)
    q=patch.astype(np.float32)
    q=q-cv2.GaussianBlur(q,(0,0),4.)
    coords=np.arange(-7,8,dtype=np.float32)*2
    xx,yy=np.meshgrid(coords,coords)
    candidates=[]
    for scale in (.75,.87,1.,1.15,1.33):
        # Largest inscribed square valid under every rotation at this scale.
        half=int(np.floor(15*scale/np.sqrt(2)/2))
        yy,xx=np.mgrid[-half:half+1,-half:half+1].astype(np.float32)*2
        for angle in np.arange(0,360,15):
            a=np.deg2rad(angle)
            mx=(15.5+(np.cos(a)*xx-np.sin(a)*yy)/scale).astype(np.float32)
            my=(15.5+(np.sin(a)*xx+np.cos(a)*yy)/scale).astype(np.float32)
            template=cv2.remap(q,mx,my,cv2.INTER_LINEAR)
            scores=cv2.matchTemplate(small,template,cv2.TM_CCOEFF_NORMED)
            for _ in range(per_pose):
                _,v,_,(x,y)=cv2.minMaxLoc(scores)
                candidates.append(((x+half)*2+.5,(y+half)*2+.5))
                scores[max(0,y-8):y+9,max(0,x-8):x+9]=-1
    return np.unique(np.array(candidates,dtype=np.float32),axis=0)
