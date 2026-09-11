import cv2
import numpy as np

def refine_candidates(image,patch,candidates):
    q=patch.astype(np.float32)
    q=cv2.GaussianBlur(q,(0,0),.8)
    q=q-cv2.GaussianBlur(q,(0,0),3)
    yy,xx=np.mgrid[:32,:32].astype(np.float32)
    mask=((xx-15.5)**2+(yy-15.5)**2<14**2).astype(np.uint8)*255
    out=[]
    for x,y,score,angle,scale in candidates:
        crop=cv2.getRectSubPix(image,(64,64),(float(x),float(y))).astype(np.float32)
        crop=cv2.GaussianBlur(crop,(0,0),.8)
        crop=crop-cv2.GaussianBlur(crop,(0,0),3*scale)
        a=np.deg2rad(angle);c=scale*np.cos(a);s=scale*np.sin(a)
        warp=np.array([[c,s,31.5-15.5*(c+s)],[-s,c,31.5-15.5*(c-s)]],np.float32)
        original=warp.copy()
        try:
            cc,warp=cv2.findTransformECC(q,crop,warp,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,1e-4),None,3)
            center=warp@np.array([15.5,15.5,1],np.float32)
            sv=np.linalg.svd(warp[:,:2],compute_uv=False)
            if np.linalg.norm(center-31.5)>5 or min(sv)<.65 or max(sv)>1.6 or max(sv)/min(sv)>1.35:warp=original
        except cv2.error:warp=original
        sampled=cv2.warpAffine(crop,warp,(32,32),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        v=q[mask>0];w=sampled[mask>0];v=v-v.mean();w=w-w.mean()
        corr=float(v@w/max(np.linalg.norm(v)*np.linalg.norm(w),1e-6))
        center=warp@np.array([15.5,15.5,1],np.float32)
        out.append((float(x+center[0]-31.5),float(y+center[1]-31.5),corr,float(angle),float(scale)))
    return sorted(out,key=lambda p:-p[2])
