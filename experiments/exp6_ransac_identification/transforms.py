from __future__ import annotations
import numpy as np

def apply_transform(points, matrix):
    return np.c_[np.asarray(points,float),np.ones(len(points))] @ np.asarray(matrix,float).T

def _similarity(src,dst,reflected=False):
    src=np.asarray(src,float); dst=np.asarray(dst,float)
    if len(src)<2 or np.linalg.norm(src[1]-src[0])<1e-9: return None
    x=src-src.mean(0); y=dst-dst.mean(0)
    if reflected: x=x*np.array([-1.,1.])
    h=x.T@y; u,_,vt=np.linalg.svd(h); r=u@vt
    if np.linalg.det(r)<0: u[:,-1]*=-1; r=u@vt
    scale=np.sum((x@r)*y)/max(np.sum(x*x),1e-12)
    a=(np.diag([-1.,1.])@r if reflected else r)*scale
    t=dst.mean(0)-src.mean(0)@a
    return np.c_[a.T,t]

def fit_transform(src,dst,family):
    src=np.asarray(src,float); dst=np.asarray(dst,float)
    if family=="similarity": return _similarity(src,dst,False)
    if family=="reflected_similarity": return _similarity(src,dst,True)
    if family=="anisotropic":
        if len(src)<3: return None
        from scipy.optimize import least_squares
        def unpack(z):
            c,s=np.cos(z[0]),np.sin(z[0]); a=np.array([[c,-s],[s,c]])@np.diag(z[1:3])
            return np.c_[a.T,z[3:5]]
        def residual(z): return (apply_transform(src,unpack(z))-dst).ravel()
        scale=max(np.ptp(dst,axis=0).mean()/max(np.ptp(src,axis=0).mean(),1e-9),1e-3)
        fit=least_squares(residual,[0,scale,scale,*list(dst.mean(0)-src.mean(0)*scale)],max_nfev=200)
        return unpack(fit.x) if fit.success else None
    if family=="affine":
        if len(src)<3: return None
        area=(src[1,0]-src[0,0])*(src[2,1]-src[0,1])-(src[1,1]-src[0,1])*(src[2,0]-src[0,0])
        if abs(area)<1e-8: return None
        coef=np.linalg.lstsq(np.c_[src,np.ones(len(src))],dst,rcond=None)[0]
        return coef.T
    raise ValueError(family)

def transform_diagnostics(matrix):
    a=np.asarray(matrix)[:,:2]; sv=np.linalg.svd(a,compute_uv=False)
    cond=float(sv[0]/max(sv[-1],1e-12)); shear=float(abs(np.dot(a[0],a[1]))/max(np.linalg.norm(a[0])*np.linalg.norm(a[1]),1e-12))
    return {"condition_number":cond,"anisotropy":cond,"shear":shear}
