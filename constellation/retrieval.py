"""Rotation-invariant radial proposals followed by masked rotation/scale verification."""
import cv2
import numpy as np
from scipy.spatial import cKDTree

def normalize(x):
    x=x-x.mean(axis=-1,keepdims=True)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-6)

def radial_maps(image):
    yy,xx=np.mgrid[-20:21,-20:21];r=np.hypot(xx,yy)
    maps=[]
    for radius in np.arange(0,19,2):
        kernel=np.exp(-.5*((r-radius)/1.15)**2).astype(np.float32);kernel/=kernel.sum()
        maps.append(cv2.filter2D(image,-1,kernel))
    return np.stack(maps,axis=-1)

def build_index(image,stride=4):
    raw=image.astype(np.float32)
    maps=radial_maps(raw)
    yy,xx=np.mgrid[20:raw.shape[0]-20:stride,20:raw.shape[1]-20:stride]
    dense=np.c_[xx.ravel(),yy.ravel()]
    blur=cv2.GaussianBlur(raw,(0,0),.8)
    dog=blur-cv2.GaussianBlur(raw,(0,0),2.5)
    maxima=(dog==cv2.dilate(dog,np.ones((5,5),np.uint8)))&(dog>1.5)
    y,x=np.where(maxima);keep=(x>=20)&(x<raw.shape[1]-20)&(y>=20)&(y<raw.shape[0]-20)
    stars=np.c_[x[keep],y[keep]]
    points=np.unique(np.vstack([dense,stars]),axis=0)
    desc=normalize(maps[points[:,1],points[:,0]])
    return points,desc,stars

def query_descriptors(patch):
    theta=np.arange(64)*2*np.pi/64
    out=[]
    for scale in (.75,.87,1.,1.15,1.33):
        radius=np.arange(0,19,2)/scale
        mx=(15.5+radius[:,None]*np.cos(theta)).astype(np.float32)
        my=(15.5+radius[:,None]*np.sin(theta)).astype(np.float32)
        values=cv2.remap(patch.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        out.append(values.mean(axis=1))
    return normalize(np.array(out))

def retrieve(patch,points,descriptors,budget=1000):
    d=query_descriptors(patch)
    scores=np.max(descriptors@d.T,axis=1)
    ids=np.argpartition(scores,-budget)[-budget:]
    return points[ids[np.argsort(scores[ids])[::-1]]]

def verify(image,patch,candidates,keep=20):
    # Sample each sky neighborhood and compare to transformed query over a
    # common circular support; float32 maps avoid OpenCV sampling dtype errors.
    yy,xx=np.mgrid[-12:13:2,-12:13:2].astype(np.float32)
    mask=(xx*xx+yy*yy<=144);xx=xx[mask];yy=yy[mask]
    xmap=candidates[:,0,None].astype(np.float32)+xx
    ymap=candidates[:,1,None].astype(np.float32)+yy
    samples=cv2.remap(image,xmap,ymap,cv2.INTER_LINEAR)
    samples=normalize(samples)
    templates=[];transforms=[]
    for scale in (.75,.87,1.,1.15,1.33):
        for angle in np.arange(0,360,15):
            a=np.deg2rad(angle)
            mx=(15.5+(np.cos(a)*xx-np.sin(a)*yy)/scale).astype(np.float32)[None,:]
            my=(15.5+(np.sin(a)*xx+np.cos(a)*yy)/scale).astype(np.float32)[None,:]
            valid=(mx>=0)&(mx<=31)&(my>=0)&(my<=31)
            if not valid.all():continue
            v=cv2.remap(patch,mx,my,cv2.INTER_LINEAR).ravel()
            templates.append(v);transforms.append((float(angle),float(scale)))
    templates=normalize(np.array(templates))
    scores=samples@templates.T
    best=scores.max(axis=1); order=np.argsort(best)[::-1]
    selected=[]
    for idx in order:
        pt=candidates[idx]
        if any(np.linalg.norm(pt-np.array(s[:2]))<8 for s in selected):continue
        angle,scale=transforms[int(scores[idx].argmax())]
        selected.append((float(pt[0]),float(pt[1]),float(best[idx]),angle,scale))
        if len(selected)>=keep:break
    # Refine each alternative around its immutable coarse pose.
    refined=[]
    for x,y,_,angle,scale in selected:
        best=(-2,None)
        for da in (-7.5,0,7.5):
            a=np.deg2rad(angle+da)
            for ds in (.94,1,1.06):
                mx=(15.5+(np.cos(a)*xx-np.sin(a)*yy)/(scale*ds)).astype(np.float32)[None,:]
                my=(15.5+(np.sin(a)*xx+np.cos(a)*yy)/(scale*ds)).astype(np.float32)[None,:]
                if mx.min()<0 or mx.max()>31 or my.min()<0 or my.max()>31:continue
                q=normalize(cv2.remap(patch,mx,my,cv2.INTER_LINEAR))
                offsets=np.array([(dx,dy) for dx in (-2,-1,0,1,2) for dy in (-2,-1,0,1,2)],np.float32)
                sm=cv2.remap(image,(x+offsets[:,0,None]+xx).astype(np.float32),(y+offsets[:,1,None]+yy).astype(np.float32),cv2.INTER_LINEAR)
                corr=normalize(sm)@q.ravel();idx=int(corr.argmax())
                if corr[idx]>best[0]:best=(float(corr[idx]),(x+float(offsets[idx,0]),y+float(offsets[idx,1]),float(corr[idx]),angle+da,scale*ds))
        if best[1] is not None:refined.append(best[1])
    return sorted(refined,key=lambda x:-x[2])

def harmonic_features(image,points,scale=1.):
    """Circular harmonic magnitudes, invariant to in-plane rotation."""
    size=int(np.ceil(17*scale)); yy,xx=np.mgrid[-size:size+1,-size:size+1]
    radius=np.hypot(xx,yy)/scale;theta=np.arctan2(yy,xx)
    features=[]
    for r in (2,5,8,11,14):
        ring=np.exp(-.5*((radius-r)/1.2)**2);ring/=ring.sum()
        for order in (0,1,2,3):
            real=cv2.filter2D(image,-1,(ring*np.cos(order*theta)).astype(np.float32))
            rv=real[points[:,1],points[:,0]]
            if order:
                imag=cv2.filter2D(image,-1,(ring*np.sin(order*theta)).astype(np.float32))
                iv=imag[points[:,1],points[:,0]]
                features.append(np.hypot(rv,iv))
            else:features.append(rv)
    feat=np.stack(features,axis=-1)
    feat[:,::4]-=feat[:,::4].mean(axis=1,keepdims=True)
    return feat/np.maximum(np.linalg.norm(feat,axis=1,keepdims=True),1e-6)

def build_harmonic_index(image,stride=4):
    raw=cv2.GaussianBlur(image.astype(np.float32),(0,0),.6)
    yy,xx=np.mgrid[20:raw.shape[0]-20:stride,20:raw.shape[1]-20:stride]
    dense=np.c_[xx.ravel(),yy.ravel()]
    dog=raw-cv2.GaussianBlur(raw,(0,0),2.5)
    maxima=(dog==cv2.dilate(dog,np.ones((5,5),np.uint8)))&(dog>1.5)
    y,x=np.where(maxima);keep=(x>=20)&(x<raw.shape[1]-20)&(y>=20)&(y<raw.shape[0]-20)
    stars=np.c_[x[keep],y[keep]]
    points=np.unique(np.vstack([dense,stars]),axis=0)
    return points,harmonic_features(raw,points),stars

def retrieve_harmonic(patch,points,descriptors,budget=2000):
    raw=cv2.GaussianBlur(patch.astype(np.float32),(0,0),.6)
    ds=np.concatenate([harmonic_features(raw,np.array([[15,15],[16,16],[15,16],[16,15]]),scale) for scale in (.75,.87,1.,1.15,1.33)])
    scores=np.max(descriptors@ds.T,axis=1)
    ids=np.argpartition(scores,-budget)[-budget:]
    return points[ids[np.argsort(scores[ids])[::-1]]]
