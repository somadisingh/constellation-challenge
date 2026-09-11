"""Classical baselines. Prediction never reads labels or derives classes from scene IDs."""
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import cv2
import numpy as np
from .contracts import ScenePrediction

@dataclass
class Config:
    mode: str = 'radial'
    threshold: float = .72
    alternatives: int = 20
    seed: int = 6643
    threads: int = 4

def preprocess(image):
    x = image.astype(np.float32)
    return x-cv2.GaussianBlur(x,(0,0),3)

def distinct_maxima(scores, count=20, radius=16):
    scores = scores.copy()
    out=[]
    for _ in range(count):
        _,v,_,(x,y)=cv2.minMaxLoc(scores)
        if not np.isfinite(v): break
        out.append((x,y,float(v)))
        scores[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]=-np.inf
    return out

def predict_scene(image, patches, patterns, config):
    cv2.setNumThreads(config.threads)
    start=time.perf_counter()
    sky=preprocess(image)
    results=[]; diagnostics=[]
    if config.mode in ('radial','harmonic','hybrid','final'):
        from .retrieval import build_index,retrieve,verify
        if config.mode in ("harmonic","hybrid","final"):
            from .retrieval import build_harmonic_index as build_index, retrieve_harmonic as retrieve
        points,descriptors,stars=build_index(image)
    for patch in patches:
        q=preprocess(patch)
        if config.mode in ("radial","harmonic","hybrid","final"):
            proposed=retrieve(patch,points,descriptors)
            if config.mode in ("hybrid","final"):
                from .dense import dense_candidates
                proposed=np.unique(np.vstack([proposed,dense_candidates(image,patch)]),axis=0)
            refined=verify(cv2.GaussianBlur(image.astype(np.float32),(0,0),.6),cv2.GaussianBlur(patch.astype(np.float32),(0,0),.6),proposed,config.alternatives)
            coarse=refined
            if config.mode == "final":
                from .refine import refine_candidates
                refined=refine_candidates(image,patch,refined)
            x,y,s,*_=refined[0]
            results.append((x,y,0) if s>=config.threshold else None)
            diagnostics.append({"candidates":refined,"appearance_score":s,"coarse_candidates":coarse})
            continue
        score=cv2.matchTemplate(sky,q,cv2.TM_CCOEFF_NORMED)
        candidates=distinct_maxima(score,config.alternatives)
        # OpenCV template locations are top-left corners. Pixel-center midpoint
        # of an even 32-pixel array is 15.5; the <=12-pixel metric is insensitive
        # to the unresolved dataset convention's possible half-pixel offset.
        candidates=[(x+(patch.shape[1]-1)/2,y+(patch.shape[0]-1)/2,s) for x,y,s in candidates]
        x,y,s=candidates[0]
        results.append((x,y,0) if s>=config.threshold else None)
        diagnostics.append({'candidates':candidates,'appearance_score':s})
    if config.mode == "final":
        from .references import extract_patterns
        from .finalize import finalize
        prediction=finalize(image,[q['candidates'] for q in diagnostics],[q['coarse_candidates'] for q in diagnostics],extract_patterns(patterns),config.threshold,config.seed)
        prediction.diagnostics.update(queries=diagnostics,runtime_seconds=time.perf_counter()-start,config=asdict(config),stage='final')
        return prediction
    from .references import extract_patterns
    from .geometry import recognize
    present_ids=[i for i,p in enumerate(results) if p is not None]
    label,members,geometry=recognize([results[i][:2] for i in present_ids],extract_patterns(patterns),config.seed)
    for j in members:
        i=present_ids[j];x,y,_=results[i];results[i]=(x,y,1)
    return ScenePrediction(results,label,{'queries':diagnostics,'runtime_seconds':time.perf_counter()-start,'config':asdict(config),'stage':config.mode,'geometry':geometry})
