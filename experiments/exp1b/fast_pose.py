"""Numerically equivalent local nine-pose search; reuse each scale's prefilter.

The reference implementation repeats a Gaussian prefilter for each of three
angles. Here each scale filters once and remaps its three angles together.
Candidate centre, angle/scale trials, masked NCC, tie order, and raw output agree.
"""
import numpy as np
import cv2
from experiments.exp1.pose import (ANGLE_TRIALS,SCALE_TRIALS,query_grid,_prefilter,masked_ncc)

_GRID=query_grid().reshape(-1,2)-15.5

def select_pose(reps,query_blur,candidate_xy,base_pose,aa_factor=.5):
    angles=np.deg2rad(np.array(ANGLE_TRIALS)+base_pose[0]);scales=np.array(SCALE_TRIALS)*base_pose[1]
    u=_GRID[:,0];v=_GRID[:,1];co=np.cos(angles)[:,None];si=np.sin(angles)[:,None]
    dx=co*u+si*v;dy=-si*u+co*v;h,w=reps.shape
    scores=np.full((3,3),-np.inf);rejected=0
    for j,scale in enumerate(scales):
        x=candidate_xy[0]+scale*dx;y=candidate_xy[1]+scale*dy
        valid=((x>=0)&(x<=w-1)&(y>=0)&(y<=h-1)).all(axis=1)
        prepared=_prefilter(reps.blur,float(scale),aa_factor)
        crops=cv2.remap(prepared,x.astype(np.float32).reshape(96,32),y.astype(np.float32).reshape(96,32),cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,borderValue=0.).reshape(3,32,32)
        for i in range(3):
            if valid[i]:
                score=masked_ncc(query_blur,crops[i])
                if np.isfinite(score):scores[i,j]=score
                else:rejected+=1
            else:rejected+=1
    if not np.isfinite(scores).any():return {'pose':None,'ncc':None,'trials':9,'rejected':rejected,'alignment_failed':True}
    i,j=np.unravel_index(np.argmax(scores),scores.shape)
    return {'pose':(float(base_pose[0]+ANGLE_TRIALS[i]),float(scales[j])),'ncc':float(scores[i,j]),'trials':9,'rejected':rejected,'alignment_failed':False}
