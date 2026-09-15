"""Extract both star nodes and the supplied green line adjacency."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

def extract_pattern_graph(path: Path) -> dict:
    rgba=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim!=3 or rgba.shape[2]<4: raise ValueError(path)
    white=((rgba[:,:,:3].min(axis=2)>190)&(rgba[:,:,3]>100)).astype(np.uint8)
    n,labels,stats,centres=cv2.connectedComponentsWithStats(white)
    keep=np.where(stats[1:,cv2.CC_STAT_AREA]>=2)[0]+1
    nodes=centres[keep].astype(float)
    b,g,r=rgba[:,:,0],rgba[:,:,1],rgba[:,:,2]
    green=((g>120)&(g>r*1.25)&(g>b*1.25)&(rgba[:,:,3]>40)).astype(np.uint8)
    dilated=cv2.dilate(green,np.ones((7,7),np.uint8))
    edges=[]
    for i in range(len(nodes)):
      for j in range(i+1,len(nodes)):
        a,c=nodes[i],nodes[j]; v=c-a; length=float(np.linalg.norm(v))
        if length<12: continue
        # Do not turn a chain through an intermediate star into one long edge.
        intermediate=False
        for k,p in enumerate(nodes):
          if k in (i,j): continue
          t=float(np.dot(p-a,v)/max(np.dot(v,v),1e-9))
          if .08<t<.92 and np.linalg.norm(p-(a+t*v))<8: intermediate=True;break
        if intermediate: continue
        ts=np.linspace(min(10/length,.2),max(1-10/length,.8),max(20,int(length/2)))
        xy=a[None,:]+ts[:,None]*v[None,:]
        xx=np.clip(np.rint(xy[:,0]).astype(int),0,dilated.shape[1]-1)
        yy=np.clip(np.rint(xy[:,1]).astype(int),0,dilated.shape[0]-1)
        coverage=float(dilated[yy,xx].mean())
        if coverage>=.70: edges.append((i,j))
    return {'name':path.stem.removesuffix('_pattern'),'nodes':nodes,
            'edges':edges,'green_pixels':int(green.sum()),'shape':rgba.shape[:2]}

def extract_all(folder: Path) -> dict:
    return {p.stem.removesuffix('_pattern'):extract_pattern_graph(p)
            for p in sorted(folder.glob('*_pattern.png'))}

