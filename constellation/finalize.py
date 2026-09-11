import cv2
import numpy as np
from .geometry import recognize
from .contracts import ScenePrediction

def auxiliary_map(image):
    raw=image.astype(np.float32)
    dog=cv2.GaussianBlur(raw,(0,0),1)-cv2.GaussianBlur(raw,(0,0),8)
    response=cv2.dilate(dog,np.ones((25,25),np.uint8))
    reference=np.sort(response.ravel()[::10])
    return (np.searchsorted(reference,response)/len(reference)).astype(np.float32)

def finalize(image,queries,raw_queries,patterns,threshold=.72,seed=6643):
    aux=auxiliary_map(image);competing=[]
    for stage,qs,cutoff in [('refined',queries,threshold),('coarse',raw_queries,.65)]:
        ids=[i for i,q in enumerate(qs) if q[0][2]>=cutoff]
        name,m,d=recognize([qs[i][0][:2] for i in ids],patterns,seed=seed,use_quads=True,shear_penalty=2.,tolerance=18.,auxiliary_map=aux)
        best=d['hypotheses'][0] if d.get('hypotheses') else {'score':-1e9,'support':0}
        competing.append((best['score'],name,d,stage))
    competing.sort(key=lambda x:(-x[0],x[1],x[3]))
    _,name,geometry,stage=competing[0]
    nodes=np.array(geometry['hypotheses'][0]['nodes']) if geometry.get('hypotheses') and 'nodes' in geometry['hypotheses'][0] else np.empty((0,2))
    patches=[]
    for q in queries:
        x,y,score,*_=q[0]
        member=int(len(nodes)>0 and np.linalg.norm(nodes-[x,y],axis=1).min()<18)
        patches.append((x,y,member) if score>=threshold else None)
    return ScenePrediction(patches,name,{'geometry':geometry,'selected_geometry_stage':stage,'competing_geometry':[{'stage':s,'name':n,'score':float(v)} for v,n,d,s in competing]})
