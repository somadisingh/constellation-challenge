from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree

def quads(points):
    points=np.asarray(points,dtype=float)
    ids=np.array(list(combinations(range(len(points)),4)),dtype=np.int32).reshape(-1,4)
    if not len(ids):return ids,np.empty((0,4))
    p=points[ids];areas=[]
    for i in range(4):
        q=np.delete(p,i,axis=1);u=q[:,1]-q[:,0];v=q[:,2]-q[:,0]
        areas.append((u[:,0]*v[:,1]-u[:,1]*v[:,0])*(-1)**i)
    a=np.stack(areas,axis=1)
    largest=np.argmax(abs(a),axis=1);a*=np.where(a[np.arange(len(a)),largest]>=0,1.,-1.)[:,None]
    a/=np.maximum(abs(a).sum(axis=1,keepdims=True),1e-12)
    order=np.argsort(a,axis=1,kind='stable');a=np.take_along_axis(a,order,axis=1);ids=np.take_along_axis(ids,order,axis=1)
    keep=(np.min(abs(a),axis=1)>.015)&(np.min(np.diff(a,axis=1),axis=1)>.002)
    return ids[keep],a[keep]

def quad_jobs(template,points,budget):
    si,sd=quads(points);ti,td=quads(template)
    if not len(sd) or not len(td):return []
    distance,index=cKDTree(sd).query(td,k=min(4,len(sd)))
    if index.ndim==1:index=index[:,None];distance=distance[:,None]
    jobs=[]
    for i in range(len(ti)):
        for j,d in zip(index[i],distance[i]):jobs.append((float(d),ti[i],si[j]))
    jobs.sort(key=lambda x:x[0]);return jobs[:budget]
