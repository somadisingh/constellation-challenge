from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Candidate:
    query_id: int
    candidate_rank: int
    physical_star_cluster_id: int
    coordinate: tuple[float, float]
    score: float

def cluster_points(points: np.ndarray, radius: float) -> np.ndarray:
    """Order-invariant single-linkage clusters; IDs follow lexicographic centroids."""
    points=np.asarray(points,float).reshape(-1,2); n=len(points); parent=list(range(n))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    for i in range(n):
        for j in range(i):
            if np.linalg.norm(points[i]-points[j]) <= radius: union(i,j)
    roots=[find(i) for i in range(n)]
    groups={r:[i for i,x in enumerate(roots) if x==r] for r in set(roots)}
    ordered=sorted(groups, key=lambda r: tuple(np.mean(points[groups[r]],axis=0)))
    remap={r:i for i,r in enumerate(ordered)}
    return np.asarray([remap[x] for x in roots],int)

def candidates_from_alternatives(alternatives, depth=5, cluster_radius=3.0):
    raw=[]
    for q,row in enumerate(alternatives):
        ranked=sorted(enumerate(row), key=lambda z:(-float(z[1][2]), z[0]))[:depth]
        for rank,(_,c) in enumerate(ranked): raw.append((q,rank,(float(c[0]),float(c[1])),float(c[2])))
    labels=cluster_points(np.asarray([x[2] for x in raw]),cluster_radius) if raw else np.empty(0,int)
    return [Candidate(q,r,int(labels[i]),xy,s) for i,(q,r,xy,s) in enumerate(raw)]

def audit_scene(alternatives) -> dict:
    raw=[]
    for q,row in enumerate(alternatives):
        for r,c in enumerate(row[:5]): raw.append((q,r,float(c[0]),float(c[1]),float(c[2])))
    pts=np.asarray([[x[2],x[3]] for x in raw],float); qs=np.asarray([x[0] for x in raw],int)
    out={"queries":len(alternatives),"candidates_per_query":[min(5,len(x)) for x in alternatives],
         "total_candidates":len(raw),"score_distribution":{}}
    scores=np.asarray([x[4] for x in raw],float)
    if len(scores): out["score_distribution"]={"min":float(scores.min()),"median":float(np.median(scores)),"max":float(scores.max())}
    for radius in (2,3,6,12):
        lab=cluster_points(pts,radius); groups=[np.where(lab==x)[0] for x in np.unique(lab)]
        out[f"radius_{radius}"]={"clusters":len(groups),"multi_query":sum(len(set(qs[g]))>1 for g in groups),
          "same_query_duplicates":sum(any(sum(qs[g]==q)>1 for q in set(qs[g])) for g in groups),
          "max_multiplicity":max((len(g) for g in groups),default=0),"raw_minus_unique":len(raw)-len(groups)}
    for k in (1,3,5):
        ids=[i for i,x in enumerate(raw) if x[1]<k]; lab=cluster_points(pts[ids],3) if ids else []
        out[f"top{k}_collision_clusters"]=sum(len(set(qs[np.asarray(ids)[np.where(lab==z)[0]]]))>1 for z in np.unique(lab)) if ids else 0
    return out

