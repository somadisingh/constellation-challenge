from __future__ import annotations
import numpy as np
from experiments.exp6_ransac_identification.duplicate_audit import Candidate,cluster_points
from .canonical import quantized_xy,stable_digest

def stable_candidates(alternatives,depth=5,cluster_radius=3.):
    raw=[]
    for row in alternatives:
        ranked=sorted(row,key=lambda c:(-float(c[2]),quantized_xy(c[:2])))[:depth]
        qkey=stable_digest(sorted((quantized_xy(c[:2]),round(float(c[2]),8)) for c in row))[:16]
        for rank,c in enumerate(ranked): raw.append((qkey,rank,(float(c[0]),float(c[1])),float(c[2])))
    labels=cluster_points(np.asarray([x[2] for x in raw]),cluster_radius) if raw else []
    out=[Candidate(q,r,int(labels[i]),xy,s) for i,(q,r,xy,s) in enumerate(raw)]
    return sorted(out,key=lambda c:(quantized_xy(c.coordinate),-c.score,c.candidate_rank,str(c.query_id)))

def describe_pairs(candidates,max_pairs=1200):
    pts=np.asarray([c.coordinate for c in candidates],float); n=len(candidates); rows=[]
    # Local signatures are coordinate/geometry based and independent of query enumeration.
    for i in range(n):
        for j in range(i):
            a,b=candidates[i],candidates[j]
            if a.query_id==b.query_id or a.physical_star_cluster_id==b.physical_star_cluster_id: continue
            d=float(np.linalg.norm(pts[i]-pts[j]));
            if d<6: continue
            local=[]
            for z in (i,j):
                dd=np.linalg.norm(pts-pts[z],axis=1); local.append(tuple(int(np.sum((dd>0)&(dd<=r))) for r in (24,60,150)))
            endpoint=sorted((i,j),key=lambda z:(quantized_xy(candidates[z].coordinate),-candidates[z].score))
            rows.append({"candidate_ids":tuple(endpoint),"separation":d,"confidence_pair":tuple(sorted((a.score,b.score),reverse=True)),
              "rank_pair":tuple(sorted((a.candidate_rank,b.candidate_rank))),"local_signatures":tuple(sorted(local)),
              "coordinate_key":tuple(quantized_xy(candidates[z].coordinate) for z in endpoint)})
    rows.sort(key=lambda x:(x["separation"],x["coordinate_key"])); total=len(rows)
    for k,x in enumerate(rows): x["separation_rank"]=k/max(total-1,1)
    rows.sort(key=lambda x:(-sum(x["confidence_pair"]),sum(x["rank_pair"]),-x["separation"],x["coordinate_key"]))
    return rows[:max_pairs]

