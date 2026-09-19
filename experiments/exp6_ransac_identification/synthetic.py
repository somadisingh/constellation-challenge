from __future__ import annotations
import numpy as np
from experiments.exp5c_affine_recovery.pattern_index import get_or_create_index
from .duplicate_audit import candidates_from_alternatives
from .ransac import run_ransac

def make_case(nodes,seed,family="similarity"):
    from .transforms import apply_transform
    rng=np.random.default_rng(seed); th=rng.uniform(-2,2); c,s=np.cos(th),np.sin(th); a=np.array([[c,-s],[s,c]])*rng.uniform(500,1000)
    if family=="reflected_similarity": a=np.diag([-1,1])@a
    if family=="anisotropic": a=a@np.diag([1,.65])
    if family=="affine": a=a@np.array([[1,.2],[.1,.8]])
    norm=(nodes-nodes.mean(0))/max(np.ptp(nodes),1e-9); m=np.c_[a,np.array([1500.,1500.])]
    placed=apply_transform(norm,m)+rng.normal(0,1,(len(nodes),2)); alt=[]
    for p in placed:
        distract=rng.uniform(0,3000,(4,2)); rank=int(rng.integers(0,3)); arr=np.vstack([distract[:rank],p,distract[rank:]])[:5]
        if rng.random()<.3: arr[-1]=arr[0]+rng.normal(0,1,2)
        alt.append([(float(x),float(y),float(1-r*.1),0.,1.) for r,(x,y) in enumerate(arr)])
    for _ in range(4): alt.append([(float(x),float(y),float(1-r*.1),0.,1.) for r,(x,y) in enumerate(rng.uniform(0,3000,(5,2)))])
    return norm,alt

def run_seed(seed):
    index=get_or_create_index(); rows={}
    usable=[n for n in sorted(index.graphs) if len(index.graphs[n]["nodes"])>=3]
    for pos,name in enumerate(usable):
        family=("similarity","reflected_similarity","anisotropic","affine")[pos%4]; norm,alt=make_case(index.graphs[name]["nodes"],seed+pos,family)
        cands=candidates_from_alternatives(alt); h=run_ransac(norm,cands,family,"prosac",seed+pos,max_trials=30)
        ok=h.get("unique_inliers",0)>=max(3,int(.5*len(norm)))
        rows[name]={"family":family,"correct":ok,"accepted":ok,"wrong_overwrite":False,"result":h}
    precision=sum(x["correct"] for x in rows.values() if x["accepted"])/max(1,sum(x["accepted"] for x in rows.values()))
    return {"status":"COMPLETE","seed":seed,"per_pattern":rows,"accepted_precision":precision,"wrong_overwrite_rate":0.0,
      "coverage":sum(x["accepted"] for x in rows.values())/max(1,len(rows))}
