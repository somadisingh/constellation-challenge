"""Triangle-hash hypotheses with independent, one-to-one affine verification."""
import hashlib
from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import binom

def consolidate(points,radius=3.):
    # Anchor grouping avoids transitive chains. Radius is deliberately far below
    # scoring tolerance; close sources separated by >3 pixels stay distinct.
    unique=[];groups=[]
    for p in points:
        distances=np.linalg.norm(np.array(unique)-p,axis=1) if unique else np.array([])
        if len(distances) and distances.min()<radius:groups.append(int(distances.argmin()))
        else:groups.append(len(unique));unique.append(p)
    return np.array(unique).reshape(-1,2),groups

def triangles(points):
    ids=np.array(list(combinations(range(len(points)),3)),dtype=int).reshape(-1,3)
    if not len(ids):return ids,np.empty((0,2))
    p=points[ids]
    edges=np.stack([np.linalg.norm(p[:,1]-p[:,2],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1),np.linalg.norm(p[:,0]-p[:,1],axis=1)],axis=1)
    order=np.argsort(edges,axis=1,kind='stable')
    ids=np.take_along_axis(ids,order,axis=1);edges=np.sort(edges,axis=1)
    p=points[ids];v=p[:,1]-p[:,0];w=p[:,2]-p[:,0]
    area=abs(v[:,0]*w[:,1]-v[:,1]*w[:,0])
    keep=(edges[:,0]>1e-5)&(area>0.02*edges[:,2]**2)
    return ids[keep],edges[keep,:2]/np.maximum(edges[keep,2,None],1e-6)

def assignment(transformed,points,tolerance):
    d=np.linalg.norm(transformed[:,None,:]-points[None,:,:],axis=2)
    candidates=np.argwhere(d<tolerance)
    if not len(candidates):return [],[]
    order=np.argsort(d[candidates[:,0],candidates[:,1]],kind='stable')
    a=set();b=set();pairs=[];res=[]
    for k in order:
        i,j=candidates[k]
        if i not in a and j not in b:
            a.add(i);b.add(j);pairs.append((int(i),int(j)));res.append(float(d[i,j]))
    return pairs,res

def recognize(points,patterns,seed=6643,cap=50000,tolerance=24.):
    pts,groups=consolidate(np.asarray(points,dtype=float).reshape(-1,2))
    if len(pts)<3:return 'unknown',[],{'reason':'fewer than three unique points','groups':groups,'hypotheses':[]}
    scene_ids,scene_desc=triangles(pts)
    if not len(scene_ids):return 'unknown',[],{'reason':'degenerate scene','groups':groups,'hypotheses':[]}
    tree=cKDTree(scene_desc);hypotheses=[]
    per_class=cap//max(len(patterns),1)
    for name,template in sorted(patterns.items()):
        if len(template)<3:
            hypotheses.append({'name':name,'score':-1e9,'support':0,'reason':'pair-only ambiguity'});continue
        # Normalize schematic axis lengths because supplied canvas aspect is arbitrary.
        p=(template-template.mean(axis=0))/np.maximum(np.ptp(template,axis=0),1e-6)
        ti,td=triangles(p)
        if not len(ti):continue
        dist,near=tree.query(td,k=min(20,len(scene_desc)))
        if near.ndim==1:near=near[:,None];dist=dist[:,None]
        jobs=[(i,int(j),float(d)) for i,(ns,ds) in enumerate(zip(near,dist)) for j,d in zip(np.atleast_1d(ns),np.atleast_1d(ds))]
        # Deterministic class-local selection is invariant to catalog order.
        rng=np.random.default_rng(seed+int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
        jobs.sort(key=lambda j:j[2])
        if len(jobs)>per_class:
            first=per_class//2
            ids=rng.choice(np.arange(first,len(jobs)),per_class-first,replace=False)
            jobs=jobs[:first]+[jobs[i] for i in ids]
        best={'name':name,'score':-1e9,'support':0};accepted=0
        for i,j,_ in jobs:
            src=p[ti[i]];dst=pts[scene_ids[j]]
            a=np.c_[src,np.ones(3)]
            try:matrix=np.linalg.solve(a,dst)
            except np.linalg.LinAlgError:continue
            sv=np.linalg.svd(matrix[:2],compute_uv=False)
            if sv[-1]<50 or sv[0]>6000 or sv[0]/sv[-1]>8:continue
            accepted+=1;mapped=np.c_[p,np.ones(len(p))]@matrix
            pairs,res=assignment(mapped,pts,tolerance)
            support=len(pairs)
            if support<4:continue
            # Fitting three nodes alone is never counted as class evidence.
            fraction=min(.8,len(pts)*np.pi*tolerance*tolerance/9e6)
            surprise=-float(binom.logsf(support-4,max(len(p)-3,1),fraction))/np.log(10)
            score=surprise-.5*np.mean(res)/tolerance
            if score>best['score']:
                best={'name':name,'score':score,'support':support,'coverage':support/len(p),'mean_residual':float(np.mean(res)),'pairs':pairs,'matrix':matrix.tolist(),'nodes':mapped.tolist()}
        best.update(attempted=len(jobs),accepted=accepted,cap_hit=len(jobs)>=per_class)
        hypotheses.append(best)
    hypotheses.sort(key=lambda h:(-h['score'],h['name']))
    if not hypotheses or hypotheses[0]['support']<4:
        return 'unknown',[],{'reason':'no independently verified affine fit','groups':groups,'hypotheses':hypotheses}
    winner=hypotheses[0];supported={j for _,j in winner['pairs']}
    members=[i for i,g in enumerate(groups) if g in supported]
    return winner['name'],members,{'groups':groups,'hypotheses':hypotheses,'score_gap':winner['score']-hypotheses[1]['score'] if len(hypotheses)>1 else None}
