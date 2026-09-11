"""Triangle-hash hypotheses with independent, one-to-one affine verification."""
import hashlib
from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import binom
from .quad import quad_jobs

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

def assignment(transformed,points,tolerance,tags=None):
    d=np.linalg.norm(transformed[:,None,:]-points[None,:,:],axis=2)
    candidates=np.argwhere(d<tolerance)
    if not len(candidates):return [],[]
    order=np.argsort(d[candidates[:,0],candidates[:,1]],kind='stable')
    a=set();b=set();pairs=[];res=[]
    for k in order:
        i,j=candidates[k]
        tag=int(tags[j]) if tags is not None else int(j)
        if i not in a and tag not in b:
            a.add(i);b.add(tag);pairs.append((int(i),int(j)));res.append(float(d[i,j]))
    return pairs,res

def recognize(points,patterns,seed=6643,cap=50000,tolerance=24.,alternatives=None,use_quads=False,shear_penalty=0.,auxiliary_map=None):
    pts,groups=consolidate(np.asarray(points,dtype=float).reshape(-1,2))
    if len(pts)<3:return 'unknown',[],{'reason':'fewer than three unique points','groups':groups,'hypotheses':[]}
    scene_ids,scene_desc=triangles(pts)
    if not len(scene_ids):return 'unknown',[],{'reason':'degenerate scene','groups':groups,'hypotheses':[]}
    pool=pts;tags=None;pool_scores=None
    if alternatives is not None:
        pool=[];tags=[];pool_scores=[]
        for i,qs in enumerate(alternatives):
            for q in qs[:5]:
                if q[2]>=qs[0][2]-.10:
                    pool.append(q[:2]);tags.append(groups[i]);pool_scores.append(q[2])
        pool=np.asarray(pool);tags=np.asarray(tags);pool_scores=np.asarray(pool_scores)
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
        if use_quads:
            qjobs=quad_jobs(p,pts,per_class)
            fitting=[(p[t[:3]],pts[u[:3]]) for _,t,u in qjobs]
            fitting += [(p[ti[i]],pts[scene_ids[j]]) for i,j,_ in jobs[:per_class-len(fitting)]]
        else:
            fitting=[(p[ti[i]],pts[scene_ids[j]]) for i,j,_ in jobs]
        best={'name':name,'score':-1e9,'support':0};accepted=0
        for src,dst in fitting:
            a=np.c_[src,np.ones(3)]
            try:matrix=np.linalg.solve(a,dst)
            except np.linalg.LinAlgError:continue
            sv=np.linalg.svd(matrix[:2],compute_uv=False)
            if sv[-1]<50 or sv[0]>6000 or sv[0]/sv[-1]>8:continue
            accepted+=1;mapped=np.c_[p,np.ones(len(p))]@matrix
            pairs,res=assignment(mapped,pool,max(24.,tolerance),tags)
            if len(pairs)>=4:
                for _ in range(2):
                    ii,jj=np.array(pairs).T
                    matrix=np.linalg.lstsq(np.c_[p[ii],np.ones(len(ii))],pool[jj],rcond=None)[0]
                    mapped=np.c_[p,np.ones(len(p))]@matrix
                    pairs,res=assignment(mapped,pool,tolerance,tags)
                    if len(pairs)<4:break
            support=len(pairs)
            if support<4:continue
            # Fitting three nodes alone is never counted as class evidence.
            fraction=min(.8,len(pool)*np.pi*tolerance*tolerance/9e6)
            surprise=-float(binom.logsf(support-4,max(len(p)-3,1),fraction))/np.log(10)
            shear=abs(matrix[0]@matrix[1])/max(np.linalg.norm(matrix[0])*np.linalg.norm(matrix[1]),1e-9)
            score=surprise-.5*np.mean(res)/tolerance-shear_penalty*shear
            auxiliary=0.
            if auxiliary_map is not None:
                inside=(mapped[:,0]>=0)&(mapped[:,1]>=0)&(mapped[:,0]<auxiliary_map.shape[1])&(mapped[:,1]<auxiliary_map.shape[0])
                supported_nodes={i for i,_ in pairs}
                values=[]
                for k,pt in enumerate(mapped):
                    if k in supported_nodes:continue
                    values.append(float(auxiliary_map[int(pt[1]),int(pt[0])]) if inside[k] else 0.)
                auxiliary=float(np.mean(values)) if values else .5
                score+=3.*(auxiliary-.5)
            if score>best['score']:
                best={'name':name,'score':score,'support':support,'coverage':support/len(p),'mean_residual':float(np.mean(res)),'shear':float(shear),'auxiliary':auxiliary,'pairs':pairs,'matrix':matrix.tolist(),'nodes':mapped.tolist(),'matched_points':[pool[j].tolist() for _,j in pairs]}
        best.update(attempted=len(jobs),accepted=accepted,cap_hit=len(jobs)>=per_class)
        hypotheses.append(best)
    hypotheses.sort(key=lambda h:(-h['score'],h['name']))
    if not hypotheses or hypotheses[0]['support']<4:
        return 'unknown',[],{'reason':'no independently verified affine fit','groups':groups,'hypotheses':hypotheses}
    winner=hypotheses[0];supported={int(tags[j]) if tags is not None else j for _,j in winner['pairs']}
    members=[i for i,g in enumerate(groups) if g in supported]
    return winner['name'],members,{'groups':groups,'hypotheses':hypotheses,'score_gap':winner['score']-hypotheses[1]['score'] if len(hypotheses)>1 else None}
