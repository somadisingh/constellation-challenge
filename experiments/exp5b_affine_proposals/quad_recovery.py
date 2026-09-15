"""Affine four-point proposal index with independent observed verification."""
from __future__ import annotations
import time
from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import binom

from . import ROOT,OUT,SCENES,NEIGHBORS,PER_CLASS_BUDGET,TOLERANCE
from .pattern_graph import extract_all
from constellation.quad import quads
from experiments.exp1.env import write_json

def _scene_inputs(scene):
    from experiments.exp4b_joint_solver.independent_support import _classical_alternatives
    from constellation.slate import slate_from_candidates
    from constellation.geometry import consolidate
    from constellation.joint import generation_points,build_pool
    alts=_classical_alternatives(scene);sl=[slate_from_candidates(x) for x in alts]
    anchors=np.asarray([s.seed_xy() for s in sl if len(s)]).reshape(-1,2)
    _,groups=consolidate(anchors);live=[s for s in sl if len(s)]
    gen=generation_points(live,groups)
    pool,tags,scores,ranks=build_pool(live,groups,top_k=8)
    return gen,pool,tags,scores,live

def _quad_descriptors_for_ids(points, ids):
    """The affine signed-area descriptor from constellation.quad, for explicit quads."""
    ids=np.asarray(ids,dtype=np.int32).reshape(-1,4)
    if not len(ids):return ids,np.empty((0,4),float)
    q=points[ids];areas=[]
    for i in range(4):
      z=np.delete(q,i,axis=1);u=z[:,1]-z[:,0];v=z[:,2]-z[:,0]
      areas.append((u[:,0]*v[:,1]-u[:,1]*v[:,0])*(-1)**i)
    a=np.stack(areas,axis=1);largest=np.argmax(abs(a),axis=1)
    a*=np.where(a[np.arange(len(a)),largest]>=0,1.,-1.)[:,None]
    a/=np.maximum(abs(a).sum(axis=1,keepdims=True),1e-12)
    order=np.argsort(a,axis=1,kind='stable');a=np.take_along_axis(a,order,axis=1)
    ids=np.take_along_axis(ids,order,axis=1)
    keep=(np.min(abs(a),axis=1)>.015)&(np.min(np.diff(a,axis=1),axis=1)>.002)
    return ids[keep],a[keep]

def one_alternative_scene_quads(slates,k=5):
    """Quads with exactly one alternate candidate and three rank-one candidates."""
    points=[];by_query=[]
    for s in slates:
      ids=[]
      for j in s.order_by_rank()[:k]:ids.append(len(points));points.append(s.xy[j])
      by_query.append(ids)
    points=np.asarray(points,float);n=len(by_query);blocks=[]
    primary=np.asarray([x[0] for x in by_query],np.int32)
    for q,ids in enumerate(by_query):
      others=[i for i in range(n) if i!=q]
      triples=np.asarray(list(combinations(others,3)),np.int32)
      if not len(triples):continue
      for alt in ids[1:]:
        blocks.append(np.c_[np.full(len(triples),alt,np.int32),primary[triples]])
    raw=np.vstack(blocks) if blocks else np.empty((0,4),np.int32)
    ids,desc=_quad_descriptors_for_ids(points,raw)
    return points,ids,desc

def _graph_support(edges, matched):
    supported=[(a,b) for a,b in edges if a in matched and b in matched]
    if not matched:return {'supported_edges':0,'supported_edge_fraction':0.,'largest_component':0}
    adj={i:set() for i in matched}
    for a,b in supported:adj[a].add(b);adj[b].add(a)
    seen=set();largest=0
    for start in matched:
      if start in seen:continue
      stack=[start];comp=set()
      while stack:
       u=stack.pop()
       if u in comp:continue
       comp.add(u);seen.add(u);stack.extend(adj[u]-comp)
      largest=max(largest,len(comp))
    possible=sum(1 for a,b in edges if a in matched or b in matched)
    return {'supported_edges':len(supported),'supported_edge_fraction':len(supported)/max(possible,1),
            'largest_component':largest}

def _score_hypothesis(name,p,ti,si,desc_dist,gen,pool,tags,pool_scores,edges):
    from constellation.joint import fit_affine,valid,assignment
    matrix=fit_affine(p[ti],gen[si])
    if matrix is None or not valid(matrix):return None
    mapped=np.c_[p,np.ones(len(p))]@matrix
    pairs,res=assignment(mapped,pool,TOLERANCE,tags)
    seed=set(map(int,ti));held=[(a,b) for a,b in pairs if a not in seed]
    if len(pairs)<4:return None
    n_groups=len(set(map(int,tags)));chance=min(.8,n_groups*np.pi*TOLERANCE**2/9e6)
    surprise=-float(binom.logsf(max(len(held)-1,0),max(len(p)-4,1),chance))/np.log(10)
    matched={a for a,_ in pairs};graph=_graph_support(edges,matched)
    appearance=float(np.mean([pool_scores[b] for _,b in pairs]))
    mean_res=float(np.mean(res)) if res else 999.
    # Descriptor distance proposes; independent support dominates ranking.
    score=(surprise-.6*mean_res/TOLERANCE+.35*graph['supported_edge_fraction']+
           .08*graph['largest_component']+.20*appearance-.15*np.log1p(desc_dist*1000))
    return {'name':name,'score':float(score),'matrix':matrix.tolist(),'nodes':mapped.tolist(),
      'pairs':pairs,'seed_nodes':list(map(int,ti)),'seed_points':list(map(int,si)),
      'support':len(pairs),'held_out_support':len(held),'mean_residual':mean_res,
      'descriptor_distance':float(desc_dist),'graph':graph,'appearance':appearance}

def run_scene(scene,say=print):
    from experiments.exp1.data import load_scene,query_records
    from constellation.contracts import read_truth
    started=time.perf_counter();graphs=extract_all(ROOT/'patterns')
    rank1,pool,tags,pool_scores,slates=_scene_inputs(scene)
    base_si,base_sd=quads(rank1)
    alt_gen,alt_si,alt_sd=one_alternative_scene_quads(slates,k=5)
    # Keep the rank-one and one-alternate proposal streams separate.  Combining
    # them in one descriptor index lets the much larger alternate stream crowd
    # valid rank-one proposals out of the fixed nearest-neighbour budget.
    streams=[('rank1',rank1,base_si,base_sd,cKDTree(base_sd)),
             ('one_alternate',alt_gen,alt_si,alt_sd,cKDTree(alt_sd))]
    best_by_class={};attempted={}
    for name,g in sorted(graphs.items()):
      nodes=g['nodes'];p=(nodes-nodes.mean(0))/np.maximum(np.ptp(nodes,axis=0),1e-6)
      ti,td=quads(p)
      if not len(ti):continue
      jobs=[]
      for stream_name,stream_gen,stream_si,stream_sd,tree in streams:
        dist,ind=tree.query(td,k=min(NEIGHBORS,len(stream_sd)));dist=np.atleast_2d(dist);ind=np.atleast_2d(ind)
        if dist.shape[0]!=len(td):dist,ind=dist.T,ind.T
        stream_jobs=[(float(d),i,int(j),stream_name,stream_gen,stream_si)
                     for i in range(len(ti)) for d,j in zip(dist[i],ind[i])]
        stream_jobs.sort(key=lambda x:(x[0],x[1],x[2]))
        jobs.extend(stream_jobs[:PER_CLASS_BUDGET])
      attempted[name]=len(jobs)
      best=None
      for d,i,j,stream_name,stream_gen,stream_si in jobs:
        h=_score_hypothesis(name,p,ti[i],stream_si[j],d,stream_gen,pool,tags,pool_scores,g['edges'])
        if h is not None:h['proposal_stream']=stream_name
        if h is not None and (best is None or h['score']>best['score']):best=h
      if best:best_by_class[name]=best
      say(f'  {scene}: {name} best={None if best is None else round(best["score"],3)}')
    ranked=sorted(best_by_class.values(),key=lambda h:(-h['score'],h['name']))
    truth=read_truth(ROOT/'train_ground_truth.csv')[scene];fig=np.array([x[:2] for x in truth.patches if x is not None and x[2]==1],float)
    for h in ranked:
      mapped=np.asarray(h['nodes']);d=np.linalg.norm(mapped[:,None,:]-fig[None,:,:],axis=2);cand=np.argwhere(d<12);order=np.argsort(d[cand[:,0],cand[:,1]]);a=set();b=set();vals=[]
      for z in order:
       i,j=cand[z]
       if i not in a and j not in b:a.add(int(i));b.add(int(j));vals.append(float(d[i,j]))
      h['n_figure_matches_12px']=len(vals);h['placement_correct']=h['name']==truth.constellation and len(vals)>=min(4,len(fig))
    names=[h['name'] for h in ranked]
    return {'scene':scene,'true':truth.constellation,'winner':ranked[0]['name'] if ranked else None,
      'correct':bool(ranked and ranked[0]['name']==truth.constellation),
      'true_rank':names.index(truth.constellation)+1 if truth.constellation in names else None,
      'true_best':next((h for h in ranked if h['name']==truth.constellation),None),
      'top10':ranked[:10],
      'n_scene_quads':{'rank1':len(base_si),'one_alternate':len(alt_si)},
      'attempted_by_class':attempted,
      'seconds':time.perf_counter()-started}

def run(say=print):
    OUT.mkdir(parents=True,exist_ok=True);graphs=extract_all(ROOT/'patterns')
    write_json(OUT/'pattern_graphs.json',{n:{'nodes':g['nodes'].tolist(),'edges':g['edges'],
      'green_pixels':g['green_pixels'],'shape':g['shape']} for n,g in graphs.items()})
    results={s:run_scene(s,say) for s in SCENES};write_json(OUT/'train_results.json',results);return results

if __name__=='__main__':run()
