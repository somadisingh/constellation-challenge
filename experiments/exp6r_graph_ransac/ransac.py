from __future__ import annotations
import time
import numpy as np
from experiments.exp6_ransac_identification.transforms import fit_transform,apply_transform,transform_diagnostics
from .assignment import unique_assignment
from .graph_descriptors import describe_graph
from .candidate_pairs import stable_candidates,describe_pairs
from .correspondences import edge_pair_proposals

def _graph_features(edges,pairs):
    nodes={a for a,_ in pairs}; supported=[(a,b) for a,b in edges if a in nodes and b in nodes]
    largest=0
    for start in nodes:
        seen=set(); stack=[start]
        while stack:
            u=stack.pop()
            if u in seen: continue
            seen.add(u); stack += [b if a==u else a for a,b in supported if a==u or b==u]
        largest=max(largest,len(seen))
    return len(supported),largest

def _assign(mapped,candidates,radius,duplicate_safe=True,one_to_one=True):
    if one_to_one: return unique_assignment(mapped,candidates,radius,duplicate_safe)
    pts=np.asarray([c.coordinate for c in candidates]); d=np.linalg.norm(np.asarray(mapped)[:,None,:]-pts[None,:,:],axis=2)
    pairs=[(i,int(np.argmin(row))) for i,row in enumerate(d) if float(np.min(row))<=radius]
    return {"pairs":pairs,"residuals":[float(d[a,b]) for a,b in pairs],"raw_inlier_count":int(np.sum(d<=radius)),"unique_assignment_inlier_count":len(pairs)}

def _refit(nodes,candidates,matrix,family,iterations=3,duplicate_safe=True,one_to_one=True):
    assignment=_assign(apply_transform(nodes,matrix),candidates,18,duplicate_safe,one_to_one)
    for _ in range(iterations):
        if len(assignment["pairs"])<2: break
        src=np.asarray([nodes[a] for a,_ in assignment["pairs"]]); dst=np.asarray([candidates[b].coordinate for _,b in assignment["pairs"]])
        refit_family=family if family in {"similarity","reflected_similarity"} else "similarity"
        new=fit_transform(src,dst,refit_family)
        if new is None: break
        matrix=new; nxt=_assign(apply_transform(nodes,matrix),candidates,18,duplicate_safe,one_to_one)
        if nxt["pairs"]==assignment["pairs"]: assignment=nxt;break
        assignment=nxt
    return matrix,_assign(apply_transform(nodes,matrix),candidates,12,duplicate_safe,one_to_one)

def solve_pattern(graph,candidates,candidate_pairs,strategy="two_edge",max_proposals=500,refit=True,families=("similarity","reflected_similarity"),duplicate_safe=True,one_to_one=True):
    nodes=np.asarray(graph["nodes"],float); gd=describe_graph(graph); proposals=edge_pair_proposals(gd,candidate_pairs,strategy,max_proposals)
    best=None; correctable=[]
    for serial,p in enumerate(proposals):
        if "pattern_nodes" in p:
            ni=list(p["pattern_nodes"]); ci=list(p["candidate_ids"])
        else:
            u,v=p["pattern_edge"]; a,b=p["candidate_pair"]
            if p["reverse"]: a,b=b,a
            ni=[u,v];ci=[a,b]
        src=nodes[ni]; dst=np.asarray([candidates[a].coordinate for a in ci])
        for family in families:
            matrix=fit_transform(src,dst,family)
            if matrix is None: continue
            if refit: matrix,assn=_refit(nodes,candidates,matrix,family,duplicate_safe=duplicate_safe,one_to_one=one_to_one)
            else: assn=_assign(apply_transform(nodes,matrix),candidates,12,duplicate_safe,one_to_one)
            residual=np.asarray(assn["residuals"]); edges,component=_graph_features(graph["edges"],assn["pairs"])
            held=max(0,len(assn["pairs"])-2); frac=len(assn["pairs"])/max(len(nodes),1); p90=float(np.quantile(residual,.9)) if len(residual) else 99.
            score=4*frac+.40*held+.18*edges+.12*component-p90/24
            row={"family":family,"matrix":np.asarray(matrix).tolist(),"pairs":assn["pairs"],"raw_inliers":assn["raw_inlier_count"],
              "unique_inliers":len(assn["pairs"]),"held_out_inliers":held,"support_fraction":frac,"graph_supported_edges":edges,
              "largest_component":component,"median_residual":float(np.median(residual)) if len(residual) else None,
              "p90_residual":p90,"score":float(score),"proposal_rank":serial+1,"strategy":strategy,
              "transform_diagnostics":transform_diagnostics(matrix)}
            key=(row["score"],row["unique_inliers"],-row["p90_residual"])
            if best is None or key>best[0]: best=(key,row)
    return best[1] if best else {"score":-1e9,"unique_inliers":0,"strategy":strategy,"status":"no_hypothesis"}

def solve_scene(alternatives,index,strategy="two_edge",depth=5,max_proposals=500,refit=True,families=("similarity","reflected_similarity"),duplicate_safe=True,one_to_one=True):
    started=time.perf_counter(); candidates=stable_candidates(alternatives,depth); cp=describe_pairs(candidates); ranked=[]
    for name in sorted(index.graphs):
        h=solve_pattern(index.graphs[name],candidates,cp,strategy,max_proposals,refit,families,duplicate_safe,one_to_one); h["name"]=name; ranked.append(h)
    ranked.sort(key=lambda x:(-x["score"],x["name"])); margin=ranked[0]["score"]-ranked[1]["score"]
    accepted=bool(ranked[0]["unique_inliers"]>=6 and ranked[0]["held_out_inliers"]>=3 and ranked[0]["support_fraction"]>=.45 and ranked[0].get("graph_supported_edges",0)>=2 and margin>=1.)
    return {"winner":ranked[0]["name"],"accepted":accepted,"margin":float(margin),"ranked":ranked,
      "runtime_seconds":time.perf_counter()-started,"classes_compared":len(ranked),"strategy":strategy}
