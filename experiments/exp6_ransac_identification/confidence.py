from __future__ import annotations
import math

def score_hypothesis(h,edges):
    if h.get("status")!="ok":
        h["score"]=-1e9
        return h["score"]
    matched={a for a,_ in h["pairs"]}; supported=sum(a in matched and b in matched for a,b in edges)
    h["graph_supported_edges"]=supported
    h["largest_connected_component"]=_component(matched,edges)
    complexity={"similarity":0.,"reflected_similarity":.1,"anisotropic":.4,"affine":.9}[h["family"]]
    residual=h["p90_residual"] if h["p90_residual"] is not None else 24.0
    h["score"]=h["support_fraction"]*4+h["held_out_inliers"]*.35+supported*.15-residual/24-complexity
    return h["score"]

def _component(nodes,edges):
    best=0
    for start in nodes:
        seen=set(); stack=[start]
        while stack:
            x=stack.pop()
            if x in seen: continue
            seen.add(x); stack += [b if a==x else a for a,b in edges if (a==x or b==x) and a in nodes and b in nodes]
        best=max(best,len(seen))
    return best

def accept(winner,margin):
    return bool(winner and winner["unique_inliers"]>=6 and winner["held_out_inliers"]>=3 and
                winner["support_fraction"]>=.45 and winner.get("graph_supported_edges",0)>=2 and margin>=1.0)
