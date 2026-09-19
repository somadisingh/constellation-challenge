from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment

def unique_assignment(mapped, candidates, radius=12.0, duplicate_safe=True):
    """One-to-one node/query/physical-cluster assignment."""
    if not len(mapped) or not candidates: return {"pairs":[],"residuals":[],"raw_inlier_count":0}
    pts=np.asarray([c.coordinate for c in candidates]); d=np.linalg.norm(np.asarray(mapped)[:,None,:]-pts[None,:,:],axis=2)
    raw=int(np.sum(d<=radius))
    # Candidate columns make node-to-observation assignment explicit.  The
    # deterministic pass below additionally enforces query and cluster partitions.
    cost=d.copy(); pick={(n,j):j for n in range(len(mapped)) for j in range(len(candidates))}
    rr,cc=linear_sum_assignment(cost)
    pairs=[]; used_q=set(); used_g=set()
    for n,j in sorted(zip(rr,cc),key=lambda x:cost[x]):
        i=pick.get((int(n),int(j))); c=candidates[i] if i is not None else None
        if c is None or cost[n,j]>radius or c.query_id in used_q or (duplicate_safe and c.physical_star_cluster_id in used_g): continue
        used_q.add(c.query_id); used_g.add(c.physical_star_cluster_id)
        pairs.append((int(n),int(i)))
    return {"pairs":pairs,"residuals":[float(d[n,i]) for n,i in pairs],"raw_inlier_count":raw,
            "unique_assignment_inlier_count":len(pairs)}
