from __future__ import annotations
import numpy as np

def correspondence_pool(nodes, candidates, strategy="confidence", max_size=4000):
    nodes=np.asarray(nodes,float); centre=nodes.mean(0); span=np.linalg.norm(nodes-centre,axis=1)
    rows=[]
    for ni in range(len(nodes)):
        for ci,c in enumerate(candidates):
            graph_bonus=float(span[ni]/max(span.max(),1e-9))
            score=c.score-0.08*c.candidate_rank+0.03*graph_bonus
            rows.append((ni,ci,score))
    rows.sort(key=lambda x:(-x[2],x[0],candidates[x[1]].query_id,candidates[x[1]].candidate_rank))
    return rows[:max_size]

def minimal_set_available(pool,candidates,size):
    return any(len({x[0] for x in rows})==size and len({candidates[x[1]].query_id for x in rows})==size and
               len({candidates[x[1]].physical_star_cluster_id for x in rows})==size
               for rows in __import__('itertools').combinations(pool[:min(40,len(pool))],size))

