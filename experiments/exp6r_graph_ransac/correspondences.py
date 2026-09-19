from __future__ import annotations
import numpy as np

def edge_pair_proposals(graph_desc,candidate_pairs,strategy="two_edge",max_proposals=1200):
    if strategy not in {"single_edge","two_edge","uniform","confidence"}: raise ValueError(strategy)
    proposals=[]
    if strategy=="two_edge":
        edges=graph_desc["edges"]
        # Two adjacent graph edges map to two observed edges sharing a physical
        # candidate.  Length-ratio and included-angle compatibility are invariant
        # to global scale, rotation and reflection.
        for i,e1 in enumerate(edges):
            for e2 in edges[i+1:]:
                shared=set(e1["endpoints"])&set(e2["endpoints"])
                if len(shared)!=1: continue
                centre=next(iter(shared)); u=next(x for x in e1["endpoints"] if x!=centre); v=next(x for x in e2["endpoints"] if x!=centre)
                pr=e1["normalized_length"]/max(e2["normalized_length"],1e-9)
                for x,p1 in enumerate(candidate_pairs[:24]):
                    for p2 in candidate_pairs[x+1:24]:
                        common=set(p1["candidate_ids"])&set(p2["candidate_ids"])
                        if len(common)!=1: continue
                        cc=next(iter(common)); a=next(z for z in p1["candidate_ids"] if z!=cc); b=next(z for z in p2["candidate_ids"] if z!=cc)
                        cr=p1["separation"]/max(p2["separation"],1e-9); ratio_error=abs(np.log(max(pr,1e-9)/max(cr,1e-9)))
                        conf=sum(p1["confidence_pair"])+sum(p2["confidence_pair"])-.08*(sum(p1["rank_pair"])+sum(p2["rank_pair"]))
                        score=conf-1.5*ratio_error+.15*(sum(e1["degree_multiset"])+sum(e2["degree_multiset"]))
                        for swap in (False,True):
                            proposals.append({"pattern_edge":e1["endpoints"],"candidate_pair":p1["candidate_ids"],"reverse":swap,
                              "pattern_nodes":(u,centre,v),"candidate_ids":(b,cc,a) if swap else (a,cc,b),
                              "score":float(score),"strategy":strategy,"edge_key":(e1["canonical_key"],e2["canonical_key"]),
                              "candidate_key":(p1["coordinate_key"],p2["coordinate_key"])})
        proposals.sort(key=lambda x:(-x["score"],x["edge_key"],x["candidate_key"],x["reverse"]))
        return proposals[:max_proposals]
    for ei,e in enumerate(graph_desc["edges"]):
        for pi,p in enumerate(candidate_pairs):
            structural=(2 if e["type"]=="branching" else 1)+(sum(e["degree_multiset"])-2)*.1
            rank_compat=1-abs(e["length_rank"]-p["separation_rank"])
            conf=sum(p["confidence_pair"])-.08*sum(p["rank_pair"])
            score=rank_compat+structural*.15+conf*.2
            if strategy=="uniform": score=0.
            elif strategy=="confidence": score=conf
            elif strategy=="two_edge": score+=.2*rank_compat*structural
            for reverse in (False,True): proposals.append({"pattern_edge":e["endpoints"],"candidate_pair":p["candidate_ids"],
              "reverse":reverse,"score":float(score),"strategy":strategy,"edge_key":e["canonical_key"],"candidate_key":p["coordinate_key"]})
    proposals.sort(key=lambda x:(-x["score"],x["edge_key"],x["candidate_key"],x["reverse"]))
    return proposals[:max_proposals]

def strategy_signature(proposals):
    return [(p.get("pattern_nodes",p["pattern_edge"]),p.get("candidate_ids",p["candidate_pair"]),p["reverse"]) for p in proposals]
