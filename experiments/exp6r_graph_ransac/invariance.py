from __future__ import annotations
import json,random,numpy as np
from experiments.exp5c_affine_recovery.pattern_index import PatternIndex,get_or_create_index
from experiments.exp5c_affine_recovery.solver import alternatives_from_prediction_json
from . import ROOT
from .ransac import solve_scene

def _summary(r): return {"winner":r["winner"],"accepted":r["accepted"],"scores":{x["name"]:round(x["score"],8) for x in r["ranked"]}}
def run(max_proposals=8):
    doc=json.loads((ROOT/"outputs/lab/joint_reproduction/pisces.json").read_text()); alt=alternatives_from_prediction_json(doc); idx=get_or_create_index()
    base=_summary(solve_scene(alt,idx,max_proposals=max_proposals))
    rev=PatternIndex.__new__(PatternIndex); rev.graphs=dict(reversed(list(idx.graphs.items())))
    tests={"pattern_dictionary_reverse":_summary(solve_scene(alt,rev,max_proposals=max_proposals))==base}
    rng=random.Random(66006); items=list(idx.graphs.items());rng.shuffle(items); sh=PatternIndex.__new__(PatternIndex);sh.graphs=dict(items)
    tests["pattern_dictionary_random"]=_summary(solve_scene(alt,sh,max_proposals=max_proposals))==base
    qa=list(alt);rng.shuffle(qa);tests["query_order"]=_summary(solve_scene(qa,idx,max_proposals=max_proposals))==base
    ca=[list(x) for x in alt]
    for x in ca:rng.shuffle(x)
    tests["candidate_order"]=_summary(solve_scene(ca,idx,max_proposals=max_proposals))==base
    # Permute every graph's node storage while remapping edges.
    pg={}
    for name,g in idx.graphs.items():
        n=len(g["nodes"]); perm=list(range(n));rng.shuffle(perm); old_to_new={old:new for new,old in enumerate(perm)}
        pg[name]={**g,"nodes":np.asarray(g["nodes"])[perm],"edges":[(old_to_new[a],old_to_new[b]) for a,b in g["edges"]]}
    pi=PatternIndex.__new__(PatternIndex);pi.graphs=pg
    tests["node_storage_order"]=_summary(solve_scene(alt,pi,max_proposals=max_proposals))==base
    tests["repeated_process_proxy"]=_summary(solve_scene(alt,idx,max_proposals=max_proposals))==base
    return {"status":"COMPLETE","scene":"pisces","tests":tests,"all_passed":all(tests.values()),"baseline":base}

