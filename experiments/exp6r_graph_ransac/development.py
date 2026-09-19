from __future__ import annotations
import json,numpy as np
from experiments.exp5c_affine_recovery import TRAIN_GROUND_TRUTH
from experiments.exp5c_affine_recovery.pattern_index import get_or_create_index
from experiments.exp5c_affine_recovery.solver import alternatives_from_prediction_json
from constellation.contracts import read_truth
from . import ROOT,SEEDS
from .candidate_pairs import stable_candidates,describe_pairs
from .graph_descriptors import describe_graph
from .correspondences import edge_pair_proposals
from .ransac import solve_scene
from experiments.exp6_ransac_identification.transforms import fit_transform,apply_transform

SOURCE=ROOT/"outputs/lab/joint_reproduction"

def _oracle_trace(graph,alternatives,figure):
    candidates=stable_candidates(alternatives); pts=np.asarray([c.coordinate for c in candidates]); figure=np.asarray(figure,float)
    available=int(sum(np.min(np.linalg.norm(pts-p,axis=1))<=12 for p in figure)) if len(pts) else 0
    cp=describe_pairs(candidates); props=edge_pair_proposals(describe_graph(graph),cp,"two_edge",500)
    best_support=0; first=None; correct_proposals=0
    for rank,p in enumerate(props,1):
        u,v=p["pattern_edge"]; a,b=p["candidate_pair"]
        if p["reverse"]: a,b=b,a
        for fam in ("similarity","reflected_similarity"):
            m=fit_transform(np.asarray(graph["nodes"])[[u,v]],[candidates[a].coordinate,candidates[b].coordinate],fam)
            if m is None: continue
            mapped=apply_transform(graph["nodes"],m); d=np.linalg.norm(mapped[:,None,:]-figure[None,:,:],axis=2)
            support=0; used=set()
            for x,y in sorted(np.argwhere(d<=12),key=lambda z:d[tuple(z)]):
                if int(x) not in used and int(y) not in used: used.add(int(x));used.add(int(y));support+=1
            if support>=2: correct_proposals+=1
            if support>best_support: best_support=support;first=rank
    return {"correct_candidate_coordinates_available":available,"correct_physical_clusters_available":available,
      "correct_graph_edge_match_exists":best_support>=2,"correct_edge_pair_proposal_rank":first,
      "correct_minimal_set_proposal_rank":first,"correct_transform_proposed":best_support>=2,
      "oracle_best_true_node_support_12px":best_support,"proposal_hypotheses_evaluated":len(props)*2,
      "correct_proposal_hypotheses":correct_proposals,"proposal_inlier_fraction":correct_proposals/max(len(props)*2,1)}

def run(max_proposals=30):
    index=get_or_create_index(); truth=read_truth(TRAIN_GROUND_TRUTH); rows={}
    for pos,scene in enumerate(("pisces","scorpius","taurus")):
        doc=json.loads((SOURCE/f"{scene}.json").read_text()); alt=alternatives_from_prediction_json(doc); result=solve_scene(alt,index,"two_edge",5,max_proposals)
        target=truth[scene]; figure=[p[:2] for p in target.patches if p is not None and int(p[2])==1]
        rank=next(i+1 for i,x in enumerate(result["ranked"]) if x["name"]==target.constellation); true_h=result["ranked"][rank-1]; win=result["ranked"][0]
        trace=_oracle_trace(index.graphs[target.constellation],alt,figure); trace.update({"true_class_best_support":true_h["unique_inliers"],
          "true_class_final_rank":rank,"winning_false_class_support":win["unique_inliers"],
          "reason_true_class_loses":"lower duplicate-safe structural score" if rank>1 else "none"})
        final=result["winner"] if result["accepted"] else doc["constellation"]
        rows[scene]={"role":"adversarial_diagnostic_only" if scene=="taurus" else "repeated_development",
          "true_name":target.constellation,"existing_name":doc["constellation"],"winner":result["winner"],"accepted":result["accepted"],
          "final_name":final,"true_rank":rank,"margin":result["margin"],"trace":trace,"runtime_seconds":result["runtime_seconds"],"all_48_scores":[{"name":x["name"],"score":x["score"]} for x in result["ranked"]]}
    return {"status":"COMPLETE","strategy":"two_edge","scenes":rows}
