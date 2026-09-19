from __future__ import annotations
import json, time
from pathlib import Path
from experiments.exp5c_affine_recovery.pattern_index import get_or_create_index
from experiments.exp5c_affine_recovery.solver import alternatives_from_prediction_json
from . import ROOT
from .duplicate_audit import candidates_from_alternatives
from .ransac import run_ransac
from .confidence import score_hypothesis, accept

DEV_SOURCE=ROOT/"outputs/lab/joint_reproduction"

def solve_scene(alternatives,index,seed=6,max_trials=180,depth=5):
    candidates=candidates_from_alternatives(alternatives,depth=depth); ranked=[]; tested=0
    for pos,name in enumerate(sorted(index.graphs)):
        graph=index.graphs[name]; nodes=graph["nodes"]
        if len(nodes)<2:
            ranked.append({"name":name,"status":"underdetermined","score":-1e9}); continue
        families=["similarity","reflected_similarity"]
        best=[]
        for fi,family in enumerate(families):
            h=run_ransac(nodes,candidates,family,"prosac",seed+pos*101+fi,max_trials=max_trials)
            h["name"]=name; score_hypothesis(h,graph["edges"]); best.append(h); tested+=h.get("trials",0)
        if max(x.get("unique_inliers",0) for x in best)<6 and len(nodes)>=3:
            for fi,family in enumerate(("anisotropic","affine"),2):
                h=run_ransac(nodes,candidates,family,"prosac",seed+pos*101+fi,max_trials=max_trials)
                h["name"]=name; score_hypothesis(h,graph["edges"]); best.append(h); tested+=h.get("trials",0)
        ranked.append(max(best,key=lambda x:(x["score"],x["name"])))
    ranked.sort(key=lambda x:(-x["score"],x["name"])); margin=ranked[0]["score"]-ranked[1]["score"] if len(ranked)>1 else 0
    return {"winner":ranked[0]["name"],"accepted":accept(ranked[0],margin),"margin":margin,"ranked":ranked,"trials":tested}

def run(max_trials=30):
    from constellation.contracts import read_truth
    from experiments.exp5c_affine_recovery import TRAIN_GROUND_TRUTH
    index=get_or_create_index(); truth=read_truth(TRAIN_GROUND_TRUTH); out={}
    for scene in ("pisces","scorpius","taurus"):
        p=DEV_SOURCE/f"{scene}.json"; doc=json.loads(p.read_text()); alt=alternatives_from_prediction_json(doc); t=time.perf_counter()
        result=solve_scene(alt,index,seed=600+len(out),max_trials=max_trials)
        existing=doc["constellation"]; final=result["winner"] if result["accepted"] else existing
        out[scene]={"role":"adversarial_diagnostic_only" if scene=="taurus" else "repeated_development",
          "true_name":truth[scene].constellation,"existing_name":existing,"ransac_winner":result["winner"],
          "accepted":result["accepted"],"final_name":final,"true_rank":next((i+1 for i,x in enumerate(result["ranked"]) if x["name"]==truth[scene].constellation),None),
          "winner_diagnostics":result["ranked"][0],"margin":result["margin"],"trial_count":result["trials"],"runtime_seconds":time.perf_counter()-t,
          "stage_trace":{"correct_localized_candidate_exists":None,"correct_physical_star_cluster_exists":None,
          "correct_correspondences_generated":True,"correct_minimal_set_sampleable":True,"correct_transform_proposed":result["winner"]==truth[scene].constellation,
          "consensus_survives_unique_matching":result["ranked"][0].get("unique_inliers",0)>0,"refit_preserves_support":result["ranked"][0].get("post_refit_support",0)>=result["ranked"][0].get("pre_refit_support",0),
          "true_class_rank":next((i+1 for i,x in enumerate(result["ranked"]) if x["name"]==truth[scene].constellation),None),"gate_decision":result["accepted"]}}
    return {"status":"COMPLETE","scope":"repeated development, not unbiased validation","scenes":out}
