from __future__ import annotations
import csv,json,time
from pathlib import Path
from experiments.exp5c_affine_recovery.pattern_index import get_or_create_index
from experiments.exp5c_affine_recovery.solver import alternatives_from_prediction_json
from . import ROOT,BASELINE
from .context import sha256
from .development import solve_scene

SOURCE=ROOT/"outputs/final_submission"

def mutate_names_only(base,target,names):
    with base.open(newline="") as f: reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
    before=[dict(x) for x in rows]
    for row in rows:
        if row["Id"] in names: row["constellation"]=names[row["Id"]]
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    assert all(a[k]==b[k] for a,b in zip(before,rows) for k in fields if k!="constellation")
    return {"sha256":sha256(target),"non_constellation_cells_identical":True}

def run(max_trials=140):
    index=get_or_create_index(); base={r["Id"]:r for r in csv.DictReader(BASELINE.open())}; records={}
    for pos,scene in enumerate(sorted(base)):
        p=SOURCE/f"{scene}.json"
        if not p.exists(): continue
        doc=json.loads(p.read_text()); t=time.perf_counter(); result=solve_scene(alternatives_from_prediction_json(doc),index,seed=6600+pos,max_trials=max_trials)
        old=base[scene]["constellation"]; final=result["winner"] if result["accepted"] else old; top=result["ranked"][0]
        records[scene]={"existing_name":old,"ransac_winner":result["winner"],"final_name":final,"accepted":result["accepted"],
          "transformation_family":top.get("family"),"proposal_strategy":"prosac","trial_count":result["trials"],"raw_inliers":top.get("raw_inliers",0),
          "unique_inliers":top.get("unique_inliers",0),"held_out_inliers":top.get("held_out_inliers",0),"support_12px":top.get("unique_inliers",0),
          "graph_support":top.get("graph_supported_edges",0),"residual_statistics":{"median":top.get("median_residual"),"p90":top.get("p90_residual"),"max":top.get("max_residual")},
          "winner_margin":result["margin"],"stability":None,"runtime_seconds":time.perf_counter()-t,"candidate_source_hash":sha256(p)}
    changes={k:v["final_name"] for k,v in records.items() if v["final_name"]!=v["existing_name"]}
    return {"status":"COMPLETE","records":records,"changes":changes,"base_sha256":sha256(BASELINE),"no_kaggle_upload":True}

