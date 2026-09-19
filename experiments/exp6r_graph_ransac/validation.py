from __future__ import annotations
import csv,json
from experiments.exp5c_affine_recovery.pattern_index import get_or_create_index
from experiments.exp5c_affine_recovery.solver import alternatives_from_prediction_json
from . import ROOT,BASELINE
from .ransac import solve_scene
from experiments.exp6_ransac_identification.context import sha256

SOURCE=ROOT/"outputs/final_submission"
def run(max_proposals=8):
    base={r["Id"]:r for r in csv.DictReader(BASELINE.open())}; idx=get_or_create_index(); rows={}
    for scene in sorted(base):
        p=SOURCE/f"{scene}.json"; doc=json.loads(p.read_text()); result=solve_scene(alternatives_from_prediction_json(doc),idx,max_proposals=max_proposals)
        old=base[scene]["constellation"]; final=result["winner"] if result["accepted"] else old
        rows[scene]={"existing_name":old,"winner":result["winner"],"accepted":result["accepted"],"final_name":final,"margin":result["margin"],
          "runtime_seconds":result["runtime_seconds"],"candidate_source_hash":sha256(p),"all_48_scores":[{"name":x["name"],"score":x["score"]} for x in result["ranked"]]}
    return {"status":"COMPLETE","records":rows,"changes":{s:r["final_name"] for s,r in rows.items() if r["final_name"]!=r["existing_name"]},"base_sha256":sha256(BASELINE),"no_kaggle_upload":True}

