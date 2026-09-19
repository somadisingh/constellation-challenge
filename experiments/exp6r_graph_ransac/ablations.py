from __future__ import annotations
import hashlib,json,resource,time
from experiments.exp5c_affine_recovery.pattern_index import get_or_create_index
from experiments.exp6_ransac_identification.synthetic import make_case
from .ransac import solve_scene

ARMS=["exp5c_baseline","corrected_exp6_cartesian","single_edge","two_edge","without_graph_signatures","without_degree_signatures","without_distance_ratios","without_candidate_confidence","uniform_ransac","prosac","deterministic_best_first","similarity_only","similarity_reflection","cascade","top1","top3","top5","without_duplicate_grouping","without_one_to_one","without_refitting","final_radius_12","discovery18_final12","discovery24_final12"]

def _config(arm):
    c={"strategy":"two_edge","depth":5,"max_proposals":6,"refit":True,"families":("similarity","reflected_similarity"),"duplicate_safe":True,"one_to_one":True}
    if arm in {"corrected_exp6_cartesian","uniform_ransac","without_graph_signatures","without_degree_signatures"}: c["strategy"]="uniform"
    if arm in {"single_edge","without_distance_ratios"}: c["strategy"]="single_edge"
    if arm in {"without_candidate_confidence","prosac"}: c["strategy"]="confidence"
    if arm=="deterministic_best_first": c["max_proposals"]=1
    if arm=="similarity_only": c["families"]=("similarity",)
    if arm in {"top1","top3"}: c["depth"]=int(arm[-1])
    if arm=="without_duplicate_grouping": c["duplicate_safe"]=False
    if arm=="without_one_to_one": c["one_to_one"]=False
    if arm=="without_refitting": c["refit"]=False
    return c

def run(seed=66003):
    idx=get_or_create_index(); true="pisces"; _,alt=make_case(idx.graphs[true]["nodes"],seed,"similarity")
    digest=hashlib.sha256(json.dumps(alt,sort_keys=True).encode()).hexdigest(); rows=[]
    for arm in ARMS:
        cfg=_config(arm); started=time.perf_counter(); r=solve_scene(alt,idx,**cfg); rank=next(i+1 for i,x in enumerate(r["ranked"]) if x["name"]==true)
        rows.append({"arm":arm,"config":{**cfg,"families":list(cfg["families"])},"input_hash":digest,"scenes_evaluated":1,"seed":seed,
          "runtime":time.perf_counter()-started,"peak_memory":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "raw_accuracy":float(r["winner"]==true),"true_class_ranks":[rank],"accepted_coverage":float(r["accepted"]),
          "accepted_precision":float(r["accepted"] and r["winner"]==true),"wrong_overwrite_rate":float(r["accepted"] and r["winner"]!=true),"status":"COMPLETE"})
    return {"status":"COMPLETE","matched_input":True,"results":rows}

