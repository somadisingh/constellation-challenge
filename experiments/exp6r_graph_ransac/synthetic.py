from __future__ import annotations
import numpy as np
from experiments.exp5c_affine_recovery.pattern_index import get_or_create_index
from experiments.exp6_ransac_identification.synthetic import make_case
from .ransac import solve_scene

FAMILIES=("similarity","reflected_similarity","anisotropic","affine")

def run_seed(seed,n_scenes=8,max_proposals=18):
    index=get_or_create_index(); names=[n for n in sorted(index.graphs) if len(index.graphs[n]["nodes"])>=3]
    chosen=[names[(i*len(names))//n_scenes] for i in range(n_scenes)]; rows=[]
    for i,name in enumerate(chosen):
        family=FAMILIES[i%4]; _,alt=make_case(index.graphs[name]["nodes"],seed+i,family)
        result=solve_scene(alt,index,"two_edge",5,max_proposals); rank=next(j+1 for j,x in enumerate(result["ranked"]) if x["name"]==name)
        baseline=names[(names.index(name)+1)%len(names)]; accepted=bool(result["accepted"]); prediction=result["winner"] if accepted else baseline
        rows.append({"scene":i,"true_class":name,"baseline_name":baseline,"winner":result["winner"],"accepted":accepted,
          "prediction":prediction,"correct":prediction==name,"correct_rescue":accepted and result["winner"]==name and baseline!=name,
          "wrong_overwrite":accepted and result["winner"]!=name,"confirmation":accepted and result["winner"]==baseline,
          "abstained":not accepted,"true_rank":rank,"top5":rank<=5,"reciprocal_rank":1/rank,"transform_family":family,
          "classes_compared":result["classes_compared"],"runtime_seconds":result["runtime_seconds"],"all_48_scores":[{"name":x["name"],"score":x["score"]} for x in result["ranked"]]})
    accepted=[x for x in rows if x["accepted"]]
    return {"status":"COMPLETE","seed":seed,"scenes":rows,"n_scenes":len(rows),"all_scenes_compared_48":all(x["classes_compared"]==48 for x in rows),
      "raw_top1_accuracy":sum(x["winner"]==x["true_class"] for x in rows)/len(rows),"top5_recall":sum(x["top5"] for x in rows)/len(rows),
      "accepted_coverage":len(accepted)/len(rows),"accepted_precision":sum(x["winner"]==x["true_class"] for x in accepted)/max(len(accepted),1),
      "wrong_overwrite_rate":sum(x["wrong_overwrite"] for x in rows)/max(len(accepted),1),"correct_rescues":sum(x["correct_rescue"] for x in rows),
      "confirmations":sum(x["confirmation"] for x in rows),"abstentions":sum(x["abstained"] for x in rows),"mean_reciprocal_rank":float(np.mean([x["reciprocal_rank"] for x in rows]))}

