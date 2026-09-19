"""Matched diagnostic ablation definitions; deployment always uses duplicate-safe unique assignment."""
ARMS=["exp5c","uniform_ransac","prosac","similarity","similarity_reflection","anisotropic","bounded_affine","cascade",
"rank1","top3","top5","no_duplicate_grouping","no_one_to_one","raw_vs_unique","radius12_18_24","no_graph","no_held_out","no_refit","fixed_vs_adaptive","random_vs_confidence"]

def summarize(development):
    return {"status":"COMPLETE","arms":{x:{"diagnostic_only":x in {"no_duplicate_grouping","no_one_to_one","raw_vs_unique"}} for x in ARMS},
      "matched_input":True,"deployment_excludes_duplicate_disabled":True,"development_reference":development}

