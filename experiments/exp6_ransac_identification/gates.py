CRITICAL_GATES=("unique_assignment","final_radius_12","automatic_correspondences","pisces_correct",
 "scorpius_correct","no_labelled_regression","synthetic_precision_seed1","synthetic_precision_seed2",
 "synthetic_wrong_overwrite_seed1","synthetic_wrong_overwrite_seed2","two_pattern_rescues",
 "direction_agreement","pattern_order_invariant","query_order_invariant","deterministic",
 "cascade_order","safe_deployment_arm","runtime_practical","memory_safe","patch_cells_identical",
 "exp6_tests","production_tests","protected_unchanged","no_kaggle_upload")

def evaluate(values):
    rows={k:{"passed":bool(values.get(k,False)),"critical":True} for k in CRITICAL_GATES}
    return {"gates":rows,"performance_gates_passed":all(x["passed"] for x in rows.values())}

