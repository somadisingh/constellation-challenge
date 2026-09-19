NAMES=["real_graph_edges","strategy_behavior","measured_ablations","synthetic_all48","accepted_truth_independent","wrong_overwrite_computed","stage_traces_measured","final_radius_12","unique_assignment","working_finalizer","pattern_order","node_order","query_order","candidate_order","process_determinism","pisces_rank5","scorpius_rank5","no_labelled_regression","synthetic_precision","wrong_overwrite_rate","seed_direction","two_pattern_families","mrr_improvement","runtime","memory","patch_identity","tests","protected_unchanged","no_upload"]
def evaluate(values):
    rows={x:{"passed":bool(values.get(x,False)),"category":"correctness" if x in NAMES[:10] else ("invariance" if x in NAMES[10:15] else ("performance" if x in NAMES[15:25] else "deployment"))} for x in NAMES}
    critical=all(rows[x]["passed"] for x in NAMES[:15]+NAMES[25:])
    return {"gates":rows,"correctness_and_invariance_passed":critical,"performance_gates_passed":all(x["passed"] for x in rows.values())}

