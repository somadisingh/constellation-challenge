from __future__ import annotations
import json
from pathlib import Path
from . import OUT

REQUIRED=["exp6_claim_audit.json","corrected_exp6_status.json","context_audit.json","config.json","source_hashes.json","protected_before.json","graph_descriptor_audit.json","correspondence_recall.json","development_results.json","ablations.json","invariance.json","synthetic_fit.json","synthetic_selection.json","synthetic_final_seed1.json","synthetic_final_seed2.json","validation_predictions.json","csv_diff.json","runtime.json","test_results.json","integrity.json","gates.json","deployment_policy.json"]
def validate_required(out=OUT):
    missing=[x for x in REQUIRED if not (out/x).exists()]
    if missing: raise RuntimeError("missing required stages: "+", ".join(missing))
    bad=[x for x in REQUIRED if json.loads((out/x).read_text()).get("status") not in {None,"COMPLETE","CORRECTED","NOT_PROMOTED","FROZEN"}]
    if bad: raise RuntimeError("incomplete stages: "+", ".join(bad))
    return {"required_artifacts":len(REQUIRED),"all_present":True}
def main():
    from experiments.exp6_ransac_identification.context import atomic_json
    audit=validate_required(); gates=json.loads((OUT/"gates.json").read_text())
    atomic_json(OUT/"completion_audit.json",{"status":"COMPLETE","audit_complete":True,"execution_complete":True,
      "result_supported":True,"performance_successful":gates["performance_gates_passed"],"submission_candidate_generated":False,**audit})
if __name__=="__main__": main()

