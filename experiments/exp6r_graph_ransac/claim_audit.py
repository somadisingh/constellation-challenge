"""Machine-readable Experiment 6 evidence audit source."""
CLAIMS=[
 ("correspondence strategy changes behavior","correspondences.py","strategy","untested","strategy argument ignored"),
 ("graph-guided proposer implemented","correspondences.py",None,"contradicted","no graph input; graph only used after fitting"),
 ("ablations measured","ablations.json","arms","contradicted","arm-name manifest without executions"),
 ("synthetic precision is multiclass identification precision","synthetic.py","accepted_precision","contradicted","solver sees only true pattern and accepted equals correct"),
 ("synthetic wrong-overwrite measured","synthetic.py","wrong_overwrite_rate","hardcoded","literal False and aggregate 0.0"),
 ("development stage trace is measured","development.py","stage_trace","hardcoded","two True placeholders and two None fields"),
 ("end-to-end finalizer exists","finalize.py",None,"non-reproducible","file contains only a docstring"),
 ("final seed 2 coverage is 0.2444","EXPERIMENT6_REPORT.md",None,"contradicted","final seed JSON is 0.533333; 0.244444 belongs to fit seed"),
 ("real duplicate audit counts","duplicate_audit.json","scenes","supported","development 580/475; validation 3340/2848"),
 ("validation accepted no overwrites","validation_predictions.json","changes","supported","16 scenes, zero accepted changes"),
 ("core constrained fits and duplicate-safe scoring exist","ransac.py","run_ransac","supported","four fit families, unique final assignment and 12-pixel rescore implemented"),
]
def records():
    return [{"claim":c,"source_file":f,"source_json_key":k,"implementation_path":f,"status":s,"evidence":e,
      "required_correction":"Experiment 6R repair and do not use unsupported result for promotion"} for c,f,k,s,e in CLAIMS]

