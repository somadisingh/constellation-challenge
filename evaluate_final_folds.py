import json,cv2
from pathlib import Path
from constellation.contracts import read_truth,evaluate
from constellation.finalize import finalize
from constellation.references import extract_patterns
cv2.setNumThreads(1);truth=read_truth('train_ground_truth.csv');patterns=extract_patterns('patterns');pred={}
cal=json.load(open('outputs/hybrid_ecc/calibration.json'))
for n in truth:
    raw=json.load(open(f'outputs/hybrid/{n}.json'));fine=json.load(open(f'outputs/hybrid_ecc/{n}.json'))
    threshold=cal['folds'][n]['threshold']
    image=cv2.imread(f'train/{n}/{n}_image.png',0)
    pred[n]=finalize(image,[q['candidates'] for q in fine['diagnostics']['queries']],[q['candidates'] for q in raw['diagnostics']['queries']],patterns,threshold=threshold)
    print(n,threshold,pred[n].constellation,flush=True)
result={'threshold_folds':cal['folds'],'metrics':evaluate(pred,truth),'scope':'Final frozen method with presence threshold fitted on the other two scenes. All scenes informed earlier method development; not an untouched estimate of model selection generalization.'}
Path('outputs/final_threshold_folds.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
