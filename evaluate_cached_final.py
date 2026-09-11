import json,cv2
from pathlib import Path
from dataclasses import asdict
from constellation.contracts import read_truth,evaluate,write_submission
from constellation.finalize import finalize
from constellation.references import extract_patterns
cv2.setNumThreads(4);truth=read_truth('train_ground_truth.csv');patterns=extract_patterns('patterns');pred={};out=Path('outputs/final_train_cached');out.mkdir(exist_ok=True)
for n in truth:
 coarse=json.load(open(f'outputs/hybrid/{n}.json'));refined=json.load(open(f'outputs/hybrid_ecc/{n}.json'));im=cv2.imread(f'train/{n}/{n}_image.png',0)
 pred[n]=finalize(im,[q['candidates'] for q in refined['diagnostics']['queries']],[q['candidates'] for q in coarse['diagnostics']['queries']],patterns)
 pred[n].diagnostics['queries']=refined['diagnostics']['queries']
 (out/f'{n}.json').write_text(json.dumps(asdict(pred[n]),indent=2));print(n,pred[n].constellation,flush=True)
metrics=evaluate(pred,truth);(out/'metrics.json').write_text(json.dumps(metrics,indent=2));write_submission(pred,'train_ground_truth.csv',out/'submission.csv');print(json.dumps(metrics,indent=2))
