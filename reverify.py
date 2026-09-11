import argparse,json
from pathlib import Path
import cv2
from dataclasses import asdict
from constellation.refine import refine_candidates
from constellation.contracts import ScenePrediction,evaluate,read_truth
from constellation.geometry import recognize
from constellation.references import extract_patterns
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
cv2.setNumThreads(4);truth=read_truth('train_ground_truth.csv');patterns=extract_patterns('patterns');pred={}
for name,t in truth.items():
    file=args.input/f'{name}.json'
    if not file.exists():continue
    saved=json.loads(file.read_text());im=cv2.imread(f'train/{name}/{name}_image.png',0);diags=[];patches=[]
    for i,d in enumerate(saved['diagnostics']['queries']):
        patch=cv2.imread(f'train/{name}/patches/patch_{i+1:02}.png',0)
        c=refine_candidates(im,patch,d['candidates']);diags.append({'candidates':c,'appearance_score':c[0][2]});x,y,s,*_=c[0];patches.append((x,y,0) if s>=.65 else None)
    ids=[i for i,p in enumerate(patches) if p is not None];label,m,g=recognize([patches[i][:2] for i in ids],patterns)
    for j in m:
        i=ids[j];patches[i]=(*patches[i][:2],1)
    pred[name]=ScenePrediction(patches,label,{'queries':diags,'geometry':g})
    (args.output/f'{name}.json').write_text(json.dumps(asdict(pred[name]),indent=2));print(name,evaluate({name:pred[name]},{name:t}),flush=True)
if len(pred)==len(truth):(args.output/'metrics.json').write_text(json.dumps(evaluate(pred,truth),indent=2))
