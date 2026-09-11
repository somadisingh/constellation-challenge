"""Leave-one-scene-out threshold calibration using saved blind candidates."""
import argparse,json
from pathlib import Path
import numpy as np
from constellation.contracts import ScenePrediction,evaluate,read_truth
from constellation.geometry import recognize
from constellation.references import extract_patterns

def prediction(saved,threshold,patterns=None):
    patches=[]
    for q in saved['diagnostics']['queries']:
        x,y,score,*_=q['candidates'][0]
        patches.append((x,y,0) if score>=threshold else None)
    label='unknown';d={}
    if patterns is not None:
        ids=[i for i,p in enumerate(patches) if p is not None]
        label,m,d=recognize([patches[i][:2] for i in ids],patterns)
        for j in m:
            i=ids[j];x,y,_=patches[i];patches[i]=(x,y,1)
    return ScenePrediction(patches,label,d)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    truth=read_truth('train_ground_truth.csv');saved={n:json.loads((args.input/f'{n}.json').read_text()) for n in truth}
    thresholds=np.arange(.65,.991,.01);scores={};all_pred={}
    # Threshold fitting deliberately excludes identification to avoid rerunning
    # expensive geometry and to keep presence calibration appearance-driven.
    for threshold in thresholds:
        pred={n:prediction(s,float(threshold)) for n,s in saved.items()}
        scores[float(threshold)]=evaluate(pred,truth)
    folds={};held={};patterns=extract_patterns('patterns')
    for name in truth:
        dev=[n for n in truth if n!=name]
        threshold=max(scores,key=lambda t:np.mean([scores[t]['scenes'][n]['score'] for n in dev]))
        held[name]=prediction(saved[name],threshold,patterns)
        folds[name]={'development_scenes':dev,'threshold':threshold}
    chosen=max(scores,key=lambda t:scores[t]['mean']['score'])
    result={'folds':folds,'held_out':evaluate(held,truth),'final_threshold':chosen,'selection':'maximize development-scene mean of 0.25 presence + 0.20 localization + 0.25 recovery; geometry evaluated after threshold selection','limitation':'All three scenes informed iterative method development; LOSO threshold results are not an untouched estimate of method selection generalization.'}
    (args.input/'calibration.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
