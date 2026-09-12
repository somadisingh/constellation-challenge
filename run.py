#!/usr/bin/env python3
"""Reproducible classical inference, evaluation and submission entrypoint."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import csv,hashlib,json,platform,resource
from pathlib import Path
import cv2
import numpy as np
from constellation.contracts import read_truth,evaluate,write_submission
from constellation.pipeline import Config,predict_scene

def process_row(root,split,row,config,smoke,output):
    scene=Path(root)/split/row['Id']
    images=list(scene.glob('*_image.png'))
    if len(images)!=1:raise ValueError(f'Expected one sky in {scene}')
    image=cv2.imread(str(images[0]),0)
    n=int(row['n_patches']);n=min(n,2) if smoke else n
    patches=[cv2.imread(str(scene/'patches'/f'patch_{i:02}.png'),0) for i in range(1,n+1)]
    if image is None or any(q is None for q in patches):raise ValueError('Missing images')
    if image.shape!=(3000,3000) or any(q.shape!=(32,32) for q in patches):raise ValueError('Unexpected dimensions')
    prediction=predict_scene(image,patches,Path(root)/'patterns',config)
    prediction.diagnostics['peak_rss_platform_units']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (Path(output)/f"{row['Id']}.json").write_text(json.dumps(asdict(prediction),indent=2))
    print(row['Id'],round(prediction.diagnostics['runtime_seconds'],2),'seconds',flush=True)
    return row['Id'],prediction

def source_hash():
    here=Path(__file__).resolve().parent
    source=b''.join(f.read_bytes() for f in sorted((here/'constellation').glob('*.py')))+Path(__file__).read_bytes()
    return hashlib.sha256(source).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('.'))
    p.add_argument('--mode',choices=['smoke','evaluate','submission'],default='evaluate')
    p.add_argument('--output',type=Path,default=Path('outputs/final'))
    p.add_argument('--pipeline',choices=['a0','radial','harmonic','hybrid','final','joint'],default='joint')
    p.add_argument('--threshold',type=float,default=.72)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--verify-rep',choices=['blur','dog'],default='blur')
    p.add_argument('--alternatives',type=int,default=20)
    args=p.parse_args()
    if not 0<=args.threshold<=1 or args.workers<1:raise ValueError('Invalid configuration')
    root=args.data.resolve()
    if not (root/'patterns').exists() and (root/'participant').exists():root=root/'participant'
    args.output.mkdir(parents=True,exist_ok=True)
    config=Config(threshold=args.threshold,mode=args.pipeline,threads=args.threads,verify_rep=args.verify_rep,alternatives=args.alternatives)
    template=root/('sample_submission.csv' if args.mode=='submission' else 'train_ground_truth.csv')
    with template.open() as f:rows=list(csv.DictReader(f))
    if len({r['Id'] for r in rows})!=len(rows):raise ValueError('Duplicate IDs')
    if args.mode=='smoke':rows=rows[:1]
    digest=source_hash()
    manifest={'config':asdict(config),'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__,'source_sha256':digest,'mode':args.mode,'workers':args.workers,'status':'running'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    split='validation' if args.mode=='submission' else 'train'
    predictions={}
    if args.workers==1:
        for row in rows:
            name,pred=process_row(root,split,row,config,args.mode=='smoke',args.output)
            predictions[name]=pred
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(process_row,root,split,row,config,args.mode=='smoke',args.output) for row in rows]
            for future in as_completed(futures):
                name,pred=future.result();predictions[name]=pred
    if source_hash()!=digest:raise RuntimeError('Source changed during inference; rerun from frozen source')
    if args.mode=='evaluate':
        metrics=evaluate(predictions,read_truth(template))
        (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
    if args.mode!='smoke':write_submission(predictions,template,args.output/'submission.csv')
    manifest.update(status='complete',peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
