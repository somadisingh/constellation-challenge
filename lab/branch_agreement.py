"""Classical cross-branch ranking ablation; production code stays unchanged.

Fits both actual coarse/refined candidate sets, caches all class hypotheses, and
compares deterministic fusion rules. Never creates synthetic independent branches.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import cv2
import numpy as np
from constellation.joint import recognize_joint
from constellation.finalize import auxiliary_map
from constellation.contracts import ScenePrediction, read_truth, evaluate
from constellation.references import extract_patterns
from lab.cache import load_train
from lab.imagebench.generate import digest, sha, write_json
from lab.imagebench.evaluate import source_digest, input_images, get_proposals, appearance, load_manifest
from constellation.pipeline import Config

POLICIES=['baseline','refined','coarse','rrf','mean_regret','min_regret','agreement_1','agreement_2','stable_1']


def fit_branches(image, refined, coarse, patterns):
    aux=auxiliary_map(image);branches=[]
    for stage,qs,cutoff in [('refined',refined,.72),('coarse',coarse,.65)]:
        ids=[i for i,q in enumerate(qs) if q and q[0][2]>=cutoff]
        if len(ids)<3:continue
        name,chosen,diag=recognize_joint([qs[i] for i in ids],patterns,seed=6643,
             tolerance=18.,gap=.03,top_k=8,cap=80000,quad_share=0.,
             aux_weight=3.,models=('affine',),shear_penalty=2.,auxiliary_map=aux,diag_top=48)
        branches.append(dict(stage=stage,ids=ids,name=name,diag=diag))
    return branches


def valid(branch):
    return [h for h in branch['diag'].get('hypotheses',[]) if h.get('support',0)>=4 and 'nodes' in h]


def select(branches,policy):
    candidates=[(h,b) for b in branches for h in valid(b)]
    if not candidates:return None
    baseline=min(candidates,key=lambda hb:(-hb[0]['score'],hb[0]['name'],hb[1]['stage']))
    if policy=='baseline':return baseline
    if policy in ('refined','coarse'):
        eligible=[x for x in candidates if x[1]['stage']==policy]
        return min(eligible,key=lambda hb:(-hb[0]['score'],hb[0]['name'])) if eligible else baseline
    maps=[{h['name']:(i,h) for i,h in enumerate(valid(b))} for b in branches]
    if len(maps)!=2:return baseline
    names=sorted(set(maps[0])|set(maps[1]))
    tops=[max((h['score'] for _,h in m.values()),default=0.) for m in maps]
    def value(name):
        hits=[m.get(name) for m in maps]
        ranks=[h[0]+1 if h else 49 for h in hits]
        regrets=[tops[i]-h[1]['score'] if h else 20. for i,h in enumerate(hits)]
        if policy=='rrf':return sum(1/(3+r) for r in ranks)
        if policy=='mean_regret':return -np.mean(regrets)
        if policy=='min_regret':return -max(regrets)
        score=max(h[1]['score'] for h in hits if h)
        shared=all(h is not None for h in hits)
        if policy=='stable_1' and shared:
            a,b=[np.array(h[1]['nodes']) for h in hits]
            shared=a.shape==b.shape and float(np.median(np.linalg.norm(a-b,axis=1)))<=18.
        bonus=2. if policy=='agreement_2' else 1.
        return score+bonus*shared
    winner=min(names,key=lambda n:(-value(n),n))
    return min([hb for hb in candidates if hb[0]['name']==winner],key=lambda hb:(-hb[0]['score'],hb[1]['stage']))


def predict(refined,branches,policy,identification_only=False):
    picked=select(branches,policy)
    base=select(branches,'baseline') if identification_only else picked
    if base is None:return ScenePrediction([None]*len(refined))
    h,b=base;rel=b['diag'].get('relocation',{});chosen={}
    if rel:
        by_tag={int(rel['tags'][j]):rel['pool'][j] for _,j in h['pairs']}
        for local,tag in rel['group_of_query'].items():
            if int(tag) in by_tag:chosen[b['ids'][int(local)]]=by_tag[int(tag)]
    nodes=np.array(h['nodes']);patches=[]
    for i,q in enumerate(refined):
        if not q or q[0][2]<.72:patches.append(None);continue
        xy=chosen.get(i,q[0][:2]);member=int(np.min(np.linalg.norm(nodes-xy,axis=1))<18)
        patches.append((float(xy[0]),float(xy[1]),member))
    return ScenePrediction(patches,picked[0]['name'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('.'));p.add_argument('--dataset',type=Path)
    p.add_argument('--split',choices=['development','calibration'],default='development')
    p.add_argument('--limit',type=int,default=6);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cv2.setNumThreads(1);root=args.data.resolve()
    patterns=extract_patterns(root/'patterns');source=digest(dict(pipeline=source_digest(),experiment=sha(__file__)))
    spec=dict(source=source,dataset=str(args.dataset),split=args.split,limit=args.limit)
    if args.dataset:spec['build']=load_manifest(args.dataset)['build_id']
    else:spec['caches']={str(f):sha(f) for folder in ['hybrid','hybrid_ecc'] for f in sorted((root/'outputs'/folder).glob('*.json'))}
    key=digest(spec);out=args.output;out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['key']!=key:raise ValueError('Changed inputs or code; use new output')
    write_json(manifest,dict(key=key,spec=spec,status='running'))
    if args.dataset:
        entries=[e for e in load_manifest(args.dataset)['scenes'] if e['split']==args.split and e['track']=='rendered'][:args.limit]
        items=[(e['id'],e) for e in entries]
    else:
        cache=load_train();items=list(cache.items())
    predictions={policy:{} for policy in POLICIES};idonly={policy:{} for policy in POLICIES};truth={};agreements=[]
    for sid,entry in items:
        path=out/(sid+'.json')
        if path.exists():result=json.loads(path.read_text())
        else:
            if args.dataset:
                image,patches=input_images(args.dataset,entry)
                cfg=Config(threads=1);proposals=get_proposals(args.dataset,entry,image,patches,args.dataset/'cache',cfg)
                coarse,refined=appearance(image,patches,proposals,cfg)
            else:
                image=cv2.imread(str(root/'train'/sid/(sid+'_image.png')),0)
                coarse,refined=entry['coarse'],entry['refined']
            result=dict(key=key,refined=refined,branches=fit_branches(image,refined,coarse,patterns))
            write_json(path,result)
        if result['key']!=key:raise ValueError('Stale scene cache')
        # Ground truth is first used after fitting the image-derived candidates.
        if args.dataset:
            if sha(args.dataset/entry['labels'])!=entry['labels_sha256']:raise ValueError('Truth hash mismatch')
            label=json.loads((args.dataset/entry['labels']).read_text());truth[sid]=ScenePrediction(label['patches'],label['constellation'])
        else:truth[sid]=read_truth(root/'train_ground_truth.csv')[sid]
        for policy in POLICIES:
            predictions[policy][sid]=predict(result['refined'],result['branches'],policy)
            idonly[policy][sid]=predict(result['refined'],result['branches'],policy,True)
        names=[b['name'] for b in result['branches']]
        agreements.append(dict(scene=sid,branches=names,agree=len(names)==2 and names[0]==names[1],true=truth[sid].constellation))
        print(sid,names,truth[sid].constellation,flush=True)
    metrics={k:evaluate(v,truth) for k,v in predictions.items()}
    report=dict(spec=spec,metrics=metrics,identification_only={k:evaluate(v,truth) for k,v in idonly.items()},agreements=agreements,
      scope='Bounded development/calibration ablation. Correlated branches; no independent-evidence claim. No confirmation access or production adoption.')
    if source!=digest(dict(pipeline=source_digest(),experiment=sha(__file__))):raise RuntimeError('Source changed')
    write_json(out/'report.json',report);write_json(manifest,dict(key=key,spec=spec,status='complete'))
    print(json.dumps({k:v['mean'] for k,v in metrics.items()},indent=2))

if __name__=='__main__':main()
