"""Fixed historical inner ranking plus independent fresh, partition-restricted banks."""
import pickle,time
import numpy as np
from experiments.exp1.env import read_json,write_json,derive_seed
from experiments.exp1.inner import load_inner_aligned,_aligned_set,inner_metrics
from experiments.exp1.stages import Paths
from experiments.exp1.splits import fold_skies,partition_records,cell_box
from experiments.exp1.search import regional_candidates,clear_index_cache
from experiments.exp1.scoring import AlignedSet,encode_batch,summarise_query,score_classical,NEG_INF
from experiments.exp1.pose import SceneReps,prepare_query,select_pose,aligned_candidate
from .data import OUT,OLD,generate,local_context,stats,source_weights,valid_source
from .stream import IGNORE

def old_bank(fold):
    doc=load_inner_aligned(Paths(OLD),fold);out={}
    for s,node in doc['skies'].items():
        out[s]=[{'aligned':_aligned_set(node,i),'kind':node['kind'][i],'centre':node['centre'][i],
                 'source_id':node['source_id'][i],'stratum':node['stratum'][i],'index':i} for i in range(len(node['kind']))]
    return out

def fresh_bank(fold,step):
    path=OUT/'folds'/fold/f'fresh_val_{step}.pkl'
    if path.exists():
        with open(path,'rb') as f:return pickle.load(f)
    from experiments.exp1.data import load_scene
    recipe=read_json(OUT/'folds'/fold/'recipe.json');manifest=read_json(OUT/'folds'/fold/'manifest.json')
    _,allowed=fold_skies(fold);records={};weights={};images={s:load_scene(s).image for s in allowed}
    targets=[p['parent'] for d in recipe['diagnostics'].values() for p in d['pairs']]
    for s in allowed:
        records[s]=partition_records(manifest['banks'][s],'val')
        weights[s]=source_weights([stats(__import__('experiments.exp1b.data',fromlist=['crop']).crop(images[s],r['xy'])) for r in records[s]],targets)
    out={};started=time.time()
    for s in allowed:
        rows=[];other=next(t for t in allowed if t!=s)
        rng=np.random.default_rng(derive_seed(31002,fold,'freshval',step,s))
        n_queries=128 if step==0 else 32
        for i in range(n_queries):
            donor=s if i<n_queries//2 else other;kind='present' if i<n_queries//2 else 'absent'
            r=records[donor][rng.choice(len(records[donor]),p=weights[donor])]
            e=generate(images[donor],r,rng,recipe);q=(e['q']*255).round().astype(np.uint8)
            candidates=regional_candidates(s,'val',q,keep=20)
            good=[]
            for x,y,score,a,sc in candidates:
                for cell in (6,7):
                    candidate={'cell':cell,'xy':[x,y]}
                    if valid_source(candidate,images[s].shape):good.append((x,y,a,sc));break
            crops=[];xy=[];poses=[];ncc=[]
            qr,qb=prepare_query(q)
            for x,y,a,sc in good:
                local,c=local_context(images[s],(x,y));reps=SceneReps(local)
                sel=select_pose(reps,qb,c,(a,sc))
                if sel['pose'] is None:continue
                p,m=aligned_candidate(reps.raw,c,sel['pose'])
                if not m.all():continue
                crops.append(p/255);xy.append((x,y));poses.append(sel['pose']);ncc.append(sel['ncc'])
            crops=np.array(crops,np.float32).reshape(-1,32,32)
            aligned=AlignedSet(f'{s}:fresh:{step}:{i}',np.array(xy,float).reshape(-1,2),crops,np.array(ncc),np.array(poses).reshape(-1,2),
                               np.ones(len(crops),bool),np.std(crops,axis=(1,2))<1e-3,9,0,qr)
            rows.append({'aligned':aligned,'kind':kind,'centre':r['xy'] if kind=='present' else None,
                         'source_id':r['source_id'],'stratum':r['stratum'],'index':i,
                         'diagnostic_positive':e['p'] if kind=='present' else None})
        out[s]=rows;print(f'fresh validation {fold}/{step}/{s} ready ({time.time()-started:.1f}s)',flush=True)
    with open(path,'wb') as f:pickle.dump(out,f,protocol=5)
    return out

def score_bank(bank,model=None,device='mps'):
    # Batch encoding removes hundreds of tiny GPU dispatches without changing scores.
    records=[(s,r) for s,rows in bank.items() for r in rows]
    scored={s:[] for s in bank};diagnostic=[];zlist=[];t=time.time()
    if model is not None:
        allc=np.concatenate([np.concatenate([r['aligned'].query_raw[None],r['aligned'].crops]) for _,r in records])
        enc=encode_batch(model,allc,device).numpy();cursor=0
    for s,r in records:
        a=r['aligned'];n=len(a)
        if model is None:scores=score_classical(a)
        else:
            zq=enc[cursor];zc=enc[cursor+1:cursor+1+n];cursor+=1+n
            d=np.sqrt(np.maximum(((zc-zq)**2).sum(axis=1),1e-12));scores=-d.astype(float)
            scores[~a.admissible|a.low_info]=NEG_INF;zlist.append(zq)
            if r['centre'] is not None and n:
                spatial=np.linalg.norm(a.xy-r['centre'],axis=1);pos=np.flatnonzero((spatial<=12)&a.admissible&~a.low_info)
                neg=np.flatnonzero((spatial>IGNORE)&a.admissible&~a.low_info)
                if len(pos) and len(neg):
                    # Nearest spatial positive, not the learned best positive.
                    p=pos[np.argmin(spatial[pos])];dn=d[neg].min();dp=d[p]
                    diagnostic.append({'dpos':float(dp),'dneg':float(dn),'active':bool(.5+dp-dn>0)})
        row=summarise_query(scores,a,r['centre']);row.update({k:r[k] for k in ('kind','source_id','stratum','index')});scored[s].append(row)
    metric=inner_metrics(scored)
    metric['triplet_diagnostic']={'n':len(diagnostic),'scope':'retrieved correct candidates within12px vs non-overlapping wrong candidates; not oracle-inserted',
                                 'active_fraction':float(np.mean([d['active'] for d in diagnostic])) if diagnostic else None,
                                 'dpos':float(np.mean([d['dpos'] for d in diagnostic])) if diagnostic else None,
                                 'dneg':float(np.mean([d['dneg'] for d in diagnostic])) if diagnostic else None,
                                 'descriptor_variance':float(np.var(zlist,axis=0).mean()) if zlist else None}
    metric['seconds']=time.time()-t
    return metric,scored

if __name__=='__main__':
    from experiments.exp1.env import pin_threads
    pin_threads()
    for fold in ('pisces','scorpius','taurus'):
        for step in (500,1000,1500,2000):fresh_bank(fold,step)
        clear_index_cache()
