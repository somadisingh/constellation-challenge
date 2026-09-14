"""Bounded fresh source search, independent anchor RNG, and explicit negative controls."""
from collections import Counter
import numpy as np
import cv2
from experiments.exp1.env import read_json,derive_seed
from experiments.exp1.splits import partition_records,fold_skies
from experiments.exp1.data import load_scene
from experiments.exp1.pose import (SceneReps,aligned_candidate,select_pose,prepare_query,support_mask)
from experiments.exp1.scoring import encode_batch
from .data import OUT,OLD,generate,local_context,valid_source
from .fast_pose import select_pose

POOL_CENTRES=128
POSES=[(a,s) for a in range(0,360,30) for s in (.85,1.,1.18)]
IGNORE=72. # >=36px and excludes overlapping 32x32 views at supported scales.
MASK=support_mask()

def normalize_ncc(crops):
    v=crops[:,MASK].astype(np.float32);v-=v.mean(axis=1,keepdims=True)
    return v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-7)

def eligible(xy,anchor):return np.linalg.norm(np.asarray(xy)-anchor,axis=-1)>IGNORE

def choose_indices(ncc,similarity,valid,generator,random_only=False,groups=None):
    """Three distinct physical negatives; random arm ignores both score arrays."""
    ok=np.flatnonzero(valid)
    groups=np.arange(len(valid)) if groups is None else np.asarray(groups)
    if len(np.unique(groups[ok]))<3:raise ValueError('fewer than three admissible sources')
    chosen=[];kinds=[]
    for j in range(3):
        if random_only or j==2:
            # Uniform over locations, then uniform over that location's poses.
            g=generator.choice(np.unique(groups[ok]));k=int(generator.choice(ok[groups[ok]==g]));kind='random'
        elif j==0:k=int(ok[np.argmax(ncc[ok])]);kind='classical'
        else:k=int(ok[np.argmax(similarity[ok])]);kind='network'
        chosen.append(k);kinds.append(kind);ok=ok[groups[ok]!=groups[k]]
    return np.array(chosen),kinds

class NegativePool:
    def __init__(self,fold,seed,records,images):
        self.fold=fold;self.seed=seed;self.records=records;self.images=images
        self.nodes={};self.logs=[];self.seen={s:set() for s in records}
    def refresh(self,round_index,model,device):
        from experiments.exp1.models import restore_train_mode
        for s,recs in self.records.items():
            rng=np.random.default_rng(derive_seed(self.seed,self.fold,'pool',round_index,s))
            ids=rng.choice(len(recs),min(POOL_CENTRES,len(recs)),replace=False)
            source_ids={recs[i]['source_id'] for i in ids}
            prior=set(self.nodes.get(s,{}).get('source_ids',[]))
            crops=[];xy=[];poses=[];sid=[]
            for i in ids:
                r=recs[i];local,c=local_context(self.images[s],r['xy'])
                for pose in POSES:
                    p,m=aligned_candidate(local,c,pose)
                    assert m.all()
                    crops.append(p/255.);xy.append(r['xy']);poses.append(pose);sid.append(r['source_id'])
            crops=np.stack(crops).astype(np.float32)
            # Descriptor refresh uses the current model on newly sampled locations.
            z=encode_batch(model,crops,device).numpy()
            blurred=np.stack([cv2.GaussianBlur(p,(0,0),.6) for p in crops])
            self.nodes[s]={'crops':crops,'xy':np.array(xy),'poses':np.array(poses),'z':z,
                           'ncc_vectors':normalize_ncc(blurred),'source_ids':sid}
            self.logs.append({'round':round_index,'scene':s,'locations':len(source_ids),
                              'new_vs_previous':len(source_ids-prior),'new_ever':len(source_ids-self.seen[s]),
                              'jaccard_previous':len(source_ids&prior)/max(1,len(source_ids|prior)),
                              'ids':sorted(source_ids),'pose_candidates':len(crops)})
            self.seen[s]|=source_ids
        restore_train_mode(model,True)
    def candidates(self,batch,model,device,step,random_only=False):
        from experiments.exp1.models import restore_train_mode
        q=np.stack([e['q'] for e in batch]);z=encode_batch(model,q,device).numpy()
        blur=np.stack([cv2.GaussianBlur(p,(0,0),.6) for p in q]);nq=normalize_ncc(blur)
        out=[None]*len(batch);origins=[None]*len(batch);difficulty=np.zeros(len(batch));selected_ids=[]
        for s in self.nodes:
            rows=[i for i,e in enumerate(batch) if e['scene']==s];node=self.nodes[s]
            nc=nq[rows]@node['ncc_vectors'].T;sim=z[rows]@node['z'].T
            for j,i in enumerate(rows):
                e=batch[i];valid=eligible(node['xy'],e['xy'])
                rng=np.random.default_rng(derive_seed(self.seed,self.fold,'neg',step,i))
                inds,kinds=choose_indices(nc[j],sim[j],valid,rng,random_only,groups=np.arange(len(node['xy']))//len(POSES))
                negs=[];qraw,qb=prepare_query(e['q'])
                for k in inds:
                    # Grid is bounded coarse search; FINAL crops use exactly the
                    # same nine-trial, fixed-centre image pose adapter as inference.
                    local,c=local_context(self.images[s],node['xy'][k]);reps=SceneReps(local)
                    sel=select_pose(reps,qb,c,node['poses'][k]);assert sel['pose'] is not None
                    p,m=aligned_candidate(reps.raw,c,sel['pose']);assert m.all()
                    negs.append(p/255.)
                out[i]=np.stack(negs);origins[i]=kinds
                difficulty[i]=float(nc[j,valid].max())
                selected_ids.append([node['source_ids'][k] for k in inds])
        restore_train_mode(model,True)
        return np.stack(out).astype(np.float32),origins,difficulty,selected_ids

class AnchorStream:
    def __init__(self,fold,arm,seed):
        self.fold=fold;self.arm=arm;self.seed=seed
        _,self.allowed=fold_skies(fold);self.fresh=arm in ('B','D','E');self.corrected=arm in ('C','D','E')
        manifest=read_json(OUT/'folds'/fold/'manifest.json');self.recipe=read_json(OUT/'folds'/fold/'recipe.json') if self.corrected else None
        self.records={s:partition_records(manifest['banks'][s],'fit') for s in self.allowed}
        self.images={s:load_scene(s).image for s in self.allowed}
        self.weights={};self.fixed={};self.replay=[];self.pending=[];self.uses=Counter();self.hashes=set();self.active_uses=Counter();self.active_hashes=set()
        self.draws=0;self.replayed=0;self.generated=0;self.anchor_digest='' ;self.generated_degenerate=0
        self.selection={}
        for s in self.allowed:
            if self.corrected:self.weights[s]=np.load(OUT/'folds'/fold/f'{s}_sources.npz')['weights']
            else:
                counts=Counter((r['source_class'],r['stratum']) for r in self.records[s])
                w=np.array([1/counts[(r['source_class'],r['stratum'])] for r in self.records[s]])
                self.weights[s]=w/w.sum()
            if not self.fresh:
                if arm=='A':
                    t=dict(np.load(OLD/'tensors'/f'{s}.npz',allow_pickle=True))
                    self.fixed[s]=[{'q':t['queries'][i],'p':t['positive'][i]/255.,'xy':t['centres'][i],
                                    'source_id':str(t['source_ids'][i]),'scene':s,'hash':__import__('hashlib').sha256(t['queries'][i].tobytes()).hexdigest(),
                                    'source_class':'historical'} for i in np.flatnonzero(t['positive_ok'])]
                else:
                    rng=np.random.default_rng(derive_seed(seed,fold,'fixed',s));ids=rng.choice(len(self.records[s]),200,replace=False,p=self.weights[s])
                    self.fixed[s]=[generate(self.images[s],self.records[s][i],rng,self.recipe) for i in ids]
                self.generated+=len(self.fixed[s])
    def sample(self,step,n=64):
        batch=[]
        # Shared D/E RNG and score-independent replay rule guarantee paired anchors.
        rng=np.random.default_rng(derive_seed(self.seed,self.fold,'anchor',step))
        for s in self.allowed:
            count=n//len(self.allowed)
            replay=[e for _,e in self.replay if e['scene']==s]
            nr=count//4 if self.fresh and replay else 0
            if self.fresh:
                ids=rng.choice(len(self.records[s]),count-nr,p=self.weights[s])
                for i in ids:
                    e=generate(self.images[s],self.records[s][i],rng,self.recipe);e['replayed']=False
                    self.generated+=1;self.generated_degenerate+=int(e['q'].std()<2/255. or ((e['q']==0)|(e['q']==1)).mean()>.25)
                    batch.append(e)
                for i in rng.choice(len(replay),nr,replace=False):
                    e=dict(replay[i]);e['replayed']=True;batch.append(e);self.replayed+=1
            else:
                ids=rng.integers(0,len(self.fixed[s]),count)
                for i in ids:batch.append({**self.fixed[s][i],'replayed':False})
        for e in batch:
            self.uses[e['source_id']]+=1;self.hashes.add(e['hash']);self.anchor_digest=__import__('hashlib').sha256((self.anchor_digest+e['hash']).encode()).hexdigest();self.draws+=1
        return batch
    def record_loss(self,batch,active,difficulty,step):
        for e,a,d in zip(batch,active,difficulty):
            if a:self.active_uses[e['source_id']]+=1;self.active_hashes.add(e['hash'])
            if self.fresh and not e['replayed']:self.pending.append((float(d),e))
        # Bounded hard replay selected only by classical scores, common to D/E.
        # Keep top candidates of each recent 16-step window plus existing replay.
        if (step+1)%16==0:
            unique={e['hash']:(d,e) for d,e in self.replay+self.pending}
            merged=sorted(unique.values(),key=lambda t:(-t[0],t[1]['hash']))
            self.replay=[]
            for s in self.allowed:self.replay.extend([e for e in merged if e[1]['scene']==s][:256])
            self.pending=[]
    def stats(self):
        return {'presentations':self.draws,'unique_source_centres':len(self.uses),'unique_query_images_entering_updates':len(self.hashes),
                'unique_sources_nonzero_loss':len(self.active_uses),'unique_queries_nonzero_loss':len(self.active_hashes),
                'nonzero_loss_presentations':sum(self.active_uses.values()),'generated_images':self.generated,
                'replay_fraction':self.replayed/max(1,self.draws),'replay_bank_size':len(self.replay),
                'source_reuse_quantiles':np.quantile(list(self.uses.values()),[0,.5,.9,1]).tolist() if self.uses else [],
                'per_sky_unique':{s:sum(k.startswith(s+':') for k in self.uses) for s in self.allowed},
                'generated_degenerate_fraction':self.generated_degenerate/max(1,self.generated),
                'anchor_digest':self.anchor_digest}
