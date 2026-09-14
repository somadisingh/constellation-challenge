"""Fold-local radiometry audit, conservative source weighting, local-only warps."""
from pathlib import Path
from functools import lru_cache
import hashlib
import numpy as np
import cv2
from experiments.exp1.env import read_json,write_json,sha256_file,sha256_json,derive_seed
from experiments.exp1.splits import partition_records,fold_skies,cell_box,MARGIN
from experiments.exp1.data import load_scene,query_records
from experiments.exp1.pose import SceneReps,prepare_query,select_pose,aligned_candidate
from experiments.exp1.augment import sample_pose,sample_degradation,synthesize_query,DEFAULT
from experiments.exp1.evaluation import load_real_aligned
from experiments.exp1.stages import Paths

OUT=Path('outputs/exp1b'); OLD=Path('outputs/exp1')
Y,X=np.mgrid[:32,:32]; R=np.hypot(X-15.5,Y-15.5)
FEATURES=['mean','std','bright_gt200','clip255','core_annulus','highfreq','width','asymmetry','neighbors']

def stats(p):
    p=np.asarray(p,np.float32)
    if p.max()<=1.5:p=p*255
    core=p[R<4];ann=p[(R>=10)&(R<=15.5)]
    hp=p-cv2.GaussianBlur(p,(0,0),1.)
    mass=np.maximum(p-float(np.median(ann)),0)*(R<=10)
    width=np.sqrt(float((mass*R**2).sum())/max(float(mass.sum()),1e-6))
    peaks=(hp==cv2.dilate(hp,np.ones((5,5),np.uint8)))&(hp>max(3,float(hp.std())))&(R>5)&(R<15)
    return dict(zip(FEATURES,map(float,[p.mean(),p.std(),(p>200).mean(),(p==255).mean(),core.mean()-ann.mean(),np.sqrt((hp**2).mean()),width,np.abs(p-p[::-1,::-1]).mean(),peaks.sum()])))

def summarise(items):
    return {k:{'mean':float(np.mean([r[k] for r in items])), 'q10_q50_q90':np.quantile([r[k] for r in items],[.1,.5,.9]).tolist()} for k in FEATURES}

def local_context(image,xy,radius=72):
    x,y=np.floor(xy).astype(int);x0=x-radius;y0=y-radius
    if min(x0,y0)<0 or x+radius>=image.shape[1] or y+radius>=image.shape[0]:raise ValueError('unsupported footprint')
    return image[y0:y+radius+1,x0:x+radius+1],np.asarray(xy)-[x0,y0]

def valid_source(record,image_shape):
    x0,y0,x1,y1=cell_box(record['cell'],*image_shape)
    x,y=record['xy']
    return (x-MARGIN>=x0 and y-MARGIN>=y0 and x+MARGIN<x1 and y+MARGIN<y1)

def crop(image,xy):
    local,c=local_context(image,xy)
    return aligned_candidate(local,c,(0,1),aa_factor=0)[0]

def generate(image,source,rng,recipe=None):
    # Filtering happens on a bounded context wholly within its assigned cell.
    local,xy=local_context(image,source['xy'])
    pose=sample_pose(rng);deg=sample_degradation(rng)
    if recipe:
        # Fit-supported ranges only; absent/present never select different laws.
        for key in ('gain','offset','blur_sigma'):
            if key in recipe.get('degradation',{}):deg[key]=float(rng.uniform(*recipe['degradation'][key]))
    synth=synthesize_query(local,xy,pose,deg)
    if not synth['ok']:raise ValueError('invalid generated footprint')
    qraw,qblur=prepare_query(synth['query']); reps=SceneReps(local)
    # Known pose initializes the same image-estimated 9-trial adapter as Exp1.
    base=(pose['angle']+rng.uniform(-10,10),pose['scale']*rng.uniform(.9,1.1))
    sel=select_pose(reps,qblur,xy,base)
    if sel['pose'] is None:raise ValueError('invalid positive alignment')
    pos,mask=aligned_candidate(reps.raw,xy,sel['pose'])
    assert mask.all()
    return {'q':qraw,'p':pos/255.,'xy':np.array(source['xy']),'source_id':source['source_id'],'scene':source['scene'],
            'pose':sel['pose'],'true_pose':pose,'degradation':deg,'source_class':source['source_class'],
            'hash':hashlib.sha256(synth['query'].tobytes()).hexdigest()}

def feature_matrix(rows):return np.array([[r[k] for k in ('mean','std','core_annulus','width')] for r in rows])

def source_weights(source_stats,targets):
    # Approximate transport: each observed parent spreads its probability over
    # 128 nearby FIT sources, with a 5% uniform floor. No query pixels are edited.
    from scipy.spatial import cKDTree
    a=feature_matrix(source_stats);b=feature_matrix(targets)
    scale=np.maximum(np.std(b,axis=0),[20,8,20,1.5])
    distances,idx=cKDTree(a/scale).query(b/scale,k=min(128,len(a)))
    affinity=np.exp(-0.5*(distances-distances[:,:1])**2)
    affinity/=affinity.sum(axis=1,keepdims=True)
    w=np.zeros(len(a));np.add.at(w,idx.ravel(),affinity.ravel()/len(b))
    return .95*w+.05/len(a)

def prepare_fold(fold):
    dest=OUT/'folds'/fold;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'recipe.json').exists():return read_json(dest/'recipe.json')
    manifest=read_json(OLD/'folds'/fold/'manifest.json');_,allowed=fold_skies(fold)
    assert tuple(manifest['allowed'])==allowed
    write_json(dest/'manifest.json',manifest)
    diagnostics={};all_parents=[];pairs=[];sheets=[];records={};source_stats={}
    for s in allowed:
        image=load_scene(s).image; recs=partition_records(manifest['banks'][s],'fit');records[s]=recs
        assert all(valid_source(r,image.shape) for r in recs)
        rows=[]
        for r in recs:rows.append(stats(crop(image,r['xy'])))
        source_stats[s]=rows
        real=load_real_aligned(Paths(OLD),s);realstats=[];parentstats=[];paired=[]
        for r in query_records(s):
            q=load_scene(s).patches[r['index']];qs=stats(q);realstats.append(qs)
            if not r['present']:continue
            # Labels/poses from permitted development skies only, used for diagnostics.
            i=r['index'];k=int(real['counts'][i]);j=int(np.linalg.norm(real['xy'][i,:k]-r['xy'],axis=1).argmin())
            local,c=local_context(image,r['xy']); reps=SceneReps(local)
            _,qb=prepare_query(q);sel=select_pose(reps,qb,c,real['poses'][i,j])
            p,_=aligned_candidate(reps.raw,c,sel['pose']);ps=stats(p)
            parentstats.append(ps);all_parents.append(ps)
            # Robust paired affine fit is DIAGNOSTIC; not forced into synthesis.
            v=p.ravel();u=q.astype(float).ravel();ok=(u>5)&(u<250)&(v>5)&(v<250)
            gain,offset=(np.linalg.lstsq(np.c_[v[ok],np.ones(ok.sum())],u[ok],rcond=None)[0] if ok.sum()>32 else [np.nan,np.nan])
            paired.append({'query_id':r['query_id'],'query':qs,'parent':ps,'ncc':sel['ncc'],'gain':gain,'offset':offset})
            if len(sheets)<12:sheets.append((q,p,crop(image,recs[len(sheets)*17]['xy'])))
        diagnostics[s]={'real_queries':summarise(realstats),'paired_parents':summarise(parentstats),'fit_sources':summarise(rows),'pairs':paired}
        pairs+=paired
    # Conservative correction: source selection first, unchanged degradations.
    # Paired radiometry estimates have alignment/clipping uncertainty; no arbitrary offset.
    recipe={'fold':fold,'allowed':list(allowed),'held_out':fold,'degradation':{},
            'correction':'128-neighbor parent-feature transport weights, 5% uniform floor; original degradation laws retained',
            'features':['mean','std','core_annulus','width'],'uniform_floor':.05,'neighbors_per_target':128,
            'fit_note':'paired real development queries used only for source-distribution fitting, never gradients; no held-out data',
            'diagnostics':diagnostics}
    for s in allowed:
        w=source_weights(source_stats[s],all_parents)
        recipe.setdefault('effective_sources',{})[s]=float(1/np.sum(w*w))
        np.savez_compressed(dest/f'{s}_sources.npz',weights=w,features=feature_matrix(source_stats[s]))
    recipe['hash']=sha256_json(recipe);write_json(dest/'recipe.json',recipe)
    # Shared fixed 0..255 scale is essential for interpretable images.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(sheets),3,figsize=(6,len(sheets)*1.5),squeeze=False)
    for row,images in zip(axes,sheets):
        for ax,p in zip(row,images):ax.imshow(p,cmap='gray',vmin=0,vmax=255);ax.axis('off')
    for ax,title in zip(axes[0],['Real query','Aligned true parent','Original FIT source']):ax.set_title(title,fontsize=8)
    fig.tight_layout();fig.savefig(dest/'paired_contact.png',dpi=130);plt.close(fig)
    print(f'{fold}: recipe frozen, effective sources={recipe["effective_sources"]}',flush=True)
    return recipe

if __name__=='__main__':
    from experiments.exp1.env import pin_threads
    pin_threads()
    for fold in ('pisces','scorpius','taurus'):prepare_fold(fold)
