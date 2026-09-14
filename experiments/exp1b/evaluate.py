"""Frozen, identical-bank calibration/integration and post-selection alignment audit."""
import time
from pathlib import Path
import numpy as np
from experiments.exp1 import SCENES
from experiments.exp1.env import read_json,write_json,sha256_file,append_jsonl
from experiments.exp1.evaluation import (load_real_aligned,_aligned_set,load_arm,frozen_geometry,integrate,evaluate_predictions,_slim,_strata_summary)
from experiments.exp1.data import query_records,load_truth,load_scene
from experiments.exp1.splits import fold_skies
from experiments.exp1.calibration import fit_calibrator,select_threshold,apply_calibrator,extract_features,fallback_threshold
from experiments.exp1.config import load_config
from experiments.exp1.stages import Paths
from experiments.exp1.scoring import encode_batch,NEG_INF
from experiments.exp1.pose import prepare_query,SceneReps,aligned_candidate,masked_ncc
from constellation.contracts import ScenePrediction,evaluate,reward
from .data import OUT,OLD,local_context
from .validation import score_bank

COMPONENTS=('presence','localization','recovery','identification','score')

def aggregate_complete(folds,required=SCENES):
    arms=sorted({a for f in folds.values() for a in f['arms']});out={}
    for arm in arms:
        rows={f:d['arms'][arm]['held_out_metrics'] for f,d in folds.items() if arm in d['arms']}
        complete=set(rows)==set(required)
        out[arm]={'complete':complete,'n_scenes':len(rows),'per_scene':rows,
                  'mean':{k:float(np.mean([r[k] for r in rows.values()])) for k in COMPONENTS} if complete else None,
                  'partial_mean_NOT_COMPARABLE':{k:float(np.mean([r[k] for r in rows.values()])) for k in COMPONENTS} if not complete else None,
                  'worst_score':min(r['score'] for r in rows.values())}
    return out

def c0_check():
    preds={s:ScenePrediction(**read_json(Path('outputs/joint_train')/f'{s}.json')) for s in SCENES}
    measured=evaluate(preds,load_truth());old=read_json(OLD/'c0.json')['metrics']
    assert measured['mean']==old['mean'],(measured,old)
    assert measured['mean']['score']==0.7286878605463721
    write_json(OUT/'c0.json',measured);return measured,preds

def real_bank(scene,node):
    return {scene:[{'aligned':_aligned_set(node,r['index']),'kind':'present' if r['present'] else 'absent','centre':r['xy'] if r['present'] else None,
                   'source_id':r['query_id'],'stratum':r['stratum'],'index':r['index']} for r in query_records(scene)]}

def score_real(scene,node,model):
    _,scored=score_bank(real_bank(scene,node),model,'mps')
    rows=scored[scene]
    for row,r in zip(rows,query_records(scene)):
        row.update({'scene':scene,'query_id':r['query_id'],'present':r['present'],'truth_xy':r['xy'],'figure':r['figure']})
    return rows

def calibration_predictions(fold,all_rows,config):
    _,allowed=fold_skies(fold)
    allowed_rows=[r for s in allowed for r in all_rows[s]]
    cal=fit_calibrator(allowed_rows,config)
    fallback=None
    if cal.get('ok'):thr=select_threshold(cal,allowed_rows,config)
    else: fallback=fallback_threshold(allowed_rows,config);thr={'fallback':fallback}
    preds={};details={}
    for s,rows in all_rows.items():
        if cal.get('ok'):probs=apply_calibrator(cal,rows);t=thr['selected']
        else:
            x,usable=extract_features(rows);probs=np.where(usable,x[:,0],-np.inf);t=fallback['selected_score_threshold'] if fallback.get('ok') else 0.
        b=integrate(s,rows,probs,t,frozen_geometry(s),18)
        preds[s]=b['prediction'];details[s]=_slim(b['detail'],rows)
    metrics=evaluate_predictions(preds)
    return {'calibration':cal,'threshold':thr,'held_out_metrics':metrics['scenes'][fold],
            'metrics':metrics,'per_query':details,'rows':all_rows,'strata':{s:_strata_summary(rows) for s,rows in all_rows.items()},
            'predictions':{s:{'patches':p.patches,'constellation':p.constellation,'diagnostics':p.diagnostics} for s,p in preds.items()}}

def query_changes(base,new,truth):
    changes=[]
    for i,(a,b,t) in enumerate(zip(base,new,truth.patches)):
        def loc(p):return float(reward(np.linalg.norm(np.array(p[:2])-t[:2]))) if p is not None and t is not None else 0.
        ap=(a is not None)==(t is not None);bp=(b is not None)==(t is not None)
        al=loc(a);bl=loc(b)
        if ap!=bp or abs(al-bl)>1e-9:
            changes.append({'index':i,'patch':f'patch_{i+1:02d}','base':a,'new':b,'truth':t,
                            'presence_change':int(bp)-int(ap),'localization_change':bl-al,'base_reward':al,'new_reward':bl})
    return {'presence_fixes':sum(r['presence_change']>0 for r in changes),'presence_regressions':sum(r['presence_change']<0 for r in changes),
            'localization_fixes':sum(r['localization_change']>0 for r in changes),'localization_regressions':sum(r['localization_change']<0 for r in changes),'changes':changes}

def correct_alignment_state(xy,truth,admissible):
    if not len(xy):return {'near_correct_exists':False,'nearest_distance':None,'valid_pose':False,'index':None}
    d=np.linalg.norm(np.asarray(xy)-truth,axis=1);i=int(d.argmin())
    return {'near_correct_exists':bool(d[i]<=12),'nearest_distance':float(d[i]),'valid_pose':bool(admissible[i]),'index':i}

def alignment_audit(fold,model,node):
    # Executed after checkpoint selection, calibration, and frozen outer scoring.
    # Truth identifies a candidate for audit only. No new coordinates enter scores.
    rows=[];image=load_scene(fold).image
    for r in query_records(fold):
        if not r['present']:continue
        a=_aligned_set(node,r['index']);state=correct_alignment_state(a.xy,r['xy'],a.admissible)
        row={'query_id':r['query_id'],**state}
        if not state['near_correct_exists'] or not state['valid_pose']:
            rows.append(row);continue
        i=state['index'];p=a.crops[i];q=a.query_raw;_,qb=prepare_query(q)
        local,c=local_context(image,a.xy[i]);reps=SceneReps(local)
        best={'ncc':float(a.ncc[i]),'pose':tuple(a.poses[i]),'crop':p}
        for da in (-5,-2.5,0,2.5,5):
            for ds in (.95,.975,1,1.025,1.05):
                pose=(a.poses[i,0]+da,a.poses[i,1]*ds)
                pc,m=aligned_candidate(reps.blur,c,pose)
                if not m.all():continue
                ncc=masked_ncc(qb,pc,m)
                if ncc>best['ncc']:
                    pc,_=aligned_candidate(reps.raw,c,pose);best={'ncc':ncc,'pose':pose,'crop':pc/255}
        z=encode_batch(model,np.concatenate([q[None],a.crops,best['crop'][None]]),'mps').numpy()
        d=np.linalg.norm(z[1:1+len(a)]-z[0],axis=1);valid=a.admissible&~a.low_info
        olddist=d[i];newdist=float(np.linalg.norm(z[-1]-z[0]));rank_before=1+int(((d<olddist)&valid).sum());d[i]=newdist
        rank_after=1+int(((d<newdist)&valid).sum())
        # Fitted linear-radiometry residual is an audit metric, not inference.
        mask=(q>0)&(q<1)&(p>0)&(p<1);v=p[mask];u=q[mask]
        beta=np.linalg.lstsq(np.c_[v,np.ones(len(v))],u,rcond=None)[0] if len(v)>3 else [0,0]
        residual=float(np.sqrt(np.mean((u-(v*beta[0]+beta[1]))**2))) if len(v) else None
        row.update({'masked_ncc':float(a.ncc[i]),'radiometry_fitted_rmse_0_1':residual,'diagnostic_ncc':best['ncc'],
                    'diagnostic_pose':best['pose'],'original_pose':a.poses[i].tolist(),'rank_before':rank_before,'rank_after':rank_after,
                    'descriptor_distance_before':float(olddist),'descriptor_distance_after':newdist})
        rows.append(row)
    return rows

def gate_result(arm,c0,c1,seed2=None):
    if not arm.get('complete'):return {'pass':False,'reason':'incomplete required folds'}
    m=arm['mean'];b=c0['mean'];c=c1['mean']
    checks={'total_gain_at_least_0.01':m['score']>=b['score']+.01,'beats_C1':m['score']>c['score'],
            'components_not_below_C0':all(m[k]>=b[k] for k in ('presence','localization','recovery')),
            'no_scene_regression_above_0.02':all(arm['per_scene'][s]['score']>=c0['scenes'][s]['score']-.02 for s in SCENES),
            'second_seed_stable':bool(seed2 and seed2.get('complete') and seed2['mean']['score']>b['score'] and seed2['mean']['score']>c['score'])}
    return {'pass':all(checks.values()),'checks':checks,'gain_C0':m['score']-b['score'],'gain_C1':m['score']-c['score']}

def run_evaluation():
    config=load_config();selection=read_json(OUT/'selection_frozen.json')
    assert set(selection)==set(SCENES)
    real={s:load_real_aligned(Paths(OLD),s) for s in SCENES};c0,c0preds=c0_check();folds={};truth=load_truth();audits={}
    classical={s:score_real(s,real[s],None) for s in SCENES}
    for fold in SCENES:
        dest=OUT/'folds'/fold/'evaluation.json'
        if dest.exists():folds[fold]=read_json(dest);continue
        node={'held_out':fold,'allowed':list(fold_skies(fold)[1]),'arms':{}}
        node['arms']['C1']=calibration_predictions(fold,classical,config)
        for arm in 'ABCDE':
            checkpoint=OUT/'runs'/fold/f'{arm}_s31004'/'best.pt'
            model,_=load_arm('hardnet','trained','mps',checkpoint)
            t=time.time();scored={s:score_real(s,real[s],model) for s in SCENES}
            result=calibration_predictions(fold,scored,config);result['seconds']=time.time()-t;result['checkpoint_sha256']=sha256_file(checkpoint)
            result['fixes_vs_C0']=query_changes(c0preds[fold].patches,result['predictions'][fold]['patches'],truth[fold])
            result['fixes_vs_C1']=query_changes(node['arms']['C1']['predictions'][fold]['patches'],result['predictions'][fold]['patches'],truth[fold])
            node['arms'][arm]=result
            print(f'OUTER {fold}/{arm}: {result["held_out_metrics"]}',flush=True)
            if arm==selection[fold]['arm']:
                node['arms']['SELECTED']=result;audits[fold]=alignment_audit(fold,model,real[fold]);write_json(OUT/'folds'/fold/'alignment_audit.json',audits[fold])
        s=selection[fold]
        if s['second_seed_required']:
            checkpoint=OUT/'runs'/fold/f'{s["arm"]}_s31005'/'best.pt'
            model,_=load_arm('hardnet','trained','mps',checkpoint)
            node['arms']['SELECTED_s31005']=calibration_predictions(fold,{s:score_real(s,real[s],model) for s in SCENES},config)
        folds[fold]=node;write_json(dest,node)
    oof=aggregate_complete(folds);old=read_json(OLD/'evaluate_s31004.json')['oof']
    # Correctness check on batched scoring/integration; C1 must reproduce exactly.
    for k in COMPONENTS:assert abs(oof['C1']['mean'][k]-old['C1']['mean'][k])<1e-10,(k,oof['C1'],old['C1'])
    gates={a:gate_result(oof[a],c0,oof['C1'],oof.get('SELECTED_s31005') if a=='SELECTED' else None) for a in 'ABCDE'}
    gates['SELECTED']=gate_result(oof['SELECTED'],c0,oof['C1'],oof.get('SELECTED_s31005'))
    metrics={'c0':c0,'oof':oof,'historical':old,'gates':gates,'selection':selection,
             'verdict':'promotion gates passed' if gates['SELECTED']['pass'] else 'no demonstrated deployable gain',
             'note':'Repeated exploratory out-of-fold evidence on three development skies; no untouched-test claim.'}
    write_json(OUT/'metrics.json',metrics)
    append_jsonl(OUT/'runs.jsonl',{'event':'outer_evaluation_complete','verdict':metrics['verdict'],'time':time.time()})
    from .report import make_report
    make_report()
if __name__=='__main__':run_evaluation()
