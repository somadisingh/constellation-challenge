"""One MPS training job; 5 arms x 3 folds, explicit hard/random loss selection."""
import json,time,pickle,os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from experiments.exp1.env import (derive_seed,seed_torch,pin_threads,write_json,read_json,sha256_file,sha256_json,append_jsonl,env_dict)
from experiments.exp1.models import build_for_training,restore_train_mode
from experiments.exp1.training import lr_at
from .data import OUT,OLD
from .stream import AnchorStream,NegativePool
from .validation import old_bank,fresh_bank,score_bank

CONFIG={'legacy_scale_bug':'A cached positive normalized /255; historical reproduction remains unchanged', 'steps':2000,'anchors':64,'eval_every':500,'pool_refresh':250,'pool_centres_per_sky':128,
        'pool_angles':list(range(0,360,30)),'pool_scales':[.85,1.,1.18],'negatives_evaluated':3,
        'loss':'margin 0.5 Euclidean triplet; hard=min(classical,network,random); E=uniform one of three uniform negatives; NO in-batch mining',
        'lr':1e-4,'weight_decay':1e-4,'warmup_lr':200,'lr_min':1e-6,'grad_clip':5,'seed':31004,
        'selection':'corrected fixed inner bank (128 queries/sky), equal-sky mean top1 localization reward; ties earliest step then arm alphabetical',
        'fresh_validation':'32 new queries per allowed sky per checkpoint, 1:1 present/absent, corrected recipe, blind regional search, diagnostic only',
        'negative_radius':72,'random':'uniform over admissible FIT source-bank locations and discrete initial poses; common NCC pose alignment',
        'replay':'25% after first16 steps; top256 per sky by classical-only false-candidate NCC; bounded 16-step insertion windows',
        'warmup_negative':'none in A-E to keep one common explicit policy; historical reproduction retains original500-step warmup',
        'gates':{'gain_C0':.01,'gain_C1':0,'max_scene_regression':.02,'nondecreasing_components':['presence','localization','recovery'],'stable_seed':True}}

def code_fingerprint():
    # Reporting/tests may be added during execution; training semantics may not.
    files=[Path(__file__).parent/p for p in ['data.py','stream.py','validation.py','train.py','fast_pose.py']]
    files+=sorted(Path('experiments/exp1').glob('*.py'))
    return {str(p):sha256_file(p) for p in files}

def fingerprint(fold,seed):
    files=[OUT/'folds'/fold/'recipe.json',OUT/'folds'/fold/'manifest.json',Path('cnn/hardnet-hf/checkpoint_liberty_with_aug.pth')]
    files+=list((OUT/'folds'/fold).glob('*_sources.npz'))
    from experiments.exp1.splits import fold_skies
    from experiments.exp1.data import load_scene
    for scene in fold_skies(fold)[1]:
        files.extend([load_scene(scene).image_path,OLD/'tensors'/f'{scene}.npz'])
    files+=[OLD/'inner'/f'{fold}_aligned.npz',OLD/'inner'/f'{fold}_aligned.meta.json']
    return sha256_json({'code':code_fingerprint(),'config':CONFIG,'seed':seed,'files':{str(p):sha256_file(p) for p in files}})

def select_negative_distance(distances,random_only,indices):
    return distances.gather(1,indices[:,None]).squeeze(1) if random_only else distances.min(dim=1).values

def training_step(model,opt,batch,negs,step,seed,random_only,total=2000):
    n=len(batch)
    data=np.concatenate([np.stack([e['q'] for e in batch]),np.stack([e['p'] for e in batch]),negs.reshape(-1,32,32)])
    z=model(torch.from_numpy(data.astype(np.float32))[:,None].to('mps'))
    za,zp=z[:n],z[n:2*n];zn=z[2*n:].reshape(n,3,-1)
    dp=torch.sqrt(torch.clamp(((za-zp)**2).sum(1),min=1e-6))
    dnall=torch.sqrt(torch.clamp(((za[:,None]-zn)**2).sum(2),min=1e-6))
    ids=torch.tensor(np.random.default_rng(derive_seed(seed,'loss',step)).integers(0,3,n),device='mps')
    dn=select_negative_distance(dnall,random_only,ids)
    winning=ids if random_only else dnall.argmin(dim=1)
    losses=torch.relu(.5+dp-dn);loss=losses.mean()
    lr=lr_at(step,total,1e-4,200,1e-6)
    for g in opt.param_groups:g['lr']=lr
    opt.zero_grad(set_to_none=True);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5));opt.step()
    if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss at {step}')
    active=(losses.detach()>0).cpu().numpy()
    return {'loss':float(loss.detach()),'active_fraction':float(active.mean()),'dpos':float(dp.mean().detach()),'dneg':float(dn.mean().detach()),
            'descriptor_variance':float(za.var(dim=0).mean().detach()),'descriptor_norm_min':float(za.norm(dim=1).min().detach()),'grad':grad,'lr':lr},active,winning.detach().cpu().numpy()

def run_arm(fold,arm,seed=31004,steps=2000,profile=False):
    out=OUT/('profile' if profile else 'runs')/fold/f'{arm}_s{seed}';out.mkdir(parents=True,exist_ok=True)
    key=fingerprint(fold,seed);cfg={**CONFIG,'fold':fold,'arm':arm,'seed':seed,'steps':steps,'fingerprint':key}
    if (out/'result.json').exists():
        result=read_json(out/'result.json')
        if result['fingerprint']!=key:raise RuntimeError(f'incompatible completed run {out}')
        return result
    if (out/'config.json').exists() and read_json(out/'config.json')!=cfg:raise RuntimeError(f'incompatible run config {out}')
    write_json(out/'config.json',cfg)
    seed_torch(derive_seed(seed,fold,'exp1b'))
    stream=AnchorStream(fold,arm,seed);model,info=build_for_training('hardnet',True,'mps',True)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    pool=NegativePool(fold,seed,stream.records,stream.images)
    start=0;history=[];evaluations=[];best={'metric':-1,'step':None};origins=Counter();elapsed_prior=0
    if (out/'last.pt').exists():
        blob=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if blob['fingerprint']!=key:raise RuntimeError('checkpoint/source/config mismatch')
        model.load_state_dict(blob['model']);opt.load_state_dict(blob['optimizer']);start=blob['step']
        history=blob['history'];evaluations=blob['evaluations'];best=blob['best'];elapsed_prior=blob['seconds']
        for name,value in blob['stream'].items():setattr(stream,name,value)
        pool.logs=blob['pool_logs'];pool.seen=blob['pool_seen'];pool.nodes={s:{'source_ids':ids} for s,ids in blob['prior_pool_ids'].items()}
        origins.update(blob['origins']);torch.set_rng_state(blob['cpu_rng']);torch.mps.set_rng_state(blob['mps_rng'])
        print(f'resume {fold}/{arm} at {start}',flush=True)
    inner=fresh_bank(fold,0) if not profile else None
    historical_inner=old_bank(fold) if not profile else None
    if inner is not None:
        c1,_=score_bank(inner);write_json(out/'inner_C1.json',c1)
    timings=Counter();t0=time.time();append_jsonl(OUT/'runs.jsonl',{'event':'start','fold':fold,'arm':arm,'seed':seed,'step':start,'fingerprint':key,'time':t0,'profile':profile})
    for step in range(start,steps):
        t=time.time()
        if step%250==0:pool.refresh(step//250,model,'mps')
        timings['pool_seconds']+=time.time()-t;t=time.time()
        batch=stream.sample(step);timings['generation_seconds']+=time.time()-t;t=time.time()
        negs,kinds,difficulty,selected_ids=pool.candidates(batch,model,'mps',step,arm=='E')
        timings['negative_seconds']+=time.time()-t;t=time.time()
        stat,active,winner=training_step(model,opt,batch,negs,step,seed,arm=='E')
        timings['gradient_seconds']+=time.time()-t
        stream.record_loss(batch,active,difficulty,step)
        for i in np.flatnonzero(active):origins[kinds[i][winner[i]]]+=1
        if step%25==0 or step+1==steps:
            history.append({'step':step+1,**stat,**stream.stats()})
        if (step+1)%50==0 or (profile and step+1==steps):
            print(f'{fold}/{arm}/{seed} step {step+1}/{steps} loss={stat["loss"]:.4f} active={stat["active_fraction"]:.3f} unique={len(stream.uses)} seconds={time.time()-t0:.0f}',flush=True)
        due=(step+1)%500==0 and not profile
        if due:
            fixed,rows=score_bank(inner,model,'mps');fresh,fr=score_bank(fresh_bank(fold,step+1),model,'mps')
            historical,historical_rows=score_bank(historical_inner,model,'mps')
            ev={'step':step+1,'fixed':fixed,'fresh':fresh,'historical_fixed':historical,'training':stat};evaluations.append(ev)
            write_json(out/f'inner_{step+1}.json',{'metrics':ev,'fixed_rows':rows,'fresh_rows':fr,'historical_rows':historical_rows})
            metric=fixed['equal_sky_top1_localization_reward']
            ckpt={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'step':step+1,'metric':metric,'fingerprint':key,'fold':fold,'arm':arm,'seed':seed}
            torch.save(ckpt,out/f'step_{step+1}.pt')
            if metric>best['metric']+1e-9:
                best={'metric':metric,'step':step+1};torch.save(ckpt,out/'best.pt')
            restore_train_mode(model,True)
            print(f'EVAL {fold}/{arm}/{seed} step={step+1} fixed={metric:.4f} fresh={fresh["equal_sky_top1_localization_reward"]:.4f} C1={c1["equal_sky_top1_localization_reward"]:.4f}',flush=True)
            # MPS tensors are saved on CPU. Optimizer state is also migrated by
            # load_state_dict; deterministic RNG/sampler state retained.
            state={k:getattr(stream,k) for k in ('replay','pending','uses','hashes','active_uses','active_hashes','draws','replayed','generated','anchor_digest','generated_degenerate')}
            blob={**ckpt,'optimizer':opt.state_dict(),'history':history,'evaluations':evaluations,'best':best,'stream':state,
                  'pool_logs':pool.logs,'pool_seen':pool.seen,'prior_pool_ids':{s:n['source_ids'] for s,n in pool.nodes.items()},'origins':dict(origins),
                  'cpu_rng':torch.get_rng_state(),'mps_rng':torch.mps.get_rng_state(),'seconds':elapsed_prior+time.time()-t0}
            torch.save(blob,out/'last.tmp.pt');(out/'last.tmp.pt').replace(out/'last.pt')
            write_json(out/'progress.json',{'step':step+1,'best':best,'counts':stream.stats(),'mining':pool.logs,'timing':dict(timings)})
    result={'fold':fold,'arm':arm,'seed':seed,'steps':steps,'fingerprint':key,'best':best,'counts':stream.stats(),'origin_nonzero_loss':dict(origins),
            'history':history,'evaluations':evaluations,'mining':pool.logs,'seconds':elapsed_prior+time.time()-t0,'timing':dict(timings),
            'model_info':info,'device_memory_bytes':torch.mps.current_allocated_memory(),'driver_memory_bytes':torch.mps.driver_allocated_memory(),
            'inner_C1':c1 if inner else None}
    write_json(out/'source_usage.json',{'counts':dict(stream.uses),'nonzero_loss_counts':dict(stream.active_uses)})
    write_json(out/'result.json',result);append_jsonl(OUT/'runs.jsonl',{'event':'complete','fold':fold,'arm':arm,'seed':seed,'steps':steps,'seconds':result['seconds'],'best':best,'profile':profile})
    del model,opt,pool,stream;torch.mps.empty_cache()
    return result

if __name__=='__main__':
    import argparse
    pin_threads();parser=argparse.ArgumentParser();parser.add_argument('--profile',action='store_true');parser.add_argument('--fold',default='pisces');parser.add_argument('--arm',default='D');parser.add_argument('--seed',type=int,default=31004);parser.add_argument('--steps',type=int,default=2000)
    args=parser.parse_args();assert torch.backends.mps.is_available(),'GPU inaccessible; do not silently run on CPU'
    if not (OUT/'environment.json').exists():write_json(OUT/'environment.json',env_dict())
    run_arm(args.fold,args.arm,args.seed,args.steps,args.profile)
