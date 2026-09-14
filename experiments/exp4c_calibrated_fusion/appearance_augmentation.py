"""Attach fold-specific frozen Exp3 evidence to immutable geometry records."""
from __future__ import annotations
import json
import numpy as np
from . import OUT,SCENES,SEEDS
from experiments.exp1.env import write_json

def _key(xy):return (round(float(xy[0]),3),round(float(xy[1]),3))

def run_fold(fold,seed,device='cpu',say=print):
    from experiments.exp4b_joint_solver.rank_fidelity import _load_fold_members
    from experiments.exp4b_joint_solver.rank_features import collect_all
    models=_load_fold_members(fold,(seed,),device)
    rows=collect_all(models,device,say=say)
    manifest={}
    for scene in SCENES:
        base=json.load(open(OUT/f'hypotheses_{scene}_s{seed}.json'))
        lookup={}
        for row in rows[scene]:
            for c in row['candidates']:
                if not c.get('valid') or c.get('exp3_pair_logit') is None:continue
                node={'logit':float(c['exp3_pair_logit']),
                      'absent':float(c['exp3_absent_logit']),
                      'disagreement':float(c.get('exp3_ensemble_disagreement') or 0.0)}
                k=_key(c['xy'])
                if k not in lookup or node['logit']>lookup[k]['logit']:lookup[k]=node
        pool=base['pool_xy']; classical=np.asarray(base['pool_classical_scores'],float)
        matched=0
        for rec in base['records']:
            js=[j for _,j in rec['total_pairs'] if 0<=j<len(pool)]
            matched_nodes=[(j,lookup.get(_key(pool[j]))) for j in js]
            matched_nodes=[x for x in matched_nodes if x[1] is not None]
            nodes=[x[1] for x in matched_nodes]
            if nodes:
                matched+=1
                logits=np.array([x['logit'] for x in nodes]);absent=np.array([x['absent'] for x in nodes])
                probs=1/(1+np.exp(np.clip(absent-logits,-40,40)))
                ncc=np.array([classical[j] for j,_ in matched_nodes])
                def z(x):return (x-x.mean())/x.std() if len(x)>1 and x.std()>1e-9 else x-x.mean()
                rec['features']['exp3_appearance']=float(logits.mean())
                rec['features']['exp3_probability']=float(probs.mean())
                rec['features']['exp3_disagreement']=float(np.mean([x['disagreement'] for x in nodes]))
                rec['features']['rank_fusion_appearance']=float(np.mean(.5*z(ncc)+.5*z(logits)))
                for k in ('exp3_appearance','exp3_probability','exp3_disagreement','rank_fusion_appearance'):
                    rec['missing'][k]=False
        path=OUT/f'augmented_{fold}_{scene}_s{seed}.json';write_json(path,base)
        manifest[scene]={'path':str(path),'n_hypotheses_with_exp3':matched,'n_total':len(base['records'])}
        say(f'  {fold}/s{seed}/{scene}: Exp3 attached to {matched}/{len(base["records"])} hypotheses')
    return manifest

def run(say=print):
    out={}
    for seed in SEEDS:
      for fold in SCENES:out[f'{fold}:s{seed}']=run_fold(fold,seed,say=say)
    write_json(OUT/'appearance_augmentation.json',out);return out
if __name__=='__main__':run()
