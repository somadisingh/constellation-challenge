"""Post-selection diagnostics for Experiment 4C.

These plots and tables are descriptive.  They are generated only after the
arm, C and fallback rule are frozen and never feed back into selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import OUT, ROOT, SCENES, SEEDS
from .evaluate import _records
from .fusion import LogisticFusion
from experiments.exp1.env import write_json


def _restore(doc: dict) -> LogisticFusion:
    n = doc['normalizer']
    model = LogisticFusion(doc['names'], n['kind'], doc['c'],
                           doc['class_balanced'], doc['constrained'],
                           doc['pairwise'], doc['synthetic_anchor'])
    model.normalizer.centre = np.asarray(n['centre'], float)
    model.normalizer.scale = np.asarray(n['scale'], float)
    model.normalizer.impute = np.asarray(n['impute'], float)
    model.coef_ = np.asarray(doc['coef'], float)
    model.intercept_ = float(doc['intercept'])
    model.fit_info = doc['fit_info']
    return model


def _corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run(say=print):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panel_dir = OUT / 'panels'; panel_dir.mkdir(parents=True, exist_ok=True)
    coefficient_rows=[]; scale_rows=[]; calibration_rows=[]; distribution_rows=[]
    bias_rows=[]; null_rows=[]; decisions=[]
    score_cache={}
    for seed, fname in ((31004, 'oof_primary.json'), (31005, 'oof_repeat.json')):
        oof=json.load(open(OUT/fname))
        for held in SCENES:
            model=_restore(oof['models'][held])
            records=_records(held,seed,held)
            scores=model.decision_function(records)
            probs=1/(1+np.exp(-np.clip(scores,-40,40)))
            score_cache[(seed,held)]=(records,scores,probs)
            out_names=model.normalizer.as_dict()['output_names']
            for name,value in zip(out_names,model.coef_):
                coefficient_rows.append({'seed':seed,'held_out':held,'feature':name,'coefficient':float(value)})
            allowed=[s for s in SCENES if s!=held]
            train=[r for s in allowed for r in _records(s,seed,held)]
            for j,name in enumerate(model.names):
                raw=np.asarray([np.nan if r['features'].get(name) is None else r['features'][name] for r in train],float)
                finite=raw[np.isfinite(raw)]
                scale_rows.append({'seed':seed,'held_out':held,'feature':name,
                    'raw_median':float(np.median(finite)) if len(finite) else None,
                    'raw_iqr':float(np.percentile(finite,75)-np.percentile(finite,25)) if len(finite) else None,
                    'normalizer_centre':float(model.normalizer.centre[j]),
                    'normalizer_scale':float(model.normalizer.scale[j]),
                    'missing_fraction':float(1-len(finite)/len(raw))})
            y=np.asarray([r['placement_correct'] for r in records],bool)
            for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
                m=(probs>=lo)&((probs<hi) if hi<1 else (probs<=hi))
                if m.any(): calibration_rows.append({'seed':seed,'held_out':held,'lo':float(lo),'hi':float(hi),
                    'n':int(m.sum()),'mean_probability':float(probs[m].mean()),'positive_fraction':float(y[m].mean())})
            # Hard negatives are the highest-scoring false hypotheses per class.
            neg=[i for i,r in enumerate(records) if not r['placement_correct']]
            neg=sorted(neg,key=lambda i:(-scores[i],records[i]['class_name']))[:64]
            for label,idx in [('positive',np.where(y)[0]),('hard_negative',np.asarray(neg,int))]:
                for i in idx:
                    distribution_rows.append({'seed':seed,'held_out':held,'kind':label,'score':float(scores[i]),
                        'held_out_support':records[i]['features']['held_out_support'],
                        'classical_appearance':records[i]['features']['classical_appearance']})
            bias_rows.append({'seed':seed,'held_out':held,
                'score_vs_node_count':_corr(scores,[r['features']['node_count'] for r in records]),
                'score_vs_attempted_hypotheses':_corr(scores,[r['features']['attempted_hypotheses'] for r in records]),
                'score_vs_pool_size':_corr(scores,[r['features']['pool_size'] for r in records])})
            for r,score in zip(records,scores):
                null_rows.append({'seed':seed,'held_out':held,'placement_correct':bool(r['placement_correct']),
                    'score':float(score),'raw':r['features']['unqueried_raw_count'],
                    'corrected':r['features']['corrected_unqueried']})
            decisions.append({'seed':seed,'scene':held,
                'raw':oof['per_scene'][held]['raw'],
                'always_calibrated':oof['per_scene'][held]['always_calibrated'],
                'confidence_gated':oof['per_scene'][held]['confidence_gated']})

    cal=json.load(open(OUT/'calibration.json'))
    diagnostic={
        'generated_after_selection_frozen':True,
        'selection_source':'outputs/exp4c_calibrated_fusion/selection_frozen.json',
        'coefficient_table':coefficient_rows,
        'feature_scale_table':scale_rows,
        'calibration_bins':calibration_rows,
        'score_distributions':distribution_rows,
        'template_and_hypothesis_bias':bias_rows,
        'matched_null_rows':null_rows,
        'override_decisions':decisions,
        'coefficient_stability':cal['per_fold'],
        'primary_repeat_disagreement':[{ 'scene':s,
            'primary_raw':next(x for x in decisions if x['seed']==31004 and x['scene']==s)['raw']['winner'],
            'repeat_raw':next(x for x in decisions if x['seed']==31005 and x['scene']==s)['raw']['winner'],
            'same_raw_winner':next(x for x in decisions if x['seed']==31004 and x['scene']==s)['raw']['winner']==next(x for x in decisions if x['seed']==31005 and x['scene']==s)['raw']['winner']}
            for s in SCENES],
    }
    write_json(OUT/'diagnostics.json',diagnostic)

    # Calibration and score distributions.
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].plot([0,1],[0,1],'k--',lw=1,label='ideal')
    for seed in SEEDS:
        rows=[x for x in calibration_rows if x['seed']==seed]
        ax[0].scatter([x['mean_probability'] for x in rows],[x['positive_fraction'] for x in rows],
                      s=[max(8,np.sqrt(x['n'])*4) for x in rows],alpha=.7,label=str(seed))
    ax[0].set(xlabel='predicted probability',ylabel='observed positive fraction',title='OOF calibration'); ax[0].legend()
    for kind,color in [('positive','tab:green'),('hard_negative','tab:red')]:
        vals=[x['score'] for x in distribution_rows if x['kind']==kind]
        if vals: ax[1].hist(vals,bins=25,alpha=.55,label=kind,color=color)
    ax[1].set(xlabel='fusion logit',ylabel='count',title='Positive vs strongest confusers');ax[1].legend()
    fig.tight_layout();fig.savefig(panel_dir/'calibration_and_scores.png',dpi=160);plt.close(fig)

    # Template/multiplicity bias and matched-null effect.
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    records,scores,_=score_cache[(31004,'pisces')]
    ax[0].scatter([r['features']['node_count'] for r in records],scores,s=7,alpha=.35)
    ax[0].set(xlabel='reference-node count',ylabel='fusion logit',title='Template-size bias (Pisces fold)')
    raw=[x['raw'] for x in null_rows if x['seed']==31004]; cor=[x['corrected'] for x in null_rows if x['seed']==31004]
    ax[1].scatter(raw,cor,s=6,alpha=.25);ax[1].set(xlabel='raw unqueried evidence',ylabel='matched-null corrected',title='Multiplicity correction')
    fig.tight_layout();fig.savefig(panel_dir/'bias_and_null.png',dpi=160);plt.close(fig)

    # Correct/wrong placement overlays on the three real skies.
    from experiments.exp1.data import load_scene
    fig,axes=plt.subplots(len(SCENES),2,figsize=(10,12))
    for row,scene in enumerate(SCENES):
        rec,scores,_=score_cache[(31004,scene)]
        winner=max(range(len(rec)),key=lambda i:scores[i])
        true_ids=[i for i,r in enumerate(rec) if r['class_name']==scene]
        true_best=max(true_ids,key=lambda i:scores[i])
        image=load_scene(scene).image
        for col,(idx,title) in enumerate(((winner,'calibrated winner'),(true_best,'best true-class hypothesis'))):
            axes[row,col].imshow(image,cmap='gray',vmin=np.percentile(image,2),vmax=np.percentile(image,99.5))
            pts=np.asarray(rec[idx]['mapped_nodes']);axes[row,col].scatter(pts[:,0],pts[:,1],s=12,facecolors='none',edgecolors='tab:red')
            axes[row,col].set_title(f'{scene}: {title}\n{rec[idx]["class_name"]}, placement={rec[idx]["placement_correct"]}')
            axes[row,col].set_xlim(0,image.shape[1]);axes[row,col].set_ylim(image.shape[0],0);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
    fig.tight_layout();fig.savefig(panel_dir/'placement_overlays.png',dpi=150);plt.close(fig)

    manifest=json.load(open(panel_dir/'manifest.json')) if (panel_dir/'manifest.json').exists() else {}
    manifest.update({
        'calibration_and_scores.png':'OOF calibration plus positive/hard-negative score distributions',
        'bias_and_null.png':'template-size bias and matched-null evidence correction',
        'placement_overlays.png':'primary-seed calibrated winner and best true-class placement on each labelled sky',
        'generated_after_selection_frozen':True,
    })
    write_json(panel_dir/'manifest.json',manifest)
    say(f'wrote {OUT/"diagnostics.json"}')
    return diagnostic


if __name__=='__main__':
    run()
