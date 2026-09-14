"""Appearance/diagnosis artifacts, independent of scores and model selection."""
import json
from pathlib import Path
import numpy as np
from experiments.exp1.env import read_json,write_json,derive_seed,sha256_file,pin_threads
from experiments.exp1.splits import fold_skies
from .data import OUT,OLD,stats,summarise
from .stream import AnchorStream

def run():
    pin_threads();out={}
    for fold in ('pisces','scorpius','taurus'):
        groups={};images={}
        for arm in ('A','C'):
            stream=AnchorStream(fold,arm,31004)
            rows=[];parents=[];qs=[];ps=[]
            for s in stream.allowed:
                for e in stream.fixed[s]:
                    rows.append(stats(e['q']));parents.append(stats(e['p']));qs.append(e['q']*255);ps.append(e['p']*255)
            groups[arm]={'queries':summarise(rows),'parents':summarise(parents),'query_stats':rows,'parent_stats':parents,
                         'n':len(rows),'actual_source_centres':len({e['source_id'] for v in stream.fixed.values() for e in v})}
            images[arm]=(qs,ps)
        recipe=read_json(OUT/'folds'/fold/'recipe.json')
        real=[r for s in recipe['allowed'] for r in __import__('experiments.exp1.data',fromlist=['load_scene']).load_scene(s).patches]
        groups['real_queries']={'queries':summarise([stats(p) for p in real]),'n':len(real)}
        out[fold]=groups
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(12,5,figsize=(9,18))
        rng=np.random.default_rng(derive_seed(31003,fold,'sheet'))
        for i in range(12):
            j=int(rng.integers(400));k=int(rng.integers(len(real)))
            for ax,p in zip(axes[i],[real[k],images['A'][0][j],images['A'][1][j],images['C'][0][j],images['C'][1][j]]):
                ax.imshow(p,cmap='gray',vmin=0,vmax=255);ax.axis('off')
        for ax,title in zip(axes[0],['Real query','Original query','Original parent','Corrected query','Corrected parent']):ax.set_title(title,fontsize=8)
        fig.tight_layout();fig.savefig(OUT/'folds'/fold/'synthesis_contact.png',dpi=140);plt.close(fig)
        print(fold,[(a,round(groups[a]['queries']['std']['mean'],2),round(groups[a]['queries']['mean']['mean'],2)) for a in ('A','C','real_queries')],flush=True)
    write_json(OUT/'appearance_audit.json',out)
    diagnosis={'verified':[
        '200 fixed tensor queries per sky, 400 allowed per fold; source manifests held about20k but gradients used only cached queries.',
        'Legacy sampler regeneration: none. Query degradation, positive and warmup positive tensors were fixed.',
        'Legacy negative refresh only re-scores existing candidate tensors.',
        'Legacy hard=False uniformly chooses cached candidates (including mined candidates), while triplet_loss still mines hardest row/column in batch.',
        'alignment_failed_all means no usable finite score among candidates; it does not certify correct-candidate pose quality.',
        'I>200 is bright-pixel fraction, not clipping. True clipping is I==255.',
        'Aggregate contrast/brightness means do not establish per-query bimodality.',
        'Legacy HardNet screened .6412 is about.0443 below C1; selected .6551 is about.0304 below C1.'
        ],'legacy_coverage':read_json(OUT/'legacy'/'actual_coverage.json'),
        'legacy_reproduction':read_json(OUT/'legacy/folds/pisces/checkpoints/hardnet_screen_s31004.json'),
        'scope':'Historical reproduction is separate from A, which has a corrected explicit negative policy.'}
    write_json(OUT/'diagnosis.json',diagnosis)
if __name__=='__main__':run()
