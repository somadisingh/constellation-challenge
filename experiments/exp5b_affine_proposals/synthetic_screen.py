"""All-pattern synthetic engineering screen for the Exp5B affine proposer.

This is useful for confidence-gate calibration and shape coverage only.  It is
not evidence of transfer to real hidden skies.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.spatial import cKDTree

from . import ROOT,OUT,NEIGHBORS,PER_CLASS_BUDGET
from .pattern_graph import extract_all
from .quad_recovery import one_alternative_scene_quads,_score_hypothesis
from constellation.quad import quads
from experiments.exp1.env import write_json,derive_seed

SEED=derive_seed(0,'exp5b-affine','synthetic-final')

def _inputs(alternatives):
    from constellation.slate import slate_from_candidates
    from constellation.geometry import consolidate
    from constellation.joint import generation_points,build_pool
    sl=[slate_from_candidates(x) for x in alternatives]
    live=[s for s in sl if len(s)]
    anchors=np.asarray([s.seed_xy() for s in live]).reshape(-1,2)
    _,groups=consolidate(anchors)
    rank1=generation_points(live,groups)
    pool,tags,scores,_=build_pool(live,groups,top_k=8)
    return live,rank1,pool,tags,scores

def solve(alternatives,graphs):
    live,rank1,pool,tags,pool_scores=_inputs(alternatives)
    base_si,base_sd=quads(rank1)
    alt_gen,alt_si,alt_sd=one_alternative_scene_quads(live,k=5)
    streams=[]
    for label,gen,si,sd in [('rank1',rank1,base_si,base_sd),
                            ('one_alternate',alt_gen,alt_si,alt_sd)]:
        if len(sd):streams.append((label,gen,si,sd,cKDTree(sd)))
    ranked=[]
    for name,g in sorted(graphs.items()):
        p=(g['nodes']-g['nodes'].mean(0))/np.maximum(np.ptp(g['nodes'],axis=0),1e-6)
        ti,td=quads(p)
        best=None
        for label,gen,si,sd,tree in streams:
            dist,ind=tree.query(td,k=min(NEIGHBORS,len(sd)))
            dist=np.atleast_2d(dist);ind=np.atleast_2d(ind)
            if dist.shape[0]!=len(td):dist,ind=dist.T,ind.T
            jobs=[(float(d),i,int(j)) for i in range(len(ti)) for d,j in zip(dist[i],ind[i])]
            jobs.sort(key=lambda x:(x[0],x[1],x[2]))
            for d,i,j in jobs[:PER_CLASS_BUDGET]:
                h=_score_hypothesis(name,p,ti[i],si[j],d,gen,pool,tags,pool_scores,g['edges'])
                if h is not None:
                    h['proposal_stream']=label
                    if best is None or h['score']>best['score']:best=h
        if best:ranked.append(best)
    ranked.sort(key=lambda h:(-h['score'],h['name']))
    return ranked

def run(n_scenes=48,say=print):
    import lab.synth as synth
    from constellation.references import extract_patterns
    patterns=extract_patterns(ROOT/'patterns');graphs=extract_all(ROOT/'patterns')
    scenes=synth.dataset(patterns,n_scenes=n_scenes,seed=SEED,min_nodes=4)
    rows=[];started=time.perf_counter()
    for i,s in enumerate(scenes):
        r=solve(s['alternatives'],graphs);top=r[0];second=r[1]
        row={'true':s['name'],'winner':top['name'],'correct':top['name']==s['name'],
             'score':top['score'],'margin':top['score']-second['score'],
             'support':top['support'],'held_out_support':top['held_out_support'],
             'proposal_stream':top['proposal_stream']}
        row['gate']=bool(row['margin']>=2 and row['support']>=9 and row['held_out_support']>=5)
        row['gated_correct']=bool(row['gate'] and row['correct'])
        rows.append(row);say(f"{i+1}/{len(scenes)} {row['true']} -> {row['winner']} gate={row['gate']}")
    gated=[x for x in rows if x['gate']]
    out={'label':'SYNTHETIC ENGINEERING SCREEN; not real-scene transfer evidence',
         'seed':int(SEED),'n_scenes':len(rows),'raw_accuracy':float(np.mean([x['correct'] for x in rows])),
         'gate':{'margin_min':2,'support_min':9,'held_out_support_min':5,
                 'coverage':len(gated)/max(len(rows),1),
                 'precision':float(np.mean([x['correct'] for x in gated])) if gated else None,
                 'accepted':len(gated)},'per_scene':rows,'seconds':time.perf_counter()-started}
    OUT.mkdir(parents=True,exist_ok=True);write_json(OUT/'synthetic_all48.json',out);return out

if __name__=='__main__':
    r=run();print(r['raw_accuracy'],r['gate'])
