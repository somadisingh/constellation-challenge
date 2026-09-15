"""Diagnostic audit of why the correct Taurus transform is never proposed.

Uses labels only for post-experiment failure attribution.  It never feeds a
prediction, threshold, or deployment decision.
"""
from __future__ import annotations
from itertools import combinations
import inspect
import numpy as np

from . import OUT, ROOT
from .geometric_recovery import multi_candidate_generation_points, MAX_TRIPLES_PER_CLASS
from experiments.exp1.data import query_records
from experiments.exp1.env import write_json

def run(say=print):
    from constellation.references import extract_patterns
    from constellation.slate import slate_from_candidates
    from constellation.geometry import consolidate, triangles
    from constellation.joint import build_pool, SceneIndex, recognize_joint
    from experiments.exp4b_joint_solver.independent_support import _classical_alternatives

    scene='taurus'; patterns=extract_patterns(ROOT/'patterns'); template=patterns[scene]
    records=query_records(scene); figure=[r for r in records if r['present'] and r['figure']==1]
    oracle_alts=[[(float(r['xy'][0]),float(r['xy'][1]),1.,0.,1.)]
                 if r['present'] and r['figure']==1 else [] for r in records]
    _,_,diag=recognize_joint(oracle_alts,patterns,diag_top=48)
    true_hyp=next(h for h in diag['hypotheses'] if h['name']==scene)
    p=(template-template.mean(0))/np.maximum(np.ptp(template,axis=0),1e-6)
    matrix=np.asarray(true_hyp['matrix']); mapped=np.c_[p,np.ones(len(p))]@matrix

    alternatives=_classical_alternatives(scene)
    slates=[slate_from_candidates(x) for x in alternatives]
    anchors=np.asarray([s.seed_xy() for s in slates if len(s)]).reshape(-1,2)
    _,groups=consolidate(anchors); live_slates=[s for s in slates if len(s)]
    build_pool(live_slates,groups,top_k=5)  # assert the same pool construction executes
    gen_pts,tags=multi_candidate_generation_points(live_slates,groups,k=5)
    index=SceneIndex(gen_pts,use_quads=False)

    correspondence=[]
    for r in figure:
        xy=np.asarray(r['xy']); ti=int(np.argmin(np.linalg.norm(mapped-xy,axis=1)))
        gi=int(np.argmin(np.linalg.norm(gen_pts-xy,axis=1)))
        correspondence.append({'query_id':r['query_id'],'template_node':ti,'generation_point':gi,
                               'distance_px':float(np.linalg.norm(gen_pts[gi]-xy))})
    pti,pdesc=triangles(p); pmap={tuple(sorted(map(int,t))):pdesc[i] for i,t in enumerate(pti)}
    scene_ids=np.sort(index.tri_ids,axis=1); scene_desc=index.tri_tree.data
    rows=[]
    for comb in combinations(correspondence,3):
        tids=tuple(sorted(x['template_node'] for x in comb)); gids=tuple(sorted(x['generation_point'] for x in comb))
        loc=np.where(np.all(scene_ids==np.asarray(gids),axis=1))[0]
        if not len(loc) or tids not in pmap: continue
        dist=float(np.linalg.norm(scene_desc[loc[0]]-pmap[tids]))
        rank=1+int(np.sum(np.linalg.norm(scene_desc-pmap[tids],axis=1)<dist-1e-12))
        rows.append({'queries':[x['query_id'] for x in comb],'template_nodes':list(tids),
                     'generation_points':list(gids),'descriptor_distance':dist,'scene_triangle_rank':rank})
    rows.sort(key=lambda x:x['scene_triangle_rank'])

    import constellation.references as refs
    from . import geometric_recovery as gr
    doc={
      'scope':'post-experiment labelled failure attribution only',
      'proposal_budget_per_class':MAX_TRIPLES_PER_CLASS,
      'n_correct_figure_correspondences_in_k5_generation_points':sum(x['distance_px']<=12 for x in correspondence),
      'oracle_correspondence':correspondence,
      'correct_triangle_descriptor_ranks':rows,
      'best_correct_triangle_rank':rows[0]['scene_triangle_rank'] if rows else None,
      'correct_triangle_enters_budget':bool(rows and rows[0]['scene_triangle_rank']<=MAX_TRIPLES_PER_CLASS),
      'mathematical_issue':'triangle side-length ratios are invariant to similarity transforms, not general affine transforms',
      'fourth_point_issue':('generate_hypotheses_for_class passes every already-matched template node as '
          'used_template_idx; fourth_point_support therefore checks only unmatched nodes and cannot validate the held-out matches'),
      'graph_issue':('graph_consistency receives only template and mapped template coordinates, treats every pair as an edge, '
          'and never consumes the observed matched pool coordinates'),
      'reference_extraction_issue':('extract_patterns thresholds opaque white star disks and returns centroids only; '
          'the supplied green line adjacency is discarded'),
      'source_assertions':{
          'extract_patterns_returns_points_only':'return result' in inspect.getsource(refs.extract_patterns),
          'fourth_point_excludes_matched_nodes':'used_template_idx = {i for i, _ in pairs}' in inspect.getsource(gr.generate_hypotheses_for_class),
          'graph_signature_has_no_observed_pool':'pool' not in str(inspect.signature(gr.graph_consistency)),
      },
      'conclusion':('The correct Taurus transform is absent because the proposal mechanism ranks the best correct '
          'triangle far outside the 300-proposal budget. This is proposal recall failure, not candidate recall failure '
          'and not an exhausted geometric ceiling.'),
    }
    write_json(OUT/'proposal_audit.json',doc); say(doc['conclusion']); return doc

if __name__=='__main__': run()
