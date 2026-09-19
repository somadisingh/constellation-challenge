"""Matched proposal/verification ablations on identical development banks."""
from __future__ import annotations

import json

from constellation.joint import recognize_joint
from constellation.references import extract_patterns
from experiments.exp5c_affine_recovery import OUT, PATTERNS_DIR, SCENES, TRAIN_GROUND_TRUTH
from experiments.exp5c_affine_recovery.pattern_index import get_or_create_index
from experiments.exp5c_affine_recovery.solver import alternatives_from_prediction_json, solve
from experiments.exp5c_affine_recovery.synthetic import atomic_json
from constellation.contracts import read_truth


def _winner_with_rescore(result, mode):
    rows = []
    for h in result['ranked']:
        z = dict(h)
        if mode == 'without_held_out':
            z['ablation_score'] = .25*h['support'] + .35*h['graph']['supported_edges'] - .55*h['residual_p90']/18
        elif mode == 'without_graph':
            z['ablation_score'] = .70*h['held_out_support'] + .25*h['support'] - .55*h['residual_p90']/18
        elif mode == 'complete_graph':
            z['ablation_score'] = .70*h['held_out_support'] + .25*h['support'] + .35*(h['support']*(h['support']-1)//2)
        elif mode == 'without_multiplicity':
            z['ablation_score'] = h['score'] + .30*(h['null_significance_raw']-h['null_significance_adjusted'])
        rows.append(z)
    rows.sort(key=lambda h: (-h['ablation_score'], h['name']))
    return rows[0]['name'] if rows else 'unknown'


def run(say=print):
    index = get_or_create_index(); patterns = extract_patterns(PATTERNS_DIR)
    truth = read_truth(TRAIN_GROUND_TRUTH); out = {}
    cfg = {'max_scene_quads': 6000, 'proposal_budget': 5000,
           'progressive_budgets': [2500, 5000]}
    for scene in SCENES:
        doc = json.loads((OUT/'reconstructed_train'/f'{scene}.json').read_text())
        alts = alternatives_from_prediction_json(doc); true = truth[scene].constellation
        triangle, _, triangle_diag = recognize_joint(alts, patterns, cap=18000, quad_share=0.)
        arms = {}
        for name, selected in {
            'rank1_only': ('rank1',), 'alternate_only': ('one_alt_top5','two_alt_top5'),
            'combined_separated': ('rank1','one_alt_top5','two_alt_top5','adaptive','prosac'),
            'random_vs_guided_proxy': ('rank1',),
            'depth1': ('rank1',), 'depth3': ('one_alt_top3','two_alt_top3'),
            'depth5': ('one_alt_top5','two_alt_top5'),
            'fixed_budget': ('rank1','one_alt_top5','adaptive'),
            'progressive_widening': ('rank1','one_alt_top5','adaptive','prosac'),
        }.items():
            r = solve(alts,index,arms=selected,config=cfg,seed=42001)
            arms[name] = {'winner':r['winner'],'correct':r['winner']==true,
                          'proposals':r['proposals_evaluated'],'seconds':r['runtime_seconds']}
            if name == 'combined_separated': combined = r
        out[scene] = {
            'true': true,
            'triangle_side_ratio': {'winner': triangle, 'correct': triangle==true,
                                    'hypotheses': sum(h.get('attempted',0) for h in triangle_diag.get('hypotheses',[]))},
            **arms,
            'without_held_out_support': {'winner':_winner_with_rescore(combined,'without_held_out')},
            'without_real_graph': {'winner':_winner_with_rescore(combined,'without_graph')},
            'complete_graph_diagnostic': {'winner':_winner_with_rescore(combined,'complete_graph')},
            'without_multiplicity_correction': {'winner':_winner_with_rescore(combined,'without_multiplicity')},
            'fixed_gate_vs_calibrated': 'reported from development_results.json and synthetic final records',
            'existing_recognizer_vs_hybrid': {'existing':doc['constellation'],'affine':combined['winner']},
        }
        say(f'{scene}: triangle={triangle}, affine={combined["winner"]}')
    result={'status':'COMPLETE','matched_candidate_banks':True,
            'scope_note':'random_vs_guided_proxy compares unguided rank-one enumeration with graph-prioritized retrieval; it is not an unrestricted random search.',
            'scenes':out}
    atomic_json(OUT/'ablations.json',result);return result


if __name__=='__main__': run()
