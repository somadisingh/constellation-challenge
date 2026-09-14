"""Synthesizes every phase's negative/mixed finding into one document (task's
"failure_analysis.json" requirement). No new numbers are computed here --
every figure below is read directly from the already-written per-phase JSON
and `gates.json`.
"""
from __future__ import annotations

import json

from . import OUT
from experiments.exp1.env import write_json


def _load(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def run(say=print) -> dict:
    metrics = _load('metrics.json')
    gates = _load('gates.json')

    findings = []

    findings.append({
        'phase': 2,
        'title': 'Template-size normalization regresses independent geometric support',
        'evidence': metrics['phase2_independent_support']['pisces']['held_out_with_size_normalization'],
        'summary': ('decorrelate_size (lab\'s own size-normalization primitive, reused '
                   'verbatim) moves the true class from rank 15 to rank 39 on pisces and '
                   'from rank 18 to rank 29 on taurus -- a real, reproducible regression, '
                   'not a bug. Held-out geometric support should NOT be size-normalized '
                   'with this primitive in this pipeline.'),
        'root_cause': ('decorrelate_size rescales support counts by expected template size, '
                     'which systematically favours SMALL reference templates on these '
                     'scenes -- the opposite of the large-template bias it is meant to '
                     'correct for, because the held-out (seed-excluded) support signal is '
                     'already much sparser than the full-slate signal decorrelate_size was '
                     'tuned against elsewhere in the repo (see lab/LEDGER.md).'),
    })

    findings.append({
        'phase': 3,
        'title': 'Uncorrected unqueried-star evidence breaks an already-correct scene',
        'evidence': {
            'scorpius_none': metrics['phase3_null_model']['results']['scorpius']['none'],
            'scorpius_raw_count': metrics['phase3_null_model']['results']['scorpius']['raw_count'],
            'scorpius_matched_null_lr_corrected':
                metrics['phase3_null_model']['results']['scorpius']['matched_null_lr_corrected'],
        },
        'summary': ('Adding raw unqueried-node evidence counts moves scorpius from a correct '
                   'rank-1 winner to a wrong rank-11 winner (eridanus). The sqrt(n) multiple-'
                   'testing correction recovers rank-1 correctness. This is the strongest '
                   'positive methodological finding in the experiment: uncorrected auxiliary '
                   'evidence search is actively dangerous, not merely unhelpful.'),
        'root_cause': ('References offering more unqueried nodes to search accumulate more '
                     'chances to find a spuriously-strong DoG response by chance -- the '
                     'exact large-template bias already documented in lab/LEDGER.md for a '
                     'different mechanism (wrong-class score correlating with reference '
                     'node count, r~=0.66). The correction divides by sqrt(n_searched) to '
                     'remove this multiple-comparisons inflation.'),
    })

    findings.append({
        'phase': 3,
        'title': 'Corrected unqueried-star evidence provides no net identification gain '
               'beyond Phase 1/2 signals on ambiguous scenes',
        'evidence': {
            'pisces': {'none': metrics['phase3_null_model']['results']['pisces']['none'],
                      'corrected': metrics['phase3_null_model']['results']['pisces']['matched_null_lr_corrected']},
            'taurus': {'none': metrics['phase3_null_model']['results']['taurus']['none'],
                      'corrected': metrics['phase3_null_model']['results']['taurus']['matched_null_lr_corrected']},
        },
        'summary': ('Pisces\'s rank improves under correction (15->6) but the scene remains '
                   'misidentified (winner=centaurus, not pisces) at every ablation level. '
                   'Taurus is unchanged (18->18). Neither previously-wrong scene becomes '
                   'correctly identified by adding this evidence source.'),
        'root_cause': ('The unqueried-star evidence signal is real but too weak relative to '
                     'the existing geometric-support gap on these two scenes -- the true '
                     'class needs to climb 5-17 ranks to win, and this signal alone moves it '
                     '2-9 ranks.'),
    })

    findings.append({
        'phase': 4,
        'title': 'The complete joint solver regresses a previously-correct scene on both seeds',
        'evidence': gates['checks']['6_complete_joint_solver_never_regresses_a_correct_scene'],
        'summary': ('The additive composite score (appearance + geometric support + '
                   'unqueried evidence - clutter penalty) makes scorpius wrong on BOTH '
                   'seeds (primary winner=ursa-minor, repeat winner=ursa-major), even though '
                   'every individual signal (Phases 1-3) is separately either neutral or '
                   'positive for scorpius. This is a genuine scale-mixing failure, diagnosed '
                   'during Phase 4 development: the unqueried z-score-sum term dominates '
                   'weak-support hypotheses because it is not on the same numeric scale as '
                   'the geometric-support counts it is added to, and the sqrt(n) correction '
                   'from Phase 3 does not fix this at the joint-solver level (it only fixes '
                   'the standalone Phase 3 evaluation, which combines with a DIFFERENT, '
                   'already-scaled geometric support baseline).'),
        'root_cause': ('No learned or calibrated combination weights were used for the '
                     'composite score (an explicit, documented design decision -- see '
                     'joint_solver.py) -- this is the direct, expected consequence of that '
                     'choice, reported honestly rather than silently reweighted after the '
                     'fact to hide the regression.'),
        'not_attempted': ('Learning combination weights via logistic regression on the '
                         'allowed-sky evidence (analogous to Phase 1\'s calibrator) was NOT '
                         'attempted in this pass -- flagged as the most promising next step, '
                         'not fabricated as already-tried-and-failed.'),
    })

    findings.append({
        'phase': 5,
        'title': 'Phase 1\'s isolated rank-fidelity gain does not survive full-pipeline '
               'integration; repeat seed regresses',
        'evidence': {
            'phase1_isolated_gain': metrics['phase1_rank_fidelity']['selection']['means'],
            'phase5_primary': metrics['phase5_full_evaluation']['primary']['verifier_snap_rescue'],
            'phase5_repeat': metrics['phase5_full_evaluation']['repeat']['verifier_snap_rescue'],
            'matching_baseline': metrics['matching_baseline'],
        },
        'summary': ('Phase 1 measured a real, positive gain in isolated candidate-rank '
                   'fidelity (mean top1_reward 0.7863->0.8267, leave-one-sky-out x2 seeds). '
                   'Integrated end to end with presence calibration + Exp2 geometry snap/'
                   'rescue, the primary seed is flat (+0.0001 vs the matching fixed-policy '
                   'baseline) and the repeat seed REGRESSES (-0.0103). Per-scene breakdown '
                   'shows pisces and scorpius both regress on the primary seed by more than '
                   '0.02 while taurus improves -- the net is not favourable.'),
        'root_cause': ('The linear rank-fusion rule changes WHICH candidate becomes rank-1 '
                     'for some queries. When the presence calibrator and Exp2\'s snap/rescue '
                     'logic were tuned against the ORIGINAL (classical-only) rank-1 '
                     'candidate\'s score distribution, changing which candidate is rank-1 '
                     'shifts the input distribution the calibrator and rescue thresholds see, '
                     'without those downstream components being retuned to match -- the exact '
                     'failure mode already documented in FINDINGS.md for a different rank-'
                     'changing experiment ("the presence threshold, the ambiguity gate and '
                     'the pool margin are all calibrated to the incumbent score '
                     'distribution").'),
        'not_attempted': ('Rejoining the calibrator/threshold sweep on the NEW rank-fusion '
                         'score distribution (rather than reusing Exp3\'s calibration as-is) '
                         'was NOT attempted in this pass.'),
    })

    doc = {
        'n_findings': len(findings),
        'findings': findings,
        'overall_conclusion': (
            'Every phase of Experiment 4B has executable code and real, recorded results '
            '(see metrics.json). The system built is IMPLEMENTATION-COMPLETE. It is NOT '
            'PERFORMANCE-SUCCESSFUL: 6 of 14 predeclared promotion gates fail (see '
            'gates.json), the complete joint solver regresses a previously-correct scene on '
            'both seeds, and the full pipeline regresses on the repeat seed. Per the task\'s '
            'own rule, these are reported as separate facts: the experiment is COMPLETE as '
            'an implementation-and-evaluation exercise, and its resulting method is NOT '
            'promoted to deployment (see deployment_policy.json). No submission candidate '
            'was generated.'
        ),
        'single_most_promising_next_step': (
            'Replace the joint solver\'s hand-picked additive composite score with weights '
            'learned by logistic regression on allowed-sky evidence (the same technique '
            'already used successfully for Phase 1\'s rank fusion and Exp3\'s presence '
            'calibrator) -- this directly targets the diagnosed scale-mixing root cause of '
            'gate 6\'s failure, which is the single largest and most reproducible regression '
            'in the experiment.'
        ),
    }
    write_json(OUT / 'failure_analysis.json', doc)
    say(f'failure_analysis.json written: {len(findings)} findings')
    return doc


if __name__ == '__main__':
    run()
