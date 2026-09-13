"""Refresh the submission record after the second pass.

The pass shipped no prediction change. This records the new source hash, confirms the
submitted CSV and training CSV both reproduce byte for byte under it, and carries the
user-reported Kaggle score as unverified.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.cache import ROOT


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_hash():
    s = b''.join(f.read_bytes() for f in sorted((ROOT / 'constellation').glob('*.py')))
    return hashlib.sha256(s + (ROOT / 'run.py').read_bytes()).hexdigest()


def main():
    prev = json.loads((ROOT / 'outputs/joint_submission_record.json').read_text())
    sub = ROOT / 'outputs/joint_submission/submission.csv'
    record = dict(prev)
    record['source_sha256'] = source_hash()
    record['previous_source_sha256'] = prev['source_sha256']
    record['source_change'] = (
        'Sixth pass: null_mode and decorrelate_size. Fifth pass: twopass.py and QuerySlate.eligible. Fourth pass: verify_adaptive plus --verify-radius flag. Third pass: QuerySlate role separation (ranking / calibration / seeding), '
        'relocation guard interfaces, two latent build_pool faults fixed, relocation '
        'context exposed in diagnostics. Earlier: verify_rep and alternatives flags, '
        'recognize_joint defaults aligned to production. No prediction change.')
    record['equivalence_after_source_change'] = {
        'train_metrics_score': 0.7286878605463721,
        'train_csv_sha256': sha(ROOT / 'outputs/lab/pass6_train/submission.csv'),
        'train_csv_matches_previous': (
            sha(ROOT / 'outputs/joint_train/submission.csv')
            == sha(ROOT / 'outputs/lab/pass6_train/submission.csv')),
        'submission_csv_sha256': sha(ROOT / 'outputs/lab/pass6_submission/submission.csv'),
        'submission_csv_matches_previous': (
            sha(sub) == sha(ROOT / 'outputs/lab/pass6_submission/submission.csv')),
    }
    record['kaggle_status'] = (
        'User-reported public score approximately 0.637, up from the confirmed 0.58189. '
        'Recorded as user-reported only: no submission entry or CSV hash has been '
        'matched to it locally, so which CSV produced it is unverified. Not used for '
        'tuning.')
    record['sixth_pass'] = {
        'production_change': 'none; baseline retained',
        'target': 'identification (0.30 weight, unmoved across five prior passes)',
        'faults_confirmed': {
            'size_bias': ('wrong-class score correlates +0.660 with reference node '
                          'count; wrong 4-6 node fits score 2.15, wrong 19+ score 4.42'),
            'chance_density': ('fraction counted ~32 query groups where the pool holds '
                               '~124 points, understating chance density fourfold')},
        'corrections': 'null_mode = groups | pool | decorrelate | pool+decorrelate',
        'paired_synthetic_n384': {
            'groups': 0.482, 'pool': 0.487, 'decorrelate': 0.482,
            'pool+decorrelate': 0.500,
            'pool+decorrelate_net_scenes': '+7 (16 gained, 9 lost)',
            'p_two_sided': 0.230,
            'note': ('+0.036 on seed 2027 did not replicate on fresh seed 4099 '
                     '(0.469 for groups, pool and pool+decorrelate alike)')},
        'real_scenes': {'all_null_modes_identification': '2/3',
                        'total': 0.7111, 'baseline_total': 0.72869},
        'ceiling_measurement': {
            'within_scene_auc': {'score': 0.857, 'support': 0.825,
                                 'coverage': 0.456, 'residual': 0.454},
            'n_wrong_fits': 7466, 'n_true_fits': 192,
            'conclusion': ('coverage and residual are at or below chance, so two of the '
                           'four score terms contribute nothing. With ~40 competing '
                           'verified classes per scene an AUC of 0.857 puts '
                           'identification near 0.5. Identification is limited by the '
                           'discriminative content of the fit features, not by the '
                           "null's calibration.")},
        'adopted': False,
        'why_not': ('No supported gain (p=0.230), lower real-scene total, and the '
                    'correction trades large-reference accuracy for small while the '
                    'class-size distribution of the unlabelled scenes is unknown; the '
                    'three labelled references have 18, 13 and 12 nodes, all large.'),
        'next': ('A new independent signal, not a re-weighting: coarse/refined branch '
                 'agreement (currently discarded by letting the higher score win), '
                 'support excluding the self-confirming seed correspondences, auxiliary '
                 'evidence by source shape/scale/background, and dropping coverage and '
                 'residual from the score.'),
        'artifacts': ['outputs/lab/null_diag.json', 'outputs/lab/null_sweep.json',
                      'outputs/lab/null_verdict.json',
                      'outputs/lab/pass6_train', 'outputs/lab/pass6_submission'],
    }
    record['fifth_pass'] = {
        'production_change': 'none; baseline retained',
        'mechanism': ('constellation/twopass.py decouples pool eligibility from the '
                      'appearance gap: hypotheses from a first pass admit any existing '
                      'candidate near a predicted node, then a second pass reverifies. '
                      'No location is manufactured from a template prediction.'),
        'results': {'baseline': 0.72869,
                    'baseline_plus_twopass': 0.7051,
                    'adaptive': 0.7019,
                    'adaptive_plus_twopass': 0.7185,
                    'adaptive_plus_twopass_recovery': 0.822,
                    'adaptive_plus_twopass_figure_localization': 0.615,
                    'note': ('figure localization .615 is the best measured, above the '
                             'production .577; total still below production')},
        'recovery_ceiling': {'baseline_top20': 0.900, 'baseline_top8': 0.811,
                             'adaptive_top20': 0.933, 'adaptive_top8': 0.844,
                             'production_achieved': 0.878,
                             'note': ('adaptive has the better ceiling but the pipeline '
                                      'extracts 88% of it against production 98%, so '
                                      'the residual is an assignment limitation; '
                                      'production exceeds the refined-only ceiling@8 '
                                      'because a winning coarse branch pools positions '
                                      'absent from the refined list')},
        'objective_met': False,
        'next': ('Identification: .30 weight, unmoved across five passes, ~.36 implied '
                 'on hidden scenes against 2/3 development, and the only component '
                 'measurable with real power. Chance-fit calibration for pool and '
                 'search multiplicity first.'),
        'artifacts': ['outputs/lab/twopass.json', 'outputs/lab/twopass_push.json',
                      'outputs/lab/recovery_ceiling.json',
                      'outputs/lab/pass5_train', 'outputs/lab/pass5_submission'],
    }
    record['fourth_pass'] = {
        'production_change': 'none; baseline retained',
        'objective_met': False,
        'objective_note': ('Target was >=.05 gain in all four components. Not achieved '
                           'and not claimed. Candidate retention and presence improved '
                           'substantially; recovery and figure localization regressed.'),
        'mechanism_found': ('retrieval.verify used a fixed radius-12 disc while the '
                            'radius a pose admits is 15.5*scale. Half of scale 0.75 was '
                            'unusable and ~2/3 of the valid area at scale 1.33 was '
                            'discarded. verify_adaptive follows the admissible radius; '
                            'available via --verify-radius adaptive, not default.'),
        'full_pool_measurement': {
            'queries': 116, 'proposals_per_query': '~2200',
            'recall12_at20': {'fixed12': 0.901, 'adaptive': 0.972},
            'recall4_at20': {'fixed12': 0.873, 'adaptive': 0.958},
            'figure_retention': {'fixed12': 0.885, 'adaptive': 1.000},
            'present_minus_absent_separation': {'fixed12': 0.074, 'adaptive': 0.117},
            'hard_seven_ranks': {'fixed12': [39, 23, 247, 50, 153, 63, 149],
                                 'adaptive': [6, 6, 88, 32, 2, 35, 323]}},
        'end_to_end': {'baseline': 0.72869, 'adaptive_shipped_calibration': 0.7019,
                       'adaptive_best_recalibration': 0.7162,
                       'adaptive_recalibration_loso': 0.6721,
                       'presence': {'baseline': 0.7173, 'adaptive': 0.7371},
                       'recovery': {'baseline': 0.8778, 'adaptive_best': 0.819}},
        'cause': ('Better verification makes queries less ambiguous, the gap gate then '
                  'supplies fewer alternatives to the pool, geometry relocates less, '
                  'and recovery falls. Baseline recovery depends partly on ambiguity '
                  'that better appearance evidence removes.'),
        'not_done': ('Workstream 2 image-level benchmark was not built. Workstreams 3-5 '
                     'not reached beyond the calibration work reported.'),
        'artifacts': ['outputs/lab/verify_full.json',
                      'outputs/lab/verify_full_hard.json',
                      'outputs/lab/adaptive_calibration.json',
                      'outputs/lab/adaptive_pool.json',
                      'outputs/lab/adaptive_margin.json',
                      'outputs/lab/adaptive_train',
                      'outputs/lab/pass4_train', 'outputs/lab/pass4_submission'],
    }
    record['third_pass'] = {
        'production_change': 'none',
        'attribution': ('Factorial with presence/ambiguity/margin/pool-membership '
                        'frozen: changing seed anchors costs identification (true '
                        'class rank 0->1); changing ranking costs off-figure '
                        'localization only. Earlier re-ranking failures were seed '
                        'disruption, not verification choices.'),
        'relocation_guards': ('Off-figure localization restored .711->.778 and all '
                              'three large errors removed, but recovery falls '
                              '.878->.711; best guard .7204 < .72869 baseline.'),
        'failure_traces': ('Pre-suppression ranks 23,39,50,63,149,153,247. '
                           'Suppression removed none; ECC never saw them. Contrast '
                           'problem, not retrieval breadth or suppression.'),
        'corrected_claim': ('Previous "correct proposal ranks 0-1 before suppression" '
                            'was a median over all present queries, not the seven '
                            'failures.'),
        'artifacts': ['outputs/lab/factorial.json', 'outputs/lab/reloc_guards.json',
                      'outputs/lab/reloc_tradeoff.json',
                      'outputs/lab/failure_traces.json',
                      'outputs/lab/pass3_train', 'outputs/lab/pass3_submission'],
    }
    record['second_pass'] = {
        'production_change': 'none',
        'rejected_end_to_end': [
            'wholesale rescoring (incumbent + dog32/annulus, + full32)',
            'scale-preserving tie-break on ambiguous queries',
            'verify keep 40', 'verify DoG (keep 20 and 40)',
            'joint threshold/gap reselection'],
        'rejection_scope': ('All of the above are scoped to "without recalibration". '
                            'The ambiguity gate, pool margin and presence threshold '
                            'are calibrated to the score distribution verify emits.'),
        'proposal_bottleneck': ('All 7 uncovered present queries reach the proposal '
                                'set and are lost inside verify. Retrieval-side '
                                'sweeps closed as unnecessary.'),
        'verify_recall_available': {'baseline': 0.901, 'dog': 0.944, 'keep40': 0.958},
        'labelled_set_resolution': {
            'identification_quantum': 0.10,
            'one_query_localization': [0.0025, 0.0037],
            'all_seven_candidates_localization': 0.0200},
        'artifacts': [
            'outputs/lab/missing_attribution.json',
            'outputs/lab/verify_variants.log',
            'outputs/lab/end_to_end_rescore.json',
            'outputs/lab/tiebreak_e2e.json',
            'outputs/lab/verify_keep40', 'outputs/lab/verify_dog20',
            'outputs/lab/verify_dog40', 'outputs/lab/baseline_recheck',
            'outputs/lab/submission_recheck', 'lab/LEDGER.md'],
    }
    benchmark_record = ROOT / 'outputs/imagebench/workstream2_record.json'
    if benchmark_record.exists():
        record['workstream_2'] = json.loads(benchmark_record.read_text())

    workstream3_record = ROOT / 'outputs/lab/workstream3/record.json'
    if workstream3_record.exists():
        record['workstream_3'] = json.loads(workstream3_record.read_text())

    out = ROOT / 'outputs/joint_submission_record.json'
    out.write_text(json.dumps(record, indent=2))

    deliv = ROOT / 'deliverables'
    for name in ('Constellation_Classical.ipynb', 'constellation_inference.py',
                 'README.md', 'FINDINGS.md'):
        shutil.copy(ROOT / name, deliv / name)
    shutil.copy(out, deliv / 'joint_submission_record.json')
    shutil.copy(ROOT / 'lab/LEDGER.md', deliv / 'EXPERIMENT_LEDGER.md')
    if benchmark_record.exists():
        shutil.copy(ROOT / 'lab/imagebench/README.md', deliv / 'IMAGE_BENCHMARK.md')
        shutil.copy(benchmark_record, deliv / 'image_benchmark_record.json')
        latest = record['workstream_2'].get('realism_revision', {}).get('dataset', 'outputs/imagebench/v1')
        shutil.copy(ROOT / latest / 'audit.json', deliv / 'image_benchmark_audit.json')
    if workstream3_record.exists():
        shutil.copy(ROOT / 'lab/WORKSTREAM3.md', deliv / 'WORKSTREAM3.md')
        shutil.copy(workstream3_record, deliv / 'workstream3_record.json')
        shutil.copy(ROOT / 'outputs/lab/workstream3/summary.json', deliv / 'workstream3_summary.json')
    print(json.dumps(record['equivalence_after_source_change'], indent=1))
    print('source', record['source_sha256'])


if __name__ == '__main__':
    main()
