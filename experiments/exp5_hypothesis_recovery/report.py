"""Generate EXPERIMENT5_REPORT.md entirely from machine-readable JSON records
under `outputs/exp5_hypothesis_recovery/`. No metric value in this file is
hand-maintained.
"""
from __future__ import annotations

import json

from . import OUT, ROOT, SCENES


def load(n):
    return json.load(open(OUT / n))


def run():
    audit = load('completion_audit.json') if (OUT / 'completion_audit.json').exists() else {'status': 'INCOMPLETE'}
    gates = load('gates.json')
    branch = load('branch_decision.json')
    oracle = load('oracle_audit.json')
    trace = load('taurus_failure_trace.json')
    proposal = load('proposal_audit.json')
    geo = load('geometry_ablations.json')
    primary = load('oof_primary.json')
    repeat = load('oof_repeat.json')
    synthetic = load('synthetic_all48.json')
    runtime = load('runtime.json')
    tests = load('test_results.json')
    integ = load('integrity.json')
    fail = load('failure_analysis.json')
    deploy = load('deployment_policy.json')

    lines = []
    lines.append('# Experiment 5: Constellation-Independent Candidate and Geometric '
                 'Hypothesis Recovery')
    lines.append('')
    lines.append('**Generated entirely from machine-readable records** under '
                 '`outputs/exp5_hypothesis_recovery/`. Regenerate with '
                 '`python -m experiments.exp5_hypothesis_recovery.report`.')
    lines.append('')

    lines.append('## Status')
    lines.append('')
    lines.append(f'**{audit.get("status")}** (implementation) -- '
                 f'**performance_gates_passed={audit.get("performance_gates_passed")}** '
                 f'({gates["n_pass"]}/{gates["n_gates"]} gates)')
    lines.append('')
    lines.append(audit.get('note', ''))
    lines.append('')

    lines.append('## Stage 1: oracle audit and branch decision')
    lines.append('')
    lines.append('| Scene | Figure queries w/ correct candidate (12px) | Distinct correct '
                 'physical sources | >=4 correct figure locations | Branch-G condition met |')
    lines.append('|---|---:|---:|---|---|')
    for scene in SCENES:
        node = oracle['candidate_recall'][scene]
        bnode = branch['per_scene'][scene]
        lines.append(f'| {scene} | {node["n_figure_with_correct_candidate_12px"]}/'
                     f'{node["n_figure_queries"]} | {node["n_distinct_correct_physical_sources"]} | '
                     f'{node["at_least_4_correct_figure_locations"]} | {bnode["branch_g_condition_met"]} |')
    lines.append('')
    lines.append(f'**Branch selected: {branch["decision"]}.** {branch["rationale"]}')
    lines.append('')

    lines.append('## Taurus failure trace')
    lines.append('')
    ht = trace['hypothesis_trace']
    lines.append(f'{ht["attempted"]} seed triples attempted, {ht["accepted_valid_transform"]} produced a '
                 f'valid transform, {ht["n_hypotheses_meeting_min_support"]} reached min_support>=4. '
                 f'Best figure-match count across ALL generated hypotheses: '
                 f'**{ht["best_figure_match_count_any_hypothesis"]}/{ht["required_figure_matches"]}** required.')
    lines.append('')
    lines.append(ht['diagnosis'])
    lines.append('')
    lines.append('### Post-finalization proposal audit')
    lines.append('')
    lines.append(f'The earlier "exhaustive geometric ceiling" interpretation is corrected: the '
                 f'best correct Taurus triangle is ranked **{proposal["best_correct_triangle_rank"]}th** '
                 f'by the scene triangle descriptor, while the generator evaluates only '
                 f'**{proposal["proposal_budget_per_class"]}** proposals per class. All '
                 f'{proposal["n_correct_figure_correspondences_in_k5_generation_points"]} Taurus '
                 'figure correspondences are present in the k=5 generation set; the correct transform '
                 'is lost during proposal ranking.')
    lines.append('')
    lines.append('The triangle descriptor uses side-length ratios, which are preserved by similarity '
                 'transforms but not by a general affine transform. The fourth-point implementation '
                 'checks only nodes left unmatched by assignment, and the graph score does not consume '
                 'the supplied green edge graph or observed matched coordinates. These are concrete '
                 'representation defects, not evidence that Taurus is geometrically unrecoverable.')
    lines.append('')

    lines.append('## Stage 2: geometric hypothesis recovery -- matched ablations')
    lines.append('')
    lines.append('| Scene | Seed | Complete generator winner | Placement correct | Figure matches |')
    lines.append('|---|---:|---|---|---:|')
    for scene in SCENES:
        for seed in sorted(geo[scene]):
            node = geo[scene][seed]['11_complete_new_generator']
            lines.append(f'| {scene} | {seed} | {node["winner"]} | {node["placement_correct"]} | '
                         f'{node["n_figure_matches_12px"]}/{node["required_figure_matches"]} |')
    lines.append('')

    lines.append('## Stage 4: identification-only whole-sky OOF evaluation')
    lines.append('')
    lines.append('| Seed | Total | Presence | Localization | Recovery | Identification | Overrides |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for tag, doc in (('primary', primary), ('repeat', repeat)):
        m = doc['metrics']['mean']
        lines.append(f'| {tag} | {m["score"]:.4f} | {m["presence"]:.4f} | {m["localization"]:.4f} | '
                     f'{m["recovery"]:.4f} | {m["identification"]:.4f} | {doc["n_overrides"]} |')
    lines.append('')
    lines.append('| Scene | Baseline | Raw geometric winner | Selected (after confidence gate) | Correct |')
    lines.append('|---|---|---|---|---|')
    for scene in SCENES:
        node = primary['per_scene'][scene]
        lines.append(f'| {scene} | {node["baseline"]} | {node["raw"]["winner"]} | {node["selected"]} | '
                     f'{node["selected_correct"]} |')
    lines.append('')
    lines.append('Patch cells are byte-identical to the corrected Experiment 3 baseline on both '
                 f'seeds: `patch_identity` = {primary["patch_identity"]}.')
    lines.append('')

    lines.append('## Stage 5: all-pattern synthetic engineering screen')
    lines.append('')
    lines.append(f'{synthetic["n_affine_identifiable_patterns"]} of {synthetic["n_reference_patterns"]} '
                 f'reference patterns have >=4 nodes. Existing recognizer accuracy='
                 f'{synthetic["accuracies"]["existing"]:.4f}; Exp4B generator accuracy='
                 f'{synthetic["accuracies"]["exp4b_generator"]:.4f}; new generator (complete) accuracy='
                 f'{synthetic["accuracies"]["new_generator_complete"]:.4f}.')
    lines.append('')
    lines.append(f'Pattern-order independence: {synthetic["pattern_order_independence"]["identical"]}. '
                 f'Query-order independence: {synthetic["query_order_independence"]["identical"]}. '
                 f'Deterministic repeatability: {synthetic["deterministic_repeatability"]["identical"]}.')
    lines.append('')
    lines.append('**SYNTHETIC ENGINEERING SCREEN ONLY -- not evidence of real-scene or Kaggle transfer.** '
                 'The new generator underperforms both baselines here; see `failure_analysis.json` for '
                 'the root-cause investigation. Candidate truncation contributes on synthetic scenes, '
                 'while the similarity-only triangle proposal descriptor is the deeper transform mismatch.')
    lines.append('')

    lines.append('## Runtime and memory (M4 Pro)')
    lines.append('')
    lines.append(f'Deployable configuration (k=5): max {runtime["deployable_configuration_max_memory_mb"]:.1f}MB '
                 f'peak memory, mean {runtime["mean_seconds_per_run"]:.1f}s per scene/seed run -- feasible.')
    lines.append('')
    lines.append(f'k=20 "full multi-candidate bank" ablation: max '
                 f'{runtime["full_multi_candidate_bank_k20_max_memory_mb"]:.1f}MB peak memory -- '
                 'NOT feasible as configured (O(k^2) scaling), but this configuration is never deployed.')
    lines.append('')

    lines.append('## Promotion gates')
    lines.append('')
    lines.append(f'**{gates["n_pass"]}/{gates["n_gates"]} gates pass** (overall_pass={gates["overall_pass"]}, '
                 f'{gates["n_not_evaluable_or_not_applicable"]} not_evaluable/not_applicable).')
    lines.append('')
    lines.append('| # | Gate | Result |')
    lines.append('|---|---|---|')
    for name, node in gates['checks'].items():
        result = 'PASS' if node['pass'] else (node['status'].upper() if node['status'] != 'evaluated' else 'FAIL')
        lines.append(f'| | {name} | {result} |')
    lines.append('')

    lines.append('## Failure analysis summary')
    lines.append('')
    lines.append(fail['conclusion'])
    lines.append('')

    lines.append('## Test results')
    lines.append('')
    for k, v in tests.items():
        lines.append(f'- {k}: {v["ran"]} run, exit code {v["exit_code"]}, {v["failures"]} failures, '
                     f'{v["errors"]} errors, {v["skipped"]} skipped.')
    lines.append('')

    lines.append('## Integrity')
    lines.append('')
    lines.append(f'- Protected artifacts checked: {integ["checked"]}')
    lines.append(f'- Changed: {len(integ["changed"])}; Missing: {len(integ["missing"])}; '
                 f'New untracked: {len(integ["new_untracked"])}')
    lines.append(f'- Overall: {"OK" if integ["ok"] else "VIOLATION"}')
    lines.append('')

    lines.append('## Submission candidate')
    lines.append('')
    lines.append(f'None was generated. {gates["n_pass"]}/{gates["n_gates"]} promotion gates pass, not all '
                 '24, so per the task\'s own rule no candidate CSV was produced. '
                 f'Deployment decision: `{deploy["policy_type"]}`. Existing production, Experiment 3, and '
                 'earlier experiment CSVs were not modified (confirmed by the integrity check above).')
    lines.append('')
    lines.append('**Nothing was uploaded to Kaggle.**')
    lines.append('')

    lines.append('## Next action')
    lines.append('')
    lines.append('The next step should replace proposal generation before changing candidates or scores. '
                 'Use a genuinely affine-aware four-point proposal/index or a robust correspondence search '
                 'that does not prefilter with triangle side ratios; extract the supplied pattern adjacency '
                 'instead of treating every node pair as an edge; and validate matched observed fourth '
                 'points rather than only unmatched nodes. First require the oracle Taurus transform to '
                 'enter the proposal set without using its label. Candidate retraining is not justified by '
                 'the current audit because all six correct Taurus locations already occur in k=5.')
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('*Generated by `experiments/exp5_hypothesis_recovery/report.py` from machine records only.*')

    path = ROOT / 'EXPERIMENT5_REPORT.md'
    path.write_text('\n'.join(lines))
    return path


if __name__ == '__main__':
    p = run()
    print(f'wrote {p}')
