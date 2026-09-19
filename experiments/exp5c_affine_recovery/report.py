"""Generate EXPERIMENT5C_REPORT.md from machine records."""
from __future__ import annotations
import json
from experiments.exp5c_affine_recovery import ROOT,OUT

def load(n):return json.loads((OUT/n).read_text())

def render():
    audit=load('completion_audit.json');g=load('gates.json');dev=load('development_results.json')
    s1=load('synthetic_final_seed1.json');s2=load('synthetic_final_seed2.json')
    val=load('validation_predictions.json');rt=load('runtime.json');integ=load('integrity.json')
    leaderboard=load('leaderboard_attribution.json')
    L=['# Experiment 5C — Efficient Affine Hypothesis Recall and Validation-Wide Identification Rescue','',
       '**Generated from machine-readable records under `outputs/exp5c_affine_recovery/`.**','',
       '## Status','',f"Implementation complete: **{audit['implementation_complete']}**. Performance gates passed: **{audit['performance_gates_passed']}** ({g['passed']}/{g['total']}). Submission candidate generated: **{audit['submission_candidate_generated']}**.",'',
       '## Mechanism','',
       'Experiment 5C replaces similarity-side-ratio proposal ordering with bounded signed-area affine quadruple retrieval. It preserves query identity and candidate rank, uses real green pattern edges, verifies one-to-one observed candidates outside the four seed nodes, adjusts the spatial null for the number of tested hypotheses, and keeps the proposal descriptor out of final confidence.','',
       '## Development diagnostics','',
       '| Scene | Role | Existing | Affine winner | Final | True rank | Accepted | Correct | Proposals | Seconds |','|---|---|---|---|---|---:|---|---|---:|---:|']
    for n,r in dev['scenes'].items():L.append(f"| {n} | {r['evaluation_role']} | {r['existing_name']} | {r['affine_winner']} | {r['final_name']} | {r['true_rank']} | {r['accepted']} | {r['correct_final']} | {r['proposals_evaluated']} | {r['runtime_seconds']:.1f} |")
    L += ['','Pisces and Scorpius are repeatedly used development skies, not an unbiased generalization estimate. Taurus is intentionally adversarial and is excluded from validation, confidence calibration, model selection, and promotion gates.','',
          '## Synthetic selective prediction','',
          '| Final seed | Raw accuracy | Primary coverage | Primary precision | Wrong-overwrite rate | Complete |','|---|---:|---:|---:|---:|---|']
    for label,s in [('seed1',s1),('seed2',s2)]:
        m=s['selective']['primary']; p='n/a' if m['precision'] is None else f"{m['precision']:.3f}"
        L.append(f"| {label} | {s['raw_accuracy']:.3f} | {m['coverage']:.3f} | {p} | {m['wrong_overwrite_rate']:.3f} | {s['status']=='COMPLETE'} |")
    L += ['','Synthetic results are engineering evidence only. Both final seeds were excluded from fitting and architecture selection.','',
          '## Validation inference','',f"Status: **{val['status']}**. Exact Experiment 3 baseline available: **{val['exact_exp3_baseline_available']}**.",'']
    for name,x in val['summary'].items():L.append(f"- {name}: accepted {x['accepted']}; changed {len(x['changed'])}; confirmed {len(x['confirmed'])}.")
    accepted=[(s,r) for s,r in val['records'].items() if r['accepted']['primary']]
    if accepted:
        L += ['','| Scene | Affine winner | Margin | Support | Held out | Confidence |','|---|---|---:|---:|---:|---:|']
        for s,r in accepted:L.append(f"| {s} | {r['affine_winner']} | {r['margin']:.3f} | {r['support']} | {r['held_out_support']} | {r['calibration_confidence']:.3f} |")
    if not val['exact_exp3_baseline_available']:
        L += ['','No CSV was generated because the exact ignored Experiment 3 baseline is absent. Substituting the reconstructed classical CSV would violate the name-only contract.']
    scores=leaderboard['public_scores'];candidate=leaderboard['recommended_candidate']
    L += ['','## Public leaderboard attribution','',
          'These user-reported Kaggle public-leaderboard scores are post-hoc deployment evidence, not validation evidence.','',
          '| Submission | Public score |','|---|---:|',
          f"| Experiment 3 baseline | {scores['experiment3_baseline']:.5f} |",
          f"| constellation_04 only | {scores['constellation_04_only']:.5f} |",
          f"| constellation_08 only | {scores['constellation_08_only']:.5f} |",
          f"| constellation_04 and constellation_08 | {scores['combined_04_and_08']:.5f} |",'',
          f"Recommended candidate: `{candidate['path']}` (SHA-256 `{candidate['sha256']}`).",
          'It changes only constellation_08 from eridanus to orion. The constellation_04 override is not deployed because its one-change ablation produced no measurable public-score gain.']
    L += ['','## Runtime and memory','']
    for n,x in rt.items():L.append(f"- {n}: {x['scenes']} scenes, {x['total_seconds']:.1f}s total, {x['max_seconds']:.1f}s maximum, peak RSS {x['peak_rss_gib']:.2f} GiB.")
    L += ['','## Promotion gates','',f"**{g['passed']}/{g['total']} passed; overall={g['overall_pass']}.**",'']
    for n,x in g['gates'].items():L.append(f"- {'PASS' if x['pass'] else 'FAIL'} — {n}")
    tests=load('test_results.json')
    L += ['','## Integrity and tests','',f"Protected files checked: {integ['checked']}; unchanged: **{integ['ok']}**.",
          f"Experiment 5C tests: {tests['exp5c']['passed']} passed, {tests['exp5c']['failed']} failed, {tests['exp5c']['errored']} errored, {tests['exp5c']['skipped']} skipped.",
          f"Full repository suite in the available non-ML environment: {tests['production']['passed']} passed, {tests['production']['failed']} failed, {tests['production']['errored']} errored, {tests['production']['skipped']} skipped. Historical ignored artifacts and PyTorch are absent.",
          'No Kaggle upload was performed. Negative results and blocked artifacts are retained in the machine records.','',
          '## Highest-value next action','']
    if val['exact_exp3_baseline_available']:
        L.append('Freeze the leaderboard-supported constellation_08 correction and investigate additional untouched validation scenes. Do not use Taurus or further public-leaderboard threshold tuning as validation evidence.')
    else:
        L.append('Restore the exact `outputs/exp3_pairwise/submission_candidate.csv` artifact and rerun only frozen validation serialization. This is the only valid way to produce and compare strict, primary and exploratory name-only CSVs without changing any patch prediction.')
    L.append('')
    text='\n'.join(L);(ROOT/'EXPERIMENT5C_REPORT.md').write_text(text);return text

if __name__=='__main__':print(render())
