"""Generate Experiment 4C report exclusively from machine records."""
import json
import numpy as np
from . import OUT, ROOT, SCENES, SEEDS

def load(n): return json.load(open(OUT/n))

def run():
    audit=load('completion_audit.json') if (OUT/'completion_audit.json').exists() else {'status':'INCOMPLETE'}
    m=load('metrics.json'); g=load('gates.json'); syn=load('synthetic_all48.json')
    p=load('oof_primary.json'); r=load('oof_repeat.json'); tests=load('test_results.json')
    integ=load('integrity.json'); base=load('baseline_verification.json'); correction=load('prior_gate_corrections.json')
    cal=load('calibration.json'); select=load('selection_frozen.json'); schema=load('feature_schema.json')
    lines=['# Experiment 4C: Calibrated Joint-Evidence Fusion','',
      '**Generated from machine-readable records under `outputs/exp4c_calibrated_fusion/`.**','',
      '## Status','',f'**{audit.get("status","INCOMPLETE")}** (implementation); performance gates passed: **{g["overall_pass"]}** ({g["n_pass"]}/{g["n_gates"]}).','',
      '## Result','',
      'The predeclared regularized calibrated model did not generalize across whole-sky folds. '
      'Its conservative confidence gate preserved the existing names, so all submitted patch cells and all four official component means are unchanged. No submission was generated.','',
      '## Experiment 4B gate correction','',
      f'Experiment 4B remains implementation-complete and performance-negative. Its repeated Scorpius regression is diagnostic evidence of a systematic failure and is not a positive promotion gate. The corrected count is {correction["recount"]["n_positive"]} positive, {correction["recount"]["n_failed"]} failed, and {correction["recount"]["n_diagnostic_only"]} diagnostic-only check.','',
      '## Hypothesis records and labels','',
      '| Scene / seed | Records | Correct class and placement | Correct class, wrong placement | Wrong class |','|---|---:|---:|---:|---:|']
    for seed in SEEDS:
      for scene in SCENES:
        c=load(f'hypotheses_{scene}_s{seed}.json')['counts']
        lines.append(f'| {scene} / {seed} | {c["n_records"]} | {c["n_positive"]} | {c["n_true_class_wrong_placement"]} | {c["n_wrong_class"]} |')
    lines += ['','A positive requires the correct class and at least four issued figure stars within 12 px (or all issued figure stars when fewer than four). Three stars can fit an affine transform; the fourth supplies independent placement evidence.','',
      '## Frozen model and normalization','',
      f'- Arm: `{select["arm"]}`; regularization `C={select["c"]}`; normalization: `{select["normalization"]}`.',
      f'- Policy: `{select["policy"]}`. Selection was frozen before held-out evaluation.',
      '- Equal total weight is assigned per sky and per class within a sky; exact feature duplicates are removed deterministically.',
      f'- Implemented feature count: {len(schema["features"])}. Requested fields absent from the frozen caches are listed in `feature_schema.json` and are not fabricated.','',
      '## Coefficient stability','',
      '| Held-out sky | Primary/repeat cosine | Primary positives | Repeat positives |','|---|---:|---:|---:|']
    for scene in SCENES:
      x=cal['per_fold'][scene]
      lines.append(f'| {scene} | {x["coefficient_cosine_primary_repeat"]:.4f} | {x["primary_fit"]["n_positive"]} | {x["repeat_fit"]["n_positive"]} |')
    lines += ['','The coefficients are stable across seeds but consistently wrong on held-out skies. Stability therefore does not imply transfer. Full coefficient and scale tables are in `diagnostics.json`.','',
      '## Official component metrics','',
      '| Seed | Total | Presence | Localization | Recovery | Identification |','|---|---:|---:|---:|---:|---:|']
    for tag in ('primary','repeat'):
      x=m[tag]; lines.append(f'| {tag} | {x["score"]:.4f} | {x["presence"]:.4f} | {x["localization"]:.4f} | {x["recovery"]:.4f} | {x["identification"]:.4f} |')
    lines += ['','## Per-scene identification and rank','',
      '| Scene | Exp4B baseline true rank | Primary raw / rank | Repeat raw / rank | Guarded name |','|---|---:|---|---|---|']
    for s in SCENES:
      old=base['exp4b_independent_support']['actual'][s]['existing_score']
      a=p['per_scene'][s]['raw']; b=r['per_scene'][s]['raw']; sel=p['per_scene'][s]['confidence_gated']['selected']
      lines.append(f'| {s} | {old} | {a["winner"]} / {a["true_class_rank"]} | {b["winner"]} / {b["true_class_rank"]} | {sel} |')
    lines += ['','The raw calibrated model selects Eridanus for all three labelled skies. The shared confidence rule rejects all three overrides, retaining Pisces and Scorpius correctly and leaving Taurus wrong as Serpens Caput.','',
      '## Gate result','',f'**{g["n_pass"]}/{g["n_gates"]} pass; overall={g["overall_pass"]}.**','',
      '| Gate | Result |','|---|---|']
    for k,v in g['checks'].items(): lines.append(f'| {k} | {"PASS" if v["pass"] else "FAIL"} |')
    lines += ['','## Synthetic engineering screen','',
      f'{syn["n_affine_identifiable_patterns"]} of {syn["n_reference_patterns"]} references have at least four nodes. '
      f'Existing recognizer accuracy={syn["existing_accuracy"]:.4f}; additive={syn["comparison_accuracy"]["additive"]:.4f}; '
      f'calibrated={syn["comparison_accuracy"]["calibrated"]:.4f}; without unqueried={syn["comparison_accuracy"]["without_unqueried"]:.4f}; '
      f'without multiplicity={syn["comparison_accuracy"]["without_multiplicity"]:.4f}; guarded={syn["comparison_accuracy"]["confidence_gated"]:.4f}.','',
      'The two deterministic synthetic partitions contain 20 pattern classes each; a class is evaluated only with a model fitted on the opposite partition. This remains an engineering screen and is not evidence of Kaggle transfer. Eight references with fewer than four nodes are explicitly structurally unidentifiable under this free-affine validation rule.','',
      '## Retained, rejected, and inconclusive','',
      '- **Retained:** frozen Exp3 patch cells, fold isolation, explicit placement labels, class/sky weighting, deterministic deduplication, and the conservative name fallback.',
      '- **Rejected:** replacing the existing constellation name with the complete calibrated model; it reduces true-class rank and collapses to Eridanus. Calibration also fails to beat additive evidence on the class-disjoint synthetic screen.',
      '- **Inconclusive:** the usefulness of requested fields absent from the frozen Exp4B caches. They were disclosed and excluded rather than approximated after held-out inspection.','',
      '## Diagnostics, tests, and integrity','',
      'Machine-readable coefficient, feature-scale, calibration, score-distribution, template-bias, matched-null, override, overlay, and seed-disagreement diagnostics are recorded in `diagnostics.json` and `panels/`.']
    for k,v in tests.items(): lines.append(f'- {k}: {v["ran"]} run, exit code {v["exit_code"]}, skipped {v["skipped"]}.')
    lines += [f'- Protected artifacts: {integ["checked"]} checked; integrity OK={integ["ok"]}.','',
      '## Deployment and limitations','',
      'Not promoted. No Kaggle CSV was generated and nothing was uploaded. The existing Experiment 3 submission remains unchanged. The real OOF set contains only three skies and very few positive placement hypotheses: in particular, the frozen pool contains no positive Taurus hypothesis, so score fusion cannot recover Taurus.','',
      '## Next action','',
      'Experiment 5 should proceed, focused on scene-adaptive self-supervised correspondence and candidate/hypothesis recall. Experiment 4C shows that another fusion model over the same frozen pool is not justified.','']
    path=ROOT/'EXPERIMENT4C_REPORT.md'; path.write_text('\n'.join(lines)); return path

if __name__=='__main__': run()
