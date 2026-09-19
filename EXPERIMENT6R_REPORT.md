# Experiment 6R: Evidence Repair and Invariant Graph-Pair RANSAC

## Question and outcome

Experiment 6R asked two sequential questions: which Experiment 6 claims are actually supported, and whether invariant real-graph edge ↔ candidate-pair hypotheses can move the true constellation near the top of a 48-class ranking. The audit found material unsupported claims. The repaired proposer is real, measured and enumeration-invariant, but performance remains negative. No CSV was promoted and nothing was uploaded to Kaggle.

This experiment is identification-only. The fallback remains `outputs/exp5c_affine_recovery/submission_recommended_constellation_08_only.csv`, SHA-256 `84fa0c3def2526694f8b89253e8e87c21b07d91bd5e0d6b798c74e64b09b4b17`, with only `constellation_08: eridanus -> orion` and public total `0.69695`. Presence, localization and recovery were not changed or rescored.

## Experiment 6 claim repair

Experiment 6 did implement constrained transforms, three-pixel duplicate grouping, unique final assignment, refitting and 12-pixel final scoring. Its real duplicate counts, development winners/ranks, zero accepted validation overwrites, runtime and test totals remain valid.

The following did not constitute valid evidence: `strategy` was ignored; no graph-guided proposer existed; `ablations.json` was an arm-name manifest; synthetic evaluation ran only the known true pattern; `accepted` equaled correctness; wrong overwrite was hardcoded false; stage traces contained placeholders; and `finalize.py` was a docstring. Accordingly, old synthetic precision and wrong-overwrite gates are invalidated and Experiment 6 is corrected to implementation-present but execution-incomplete and result-unsupported.

The authoritative old final-seed-2 artifact says coverage `0.533333`, not `0.244444`. The incorrect value came from `synthetic_fit.json` and was copied into the report and ledger. Even `0.533333` is not valid multiclass coverage because the old synthetic solver knew the true class. The complete claim table is in `exp6_claim_audit.json`.

## Graph-edge proposer

All 48 extracted patterns contribute their actual green edges: 420 total and no zero-edge pattern. Each edge records enumeration-independent endpoint degree multisets, recursive local graph/path signatures, a normalized length and length rank, neighborhood degree histograms, edge type, distances to junctions and a canonical endpoint signature. Candidate pairs require distinct queries and three-pixel physical clusters and record separation/rank, confidence/rank pairs, multiscale neighbor counts and coordinate-canonical endpoints.

The proposer creates both endpoint orientations for single-edge hypotheses and additionally joins adjacent pattern edges against candidate-edge pairs sharing one physical cluster. Two-edge ranking uses scale-free length ratios, shared-node structure and candidate confidence. Similarity and reflected similarity are fitted explicitly, followed by duplicate-safe assignment, iterative consensus refit and an inclusive 12-pixel final rescore. The measured labelled-development two-edge proposal-level inlier fractions were 0.000 Pisces, 0.027 Scorpius and 0.003 Taurus. At 0.99 confidence these imply no finite budget for Pisces under the measured pool, approximately 169 trials for Scorpius and 1,533 for Taurus. Consensus fraction was not substituted for proposal inlier fraction.

## Invariance

The complete bounded solver produced identical winners, accept/reject decisions and per-class scores after reversing and randomizing pattern dictionary order, permuting graph node storage with remapped edges, permuting queries, permuting candidates and repeating in two fresh processes. The two process result hashes were identical. Symmetric node hypotheses and both endpoint orientations are retained rather than resolved by node index.

## Development and ranking

The old Cartesian Experiment 6 ranks were Pisces 48, Scorpius 42 and Taurus 36. Graph-pair ranks were Pisces 42, Scorpius 14 and Taurus 35. Mean reciprocal rank improved from `0.02414` to `0.04127`, but the predeclared Pisces/Scorpius top-five targets both failed. Winners were Reticulum, Chamaeleon and Caelum respectively, and all development overwrites were rejected, preserving the correct existing Pisces and Scorpius names. Truth was used only for these traces and ranking diagnostics.

## Honest 48-class synthetic evaluation

Each synthetic scene was evaluated against all 48 patterns and acceptance came only from the frozen solver gate. Wrong overwrite was computed from the accepted winner and a defined incorrect fallback baseline.

Final seed 66004 had raw top-1 accuracy 0.25, top-five recall 0.50 and MRR `0.33766`; seed 66005 had raw top-1 accuracy 0, top-five recall 0.25 and MRR `0.07637`. Accepted coverage was 0 on both. Accepted precision and wrong-overwrite rate are consequently reported as 0 rather than as vacuous success. The system abstained on every scene and supplied no correct accepted prediction. These results directly contradict the optimistic but tautological Experiment 6 synthetic metrics.

## Measured ablations

`ablations.json` contains 23 executed records on one matched synthetic input. Every record includes configuration, input hash, seed, runtime, peak memory, raw accuracy, true-class rank, coverage, accepted precision, wrong-overwrite rate and status. The arms cover the corrected Cartesian control, single/two-edge proposals, signature removals, ordering variants, model variants, ranks 1/3/5, duplicate/assignment diagnostics, refitting and radius policies. The screen is small and suitable only for mechanism comparison, not precision estimation.

## Validation, runtime and gates

All 16 validation scenes were evaluated with all 48 class scores retained. No overwrite passed the local confidence rule, and the global promotion gates also failed, so no submission candidate was generated. Every baseline patch cell therefore remains untouched.

Validation took 19.43 seconds total; the median was 0.74 seconds per scene. Development took 4.57 seconds and the measured ablations took 18.70 seconds. Process peak RSS was recorded as 23,887,872 platform units.

Correctness, invariance, integrity, runtime and test gates passed. Performance gates failed for Pisces top-five rank, Scorpius top-five rank, nonzero honest synthetic accepted precision and two correctly accepted pattern families. The relevant focused suite passed 38/38. The full suite ran 384 tests: 322 passed, 62 skipped, 0 failed and 0 errored. Protected artifacts remained hash-identical.

## Reproducibility and limitations

The finalizer now validates 22 required upstream artifacts, fails on missing or incomplete stages, and generates `completion_audit.json`. The repaired experiment is audit-complete, execution-complete and evidence-supported, but performance-unsuccessful.

The principal limitation is semantic ambiguity: graph degrees and relative edge length still do not tell the system which observed pair belongs to a particular graph edge in dense clutter. The synthetic screen contains four true pattern families per partition, which is enough to expose failure but not to estimate fine-grained per-family precision.

The highest-value next action is to learn or engineer a truth-free local star-neighborhood descriptor around each physical cluster and compare it jointly with two adjacent pattern edges; another global score or looser acceptance threshold is not justified.
