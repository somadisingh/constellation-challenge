# Experiment 6: Duplicate-Safe RANSAC for Constellation Identification

## Research question and isolation

Can automatically generated, duplicate-safe RANSAC consensus improve constellation names while preserving every Experiment 5C presence, coordinate, membership and recovery decision? Experiment 6 is identification-only. It did not retrain the verifier, rerun localization, change patch cells, inspect validation labels, or upload to Kaggle.

The frozen CSV baseline is `outputs/exp5c_affine_recovery/submission_recommended_constellation_08_only.csv` (SHA-256 `84fa0c3def2526694f8b89253e8e87c21b07d91bd5e0d6b798c74e64b09b4b17`). The prior public-score evidence remains: Experiment 3 `0.64695`; constellation_04-only `0.64695`; constellation_08-only `0.69695`; both `0.69695`. These are public totals, not hidden component scores. The sole supported deployed change remains `constellation_08: eridanus -> orion`.

## Reused artifacts and notebook interpretation

The experiment reused the 48 extracted pattern graphs and actual green edges, the Experiment 5C pattern index, cached candidate banks in `outputs/final_submission` and `outputs/lab/joint_reproduction`, and the Experiment 3/5C reports, policies, submissions and attribution records. Exact inputs are in `source_hashes.json`; the source-folder substitution is recorded in `context_audit.json`.

Lecture notebook cells 15–23 demonstrate two-point similarity proposals, Euclidean inlier counting at 12 pixels, consensus selection, all-inlier refitting and the standard trial formula. They are not deployable: the notebook manually supplies ten correct Scorpius node-to-star correspondences using ground truth, then adds 30 false pairs. Experiment 6 generates correspondences automatically and never imports train truth in validation inference.

## Candidate and duplicate audit

The audit preserves `query_id`, `candidate_rank`, `physical_star_cluster_id` and coordinate. At 3 pixels, the three development scenes contain 580 raw top-five candidates and 475 physical clusters (105 duplicate coordinates removed conceptually, not destructively). The 16 validation scenes contain 3,340 raw candidates and 2,848 physical clusters (492 duplicates). Detailed radius-2/3/6/12 counts and collision statistics are in `duplicate_audit.json`. Rank-one collisions already occur, and top-three/top-five collisions are substantially more common. Twelve pixels is therefore unsuitable as a blind merge radius; deployment grouping uses 3 pixels, while 12 pixels is reserved for final scoring.

## Correspondences, models and RANSAC

The bounded pool ranks automatic pattern-node/candidate pairs by candidate confidence, rank and node baseline. Graph descriptors and long-baseline ordering are inherited from the audited Experiment 5C index; the Cartesian arm is bounded and diagnostic only. PROSAC progressively widens the ranked pool. Every minimal set requires distinct pattern nodes, queries and physical clusters.

Four transform families share one engine: similarity (2 points), reflected similarity (2), rotation plus independent x/y scales (3), and bounded affine (3 non-collinear points). The fixed cascade evaluates similarity and reflection first, adding anisotropic and affine only when simpler support is insufficient. Affine hypotheses are bounded by condition number and shear. Discovery radii 12, 18 and 24 always lead to a fresh inclusive `<=12` final assignment.

Consensus uses explicit one-node/one-candidate matching plus deterministic query and cluster partition constraints. It records raw and unique support, held-out support, coverage, real graph edges, largest component, residual quantiles, refit stability and complexity. Minimal samples only propose a pose; consensus is refit for up to three iterations and rescored at 12 pixels. The measured synthetic median consensus fraction was 0.4286, implying 23 similarity or 57 three-point trials for 0.99 success under the idealized independence formula. The real pass used bounded budgets and therefore treats misses conservatively as abstentions.

## Results and ablations

The matched arms are enumerated in `ablations.json`, including uniform versus PROSAC, all four models and cascade, ranks 1/3/5, duplicate and assignment disablements, raw versus unique support, three discovery radii, graph/held-out/refit removal, fixed/adaptive trials and random/confidence ordering. Duplicate-disabled and no-assignment arms are diagnostic and are never deployable. Unit tests directly demonstrate that duplicate observations cannot inflate unique support and that one sky star cannot support multiple nodes.

On the repeatedly used development scenes, ungated RANSAC winners were wrong (`indus` for Pisces and `horologium` for Scorpius). Both overwrites were rejected, so the frozen fallback preserved Pisces and Scorpius. Taurus remained an adversarial diagnostic: its proposed overwrite was also rejected. This is a negative identification result, not an unbiased validation estimate.

The two untouched synthetic final seeds each achieved 1.00 precision among accepted cases and 0 wrong-overwrite rate, at limited coverage: 0.2667 (seed 60403) and 0.2444 (seed 60404). Per-pattern and per-family records are in the two final-seed JSON files. Synthetic data is an engineering screen, not proof of Kaggle transfer.

The frozen policy ran over all 16 validation scenes. It accepted zero overwrites and made zero name changes. Thus no novel submission candidate was generated. In particular, the deployed `constellation_08 -> orion` correction remains untouched. Presence, localization and recovery remain unchanged by construction; no new component scores are claimed.

## Runtime, gates and integrity

Validation inference took 15.76 seconds total in the recorded bounded run; detailed per-scene time, trial count and process peak-RSS units are in `runtime.json`. The focused relevant suite passed 44/44. The full repository suite passed 310 tests, skipped 62 because optional dependencies such as Torch were unavailable, and had zero failures or errors (372 total).

Performance promotion failed because pattern-order and query-order invariance were not independently established for the complete real-scene solver. All other recorded critical gates passed, including 12-pixel final scoring, unique matching, two final synthetic seeds, labelled fallback safety, full tests, protected-file integrity and no upload. `implementation_complete=true`, `performance_gates_passed=false`, and `submission_candidate_generated=false`.

## Limitations and next action

The automatic correspondence ordering still has low real-scene discriminative power; the wrong ungated development winners show that duplicate safety alone does not solve correspondence ambiguity. The bounded validation budget prioritizes safe abstention over exhaustive recall. Some ablation arms are structural diagnostics rather than statistically powered comparisons, and the synthetic generator only partially models real candidate-score behavior.

The single highest-value next action is to add a graph-edge-pair correspondence proposer whose ordering is provably invariant to query and pattern enumeration, then rerun the two missing invariance gates before considering any name overwrite.
