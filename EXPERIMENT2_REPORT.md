# Experiment 2 — learned evidence in geometric recovery

## Result

This experiment keeps Experiment 1B's frozen learned presence decisions and uses
C0's independently verified geometric correspondences for coordinate recovery.
The selected rule snaps a learned-present query and rescues a learned-absent query
only when the production recognizer relocated that query onto its winning fit.
The constellation name remains the production geometric winner.

| rule | total | presence | localization | recovery | identification |
|---|---:|---:|---:|---:|---:|
| learned_raw | 0.721428 | 0.906413 | 0.724122 | 0.600000 | 0.666667 |
| snap_relocated | 0.744695 | 0.906413 | 0.673789 | 0.733333 | 0.666667 |
| rescue_relocated | 0.730358 | 0.865213 | 0.736942 | 0.666667 | 0.666667 |
| snap_and_rescue_relocated | 0.761959 | 0.865213 | 0.686610 | 0.833333 | 0.666667 |
| snap_and_rescue_member | 0.759188 | 0.854129 | 0.686610 | 0.833333 | 0.666667 |

C0 is 0.728688. The selected primary-seed rule is 0.761959 (+0.033271); the independent training-seed repeat is 0.751275 (+0.022587).

## Candidate-level geometry rerun

The stricter arm maps descriptor scores back to every refined/coarse candidate,
while freezing classical eligibility, ambiguity, pool membership, and hypothesis
seed coordinates. Learned within-query rank is the only new geometric signal.

| configuration | total | presence | localization | recovery | identification |
|---|---:|---:|---:|---:|---:|
| rank_weight_0.00:snap | 0.743555 | 0.906413 | 0.668091 | 0.733333 | 0.666667 |
| rank_weight_0.00:snap_rescue | 0.752870 | 0.833418 | 0.680912 | 0.833333 | 0.666667 |
| rank_weight_0.00:snap_keep_name | 0.743555 | 0.906413 | 0.668091 | 0.733333 | 0.666667 |
| rank_weight_0.00:snap_rescue_keep_name | 0.752870 | 0.833418 | 0.680912 | 0.833333 | 0.666667 |
| rank_weight_0.05:snap | 0.743555 | 0.906413 | 0.668091 | 0.733333 | 0.666667 |
| rank_weight_0.05:snap_rescue | 0.752870 | 0.833418 | 0.680912 | 0.833333 | 0.666667 |
| rank_weight_0.05:snap_keep_name | 0.743555 | 0.906413 | 0.668091 | 0.733333 | 0.666667 |
| rank_weight_0.05:snap_rescue_keep_name | 0.752870 | 0.833418 | 0.680912 | 0.833333 | 0.666667 |
| rank_weight_0.10:snap | 0.743555 | 0.906413 | 0.668091 | 0.733333 | 0.666667 |
| rank_weight_0.10:snap_rescue | 0.752870 | 0.833418 | 0.680912 | 0.833333 | 0.666667 |
| rank_weight_0.10:snap_keep_name | 0.743555 | 0.906413 | 0.668091 | 0.733333 | 0.666667 |
| rank_weight_0.10:snap_rescue_keep_name | 0.752870 | 0.833418 | 0.680912 | 0.833333 | 0.666667 |
| rank_weight_0.20:snap | 0.743555 | 0.906413 | 0.668091 | 0.733333 | 0.666667 |
| rank_weight_0.20:snap_rescue | 0.752870 | 0.833418 | 0.680912 | 0.833333 | 0.666667 |
| rank_weight_0.20:snap_keep_name | 0.743555 | 0.906413 | 0.668091 | 0.733333 | 0.666667 |
| rank_weight_0.20:snap_rescue_keep_name | 0.752870 | 0.833418 | 0.680912 | 0.833333 | 0.666667 |
| rank_weight_0.40:snap | 0.743555 | 0.906413 | 0.668091 | 0.733333 | 0.666667 |
| rank_weight_0.40:snap_rescue | 0.752870 | 0.833418 | 0.680912 | 0.833333 | 0.666667 |
| rank_weight_0.40:snap_keep_name | 0.743555 | 0.906413 | 0.668091 | 0.733333 | 0.666667 |
| rank_weight_0.40:snap_rescue_keep_name | 0.752870 | 0.833418 | 0.680912 | 0.833333 | 0.666667 |

Best primary candidate-level diagnostic: `rank_weight_0.00:snap_rescue` at 0.752870; the same fixed configuration scores 0.744654 on the repeat seed.

The learned rank term itself is rejected. Weights 0.05–0.40 make no change
to any primary-seed prediction; at 0.40 the repeat seed changes and regresses.
The useful result is the support-gated hybrid above, not learned reweighting
of class hypotheses. All 3,980 candidate mappings are exact (maximum 0 px).

## Per-scene result

| scene | C0 | selected | repeat |
|---|---:|---:|---:|
| pisces | 0.822894 | 0.948324 | 0.932603 |
| scorpius | 0.876395 | 0.910264 | 0.909936 |
| taurus | 0.486774 | 0.427288 | 0.411285 |

## Interpretation

The learned matcher and geometry have complementary strengths. Experiment 1B
improved presence and ordinary localization but lost figure recovery because it
could not consume C0's relocation map. The selected integration restores much of
that recovery without allowing learned ranking to change hypothesis seeds, the
candidate pool, or the winning constellation.

The rule was chosen from the causal primary-development comparison, then held
fixed before inspecting the repeat seed. It passes the development gate on both
seeds, but Taurus regresses by 0.05949 and the three scenes are repeatedly used.
All alternatives are reported for transparency. These three skies have been used
repeatedly, so this is strong development evidence rather than an untouched test
estimate. Production and the Kaggle submission remain unchanged. All 140 tests
pass; every protected computational artifact remains byte-identical. Only
`README.md` and `FINDINGS.md` changed in the inherited 330-file protected set,
as expected for this report update.

Machine record: `outputs/exp2_geometry/record.json`.
