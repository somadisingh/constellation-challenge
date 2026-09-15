# Experiment 5B — Affine-aware hypothesis recovery

## Why this experiment was necessary

The Experiment 5 proposal stage compared triangle side ratios before fitting an
affine transform. Side ratios are invariant to similarity transforms, but the
competition diagrams explicitly permit anisotropic scale and shear. The audit
found all six correct Taurus figure correspondences in the candidate bank, yet
the best correct triangle ranked 6,698 against a budget of 300. The search was
therefore exhaustive only over an unsuitable proposal ordering.

Two advertised verification signals were also ineffective. Fourth-point support
excluded every already matched template node before checking support, and graph
consistency treated all node pairs as edges because the old pattern reader had
discarded the green diagram lines.

## Implementation

`experiments/exp5b_affine_proposals/` implements:

- signed-area affine invariants over four points;
- separate rank-one and one-alternate proposal streams, preventing the larger
  alternate index from crowding valid rank-one proposals out of its budget;
- one-to-one observed candidate assignment with proposal nodes excluded from
  independent held-out support;
- graph extraction from the actual green lines in all 48 pattern PNGs;
- a fixed high-confidence rescue gate: margin at least 2.0, total support at
  least 9, and held-out support at least 5.

## Measured real development result

| Scene | Affine winner | True rank | Winner score | Margin | Support | Held out | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Pisces | pisces | 1 | 11.679 | 3.142 | 10 | 6 | accept |
| Scorpius | orion | 20 | 6.396 | 0.180 | 8 | 4 | reject; retain classical |
| Taurus | taurus | 1 | 16.967 | 3.329 | 11 | 7 | accept |

The fixed hybrid is therefore 3/3 on the labelled scenes: it retains the
already correct classical Scorpius result and accepts the strong Pisces and
Taurus affine placements. This is a development result on only three skies and
must not be represented as out-of-sample accuracy.

## Synthetic engineering screen

The initial 12 distinct pattern shapes produced raw accuracy 3/12. The rescue
gate accepted 2/12 and both accepted cases were correct. A subsequent full run
was stopped at 46/48 after exceeding the runtime bound; among the 46 completed
cases, the gate accepted 5 and all 5 were correct. Raw accuracy is deliberately
not the deployment criterion: this component is a selective rescue layer.

The 46-case observation is partial and was not written as a completed all-48
artifact. `outputs/exp5b_affine_proposals/synthetic_all48.json` contains the
complete 12-case pilot. A future screen must checkpoint each scene before it is
rerun.

## Honest conclusion

This is the first recent experiment to repair the actual missing-hypothesis
mechanism and recover Taurus rather than fit another scoring model over the same
frozen pool. It does not validate hidden-scene transfer. The correct next step is
to run this fixed gate on validation as a constellation-name rescue while leaving
Exp3 presence/localization coordinates unchanged, inspect how many scenes it
changes, and submit it as a controlled A/B candidate only if the acceptance rate
is suitably selective.

## Validation inference

The frozen gate was applied to all 16 validation scenes using cached blind
candidate banks. It accepted four scenes, confirmed three existing labels, and
changed one: `constellation_04` from `corona-australis` to `canis-major`. That
hypothesis has score 13.321, runner-up margin 6.908, 11 one-to-one matches and 7
held-out matches.

The name-only candidate is
`outputs/exp5b_affine_proposals/submission_candidate_name_rescue.csv`. Every
cell except that single constellation name is identical to the corrected Exp3
candidate. Its SHA-256 is
`77bdbcccb5b898151510be5d5ad17b9ad2cbe00180d09d81af9f398c9b40cdbf`.
No Kaggle upload was performed.
