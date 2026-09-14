"""Drive `held_out.evaluate_fold` across all three folds for one seed (repair task
§3.6). Referenced by the original EXPERIMENT3_REPORT.md's Appendix B but never
implemented; the JSON artifacts it should have produced
(`outputs/exp3_pairwise/held_out_s{seed}.json`) existed from an ad-hoc,
uncommitted script. This is the reconstructed, committed driver.

Run:
    python -c "from experiments.exp3_pairwise.held_out_runner import run_all_folds; \
               run_all_folds(seed=31004, device='mps')"
"""
from __future__ import annotations

import json
from pathlib import Path

from . import OUT, SCENES
from .held_out import evaluate_fold
from .serialize import rows_to_json, slim_eval_result
from experiments.exp1.env import append_jsonl, write_json


def run_all_folds(seed: int, device: str, data: str | None = None,
                  say=print) -> dict:
    """Evaluate every fold's SELECTED arm (from `matrix_s{seed}.json`) on its own
    held-out sky exactly once. Writes per-fold and aggregate JSON."""
    selections = json.load(open(OUT / f'matrix_s{seed}.json'))
    fold_arm = {f: d['selection']['selected'] for f, d in selections.items()}
    say(f'Selected arms for seed {seed}: {fold_arm}')

    all_results = {}
    for fold in SCENES:
        arm = fold_arm[fold]
        say(f'\n[{fold}] evaluating held-out={fold} arm={arm}')
        result = evaluate_fold(fold, arm, seed=seed, device=device, data=data, say=say)
        slim = slim_eval_result(result)
        fold_dir = OUT / 'folds' / fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_json(fold_dir / f'held_out_s{seed}.json', slim)
        write_json(fold_dir / f'rows_s{seed}.json', rows_to_json(result['rows']))
        all_results[fold] = slim
        for stage, info in slim['stages'].items():
            m = info['held_out_metrics']
            say(f'  {stage:25s} score={m["score"]:.4f} pres={m["presence"]:.3f} '
               f'loc={m["localization"]:.3f} rec={m["recovery"]:.3f}')

    write_json(OUT / f'held_out_s{seed}.json', all_results)
    append_jsonl(OUT / 'runs.jsonl', {'event': 'held_out_eval', 'seed': seed,
                                      'folds': list(all_results.keys())})
    say(f'HELD-OUT EVALUATION SEED {seed} COMPLETE')
    return all_results


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--device', default='mps')
    args = ap.parse_args()
    run_all_folds(seed=args.seed, device=args.device)
