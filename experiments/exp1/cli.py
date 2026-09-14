"""Command line interface for Experiment 1 (plan §13).

    python -m experiments.exp1 <subcommand> [options]

Every subcommand exists and answers `--help`. Stages that are not yet reached by
the state machine report `status: not_run` rather than pretending to succeed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import SCENES
from .env import pin_threads

SUBCOMMANDS = ('preflight', 'download-models', 'prepare', 'cache-candidates', 'mine',
               'screen', 'train', 'calibrate', 'evaluate', 'report', 'export',
               'predict', 'run-all')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m experiments.exp1',
        description='Experiment 1: learned verification of real sky patches.')
    parser.add_argument('--version', action='version', version='exp1')
    sub = parser.add_subparsers(dest='command', metavar='SUBCOMMAND')

    def common(p, fold=False, model=False, seed=False, device=False, resume=False):
        p.add_argument('--data', default=None,
                       help='directory holding patterns/, train/, validation/')
        p.add_argument('--output', default='outputs/exp1', help='artifact root')
        p.add_argument('--config', default=None, help='YAML config override')
        if fold:
            p.add_argument('--fold', default=None, choices=SCENES,
                           help='outer held-out sky; default all folds')
        if model:
            p.add_argument('--model', default=None,
                           help='backbone name (hardnet, hynet, sosnet)')
        if seed:
            p.add_argument('--seed', type=int, default=None,
                           help='model seed; default 31004')
        if device:
            p.add_argument('--device', default='mps', choices=('mps', 'cpu'))
        if resume:
            p.add_argument('--resume', action='store_true',
                           help='continue from the persisted stage state')
        p.add_argument('--quiet', action='store_true')
        return p

    p = sub.add_parser('preflight', help='environment, device, parity and smoke tests')
    common(p, device=True)
    p.add_argument('--quick', action='store_true', help='shorter smoke, no throughput')

    p = sub.add_parser('download-models', help='verify/acquire backbone weights')
    common(p)
    p.add_argument('--allow-download', action='store_true',
                   help='permit fetching a missing public checkpoint')

    p = sub.add_parser('prepare', help='fold manifests, source banks, split tests')
    common(p, fold=True)

    p = sub.add_parser('cache-candidates', help='freeze the blind candidate banks')
    common(p, fold=True)

    p = sub.add_parser('mine', help='build the hard-negative banks on fit cells')
    common(p, fold=True, device=True)
    p.add_argument('--refresh', action='store_true',
                   help='recompute even if a bank exists')

    p = sub.add_parser('screen', help='2,000-step screening of T-H and T-Y')
    common(p, fold=True, model=True, seed=True, device=True, resume=True)

    p = sub.add_parser('train', help='continue the selected backbone to 8,000 steps')
    common(p, fold=True, model=True, seed=True, device=True, resume=True)
    p.add_argument('--steps', type=int, default=None)
    p.add_argument('--scratch', action='store_true', help='scratch-init control')
    p.add_argument('--random-negatives', action='store_true',
                   help='random-negative control')

    p = sub.add_parser('calibrate', help='fit presence calibration on allowed skies')
    common(p, fold=True, model=True, seed=True, device=True)

    p = sub.add_parser('evaluate', help='controls, inner banks and the outer sky')
    common(p, fold=True, model=True, seed=True, device=True)
    p.add_argument('--controls-only', action='store_true')

    p = sub.add_parser('report', help='write EXPERIMENT1_REPORT.md and record.json')
    common(p)

    p = sub.add_parser('export', help='export the frozen recipe and weights')
    common(p, model=True, seed=True, device=True)

    p = sub.add_parser('predict', help='optional validation submission candidate')
    common(p, device=True)
    p.add_argument('--split', default='validation')

    p = sub.add_parser('run-all', help='execute the stage machine (plan §10)')
    common(p, fold=True, device=True, resume=True)
    p.add_argument('--stop-after', default=None, choices=SUBCOMMANDS)
    return parser


def main(argv=None) -> int:
    pin_threads(1)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2

    from . import stages
    started = time.perf_counter()
    handler = {
        'preflight': stages.cmd_preflight,
        'download-models': stages.cmd_download_models,
        'prepare': stages.cmd_prepare,
        'cache-candidates': stages.cmd_cache_candidates,
        'mine': stages.cmd_mine,
        'screen': stages.cmd_screen,
        'train': stages.cmd_train,
        'calibrate': stages.cmd_calibrate,
        'evaluate': stages.cmd_evaluate,
        'report': stages.cmd_report,
        'export': stages.cmd_export,
        'predict': stages.cmd_predict,
        'run-all': stages.cmd_run_all,
    }[args.command]
    code = handler(args)
    if not getattr(args, 'quiet', False):
        print(f'[exp1] {args.command} finished in {time.perf_counter() - started:.1f}s '
              f'-> exit {code}')
    return int(code or 0)


if __name__ == '__main__':
    sys.exit(main())
