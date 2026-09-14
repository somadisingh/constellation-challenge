"""Stage machine and command handlers (plan §10, §13).

Stage state is persisted so an interruption resumes without duplicating an
experiment, and a resume refuses to proceed when the code hash, data hashes or
config no longer match what produced the existing artifacts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from . import SCENES
from .config import load_config
from .env import (ROOT, append_jsonl, code_hash, env_dict, read_json, sha256_json,
                  write_json, resolve_device, SEED_MODEL, SEED_MODEL_SECOND)


# --- artifact layout --------------------------------------------------------------
@dataclass
class Paths:
    root: Path

    @property
    def state(self):
        return self.root / 'state.json'

    @property
    def runs(self):
        return self.root / 'runs.jsonl'

    @property
    def environment(self):
        return self.root / 'environment.json'

    @property
    def protocol(self):
        return self.root / 'protocol.json'

    @property
    def record(self):
        return self.root / 'record.json'

    @property
    def metrics(self):
        return self.root / 'metrics.json'

    def fold(self, fold: str) -> Path:
        return self.root / 'folds' / fold

    def bank(self, scene: str) -> Path:
        return self.root / 'banks' / f'{scene}.json'

    def checkpoints(self, fold: str) -> Path:
        return self.fold(fold) / 'checkpoints'

    def ensure(self):
        self.root.mkdir(parents=True, exist_ok=True)
        return self


def paths_for(args) -> Paths:
    return Paths(Path(args.output).resolve()).ensure()


def log_run(paths: Paths, stage: str, payload: dict) -> None:
    append_jsonl(paths.runs, {'stage': stage, 'timestamp': time.time(), **payload})


# --- resumable stage state --------------------------------------------------------
class StageState:
    def __init__(self, paths: Paths, config: dict, data_prov: dict | None = None):
        self.paths = paths
        self.config = config
        self.identity = {
            'code_hash': code_hash(),
            'config_sha256': sha256_json(config),
            'data_sha256': sha256_json(data_prov) if data_prov else None,
        }
        self.data = {'identity': self.identity, 'stages': {}}
        if paths.state.exists():
            self.data = read_json(paths.state)

    def compatible(self) -> tuple:
        stored = self.data.get('identity', {})
        mismatches = [k for k, v in self.identity.items()
                      if v is not None and stored.get(k) not in (None, v)]
        return (not mismatches), mismatches

    def refuse_if_incompatible(self, resume: bool) -> None:
        ok, mismatches = self.compatible()
        if not ok:
            raise SystemExit(
                f'[exp1] refusing to resume: {", ".join(mismatches)} changed since the '
                f'existing artifacts were produced. Use a fresh --output directory.')
        if not resume:
            return

    def get(self, key: str) -> dict | None:
        return self.data.get('stages', {}).get(key)

    def done(self, key: str) -> bool:
        entry = self.get(key)
        return bool(entry and entry.get('status') == 'complete')

    def mark(self, key: str, status: str, **payload) -> None:
        self.data.setdefault('stages', {})[key] = {
            'status': status, 'timestamp': time.time(), **payload}
        self.data['identity'] = self.identity
        write_json(self.paths.state, self.data)

    def save(self) -> None:
        write_json(self.paths.state, self.data)


def folds_for(args) -> list:
    return [args.fold] if getattr(args, 'fold', None) else list(SCENES)


def _device(args, config) -> str:
    request = getattr(args, 'device', None) or config['train']['device']
    device, info = resolve_device(request)
    if device == 'unavailable':
        raise SystemExit(f'[exp1] {info.get("cause", "device unavailable")}. '
                         f'Rerun with --device cpu to accept CPU explicitly.')
    return device


def _seed(args, config) -> int:
    return int(getattr(args, 'seed', None) or config['seeds']['model'])


# --- stage: preflight -------------------------------------------------------------
def cmd_preflight(args) -> int:
    from . import preflight
    paths = paths_for(args)
    config = load_config(args.config)
    report = preflight.run(getattr(args, 'device', 'mps'),
                           quick=getattr(args, 'quick', False))
    write_json(paths.root / 'preflight.json', report)
    write_json(paths.environment, {'environment': env_dict(),
                                   'device': report.get('device')})
    log_run(paths, 'preflight', {'ok': report.get('ok'),
                                 'problems': report.get('problems', [])})
    if not args.quiet:
        _print_preflight(report)
    return 0 if report.get('ok') else 1


def _print_preflight(report: dict) -> None:
    env = report['environment_check']['environment']
    print(f"machine   {env['machine']} macOS {env['mac_version']} "
          f"{env['cpu_brand']} {env['ram_gib']:.1f} GiB RAM "
          f"{env['free_disk_gib']:.0f} GiB free")
    print(f"python    {env['python']}  torch {env['torch']}  kornia {env['kornia']}")
    print(f"device    {report['device'].get('resolved')} "
          f"(built={report['device'].get('mps_built')} "
          f"available={report['device'].get('mps_available')})")
    if report.get('blocked'):
        print(f"BLOCKED   {report['blocked']}")
        return
    for name, entry in report.get('models', {}).items():
        if 'skipped' in entry:
            print(f'  {name:8s} SKIPPED {entry["skipped"]}')
            continue
        par = entry.get('parity', {})
        ov = entry['overfit']
        tp = entry.get('throughput', {})
        print(f'  {name:8s} params={entry["forward_backward"]["cpu"]["n_parameters"]:>9,} '
              f'parity cos>={par.get("cosine_min", float("nan")):.5f} '
              f'rank={par.get("top1_rank_agreement", float("nan")):.2f} '
              f'overfit {ov["loss_first"]:.4f}->{ov["loss_last"]:.4f} '
              + (f'{tp.get("steps_per_second", 0):.1f} steps/s' if tp else ''))
    print(f'ok        {report.get("ok")}')
    for problem in report.get('problems', []):
        print(f'  PROBLEM {problem}')


# --- stage: download-models -------------------------------------------------------
def cmd_download_models(args) -> int:
    from .models import BACKBONES, SPECS, weight_provenance
    paths = paths_for(args)
    entries = {}
    missing = []
    for name in BACKBONES:
        entry = weight_provenance(name)
        if not entry['present'] and getattr(args, 'allow_download', False):
            entry['download_attempted'] = _try_download(name)
            entry = {**weight_provenance(name),
                     'download_attempted': entry['download_attempted']}
        entries[name] = entry
        if not entry['present'] or not entry.get('sha256_matches'):
            missing.append(name)
    doc = {'models': entries, 'missing': missing,
           'note': ('Weights are local files under cnn/. Checkpoint families are '
                    'never substituted: a missing file is reported, not replaced.')}
    write_json(paths.root / 'models.json', doc)
    log_run(paths, 'download-models', {'missing': missing})
    if not args.quiet:
        for name, e in entries.items():
            state = ('ok' if e.get('sha256_matches')
                     else ('DIGEST MISMATCH' if e['present'] else 'MISSING'))
            print(f'  {name:8s} {state:16s} {e["weights_path"]}')
            print(f'           sha256 {e.get("actual_sha256", "-")}')
            print(f'           source {e["source_url"]}')
            print(f'           licence {e["license"]}')
    return 0 if not missing else 1


def _try_download(name: str) -> dict:
    """Only used with --allow-download, and only for the documented public URL."""
    import shutil
    import torch
    from .models import SPECS
    spec = SPECS[name]
    target = ROOT / spec.weights
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if name == 'sosnet':
            from kornia.feature import SOSNet
            SOSNet(pretrained=True)
            cached = Path(torch.hub.get_dir()) / 'checkpoints' / 'sosnet_32x32_liberty.pth'
            if cached.exists():
                shutil.copy2(cached, target)
                return {'ok': True, 'via': 'kornia SOSNet(pretrained=True)'}
        return {'ok': False, 'reason': 'no automated source configured for this model'}
    except Exception as exc:
        return {'ok': False, 'reason': repr(exc)}


# --- stage: prepare ---------------------------------------------------------------
def cmd_prepare(args) -> int:
    from . import splits
    from .data import data_provenance, load_scene, query_records
    paths = paths_for(args)
    config = load_config(args.config)
    prov = data_provenance(args.data)
    state = StageState(paths, config, prov)
    state.refuse_if_incompatible(getattr(args, 'resume', False))

    height, width = load_scene(SCENES[0], args.data).image.shape
    geometry = splits.check_partition_disjoint(height, width)
    protocol = {
        'config': config,
        'data': prov,
        'geometry_check': geometry,
        'seeds': config['seeds'],
        'code_hash': code_hash(),
        'queries': {s: len(query_records(s, args.data)) for s in SCENES},
    }
    write_json(paths.protocol, protocol)

    summary = {}
    for fold in folds_for(args):
        manifest = splits.fold_manifest(fold, args.data,
                                        config['protocol']['source_targets'])
        checks = {s: splits.check_bank(manifest['banks'][s])
                  for s in manifest['allowed']}
        manifest['bank_checks'] = checks
        out = paths.fold(fold) / 'manifest.json'
        write_json(out, manifest)
        ok = geometry['ok'] and all(c['ok'] for c in checks.values())
        summary[fold] = {
            'ok': ok,
            'held_out': manifest['held_out'],
            'allowed': manifest['allowed'],
            'unique_sources': manifest['unique_sources'],
            'problems': ([] if geometry['ok'] else geometry['problems'])
                        + [p for c in checks.values() for p in c['problems']],
            'manifest': str(out.relative_to(paths.root)),
            'manifest_sha256': manifest['manifest_sha256'],
        }
        state.mark(f'prepare:{fold}', 'complete' if ok else 'failed',
                   **{k: summary[fold][k] for k in ('unique_sources', 'problems')})
        if not args.quiet:
            print(f'fold {fold:9s} held-out={manifest["held_out"]:9s} '
                  f'allowed={",".join(manifest["allowed"])}')
            for scene in manifest['allowed']:
                t = manifest['banks'][scene]['totals']
                print(f'   {scene:9s} fit={t["fit"]:6d} val={t["val"]:5d} '
                      f'synthcal={t["synthcal"]:5d}')
            for problem in summary[fold]['problems']:
                print(f'   PROBLEM {problem}')
    write_json(paths.root / 'prepare.json', {'geometry_check': geometry,
                                             'folds': summary})
    log_run(paths, 'prepare', {'folds': {k: v['ok'] for k, v in summary.items()}})
    if not args.quiet:
        print(f'geometry check ok={geometry["ok"]} '
              f'(footprint radius {geometry["radius"]:.1f}px)')
    return 0 if all(v['ok'] for v in summary.values()) else 1


# --- stage: cache-candidates ------------------------------------------------------
def cmd_cache_candidates(args) -> int:
    from . import banks
    from .data import query_records
    paths = paths_for(args)
    config = load_config(args.config)
    missing = banks.missing_branches()
    if missing:
        for m in missing:
            print(f'[exp1] MISSING branch cache: {m["scene"]}/{m["branch"]} -> {m["path"]}')
        print('[exp1] regenerate with, for example:\n'
              '  OPENBLAS_NUM_THREADS=1 .venv/bin/python run.py --data . --mode evaluate '
              '--pipeline joint --verify-radius adaptive --output outputs/lab/adaptive_train')
        return 1

    summary = {}
    for scene in SCENES:
        records = query_records(scene, args.data)
        bank = banks.build_bank(scene)
        recall = banks.branch_recall(scene, records)
        oracle = banks.check_no_oracle(bank, records)
        bank['recall'] = recall
        bank['no_oracle_check'] = oracle
        write_json(paths.bank(scene), bank)
        summary[scene] = {'sizes': bank['sizes'], 'recall': recall,
                          'no_oracle_ok': oracle['ok']}
        if not args.quiet:
            u = recall['union']
            print(f'{scene:9s} n_queries={bank["n_queries"]:3d} '
                  f'bank size mean={bank["sizes"]["mean"]:.1f} '
                  f'min={bank["sizes"]["min"]} max={bank["sizes"]["max"]}')
            print(f'   union   recall@4={u["recall@4"]:.3f} '
                  f'@12={u["recall@12"]:.3f} @36={u["recall@36"]:.3f}')
            for branch in ('fixed_coarse', 'fixed_ecc', 'adaptive'):
                b = recall[branch]
                print(f'   {branch:12s} @4={b["recall@4"]:.3f} '
                      f'@12={b["recall@12"]:.3f} @36={b["recall@36"]:.3f}')
    write_json(paths.root / 'candidate_banks.json', summary)
    log_run(paths, 'cache-candidates', {'scenes': list(summary)})
    return 0


# --- stages implemented in dedicated modules --------------------------------------
def cmd_mine(args) -> int:
    from . import mining
    return mining.run_cli(args, paths_for(args), load_config(args.config))


def cmd_screen(args) -> int:
    from . import training
    return training.run_screen(args, paths_for(args), load_config(args.config))


def cmd_train(args) -> int:
    """Default is the full plan §10 schedule; the flags select a single control run."""
    from . import training
    paths, config = paths_for(args), load_config(args.config)
    if getattr(args, 'scratch', False) or getattr(args, 'random_negatives', False) \
            or getattr(args, 'steps', None) or getattr(args, 'model', None):
        return training.run_train(args, paths, config)
    return training.run_schedule(args, paths, config)


def cmd_calibrate(args) -> int:
    from . import calibration
    return calibration.run_cli(args, paths_for(args), load_config(args.config))


def cmd_evaluate(args) -> int:
    from . import evaluation
    return evaluation.run_cli(args, paths_for(args), load_config(args.config))


def cmd_report(args) -> int:
    from . import report
    return report.run_cli(args, paths_for(args), load_config(args.config))


def cmd_export(args) -> int:
    from . import exporting
    return exporting.run_cli(args, paths_for(args), load_config(args.config))


def cmd_predict(args) -> int:
    from . import exporting
    return exporting.run_predict(args, paths_for(args), load_config(args.config))


# --- stage: run-all ---------------------------------------------------------------
RUN_ALL_ORDER = ('preflight', 'download-models', 'prepare', 'cache-candidates',
                 'mine', 'screen', 'train', 'calibrate', 'evaluate', 'report')


def cmd_run_all(args) -> int:
    """Execute the stage machine, stopping only at a genuine blocker (plan §10)."""
    paths = paths_for(args)
    config = load_config(args.config)
    order = list(RUN_ALL_ORDER)
    if getattr(args, 'stop_after', None) and args.stop_after in order:
        order = order[:order.index(args.stop_after) + 1]

    handlers = {
        'preflight': cmd_preflight, 'download-models': cmd_download_models,
        'prepare': cmd_prepare, 'cache-candidates': cmd_cache_candidates,
        'mine': cmd_mine, 'screen': cmd_screen, 'train': cmd_train,
        'calibrate': cmd_calibrate, 'evaluate': cmd_evaluate, 'report': cmd_report,
    }
    results = {}
    for stage in order:
        sub = _clone_args(args, stage)
        print(f'\n===== run-all: {stage} =====')
        try:
            code = handlers[stage](sub)
        except SystemExit as exc:
            print(f'[exp1] run-all stopped at {stage}: {exc}')
            results[stage] = 'blocked'
            write_json(paths.root / 'run_all.json', {'results': results})
            return 1
        results[stage] = 'ok' if code == 0 else f'exit {code}'
        if code != 0 and stage in ('preflight', 'download-models', 'prepare',
                                   'cache-candidates'):
            print(f'[exp1] run-all stopped: {stage} is a hard dependency.')
            write_json(paths.root / 'run_all.json', {'results': results})
            return 1
    write_json(paths.root / 'run_all.json', {'results': results})
    return 0


def _clone_args(args, stage: str):
    import copy
    sub = copy.copy(args)
    defaults = {
        'quick': False, 'allow_download': False, 'refresh': False,
        'model': None, 'seed': None, 'steps': None, 'scratch': False,
        'random_negatives': False, 'controls_only': False, 'split': 'validation',
        'stop_after': None,
    }
    for key, value in defaults.items():
        if not hasattr(sub, key):
            setattr(sub, key, value)
    return sub
