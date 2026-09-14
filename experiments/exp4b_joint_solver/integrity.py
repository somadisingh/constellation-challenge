"""Protected-artifact hashing for Experiment 4B. Broader than Experiment 4's
own list: 4B must additionally never modify Experiment 4's own outputs, since
4B reads them read-only (the headroom/attribution records) to ground its
corrections.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.exp1.env import ROOT, sha256_file

PROTECTED_ROOTS = (
    'constellation',
    'lab',
    'run.py',
    'outputs/joint_train',
    'outputs/joint_submission',
    'outputs/exp1',
    'outputs/exp1b',
    'outputs/exp2_geometry',
    'outputs/exp3_pairwise',
    'outputs/exp4_joint_identification',
    'experiments/exp1',
    'experiments/exp1b',
    'experiments/exp2_geometry',
    'experiments/exp3_pairwise',
    'experiments/exp4_joint_identification',
)


def _iter_files(root: Path):
    """Yield protected files under `root`, excluding `__pycache__` bytecode and
    top-level Markdown documentation (`*.md`) -- documentation under `lab/`
    (e.g. `lab/LEDGER.md`) is meant to be updated by every experiment's final
    write step, exactly like top-level README.md/FINDINGS.md are never
    protected at all."""
    if root.is_file():
        if root.suffix.lower() != '.md':
            yield root
        return
    if not root.exists():
        return
    for p in sorted(root.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix.lower() != '.md':
            yield p


def snapshot_protected(out_path: str | Path) -> dict:
    hashes = {}
    for rel in PROTECTED_ROOTS:
        for f in _iter_files(ROOT / rel):
            hashes[str(f.relative_to(ROOT))] = sha256_file(f)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(hashes, indent=2, sort_keys=True))
    return hashes


def verify_protected(before_path: str | Path) -> dict:
    before = json.loads(Path(before_path).read_text())
    after = {}
    for rel in PROTECTED_ROOTS:
        for f in _iter_files(ROOT / rel):
            after[str(f.relative_to(ROOT))] = sha256_file(f)

    changed, missing = [], []
    for path, expected in before.items():
        if path not in after:
            missing.append(path)
        elif after[path] != expected:
            changed.append(path)
    new_untracked = [p for p in after if p not in before]

    ok = not changed and not missing and not new_untracked
    return {
        'ok': ok, 'checked': len(before),
        'changed': changed, 'missing': missing, 'new_untracked': new_untracked,
        'before_path': str(before_path),
    }
