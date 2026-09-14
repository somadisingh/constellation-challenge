"""Protected-artifact hashing and verification (repair task §3.6, rule 14).

`snapshot_protected` hashes every file under the directories Experiment 3 must
never modify: `constellation/`, `lab/`, `run.py`, and the frozen outputs of
Experiment 1/1B/2/production (`outputs/joint_train/`, `outputs/exp1/`,
`outputs/exp1b/`, `outputs/exp2_geometry/`). `verify_protected` re-hashes and
diffs against a prior snapshot; it must be called AFTER all intended writes for
the current run, per rule 14 ("Protected-file checks must be executed after all
intended writes").
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
    'outputs/exp1',
    'outputs/exp1b',
    'outputs/exp2_geometry',
)


def _iter_files(root: Path):
    if root.is_file():
        yield root
        return
    if not root.exists():
        return
    for p in sorted(root.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            yield p


def snapshot_protected(out_path: str | Path) -> dict:
    """Hash every file under `PROTECTED_ROOTS`; write `{relpath: sha256}` to `out_path`."""
    hashes = {}
    for rel in PROTECTED_ROOTS:
        for f in _iter_files(ROOT / rel):
            hashes[str(f.relative_to(ROOT))] = sha256_file(f)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(hashes, indent=2, sort_keys=True))
    return hashes


def verify_protected(before_path: str | Path) -> dict:
    """Re-hash `PROTECTED_ROOTS` now and diff against a `snapshot_protected` file.

    Returns `{'ok': bool, 'checked': int, 'changed': [...], 'missing': [...],
    'new_untracked': [...]}`. `new_untracked` lists files that exist now under a
    protected root but were absent from the snapshot -- also a violation, since it
    means something wrote a brand-new file into a read-only area.
    """
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
