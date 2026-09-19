"""Protected-artifact hashing and integrity verification for Experiment 5C."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.exp5c_affine_recovery import ROOT

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
    'outputs/exp4b_joint_solver',
    'outputs/exp4c_calibrated_fusion',
    'outputs/exp5_hypothesis_recovery',
    'outputs/exp5b_affine_proposals',
    'experiments/exp1',
    'experiments/exp1b',
    'experiments/exp2_geometry',
    'experiments/exp3_pairwise',
    'experiments/exp4_joint_identification',
    'experiments/exp4b_joint_solver',
    'experiments/exp4c_calibrated_fusion',
    'experiments/exp5_hypothesis_recovery',
    'experiments/exp5b_affine_proposals',
)


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(root: Path):
    if root.is_file():
        if root.suffix.lower() != '.md':
            yield root
        return
    if not root.exists():
        return
    for p in sorted(root.rglob('*')):
        if (p.is_file() and p.name != '.DS_Store' and '__pycache__' not in p.parts
                and p.suffix.lower() != '.md'):
            yield p


def snapshot_protected(out_path: str | Path) -> dict:
    hashes = {}
    for rel in PROTECTED_ROOTS:
        for f in _iter_files(ROOT / rel):
            hashes[str(f.relative_to(ROOT))] = sha256_file(f)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(hashes, indent=2, sort_keys=True))
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
        'ok': ok,
        'checked': len(before),
        'changed': changed,
        'missing': missing,
        'new_untracked': new_untracked,
        'before_path': str(before_path),
    }


if __name__ == '__main__':
    from experiments.exp5c_affine_recovery import OUT
    target = OUT / 'protected_before.json'
    snap = snapshot_protected(target)
    print(f"Snapshot taken: {len(snap)} files protected in {target}")
