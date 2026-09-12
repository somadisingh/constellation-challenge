"""Record hashes and provenance for the joint-stage submission candidate."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.cache import ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sub = ROOT / 'outputs/joint_submission/submission.csv'
    manifest = json.loads((ROOT / 'outputs/joint_submission/manifest.json').read_text())
    train = json.loads((ROOT / 'outputs/joint_train/metrics.json').read_text())
    folds = json.loads((ROOT / 'outputs/lab/joint_threshold_folds.json').read_text())
    record = {
        'artifact': 'outputs/joint_submission/submission.csv',
        'submission_file_sha256': sha(sub),
        'source_sha256': manifest['source_sha256'],
        'pipeline': 'joint',
        'config': manifest['config'],
        'dependencies': {k: manifest[k] for k in ('python', 'numpy', 'opencv')},
        'development_metrics_three_labelled_scenes': {
            'mean': train['mean'], 'worst_score': train['worst_score'],
            'scope': ('All three scenes informed method development. Not an '
                      'untouched estimate of generalization.')},
        'threshold_holdout': {
            'mean': folds['metrics']['mean']['score'],
            'worst_score': folds['metrics']['worst_score'],
            'thresholds': {k: v['threshold'] for k, v in folds['threshold_folds'].items()},
            'scope': folds['scope']},
        'synthetic_identification_benchmark': {
            'scenes': 192, 'seed': 99, 'seed_used_for_tuning': 17,
            'min_reference_nodes': 7,
            'frozen_stage': 0.125, 'joint_stage': 0.474,
            'relocation_fixes': 230, 'relocation_regressions': 0,
            'scope': ('Synthesised candidate lists with noise measured from the '
                      'labelled scenes. Tests geometry only, supplies no auxiliary '
                      'star map, and is not evidence about the appearance stage.')},
        'reproduction': {
            'standalone_matches_module_csv': True,
            'train_csv_sha256': sha(ROOT / 'outputs/joint_train/submission.csv')},
        'validation': json.loads(
            (ROOT / 'outputs/lab/validate_joint.json').read_text())
        if (ROOT / 'outputs/lab/validate_joint.json').exists() else None,
        'kaggle_status': ('Not uploaded. No public score exists for this CSV. The '
                          'only confirmed score remains 0.58189 for the previous '
                          'stage.'),
    }
    out = ROOT / 'outputs/joint_submission_record.json'
    out.write_text(json.dumps(record, indent=2))

    # Refresh the hand-in bundle.
    deliv = ROOT / 'deliverables'
    for name in ('Constellation_Classical.ipynb', 'constellation_inference.py',
                 'README.md', 'FINDINGS.md', 'requirements.txt',
                 'requirements-lock.txt'):
        shutil.copy(ROOT / name, deliv / name)
    shutil.copy(sub, deliv / 'submission.csv')
    shutil.copy(out, deliv / 'joint_submission_record.json')
    shutil.copy(ROOT / 'outputs/lab/joint_threshold_folds.json',
                deliv / 'joint_threshold_folds.json')
    shutil.copy(ROOT / 'outputs/joint_train/metrics.json', deliv / 'joint_train_metrics.json')
    print(json.dumps({k: record[k] for k in
                      ('submission_file_sha256', 'source_sha256')}, indent=1))
    print('deliverables refreshed')


if __name__ == '__main__':
    main()
