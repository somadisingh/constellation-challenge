"""Refresh the submission record after the second pass.

The pass shipped no prediction change. This records the new source hash, confirms the
submitted CSV and training CSV both reproduce byte for byte under it, and carries the
user-reported Kaggle score as unverified.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.cache import ROOT


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_hash():
    s = b''.join(f.read_bytes() for f in sorted((ROOT / 'constellation').glob('*.py')))
    return hashlib.sha256(s + (ROOT / 'run.py').read_bytes()).hexdigest()


def main():
    prev = json.loads((ROOT / 'outputs/joint_submission_record.json').read_text())
    sub = ROOT / 'outputs/joint_submission/submission.csv'
    record = dict(prev)
    record['source_sha256'] = source_hash()
    record['previous_source_sha256'] = prev['source_sha256']
    record['source_change'] = (
        'Configuration fields added (verify_rep, --alternatives, --verify-rep) and '
        'stale recognize_joint defaults aligned to production. No prediction change.')
    record['equivalence_after_source_change'] = {
        'train_metrics_score': 0.7286878605463721,
        'train_csv_sha256': sha(ROOT / 'outputs/lab/baseline_recheck/submission.csv'),
        'train_csv_matches_previous': (
            sha(ROOT / 'outputs/joint_train/submission.csv')
            == sha(ROOT / 'outputs/lab/baseline_recheck/submission.csv')),
        'submission_csv_sha256': sha(ROOT / 'outputs/lab/submission_recheck/submission.csv'),
        'submission_csv_matches_previous': (
            sha(sub) == sha(ROOT / 'outputs/lab/submission_recheck/submission.csv')),
    }
    record['kaggle_status'] = (
        'User-reported public score approximately 0.637, up from the confirmed 0.58189. '
        'Recorded as user-reported only: no submission entry or CSV hash has been '
        'matched to it locally, so which CSV produced it is unverified. Not used for '
        'tuning.')
    record['second_pass'] = {
        'production_change': 'none',
        'rejected_end_to_end': [
            'wholesale rescoring (incumbent + dog32/annulus, + full32)',
            'scale-preserving tie-break on ambiguous queries',
            'verify keep 40', 'verify DoG (keep 20 and 40)',
            'joint threshold/gap reselection'],
        'rejection_scope': ('All of the above are scoped to "without recalibration". '
                            'The ambiguity gate, pool margin and presence threshold '
                            'are calibrated to the score distribution verify emits.'),
        'proposal_bottleneck': ('All 7 uncovered present queries reach the proposal '
                                'set and are lost inside verify. Retrieval-side '
                                'sweeps closed as unnecessary.'),
        'verify_recall_available': {'baseline': 0.901, 'dog': 0.944, 'keep40': 0.958},
        'labelled_set_resolution': {
            'identification_quantum': 0.10,
            'one_query_localization': [0.0025, 0.0037],
            'all_seven_candidates_localization': 0.0200},
        'artifacts': [
            'outputs/lab/missing_attribution.json',
            'outputs/lab/verify_variants.log',
            'outputs/lab/end_to_end_rescore.json',
            'outputs/lab/tiebreak_e2e.json',
            'outputs/lab/verify_keep40', 'outputs/lab/verify_dog20',
            'outputs/lab/verify_dog40', 'outputs/lab/baseline_recheck',
            'outputs/lab/submission_recheck', 'lab/LEDGER.md'],
    }
    out = ROOT / 'outputs/joint_submission_record.json'
    out.write_text(json.dumps(record, indent=2))

    deliv = ROOT / 'deliverables'
    for name in ('Constellation_Classical.ipynb', 'constellation_inference.py',
                 'README.md', 'FINDINGS.md'):
        shutil.copy(ROOT / name, deliv / name)
    shutil.copy(out, deliv / 'joint_submission_record.json')
    shutil.copy(ROOT / 'lab/LEDGER.md', deliv / 'EXPERIMENT_LEDGER.md')
    print(json.dumps(record['equivalence_after_source_change'], indent=1))
    print('source', record['source_sha256'])


if __name__ == '__main__':
    main()
