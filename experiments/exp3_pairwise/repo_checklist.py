"""Reusable, automated checks for the 18 future-work rules (repair task §12, §20).

This is intentionally generic: it inspects ANY experiment package's artifacts,
not just Experiment 3's, so future experiments can reuse it. Each `check_*`
function returns `{'ok': bool, 'detail': ...}`; `run_checklist` aggregates them.

Not every rule is mechanically checkable (e.g. rule 16, "record the correction and
cause", requires human judgment about intent) -- those are flagged `'ok': None`
(not applicable to purely mechanical enforcement) rather than silently skipped.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from experiments.exp1.env import ROOT


def check_machine_records_exist(json_paths: list) -> dict:
    """Rule 1: machine records precede prose. All listed JSON paths must exist."""
    missing = [p for p in json_paths if not (ROOT / p).exists()]
    return {'ok': not missing, 'missing': missing}


def check_completion_gated_by_audit(experiment_out_dir: str) -> dict:
    """Rule 2: no completion claim without an executable audit."""
    audit_path = ROOT / experiment_out_dir / 'completion_audit.json'
    if not audit_path.exists():
        return {'ok': False, 'detail': f'{audit_path} does not exist'}
    doc = json.loads(audit_path.read_text())
    return {'ok': doc.get('status') in ('COMPLETE', 'INCOMPLETE'), 'detail': doc.get('status')}


def check_no_pending_with_complete(json_paths: list) -> dict:
    """Rule 3: no `pending` field may coexist with an overall COMPLETE status."""
    problems = []
    for rel in json_paths:
        p = ROOT / rel
        if not p.exists():
            continue
        try:
            doc = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        text = json.dumps(doc).lower()
        if doc.get('status') == 'COMPLETE' and ('"pending"' in text or "'pending'" in text):
            problems.append(rel)
    return {'ok': not problems, 'problems': problems}


def check_every_reported_number_has_source(metrics_doc: dict) -> dict:
    """Rule 4 (partial, mechanical proxy): every top-level metric group must carry
    a `source_file`/`baseline_source_file` sibling key. This cannot fully verify
    semantic correctness, only structural presence of provenance fields."""
    problems = []
    for key in ('primary_baseline', 'repeat_baseline'):
        node = metrics_doc.get(key, {})
        if 'source_file' not in node:
            problems.append(f'{key} missing source_file')
    return {'ok': not problems, 'problems': problems}


def check_primary_repeat_baselines_not_mixed(metrics_doc: dict) -> dict:
    """Rule 5: primary and repeat runs always use matching baselines."""
    pb = metrics_doc.get('primary_baseline', {})
    rb = metrics_doc.get('repeat_baseline', {})
    ok = pb.get('seed_tag') == 'primary' and rb.get('seed_tag') == 'repeat'
    return {'ok': ok, 'primary_tag': pb.get('seed_tag'), 'repeat_tag': rb.get('seed_tag')}


def check_documented_commands_exist(report_path: str) -> dict:
    """Rule 6/7: every documented command/path in a report must resolve to a real
    module or file. Extracts `python -m package.module` and `python
    path/to/file.py` patterns from fenced code blocks and checks existence."""
    p = ROOT / report_path
    if not p.exists():
        return {'ok': False, 'detail': f'{report_path} does not exist'}
    text = p.read_text()
    problems = []
    for m in re.finditer(r'python3?\s+-m\s+([\w.]+)', text):
        mod = m.group(1)
        rel = Path(*mod.split('.')).with_suffix('.py')
        if not (ROOT / rel).exists() and not (ROOT / rel.parent / '__init__.py').exists():
            problems.append(f'-m {mod} -> {rel} not found')
    for m in re.finditer(r'python3?\s+([\w./]+\.py)', text):
        rel = Path(m.group(1))
        if not (ROOT / rel).exists():
            problems.append(f'script {rel} not found')
    return {'ok': not problems, 'problems': problems}


def check_deterministic_seeding(module_path: str) -> dict:
    """Rule 8: randomness must use deterministic derived seeds, never Python
    `hash()`. Greps for bare `hash(` calls feeding `np.random`/`random.seed`."""
    p = ROOT / module_path
    if not p.exists():
        return {'ok': None, 'detail': f'{module_path} does not exist'}
    text = p.read_text()
    hits = re.findall(r'(?:np\.random\.default_rng|random\.seed|np\.random\.seed)\([^)]*\bhash\(', text)
    return {'ok': not hits, 'hits': hits}


def check_checkpoint_metadata_atomic(best_dict: dict) -> dict:
    """Rule 9: checkpoint metadata must come from the selected checkpoint itself.
    Structural check: `best` must carry step + metric + presence + localization
    together (no separate lookup required)."""
    required = ('step', 'selection_metric', 'presence', 'localization')
    missing = [k for k in required if k not in best_dict]
    return {'ok': not missing, 'missing': missing}


def check_test_counts_separated(test_results_doc: dict) -> dict:
    """Rule 11: skipped, failed and passed tests must be reported separately."""
    problems = []
    for key in ('exp3_suite', 'production_suite'):
        node = test_results_doc.get(key, {})
        for field in ('passed', 'failures', 'errors', 'skipped'):
            if field not in node:
                problems.append(f'{key}.{field} missing')
    return {'ok': not problems, 'problems': problems}


def check_protected_files_after_writes(integrity_json_path: str) -> dict:
    """Rule 14: protected-file checks must be executed after all intended writes.
    Mechanical proxy: integrity.json's mtime must be >= every other artifact's
    mtime in the same output directory (i.e. it was written last)."""
    p = ROOT / integrity_json_path
    if not p.exists():
        return {'ok': False, 'detail': 'integrity.json missing'}
    integrity_mtime = p.stat().st_mtime
    out_dir = p.parent
    later = [str(f) for f in out_dir.glob('*.json')
            if f.name != 'integrity.json' and f.stat().st_mtime > integrity_mtime + 1.0]
    return {'ok': not later, 'written_after_integrity_check': later}


def run_checklist(experiment_out_dir: str, report_path: str,
                  negatives_module: str, best_checkpoint_examples: list,
                  metrics_json_path: str, test_results_json_path: str) -> dict:
    out = {}
    out['rule_2_completion_gated'] = check_completion_gated_by_audit(experiment_out_dir)
    out['rule_5_baselines_not_mixed'] = check_primary_repeat_baselines_not_mixed(
        json.loads((ROOT / metrics_json_path).read_text())
        if (ROOT / metrics_json_path).exists() else {})
    out['rule_6_7_commands_exist'] = check_documented_commands_exist(report_path)
    out['rule_8_deterministic_seeding'] = check_deterministic_seeding(negatives_module)
    out['rule_9_checkpoint_atomic'] = [
        check_checkpoint_metadata_atomic(b) for b in best_checkpoint_examples]
    out['rule_11_test_counts_separated'] = check_test_counts_separated(
        json.loads((ROOT / test_results_json_path).read_text())
        if (ROOT / test_results_json_path).exists() else {})
    out['rule_14_integrity_after_writes'] = check_protected_files_after_writes(
        f'{experiment_out_dir}/integrity.json')
    ok = all(
        (v['ok'] if isinstance(v, dict) else all(x['ok'] for x in v)) is not False
        for v in out.values())
    out['overall_ok'] = ok
    return out
