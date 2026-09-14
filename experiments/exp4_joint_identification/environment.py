"""Environment / source-hash recording (rule 8)."""
from __future__ import annotations

import platform
import subprocess
import sys

from . import OUT, ROOT
from experiments.exp1.env import code_hash, write_json


def record_environment(say=print) -> dict:
    rec = {'python': sys.version, 'platform': platform.platform(),
          'exp1_code_hash': code_hash()}
    try:
        import torch
        rec['torch'] = torch.__version__
        rec['mps_available'] = bool(torch.backends.mps.is_available())
    except ImportError:
        rec['torch'] = None
        rec['mps_available'] = None
    try:
        import numpy
        rec['numpy'] = numpy.__version__
    except ImportError:
        rec['numpy'] = None
    try:
        import scipy
        rec['scipy'] = scipy.__version__
    except ImportError:
        rec['scipy'] = None
    try:
        rec['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        rec['git_commit'] = None
    write_json(OUT / 'environment.json', rec)
    say(f'environment recorded: python={rec["python"].split()[0]} torch={rec["torch"]}')
    return rec


def record_source_hashes(say=print) -> dict:
    from experiments.exp1.env import sha256_file
    files = sorted((ROOT / 'experiments' / 'exp4_joint_identification').glob('*.py'))
    hashes = {str(f.relative_to(ROOT)): sha256_file(f) for f in files}
    write_json(OUT / 'source_hashes.json', hashes)
    say(f'source hashes recorded for {len(hashes)} files')
    return hashes


if __name__ == '__main__':
    record_environment()
    record_source_hashes()
