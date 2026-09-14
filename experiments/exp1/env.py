"""Environment facts, provenance hashing, seeding and device selection (plan §4, §13).

Every cache/checkpoint key must include the code hash, data/source hashes, fold
partitions, augmentation/pose settings, model identity and seed. `key_for` builds
that digest so a stale artifact can never silently satisfy a later stage.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent

# Thread pinning per plan §4: one training process, no oversubscription.
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def pin_threads(n: int = 1) -> None:
    """Set single-threaded BLAS/OpenMP before heavy imports where possible."""
    for var in THREAD_VARS:
        os.environ.setdefault(var, str(n))
    try:
        import cv2
        cv2.setNumThreads(n)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj) -> str:
    """Stable digest of a JSON-serialisable object."""
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                   default=_fallback).encode())


def _fallback(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f'unserialisable {type(o)}')


def code_hash() -> str:
    """Digest of every source file in this package, so caches track code changes."""
    parts = []
    for p in sorted(PKG.rglob('*.py')):
        parts.append((str(p.relative_to(PKG)), sha256_file(p)))
    return sha256_json(parts)


def key_for(**parts) -> str:
    """Cache key: 16 hex chars over code hash plus the supplied identity."""
    return sha256_json({'code': code_hash(), **parts})[:16]


# --- seeding ---------------------------------------------------------------------
# Plan §5 fixes the manifest seeds. Fold IDs are folded into the derivation without
# ever becoming a filename or a network input.
SEED_SAMPLING = 31001
SEED_INNER_EVAL = 31002
SEED_AUGMENT = 31003
SEED_MODEL = 31004
SEED_MODEL_SECOND = 31005


def derive_seed(base: int, *tags) -> int:
    """Deterministic child seed from a manifest seed plus string/int tags."""
    digest = sha256_json([int(base), [str(t) for t in tags]])
    return int(digest[:8], 16) % (2 ** 31 - 1)


def rng(base: int, *tags) -> np.random.Generator:
    return np.random.default_rng(derive_seed(base, *tags))


def seed_torch(seed: int) -> None:
    import torch
    import random
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# --- device ----------------------------------------------------------------------
def resolve_device(requested: str = 'mps') -> tuple[str, dict]:
    """Return (device, diagnosis). Never silently downgrade mps -> cpu."""
    import torch
    info = {
        'requested': requested,
        'mps_built': bool(torch.backends.mps.is_built()),
        'mps_available': bool(torch.backends.mps.is_available()),
    }
    if requested == 'cpu':
        return 'cpu', info
    if requested == 'mps':
        if not info['mps_built']:
            info['cause'] = 'this torch wheel was not built with MPS support'
            return 'unavailable', info
        if not info['mps_available']:
            info['cause'] = ('MPS built but unavailable: needs Apple silicon and '
                             'macOS 14+ with command-line tools installed')
            return 'unavailable', info
        return 'mps', info
    return requested, info


def mps_sync(device: str) -> None:
    import torch
    if str(device).startswith('mps'):
        torch.mps.synchronize()


def peak_memory() -> dict:
    """Process RSS and, where exposed, MPS driver allocation."""
    out = {}
    try:
        import psutil
        out['process_rss_gib'] = psutil.Process().memory_info().rss / 2 ** 30
    except Exception:
        pass
    try:
        import torch
        if torch.backends.mps.is_available():
            out['mps_current_alloc_gib'] = torch.mps.current_allocated_memory() / 2 ** 30
            out['mps_driver_alloc_gib'] = torch.mps.driver_allocated_memory() / 2 ** 30
    except Exception:
        pass
    return out


# --- environment record ----------------------------------------------------------
@dataclass
class Environment:
    machine: str
    platform: str
    mac_version: str
    cpu_brand: str
    ram_gib: float
    free_disk_gib: float
    python: str
    executable: str
    torch: str
    kornia: str
    numpy: str
    scipy: str
    cv2: str
    mps_built: bool
    mps_available: bool
    xcode_clt: str
    code_hash: str
    packages: dict


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(['sysctl', '-n', name], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ''


def describe_environment() -> Environment:
    import shutil
    import torch
    import kornia
    import scipy
    import cv2
    from importlib.metadata import distributions

    ram = _sysctl('hw.memsize')
    try:
        mac = platform.mac_ver()[0]
    except Exception:
        mac = ''
    try:
        clt = subprocess.run(['xcode-select', '-p'], capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except Exception:
        clt = ''
    packages = {}
    for dist in distributions():
        name = dist.metadata['Name']
        if name:
            packages[name.lower()] = dist.version
    return Environment(
        machine=platform.machine(),
        platform=sys.platform,
        mac_version=mac,
        cpu_brand=_sysctl('machdep.cpu.brand_string'),
        ram_gib=(int(ram) / 2 ** 30) if ram.isdigit() else float('nan'),
        free_disk_gib=shutil.disk_usage(ROOT).free / 2 ** 30,
        python=platform.python_version(),
        executable=sys.executable,
        torch=torch.__version__,
        kornia=kornia.__version__,
        numpy=np.__version__,
        scipy=scipy.__version__,
        cv2=cv2.__version__,
        mps_built=bool(torch.backends.mps.is_built()),
        mps_available=bool(torch.backends.mps.is_available()),
        xcode_clt=clt,
        code_hash=code_hash(),
        packages=dict(sorted(packages.items())),
    )


# --- small IO helpers ------------------------------------------------------------
def write_json(path, obj, indent=2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=indent, sort_keys=False, default=_fallback)
    tmp.replace(path)
    return path


def read_json(path):
    with open(path) as f:
        return json.load(f)


def append_jsonl(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(obj, default=_fallback) + '\n')


def env_dict() -> dict:
    return asdict(describe_environment())
