from __future__ import annotations
import hashlib, json
from pathlib import Path
from . import ROOT, OUT

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic_json(path: Path, value) -> None:
    import os, tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def source_manifest(paths: list[Path]) -> dict:
    return {str(p.relative_to(ROOT)): {"sha256": sha256(p), "size": p.stat().st_size}
            for p in paths if p.exists()}

