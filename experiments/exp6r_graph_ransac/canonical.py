from __future__ import annotations
import hashlib, json
import numpy as np

def quantized_xy(xy,step=.01): return tuple(np.rint(np.asarray(xy,float)/step).astype(int))
def stable_digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def canonical_edge(a,b,signatures):
    ka,kb=signatures[a],signatures[b]
    return (a,b) if ka<kb else ((b,a) if kb<ka else (a,b))
def geometric_matrix_key(matrix,step=1e-6): return tuple(np.rint(np.asarray(matrix)/step).astype(np.int64).ravel())

