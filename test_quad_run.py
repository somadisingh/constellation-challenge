import json
from pathlib import Path
from constellation.contracts import read_truth
from constellation.geometry import recognize
from constellation.references import extract_patterns
patterns=extract_patterns('patterns')
for n,t in read_truth('train_ground_truth.csv').items():
    for mode in ('oracle','blind'):
        f=Path(f'outputs/hybrid_ecc/{n}.json')
        if mode=='blind' and not f.exists():continue
        if mode=='oracle':points=[p[:2] for p in t.patches if p is not None]
        else:
            d=json.loads(f.read_text());points=[p[:2] for p in d['patches'] if p is not None]
        name,m,g=recognize(points,patterns,use_quads=True,shear_penalty=2.,tolerance=12.)
        print(n,mode,name,[(h['name'],h['support'],round(h['score'],2)) for h in g['hypotheses'][:3]],flush=True)
