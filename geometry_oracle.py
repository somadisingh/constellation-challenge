import json
from pathlib import Path
from constellation.contracts import read_truth
from constellation.references import extract_patterns
from constellation.geometry import recognize
truth=read_truth('train_ground_truth.csv');patterns=extract_patterns('patterns');out={}
print({n:len(p) for n,p in patterns.items()},flush=True)
for n,t in truth.items():
    out[n]={}
    for mode in ('present','figure'):
        points=[p[:2] for p in t.patches if p is not None and (mode=='present' or p[2]==1)]
        label,m,d=recognize(points,patterns)
        out[n][mode]={'prediction':label,'diagnostics':d}
        print(n,mode,label,[(h['name'],h['support'],round(h['score'],2)) for h in d['hypotheses'][:3]],flush=True)
Path('outputs/geometry_oracles.json').write_text(json.dumps(out,indent=2))
