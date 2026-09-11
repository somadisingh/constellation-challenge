import json,argparse
from pathlib import Path
from constellation.contracts import ScenePrediction,read_truth,evaluate
from constellation.geometry import recognize
from constellation.references import extract_patterns
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args();patterns=extract_patterns('patterns');truth=read_truth('train_ground_truth.csv');out={}
for n,t in truth.items():
    file=a.input/f'{n}.json'
    if not file.exists():continue
    d=json.loads(file.read_text());qs=[q['candidates'] for q in d['diagnostics']['queries']];ids=[i for i,q in enumerate(qs) if q[0][2]>=.65]
    name,m,g=recognize([qs[i][0][:2] for i in ids],patterns,alternatives=[qs[i] for i in ids])
    print(n,name,[(h['name'],h['support'],round(h['score'],2)) for h in g['hypotheses'][:5]],flush=True);out[n]=g
(a.input/'alternative_geometry.json').write_text(json.dumps(out,indent=2))
