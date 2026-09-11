import cv2,json
import numpy as np
from pathlib import Path
from constellation.contracts import read_truth
from constellation.retrieval import build_index,retrieve
cv2.setNumThreads(4)
truth=read_truth('train_ground_truth.csv');report={}
for name,t in truth.items():
    im=cv2.imread(f'train/{name}/{name}_image.png',0);points,desc,stars=build_index(im)
    d=json.loads(Path(f'outputs/radial/{name}.json').read_text())
    rows=[]
    for i,p in enumerate(t.patches):
        if p is None:continue
        q=cv2.imread(f'train/{name}/patches/patch_{i+1:02}.png',0)
        c=retrieve(q,points,desc,2000)
        dist=np.linalg.norm(c-np.array(p[:2]),axis=1)
        refined=np.array(d['diagnostics']['queries'][i]['candidates'])
        rows.append({'query':i+1,'star_distance':float(np.linalg.norm(stars-np.array(p[:2]),axis=1).min()),'proposal_distances':{str(b):float(dist[:b].min()) for b in (200,1000,2000)},'refined_best_distance':float(np.linalg.norm(refined[0,:2]-p[:2])),'refined_any_distance':float(np.linalg.norm(refined[:,:2]-p[:2],axis=1).min())})
    report[name]=rows
    print(name,{str(b):sum(r['proposal_distances'][str(b)]<=4 for r in rows)/len(rows) for b in (200,1000,2000)},flush=True)
Path('outputs/proposal_diagnostics.json').write_text(json.dumps(report,indent=2))
