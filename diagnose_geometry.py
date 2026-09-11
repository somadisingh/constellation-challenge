import json,numpy as np
from constellation.references import extract_patterns
from constellation.quad import quads
from scipy.spatial import cKDTree
from constellation.contracts import read_truth
truth=read_truth('train_ground_truth.csv');patterns=extract_patterns('patterns');oracle=json.load(open('outputs/geometry_oracles.json'))
for name,t in truth.items():
    h=oracle[name]['figure']['diagnostics']['hypotheses'][0]
    p=patterns[name];p=(p-p.mean(0))/np.ptp(p,axis=0);mapped=np.c_[p,np.ones(len(p))]@np.array(h['matrix'])
    scene=np.array([v[:2] for v in t.patches if v is not None])
    ti,td=quads(p);si,sd=quads(scene);d,ix=cKDTree(sd).query(td,k=4)
    jobs=sorted([(dist,ti[i],si[j]) for i in range(len(ti)) for dist,j in zip(d[i],ix[i])],key=lambda x:x[0])
    ranks=[]
    for k,(dist,a,b) in enumerate(jobs):
        if np.max(np.linalg.norm(mapped[a]-scene[b],axis=1))<24:ranks.append(k)
    print(name,'correct hash candidate ranks',ranks[:20],flush=True)
