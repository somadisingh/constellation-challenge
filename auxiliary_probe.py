import json,cv2,numpy as np
from constellation.contracts import read_truth
cv2.setNumThreads(4)
for n,t in read_truth('train_ground_truth.csv').items():
 im=cv2.imread(f'train/{n}/{n}_image.png',0).astype(np.float32)
 dog=cv2.GaussianBlur(im,(0,0),1)-cv2.GaussianBlur(im,(0,0),8)
 # Local query star detector; diagnostics only.
 response=cv2.dilate(dog,np.ones((25,25),np.uint8))
 sorted_vals=np.sort(response.ravel()[::10]);rank=np.searchsorted(sorted_vals,response)/len(sorted_vals)
 oracle=json.load(open('outputs/geometry_oracles.json'))[n]['figure']['diagnostics']['hypotheses'][0]
 for label,h in [('true',oracle)]+[(x['name'],x) for x in json.load(open(f'outputs/hybrid_ecc/{n}.json'))['diagnostics']['geometry']['hypotheses'][:5]]:
  pts=np.array(h['nodes']);inside=(pts[:,0]>=0)&(pts[:,1]>=0)&(pts[:,0]<3000)&(pts[:,1]<3000);pts=pts[inside].astype(int)
  print(n,label,len(pts),np.round(np.mean(rank[pts[:,1],pts[:,0]]),3))
