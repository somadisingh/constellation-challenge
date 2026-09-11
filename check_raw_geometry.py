import json,cv2,numpy as np
from constellation.contracts import read_truth
from constellation.geometry import recognize
from constellation.references import extract_patterns
cv2.setNumThreads(4);patterns=extract_patterns('patterns')
for n,t in read_truth('train_ground_truth.csv').items():
 im=cv2.imread(f'train/{n}/{n}_image.png',0).astype(np.float32)
 dog=cv2.GaussianBlur(im,(0,0),1)-cv2.GaussianBlur(im,(0,0),8)
 response=cv2.dilate(dog,np.ones((25,25),np.uint8));sv=np.sort(response.ravel()[::10]);rank=(np.searchsorted(sv,response)/len(sv)).astype(np.float32)
 s=json.load(open(f'outputs/hybrid/{n}.json'));qs=[q['candidates'] for q in s['diagnostics']['queries']]
 pts=[q[0][:2] for q in qs if q[0][2]>=.65]
 label,m,g=recognize(pts,patterns,use_quads=True,shear_penalty=2.,tolerance=18.,auxiliary_map=rank)
 print(n,label,[(h['name'],h['support'],round(h['score'],2),round(h.get('auxiliary',0),2)) for h in g['hypotheses'][:5]],flush=True)
