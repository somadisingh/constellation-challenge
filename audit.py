from pathlib import Path
import csv,hashlib,json
import cv2
import numpy as np
from PIL import Image,ImageDraw
from constellation.contracts import read_truth
out=Path('outputs/audit');out.mkdir(parents=True,exist_ok=True)
rows=[]; hashes={}; duplicates=[]
for path in sorted(Path('.').glob('**/*.png')):
    if path.parts[0] not in ('train','validation','patterns'):continue
    a=np.array(Image.open(path));h=hashlib.sha256(a.tobytes()).hexdigest()
    if h in hashes:duplicates.append([str(path),hashes[h]])
    hashes[h]=str(path)
    rows.append(dict(path=str(path),shape=str(a.shape),dtype=str(a.dtype),sha256=h))
with (out/'inventory.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
truth=read_truth('train_ground_truth.csv')
summary={n:dict(total=len(p.patches),figure=sum(x is not None and x[2]==1 for x in p.patches),off_figure=sum(x is not None and x[2]==0 for x in p.patches),absent=sum(x is None for x in p.patches)) for n,p in truth.items()}
(out/'summary.json').write_text(json.dumps(dict(files=len(rows),duplicates=duplicates,labels=summary),indent=2))
paths=sorted(Path('patterns').glob('*.png'))
canvas=Image.new('RGB',(1200,1600),'#202030');draw=ImageDraw.Draw(canvas)
for i,path in enumerate(paths):
    im=Image.open(path).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);bg=bg.convert('RGB');bg.thumbnail((185,170))
    x=(i%6)*200;y=(i//6)*200;canvas.paste(bg,(x,y+20));draw.text((x+2,y+2),path.stem.replace('_pattern',''),fill='white')
canvas.save(out/'patterns.jpg')
print(json.dumps(summary,indent=2));print('Files',len(rows),'duplicates',len(duplicates))
