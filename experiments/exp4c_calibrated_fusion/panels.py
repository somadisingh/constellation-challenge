"""Compact post-selection diagnostics generated only from frozen JSON."""
import json
from PIL import Image,ImageDraw
from . import OUT,SCENES
from experiments.exp1.env import write_json

def run():
    OUT.joinpath('panels').mkdir(parents=True,exist_ok=True)
    p=json.load(open(OUT/'oof_primary.json'));r=json.load(open(OUT/'oof_repeat.json'))
    img=Image.new('RGB',(1200,520),'#111827');d=ImageDraw.Draw(img)
    d.text((30,20),'Experiment 4C calibrated-fusion OOF decisions',fill='white')
    y=70
    for scene in SCENES:
        a=p['per_scene'][scene]['raw'];b=r['per_scene'][scene]['raw']
        lines=[f'{scene}: true={a["true"]}',
               f'  primary raw={a["winner"]}, rank={a["true_class_rank"]}, margin={a["margin"]:.3f}',
               f'  repeat  raw={b["winner"]}, rank={b["true_class_rank"]}, margin={b["margin"]:.3f}',
               f'  gated primary={p["per_scene"][scene]["confidence_gated"]["selected"]}']
        for line in lines:d.text((40,y),line,fill='#d1d5db');y+=24
        y+=18
    path=OUT/'panels'/'oof_decisions.png';img.save(path)
    m={'panels':[{'path':str(path),'source':['oof_primary.json','oof_repeat.json']}],
       'generated_after_selection_frozen':True};write_json(OUT/'panels'/'manifest.json',m);return m
if __name__=='__main__':run()
