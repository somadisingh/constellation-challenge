"""Apply the frozen Exp5B gate to validation and create a name-only A/B CSV."""
from __future__ import annotations
import csv,json,time
from pathlib import Path

from . import ROOT,OUT
from .pattern_graph import extract_all
from .synthetic_screen import solve
from experiments.exp1.env import write_json,sha256_file

BASE=ROOT/'outputs'/'exp3_pairwise'/'submission_candidate.csv'
TARGET=OUT/'submission_candidate_name_rescue.csv'

def alternatives(scene_id):
    p=ROOT/'outputs'/'joint_submission'/f'{scene_id}.json'
    d=json.loads(p.read_text());out=[]
    for q in d['diagnostics']['queries']:
        out.append([tuple(map(float,c[:5])) for c in (q.get('candidates') or [])])
    return out,sha256_file(p)

def run(say=print):
    OUT.mkdir(parents=True,exist_ok=True);graphs=extract_all(ROOT/'patterns')
    with BASE.open(newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    records={};checkpoint=OUT/'validation_rescue.json'
    for pos,row in enumerate(rows,1):
        scene=row['Id'];started=time.perf_counter();alts,digest=alternatives(scene)
        ranked=solve(alts,graphs);top,second=ranked[:2]
        margin=float(top['score']-second['score'])
        accepted=bool(margin>=2 and top['support']>=9 and top['held_out_support']>=5)
        old=row['constellation'];new=top['name'] if accepted else old
        row['constellation']=new
        records[scene]={'base':old,'affine_winner':top['name'],'final':new,
          'accepted':accepted,'changed':new!=old,'score':top['score'],'margin':margin,
          'support':top['support'],'held_out_support':top['held_out_support'],
          'stream':top['proposal_stream'],'runner_up':second['name'],
          'candidate_source_sha256':digest,'seconds':time.perf_counter()-started}
        write_json(checkpoint,{'status':'RUNNING','completed':pos,'total':len(rows),
          'base_csv':str(BASE.relative_to(ROOT)),'records':records})
        say(f"{pos}/{len(rows)} {scene}: {old} -> {new} accepted={accepted} "
            f"margin={margin:.3f} support={top['support']}/{top['held_out_support']}")
    with TARGET.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    # Mechanical guarantee: every non-name cell is identical to the base.
    with BASE.open(newline='') as f:base=list(csv.DictReader(f))
    immutable=[k for k in fields if k!='constellation']
    assert all(a[k]==b[k] for a,b in zip(base,rows) for k in immutable)
    changed=[s for s,x in records.items() if x['changed']]
    final={'status':'COMPLETE','completed':len(rows),'total':len(rows),
      'base_csv':str(BASE.relative_to(ROOT)),'base_sha256':sha256_file(BASE),
      'candidate_csv':str(TARGET.relative_to(ROOT)),'candidate_sha256':sha256_file(TARGET),
      'non_constellation_cells_identical':True,'n_changed':len(changed),
      'changed_scenes':changed,'records':records}
    write_json(checkpoint,final);return final

if __name__=='__main__':
    r=run();print('COMPLETE',r['n_changed'],r['candidate_csv'])
