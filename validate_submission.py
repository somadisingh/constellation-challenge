import argparse,csv,json
from pathlib import Path
from constellation.contracts import parse_cell
p=argparse.ArgumentParser();p.add_argument('submission',type=Path);p.add_argument('--data',type=Path,default=Path('.'));a=p.parse_args()
with (a.data/'sample_submission.csv').open() as f:r=csv.DictReader(f);fields=r.fieldnames;sample=list(r)
with a.submission.open() as f:r=csv.DictReader(f);actual_fields=r.fieldnames;actual=list(r)
assert actual_fields==fields,'Column order mismatch'
assert len(actual)==len(sample),'Row count mismatch'
classes={p.stem.removesuffix('_pattern') for p in (a.data/'patterns').glob('*_pattern.png')}|{'unknown'}
real=present=0
for row,template in zip(actual,sample):
    assert row['Id']==template['Id'],'Scene order mismatch'
    assert row['n_patches']==template['n_patches'],'Patch count mismatch'
    assert row['constellation'] in classes,'Unknown class'
    count=int(row['n_patches']);real+=count
    for col in fields:
        if not col.startswith('patch_'):continue
        idx=int(col.split('_')[1]);point=parse_cell(row[col])
        if idx>count:assert point is None,'Nonempty padding'
        elif point is not None:
            assert 0<=point[0]<3000 and 0<=point[1]<3000,'Out-of-bounds coordinate'
            present+=1
print(json.dumps({'valid':True,'scenes':len(actual),'real_queries':real,'reported_present':present,'reported_absent':real-present,'columns':len(fields)},indent=2))
