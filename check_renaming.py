from pathlib import Path
import tempfile,csv,subprocess,sys,json,os
root=Path('.').resolve()
with tempfile.TemporaryDirectory(prefix='constellation-rename-') as td:
    data=Path(td);(data/'patterns').symlink_to(root/'patterns',target_is_directory=True)
    scene=data/'train'/'arbitrary_scene';scene.mkdir(parents=True)
    (scene/'unrelated_image.png').symlink_to(root/'train/pisces/pisces_image.png')
    (scene/'patches').symlink_to(root/'train/pisces/patches',target_is_directory=True)
    with (root/'train_ground_truth.csv').open() as f:r=csv.DictReader(f);fields=r.fieldnames;row=next(r)
    row['Id']='arbitrary_scene'
    with (data/'train_ground_truth.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(row)
    subprocess.check_call([sys.executable,str(root/'constellation_inference.py'),'--data',str(data),'--mode','smoke','--threads','1','--output',str(data/'out')],env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})
    got=json.loads((data/'out/arbitrary_scene.json').read_text())
    expected=json.loads((root/'outputs/notebook_smoke/pisces.json').read_text())
    result={'scene_and_sky_renaming_invariant':got['patches']==expected['patches'] and got['constellation']==expected['constellation'],'scope':'two-query end-to-end smoke fixture'}
    (root/'outputs/renaming_check.json').write_text(json.dumps(result,indent=2));print(result)
