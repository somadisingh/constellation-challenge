"""Development-only source-style fitting and distribution comparison for imagebench.

Fits source-selection summaries, not a predictor or query histogram-matching rule.
No calibration/confirmation pixels are used to fit the model.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.stats import wasserstein_distance

FEATURES = ('mean', 'std', 'centre_annulus', 'highpass_std')


def fit(data, output):
    from .generate import source_regions, patch_stats, sha, write_json
    from constellation.contracts import read_truth
    data=Path(data); images={str(p.relative_to(data)):cv2.imread(str(p),0)
                           for p in sorted((data/'train').glob('*/*_image.png'))}
    regions=source_regions(images)
    rows=[]
    for scene,t in read_truth(data/'train_ground_truth.csv').items():
        source=f'train/{scene}/{scene}_image.png';image=images[source]
        for index,tr in enumerate(t.patches):
            if tr is None: continue
            x,y=tr[:2]
            eligible=[r for r in regions if r['source']==source and r['split']=='development'
                      and r['box'][0]+48<=x<r['box'][2]-48
                      and r['box'][1]+48<=y<r['box'][3]-48]
            if not eligible:continue
            crop=cv2.getRectSubPix(image,(32,32),(float(x),float(y)))
            stats=patch_stats(crop);stats['background']=float(np.percentile(crop,20))
            rows.append(dict(source=source,region=eligible[0]['id'],centre=[x,y],
                             query_index=index+1,stats=stats))
    if len(rows)<8:raise ValueError('Too few development sources to fit a style model')
    model=dict(version=2,fit_split='development',rows=rows,
       features=list(FEATURES),source_sha256={n:sha(data/n) for n in images},
       truth_sha256=sha(data/'train_ground_truth.csv'),
       background_quantiles=np.percentile([r['stats']['background'] for r in rows],[10,50,90]).tolist(),
       renderer=dict(sigma=[1.,4.],amplitude_median=180.,amplitude_log_sigma=.8,
                     amplitude_bounds=[20.,600.],halo_fraction=[.1,.35],halo_scale=3.5),
       scope=f'{len(rows)}-source development fit: source-style hypothesis, not recovered competition generator. Renderer ranges are engineering choices.',
       selection='Match source-crop summaries before any query degradation, using the same rule for present and absent.')
    write_json(output,model)
    return model


def choose_sources(image, points, count, rng, model):
    """Sample target source styles, then nearby candidates in summary space.

    Selection is independent of query membership and degradation. It does not edit
    source pixels or force each generated query to have predetermined statistics.
    """
    from .generate import patch_stats
    summaries=[patch_stats(cv2.getRectSubPix(image,(32,32),tuple(map(float,p)))) for p in points]
    features=np.array([[s[k] for k in FEATURES] for s in summaries])
    if not len(points): raise ValueError('No candidate sources')
    targets=np.array([[r['stats'][k] for k in FEATURES] for r in model['rows']])
    scale=np.maximum(np.percentile(targets,75,axis=0)-np.percentile(targets,25,axis=0),[15,5,10,3])
    result=[]
    for _ in range(count):
        target=targets[int(rng.integers(len(targets)))]
        distance=np.mean(((features-target)/scale)**2,axis=1)
        ids=np.argsort(distance,kind='stable')[:min(12,len(points))]
        weights=np.exp(-(distance[ids]-distance[ids[0]])/.3)
        result.append(int(rng.choice(ids,p=weights/weights.sum())))
    return result


def compare(data, datasets, output):
    """Development-only raw-query comparison; never evaluates confirmation predictions."""
    from .generate import patch_stats,write_json
    from constellation.contracts import read_truth
    data=Path(data);reference=[]
    # Use queries whose known parent footprint is inside development regions.
    model=fit(data,Path(output).with_name('comparison_source_model.json'))
    for r in model['rows']:
        scene=Path(r['source']).parent.name
        image=cv2.imread(str(data/'train'/scene/'patches'/f'patch_{r["query_index"]:02}.png'),0)
        reference.append(patch_stats(image))
    def measure(rows):
        out={}
        for k in reference[0]:
            ref=np.array([r[k] for r in reference]);vs=np.array([r[k] for r in rows])
            scale=max(float(np.percentile(ref,90)-np.percentile(ref,10)),1e-3)
            out[k]=dict(reference_p10_p50_p90=np.percentile(ref,[10,50,90]).tolist(),
                        generated_p10_p50_p90=np.percentile(vs,[10,50,90]).tolist(),
                        normalized_wasserstein=float(wasserstein_distance(ref,vs)/scale))
        return out
    results={}
    for dataset in datasets:
        root=Path(dataset);m=json.loads((root/'manifest.json').read_text())
        trackrows={t:[] for t in ['real','rendered']}
        for e in m['scenes']:
            if e['split']!='development':continue
            for f in e['files'][1:]:
                trackrows[e['track']].append(patch_stats(cv2.imread(str(root/f['path']),0)))
        results[str(root)]={t:dict(n=len(rs),features=measure(rs)) for t,rs in trackrows.items()}
    report=dict(reference_n=len(reference),reference_split='development only; known-present queries',
                datasets=results,scope='Descriptive fitted-development comparison, not distribution equivalence or independent validation.')
    write_json(output,report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('fit');a.add_argument('--data',type=Path,default=Path('.'));a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('compare');a.add_argument('--data',type=Path,default=Path('.'));a.add_argument('--datasets',type=Path,nargs='+',required=True);a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=fit(args.data,args.output) if args.command=='fit' else compare(args.data,args.datasets,args.output)
    print(json.dumps(result if args.command=='compare' else {k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
