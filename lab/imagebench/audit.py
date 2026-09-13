"""Integrity, split isolation, centre invariants, and descriptive realism checks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np
from constellation.contracts import read_truth
from .generate import sha, write_json, patch_stats, SPLITS


def overlap(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def quantiles(rows):
    if not rows:
        return {}
    return {key:dict(zip(['p10','p50','p90'], map(float,np.percentile([r[key] for r in rows], [10,50,90]))))
            for key in rows[0]}


def audit(root, data=None, evaluation=None):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Incomplete dataset')
    regions = {r['id']:r for r in manifest['regions']}
    model = manifest['spec'].get('source_model')
    if model:
        if model['fit_split'] != 'development' or model['source_sha256'] != manifest['spec']['sources']:
            raise ValueError('Invalid source-style fit provenance')
        for row in model['rows']:
            region = regions[row['region']]
            x, y = row['centre']; a, b, c, d = region['box']
            if region['split'] != 'development' or region['source'] != row['source'] or not (a+48 <= x < c-48 and b+48 <= y < d-48):
                raise ValueError('Source-style fit crosses development boundary')
    for a in regions.values():
        for b in regions.values():
            if a['source']==b['source'] and a['split']!=b['split'] and overlap(a['box'],b['box']):
                raise ValueError('Cross-split source overlap')
    for name, checksum in manifest['spec']['patterns'].items():
        if sha(root/'patterns'/name) != checksum:
            raise ValueError('Reference hash mismatch')
    constant_queries = []
    seen_ids, seen_patches, counts, classes = set(), {}, Counter(), {s:set() for s in SPLITS}
    stats = {'real':[], 'rendered':[]}
    summaries, profiles, repeated, close_pairs = [], Counter(), 0, 0
    for entry in manifest['scenes']:
        if entry['id'] in seen_ids:
            raise ValueError('Duplicate scene ID')
        seen_ids.add(entry['id'])
        if sha(root/entry['labels']) != entry['labels_sha256']:
            raise ValueError('Corrupt labels')
        labels = json.loads((root/entry['labels']).read_text())
        if len(labels['queries']) != entry['n_queries'] or len(labels['patches']) != entry['n_queries']:
            raise ValueError('Query count mismatch')
        uses = [labels['target_region']] + ([labels['donor_region']] if 'donor_region' in labels else [])
        if any(regions[r]['split']!=entry['split'] for r in uses):
            raise ValueError('Source region from wrong split')
        if 'donor_region' in labels and regions[labels['target_region']]['source']==regions[labels['donor_region']]['source']:
            raise ValueError('Real absent queries must come from another sky')
        identities = Counter(q['physical_id'] for q in labels['queries'] if q['category']!='absent')
        repeated += sum(n-1 for n in identities.values() if n>1)
        if entry['track']=='rendered':
            classes[entry['split']].add(labels['constellation'])
            close_pairs += 1
            if labels['issued_unique'] >= labels['reference_nodes']:
                raise ValueError('Rendered figure should be partial')
        for i, f in enumerate(entry['files']):
            path = root/f['path']
            if sha(path) != f['sha256']:
                raise ValueError(f'Corrupt input {path}')
            image = cv2.imread(str(path), 0)
            expected = tuple(entry['shape']) if i==0 else (32,32)
            if image is None or image.dtype!=np.uint8 or image.shape!=expected:
                raise ValueError('Image format/shape failure')
            if i==0:
                continue
            q, truth = labels['queries'][i-1], labels['patches'][i-1]
            centre = np.asarray(q['patch_to_source']) @ [15.5,15.5,1]
            if not np.allclose(centre, q['source_centre'], atol=1e-6):
                raise ValueError('Patch centre transform is incorrect')
            if (truth is None) != (q['category']=='absent'):
                raise ValueError('Presence label inconsistency')
            if truth is not None:
                if not np.allclose(centre, truth[:2], atol=1e-6):
                    raise ValueError('Ground-truth centre mismatch')
                if not (0<=truth[0]<entry['shape'][1] and 0<=truth[1]<entry['shape'][0]):
                    raise ValueError('Truth outside image')
            pixel_hash = __import__('hashlib').sha256(image.tobytes()).hexdigest()
            previous = seen_patches.get(pixel_hash)
            if int(image.min()) == int(image.max()):
                constant_queries.append(f['path'])
            elif previous and previous != entry['split']:
                raise ValueError('Exact decoded nonconstant query duplicate across splits')
            seen_patches[pixel_hash] = entry['split']
            counts[(entry['track'],entry['split'],q['category'])] += 1
            profiles[(entry['track'],entry['split'],q['profile'])] += 1
            # Confirmation inputs are checked for integrity, never scored or summarized.
            if entry['split']=='development':
                stats[entry['track']].append(patch_stats(image))
        summaries.append(dict(id=entry['id'], track=entry['track'], split=entry['split'],
                              n_queries=entry['n_queries']))
    report = dict(build_id=manifest['build_id'], integrity='passed', split_overlap='none',
                  constant_queries=constant_queries,
                  scenes=len(summaries), queries=sum(counts.values()), repeated_views=repeated,
                  rendered_close_pair_scenes=close_pairs,
                  counts={'/'.join(k):v for k,v in sorted(counts.items())},
                  profile_counts={'/'.join(k):v for k,v in sorted(profiles.items())},
                  class_coverage={s:len(c) for s,c in classes.items()},
                  development_patch_statistics={k:quantiles(v) for k,v in stats.items()},
                  caveats=[
                      'Region split is not unseen-real-scene validation; each original sky contributes to all splits.',
                      'Rendered scenes reuse split-specific background regions; scenes are not fully independent.',
                      'Degradation ranges are plausible hypotheses, not estimates of the unknown competition generator.',
                      'Absent means distinct source provenance; accidental visually similar matches remain possible.',
                      'Confirmation integrity checks do not constitute model evaluation or release confirmation scores.'])
    if data:
        data = Path(data)
        truth = read_truth(data/'train_ground_truth.csv')
        realstats=[]
        for scene,t in truth.items():
            for i in range(len(t.patches)):
                q=cv2.imread(str(data/'train'/scene/'patches'/f'patch_{i+1:02}.png'),0)
                realstats.append(patch_stats(q))
        reference=quantiles(realstats)
        report['actual_query_statistics']=reference
        report['realism_flags']={}
        for track,values in stats.items():
            medians=quantiles(values)
            report['realism_flags'][track]=[k for k in reference if medians and
                not reference[k]['p10']<=medians[k]['p50']<=reference[k]['p90']]
        report['realism_interpretation']='Flags mark synthetic medians outside actual p10-p90, not statistical equivalence tests. All-query statistics are descriptive only. Source-style fitting, when present, uses development parent crops; see the manifest and development comparison.'
    if evaluation:
        evalrows=json.loads((Path(evaluation)/'queries.json').read_text())
        if any(r['split']!='development' for r in evalrows):
            raise ValueError('This realism report only consumes development evaluation')
        from .evaluate import summarize
        report['development_matching_statistics']=summarize(evalrows)
        if data:
            cached=[]
            for scene,t in truth.items():
                p=Path(data)/'outputs/joint_train'/f'{scene}.json'
                if not p.exists(): continue
                obj=json.loads(p.read_text())
                for i,q in enumerate(obj['diagnostics']['queries']):
                    cs=q['candidates'];tr=t.patches[i]
                    cached.append(dict(scene=scene,physical_id=str(i), category='present' if tr else 'absent',
                        score=cs[0][2],gap=cs[0][2]-cs[1][2] if len(cs)>1 else None,
                        predicted_present=obj['patches'][i] is not None,present=tr is not None))
            report['actual_cached_matching_statistics']=summarize(cached) if cached else None
    write_json(root/'audit.json', report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('outputs/imagebench/v1'))
    p.add_argument('--data',type=Path,default=Path('.'))
    p.add_argument('--evaluation',type=Path)
    a=p.parse_args();r=audit(a.dataset,a.data,a.evaluation)
    print(json.dumps({k:v for k,v in r.items() if k not in ['counts','profile_counts']},indent=2))

if __name__=='__main__':main()
