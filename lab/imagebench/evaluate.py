"""Blind benchmark runner with reusable candidate caches and explicit confirmation access.

Real-region images measure correspondence only. Full scoring is available only on
3000x3000 rendered scenes (production geometry assumes that field area).
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time

import cv2
import numpy as np
from constellation.contracts import ScenePrediction, evaluate, reward
from constellation.pipeline import Config
from constellation.retrieval import build_harmonic_index, retrieve_harmonic, verify, verify_adaptive
from constellation.refine import refine_candidates
from constellation.references import extract_patterns
from constellation.finalize import finalize_joint
from .generate import digest, sha, write_json, SPLITS


def source_digest():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root/'constellation').glob('*.py')) + sorted(Path(__file__).parent.glob('*.py'))
    return digest({str(p.relative_to(root)): sha(p) for p in paths})


def load_manifest(root):
    m = json.loads((root/'manifest.json').read_text())
    if m['status'] != 'complete':
        raise ValueError('Dataset generation must finish before evaluation')
    return m


def input_images(root, entry):
    files = entry['files']
    for f in files:
        if sha(root/f['path']) != f['sha256']:
            raise ValueError(f'Input hash mismatch: {f["path"]}')
    arrays = [cv2.imread(str(root/f['path']), 0) for f in files]
    if any(x is None for x in arrays):
        raise ValueError('Unreadable benchmark input')
    return arrays[0], arrays[1:]


def get_proposals(root, entry, image, patches, cache, config):
    """No label access. Cache proposal stage independently of verification policy."""
    key = digest(dict(files=entry['files'], stage='harmonic+dense-v1', source=source_digest()))
    path = cache/'proposals'/f'{key}.json'
    if path.exists():
        return json.loads(path.read_text())['proposals']
    points, desc, _ = build_harmonic_index(image)
    from constellation.dense import dense_candidates
    proposals = []
    for patch in patches:
        p = retrieve_harmonic(patch, points, desc)
        p = np.unique(np.vstack([p, dense_candidates(image, patch)]), axis=0)
        proposals.append(p.tolist())
    write_json(path, dict(key=key, proposals=proposals))
    return proposals


def appearance(image, patches, proposals, config):
    coarse, refined = [], []
    scene_rep, _ = config.verify_pair(image, patches[0])
    for patch, points in zip(patches, proposals):
        _, patch_rep = config.verify_pair(np.zeros((1, 1), np.float32), patch)
        args = (scene_rep, patch_rep, np.asarray(points), config.alternatives)
        c = verify(*args) if config.verify_radius == 'fixed' else verify_adaptive(*args, config.verify_radius)
        coarse.append(c)
        refined.append(refine_candidates(image, patch, c))
    return coarse, refined


def local_prediction(refined, threshold):
    return ScenePrediction([tuple([*q[0][:2], 0]) if q and q[0][2] >= threshold else None
                            for q in refined])


def measurement_rows(entry, labels, prediction, proposals, coarse, refined):
    rows = []
    for i, (truth, meta, got, pp, cc, rr) in enumerate(zip(
            labels['patches'], labels['queries'], prediction.patches, proposals, coarse, refined)):
        row = dict(scene=entry['id'], index=i, track=entry['track'], split=entry['split'],
                   category=meta['category'], profile=meta['profile'], physical_id=meta['physical_id'],
                   present=truth is not None, predicted_present=got is not None,
                   score=float(rr[0][2]) if rr else -1.,
                   gap=float(rr[0][2]-rr[1][2]) if len(rr)>1 else None,
                   proposal_count=len(pp), coarse_count=len(cc), refined_count=len(rr))
        if truth is not None:
            xy = np.array(truth[:2])
            def distances(q):
                return np.linalg.norm(np.asarray(q)[:, :2]-xy, axis=1) if len(q) else np.array([])
            pd, cd, rd = distances(pp), distances(cc), distances(rr)
            for radius in (4, 12):
                row[f'proposal_recall{radius}'] = bool(np.any(pd <= radius))
                row[f'coarse_recall{radius}'] = bool(np.any(cd <= radius))
                row[f'refined_recall{radius}'] = bool(np.any(rd <= radius))
                row[f'union_recall{radius}'] = bool(np.any(np.r_[cd, rd] <= radius))
            hits = np.where(rd <= 12)[0]
            row.update(correct_rank=int(hits[0]) if len(hits) else None,
                       top1_error=float(rd[0]) if len(rd) else None,
                       localization=float(reward(np.linalg.norm(np.asarray(got[:2])-xy))) if got else 0.,
                       top1_reward=float(reward(rd[0])) if len(rd) else 0.)
        rows.append(row)
    return rows


def summarize(rows):
    def mean(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    def quant(values):
        values = [v for v in values if v is not None]
        return dict(zip(['p10','p50','p90'], map(float, np.percentile(values, [10,50,90])))) if values else None
    out = {}
    for category in sorted({r['category'] for r in rows}):
        rs = [r for r in rows if r['category'] == category]
        out[category] = dict(n_queries=len(rs), n_scenes=len({r['scene'] for r in rs}),
                            n_physical_sources=len({(r['scene'],r['physical_id']) for r in rs}),
                            score=quant([r['score'] for r in rs]), gap=quant([r['gap'] for r in rs]),
                            predicted_present_rate=mean([r['predicted_present'] for r in rs]))
        if rs[0]['present']:
            out[category].update({k: mean([r.get(k) for r in rs]) for k in [
                'localization','top1_reward','proposal_recall12','coarse_recall12',
                'refined_recall12','refined_recall4','union_recall12']})
    return out


def run(root, output, split, track, limit, config, full=False, allow_confirmation=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if split == 'confirmation' and not allow_confirmation:
        raise ValueError('Confirmation is held back. Freeze a configuration and explicitly use --allow-confirmation.')
    manifest = load_manifest(root)
    entries = [e for e in manifest['scenes'] if e['split']==split and e['track']==track]
    if limit:
        entries = entries[:limit]
    if not entries:
        raise ValueError('No scenes selected')
    if full and (track != 'rendered' or any(e['shape'] != [3000,3000] for e in entries)):
        raise ValueError('Full competition scoring requires rendered 3000x3000 scenes')
    spec = dict(build_id=manifest['build_id'], source=source_digest(), config=asdict(config),
                split=split, track=track, scenes=[e['id'] for e in entries], full=full)
    run_id = digest(spec)
    run_manifest = output/'manifest.json'
    if run_manifest.exists() and json.loads(run_manifest.read_text())['run_id'] != run_id:
        raise ValueError('Different configuration already uses this output directory')
    output.mkdir(parents=True, exist_ok=True)
    if split == 'confirmation':
        with (root/'confirmation_access.jsonl').open('a') as f:
            f.write(json.dumps(dict(run_id=run_id, time=time.time(), output=str(output), config=asdict(config)))+'\n')
    write_json(run_manifest, dict(run_id=run_id, spec=spec, status='running'))
    cv2.setNumThreads(config.threads)
    patterns = extract_patterns(root/'patterns') if full else None
    predictions, truths, allrows, branches, scene_metrics = {}, {}, [], [], {}
    start = time.perf_counter()
    for entry in entries:
        sid = entry['id']; path = output/f'{sid}.json'
        if path.exists():
            result = json.loads(path.read_text())
            if result['run_id'] != run_id:
                raise ValueError('Stale scene result')
        else:
            image, patches = input_images(root, entry)
            begin = time.perf_counter()
            proposals = get_proposals(root, entry, image, patches, root/'cache', config)
            coarse, refined = appearance(image, patches, proposals, config)
            pred = (finalize_joint(image, refined, coarse, patterns, config.threshold,
                                   tolerance=config.tolerance, gap=config.gap, top_k=config.top_k,
                                   cap=config.cap, quad_share=config.quad_share)
                    if full else local_prediction(refined, config.threshold))
            # Only after blind prediction do labels enter this runner.
            result = dict(run_id=run_id, prediction=asdict(pred), coarse=coarse, refined=refined,
                          proposals=proposals, seconds=time.perf_counter()-begin)
            write_json(path, result)
        if sha(root/entry['labels']) != entry['labels_sha256']:
            raise ValueError('Label hash mismatch')
        labels = json.loads((root/entry['labels']).read_text())
        pred = ScenePrediction(**result['prediction'])
        predictions[sid] = pred
        truths[sid] = ScenePrediction(labels['patches'], labels['constellation'] or 'unknown')
        rows = measurement_rows(entry, labels, pred, result['proposals'], result['coarse'], result['refined'])
        allrows.extend(rows)
        metric = evaluate({sid:pred}, {sid:truths[sid]})['scenes'][sid]
        if not full:
            metric = {k:metric[k] for k in ('presence','localization')}
        scene_metrics[sid] = metric
        b = pred.diagnostics.get('competing_geometry', [])
        if full:
            branches.append(dict(scene=sid, branches=b,
                agreement=len(b)==2 and b[0]['name']==b[1]['name'],
                both_available=len(b)==2, correct=pred.constellation==labels['constellation'],
                reference_nodes=labels['reference_nodes'], issued_unique=labels['issued_unique'],
                profile=labels['queries'][0]['profile']))
        print(f'evaluated {sid}: {metric}', flush=True)
    means = {k:float(np.mean([v[k] for v in scene_metrics.values()])) for k in next(iter(scene_metrics.values()))}
    strata = {}
    if full:
        for b in branches:
            for key in [f'nodes:{"2-3" if b["reference_nodes"]<4 else "4-6" if b["reference_nodes"]<7 else "7+"}',
                        f'issued:{"1-4" if b["issued_unique"]<5 else "5-8" if b["issued_unique"]<9 else "9+"}',
                        'profile:'+b['profile']]:
                strata.setdefault(key, []).append(scene_metrics[b['scene']])
        strata = {k:dict(n_scenes=len(v), mean={m:float(np.mean([x[m] for x in v])) for m in means}) for k,v in strata.items()}
    report = dict(run_id=run_id, spec=spec, scenes=scene_metrics, mean=means,
                  category_query_weighted=summarize(allrows), strata=strata, branches=branches,
                  worst={k:min(v[k] for v in scene_metrics.values()) for k in means},
                  seconds=time.perf_counter()-start,
                  peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Development evidence only; region and rendered-scene reuse is dependent. No query-level confidence intervals.',
                  prediction_seconds=sum(json.loads((output/f'{e["id"]}.json').read_text())['seconds'] for e in entries))
    write_json(output/'metrics.json', report); write_json(output/'queries.json', allrows)
    if source_digest() != spec['source']:
        raise RuntimeError('Code changed during run; results must be regenerated')
    write_json(run_manifest, dict(run_id=run_id, spec=spec, status='complete'))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path('outputs/imagebench/v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=SPLITS, default='development')
    p.add_argument('--track', choices=['real','rendered'], default='real')
    p.add_argument('--limit-scenes', type=int, default=0)
    p.add_argument('--full', action='store_true')
    p.add_argument('--allow-confirmation', action='store_true')
    p.add_argument('--verify-radius', choices=['fixed','adaptive'], default='fixed')
    p.add_argument('--verify-rep', choices=['blur','dog'], default='blur')
    p.add_argument('--alternatives', type=int, default=20)
    p.add_argument('--threshold', type=float, default=.72)
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    if args.limit_scenes < 0 or not 0<=args.threshold<=1 or args.alternatives<1 or args.threads<1:
        raise ValueError('Invalid evaluation parameters')
    config = Config(verify_radius=args.verify_radius, verify_rep=args.verify_rep,
                    alternatives=args.alternatives, threshold=args.threshold, threads=args.threads)
    run(args.dataset, args.output, args.split, args.track, args.limit_scenes, config,
        args.full, args.allow_confirmation)

if __name__ == '__main__':
    main()
