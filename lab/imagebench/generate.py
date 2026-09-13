"""Deterministic, provenance-tracked real-crop and rendered-image benchmarks.

Run with python -m lab.imagebench.generate --help. Labels are separate from inputs.
No validation images, labels, or cached match results are used during generation.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from constellation.references import extract_patterns

SPLITS = ('development', 'calibration', 'confirmation')
VERSION = 2
PROFILES = {
    'mild': dict(scale=(.85, 1.18), blur=(.25, .65), gain=(.8, 1.2),
                 offset=(-5., 5.), drift=(0., 5.), noise=(.5, 2.), jpeg=(92, 100)),
    'nominal': dict(scale=(.75, 1.33), blur=(.4, 1.1), gain=(.65, 1.35),
                    offset=(-10., 10.), drift=(2., 12.), noise=(1., 4.), jpeg=(80, 98)),
    'stress': dict(scale=(.65, 1.5), blur=(.8, 1.7), gain=(.45, 1.6),
                   offset=(-15., 15.), drift=(5., 20.), noise=(3., 7.), jpeg=(65, 90)),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def rng_for(seed, *parts):
    return np.random.default_rng(int(digest([seed, *parts])[:16], 16))


def source_regions(images, guard=96):
    """3x3 cells per sky. Inset all uses; different splits never share pixels.

    Each sky contributes three regions to each split. Thus this is region-held-out,
    not scene-held-out. Processing is performed AFTER cropping to the recorded box.
    """
    regions = []
    for si, (source, image) in enumerate(sorted(images.items())):
        h, w = image.shape
        for row in range(3):
            for col in range(3):
                box = [col*w//3+guard, row*h//3+guard,
                       (col+1)*w//3-guard, (row+1)*h//3-guard]
                if min(box[2]-box[0], box[3]-box[1]) < 128:
                    raise ValueError('Source too small for buffered region partition')
                regions.append(dict(id=f'r{si:02}_{row}{col}', source=source, box=box,
                                    split=SPLITS[(row+col+si) % 3], guard=guard))
    return regions


def region_image(images, region):
    x0, y0, x1, y1 = region['box']
    return images[region['source']][y0:y1, x0:x1].copy()


def peaks(image, margin=52):
    a = image.astype(np.float32)
    dog = cv2.GaussianBlur(a, (0, 0), .8)-cv2.GaussianBlur(a, (0, 0), 2.5)
    threshold = max(1.5, float(np.percentile(dog, 96)))
    mask = (dog == cv2.dilate(dog, np.ones((5, 5), np.uint8))) & (dog > threshold)
    y, x = np.where(mask)
    keep = (x >= margin) & (x < image.shape[1]-margin) & (y >= margin) & (y < image.shape[0]-margin)
    return np.c_[x[keep], y[keep]].astype(float)


def draw_params(rng, profile):
    ranges = PROFILES[profile]
    p = {k: float(rng.uniform(*v)) for k, v in ranges.items() if k != 'jpeg'}
    p.update(angle=float(rng.uniform(0, 360)), drift_angle=float(rng.uniform(0, 2*np.pi)),
             jpeg=int(rng.integers(ranges['jpeg'][0], ranges['jpeg'][1]+1)),
             shot=float(rng.uniform(0., .15)))
    return p


def sample_maps(center, angle, scale, size=32):
    """Query pixel centres -> source coordinates. Midpoint 15.5 maps to centre."""
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    xx -= (size-1)/2; yy -= (size-1)/2
    a = np.deg2rad(angle); c, s = np.cos(a), np.sin(a)
    return ((center[0]+scale*(c*xx-s*yy)).astype(np.float32),
            (center[1]+scale*(s*xx+c*yy)).astype(np.float32))


def degrade(image, center, params, rng):
    """Generate from extra real context, then crop; never pad the query with fake pixels.

    Arbitrary subpixel source centres give sampling-phase variation without shifting
    the supervision target. No independent unrecorded translation is applied.
    """
    mx, my = sample_maps(center, params['angle'], params['scale'], size=48)
    if mx.min() < 1 or my.min() < 1 or mx.max() > image.shape[1]-2 or my.max() > image.shape[0]-2:
        raise ValueError('Degradation footprint leaves source; choose an interior centre')
    q = cv2.remap(image.astype(np.float32), mx, my, cv2.INTER_LINEAR)
    if params['blur'] > 0:
        q = cv2.GaussianBlur(q, (0, 0), params['blur'])
    q = q[8:40, 8:40]
    y, x = np.mgrid[-1:1:32j, -1:1:32j]
    plane = np.cos(params['drift_angle'])*x + np.sin(params['drift_angle'])*y
    q = q*params['gain'] + params['offset'] + params['drift']*plane
    sigma = np.sqrt(params['noise']**2 + params.get('shot', 0.)*np.maximum(q, 0))
    q += rng.normal(size=q.shape)*sigma
    q = np.uint8(np.clip(np.rint(q), 0, 255))
    if params['jpeg'] < 100:
        ok, encoded = cv2.imencode('.jpg', q, [cv2.IMWRITE_JPEG_QUALITY, params['jpeg']])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        q = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    a = np.deg2rad(params['angle']); s = params['scale']
    matrix = np.array([[s*np.cos(a), -s*np.sin(a)], [s*np.sin(a), s*np.cos(a)]])
    transform = np.c_[matrix, np.asarray(center)-matrix@np.array([15.5, 15.5])]
    return q, dict(params=params, patch_to_source=transform.tolist(),
                   footprint=[float(mx.min()), float(my.min()), float(mx.max()), float(my.max())])


def patch_stats(q):
    f = q.astype(np.float32)
    y, x = np.mgrid[:32, :32]; r = np.hypot(x-15.5, y-15.5)
    return dict(mean=float(f.mean()), std=float(f.std()), maximum=float(f.max()),
                saturated=float(np.mean(f >= 254)),
                centre_annulus=float(f[r < 5].mean()-f[(r > 8) & (r < 14)].mean()),
                highpass_std=float((f-cv2.GaussianBlur(f, (0, 0), 2.5)).std()))


def add_sources(image, points, rng, model=None):
    """Mixture of elliptical Gaussian and winged Moffat sources, all categories alike."""
    records = []
    for point in points:
        x, y = map(float, point)
        sigma = float(rng.uniform(*(model['renderer']['sigma'] if model else (.7, 2.6)))); ellipticity = float(rng.uniform(1., 1.5))
        amplitude = float(np.clip(rng.lognormal(np.log(model['renderer']['amplitude_median'] if model else 65), .8), *(model['renderer']['amplitude_bounds'] if model else (8, 240))))
        angle = float(rng.uniform(0, 2*np.pi)); winged = bool(rng.random() < .35)
        radius = int(np.ceil(sigma*ellipticity*7))
        x0, x1 = max(0, int(x)-radius), min(image.shape[1], int(x)+radius+1)
        y0, y1 = max(0, int(y)-radius), min(image.shape[0], int(y)+radius+1)
        yy, xx = np.mgrid[y0:y1, x0:x1]; xx = xx-x; yy = yy-y
        u = (np.cos(angle)*xx+np.sin(angle)*yy)/sigma
        v = (-np.sin(angle)*xx+np.cos(angle)*yy)/(sigma*ellipticity)
        r2 = u*u+v*v
        source = (1+r2/2)**-2.5 if winged else np.exp(-r2/2)
        if model:
            halo = rng.uniform(*model['renderer']['halo_fraction'])
            source = source + halo*np.exp(-r2/(2*model['renderer']['halo_scale']**2))
        image[y0:y1, x0:x1] += amplitude*source
        records.append(dict(x=x, y=y, sigma=sigma, axis_ratio=ellipticity,
                            amplitude=amplitude, family='moffat' if winged else 'gaussian'))
    return records


def background(tile, size, rng, model=None):
    # Crop before processing guarantees filters cannot borrow pixels across splits.
    base = cv2.resize(cv2.GaussianBlur(tile.astype(np.float32), (0, 0), 6),
                      (size, size), interpolation=cv2.INTER_LINEAR)
    base = base*rng.uniform(.5, 1.) + rng.uniform(0, 12)
    field = cv2.resize(rng.normal(0, 3, (12, 12)).astype(np.float32), (size, size),
                       interpolation=cv2.INTER_CUBIC)
    base += field
    if model:
        lo, mid, hi = np.percentile(base, [2, 50, 98])
        base = np.interp(base, [lo, mid, hi], model['background_quantiles']).astype(np.float32)
    # A seam and a short streak supplement the retained real background structure.
    seam = int(rng.integers(size//4, 3*size//4)); base[:, seam:] += rng.uniform(-8, 8)
    p1 = tuple(rng.integers(0, size, 2).tolist()); p2 = tuple(rng.integers(0, size, 2).tolist())
    cv2.line(base, p1, p2, float(np.median(base)+rng.uniform(8, 35)), 1)
    return base


def place_template(template, size, rng, schematic_sigma):
    p = (template-template.mean(0))/np.maximum(np.ptp(template, axis=0), 1)
    a = rng.uniform(0, 2*np.pi)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    affine = np.array([[rng.uniform(.8, 1.25), rng.uniform(-.12, .12)],
                       [0, rng.choice([-1., 1.])]])@rot*size*rng.uniform(.32, .55)
    points = p@affine
    factor = min(1., (size-200)/max(np.ptp(points, axis=0)))
    affine *= factor; points = p@affine
    low, high = points.min(0), points.max(0)
    shift = np.array([rng.uniform(75-low[k], size-75-high[k]) for k in (0, 1)])
    ideal = points+shift
    actual = ideal+rng.normal(0, schematic_sigma, ideal.shape)
    actual = np.clip(actual, 55, size-55)
    return actual, dict(affine=affine.tolist(), translation=shift.tolist(),
                        ideal_nodes=ideal.tolist(), schematic_sigma=schematic_sigma,
                        actual_nodes=actual.tolist())


@dataclass
class BuildConfig:
    seed: int = 610120
    guard: int = 96
    rendered_per_split: int = 48
    real_queries: int = 24
    size: int = 3000
    source_model: str | None = None


def real_scene(images, region, donor, config, rng, model=None):
    image = region_image(images, region); other = region_image(images, donor)
    pts, dp = peaks(image), peaks(other)
    if not len(pts) or not len(dp):
        raise ValueError('No interior peaks in region')
    queries, truth, records = [], [], []
    n_present = round(config.real_queries*.67)
    # Repeated query views are explicitly tracked and never straddle splits.
    if model:
        from .realism import choose_sources
        chosen = choose_sources(image, pts, n_present-1, rng, model)
        absent_ids = choose_sources(other, dp, config.real_queries-n_present, rng, model)
    else:
        chosen = rng.choice(len(pts), n_present-1, replace=len(pts) < n_present-1).tolist()
        absent_ids = rng.integers(len(dp), size=config.real_queries-n_present).tolist()
    chosen.append(chosen[0])
    for i in range(config.real_queries):
        present = i < n_present
        idx = chosen[i] if present else absent_ids[i-n_present]
        centre = (pts if present else dp)[idx].copy()
        # Fixed phase per physical source. Repeated views keep identical supervision.
        phase = rng_for(config.seed, region['id'] if present else donor['id'], idx).uniform(-.35, .35, 2)
        centre += phase
        profile = tuple(PROFILES)[i % 3]
        patch, meta = degrade(image if present else other, centre, draw_params(rng, profile), rng)
        queries.append(patch); truth.append([*centre.tolist(), 0] if present else None)
        records.append(dict(category='present' if present else 'absent', profile=profile,
                            source_region=region['id'] if present else donor['id'],
                            source_centre=centre.tolist(), physical_id=f'{region["id"] if present else donor["id"]}:{idx}',
                            **meta))
    order = rng.permutation(len(queries))
    return image, [queries[i] for i in order], dict(
        patches=[truth[i] for i in order], constellation=None,
        queries=[records[i] for i in order], target_region=region['id'], donor_region=donor['id'],
        note='Real-crop track scores presence/localization only; membership and class unknown.')


def rendered_scene(images, region, patterns, name, size, rng, profile, model=None):
    tile = region_image(images, region)
    image = background(tile, size, rng, model)
    sigma = float(rng.choice([2., 4.1, 8.]))  # plausible regimes, NOT a fitted distribution
    figure, placement = place_template(patterns[name], size, rng, sigma)
    n_field = max(100, int(size*size*rng.uniform(.00045, .0009)))
    field = rng.uniform(55, size-55, (n_field, 2))
    # Cluster some field stars; distribution is unrelated to query membership.
    count = n_field//5
    field[:count] = np.clip(rng.normal(rng.uniform(.25*size, .75*size, 2), size*.09,
                                      (count, 2)), 55, size-55)
    other_name = rng.choice([n for n in sorted(patterns) if n != name])
    other, _ = place_template(patterns[other_name], size, rng, sigma)
    fragment = other[rng.choice(len(other), min(3, len(other)), replace=False)]
    off = np.vstack([field, fragment])
    close = figure[0]+np.array([rng.uniform(3, 8), rng.uniform(-2, 2)])
    off = np.vstack([off, close])
    # Same rendering law for all sources; no brightness cue for figure membership.
    source_records = add_sources(image, np.vstack([figure, off]), rng, model)
    image += rng.normal(0, rng.uniform(.7, 2.5), image.shape).astype(np.float32)
    image = np.uint8(np.clip(np.rint(image), 0, 255))
    # Absent queries have a separately generated donor star field.
    donor = background(np.flipud(tile).copy(), size, rng, model)
    donor_points = rng.uniform(55, size-55, (max(100, n_field//2), 2))
    add_sources(donor, donor_points, rng, model)
    donor += rng.normal(0, 1.5, donor.shape).astype(np.float32)
    donor = np.uint8(np.clip(np.rint(donor), 0, 255))
    fraction = float(rng.choice([.4, .6, .8]))
    n_issue = max(1, min(len(figure)-1, round(len(figure)*fraction)))
    ids = rng.choice(len(figure), n_issue, replace=False).tolist()
    if 0 not in ids:
        ids[0] = 0  # ensures queried close-pair anchor; indices have no inference meaning
    items = [(figure[i], 'figure', f'figure:{i}', image) for i in ids]
    off_ids = rng.choice(len(field), int(rng.integers(8, 15)), replace=False).tolist()
    off_ids += list(range(len(field), len(off)))
    items += [(off[i], 'off-figure', f'off:{i}', image) for i in off_ids]
    items.append(items[0])  # independently degraded repeated view
    items += [(donor_points[i], 'absent', f'donor:{i}', donor)
              for i in rng.choice(len(donor_points), int(rng.integers(8, 15)), replace=False)]
    queries, truth, records = [], [], []
    for xy, category, physical_id, source in items:
        patch, meta = degrade(source, xy, draw_params(rng, profile), rng)
        queries.append(patch)
        truth.append([*map(float, xy), int(category == 'figure')] if category != 'absent' else None)
        records.append(dict(category=category, profile=profile, physical_id=physical_id,
                            source_centre=list(map(float, xy)), **meta))
    order = rng.permutation(len(queries))
    return image, [queries[i] for i in order], dict(
        patches=[truth[i] for i in order], constellation=name,
        queries=[records[i] for i in order], target_region=region['id'], placement=placement,
        reference_nodes=len(figure), issued_unique=n_issue, fragment_class=other_name,
        fragment_nodes=fragment.tolist(), close_pair=[figure[0].tolist(), close.tolist()],
        sources=source_records, field_sources=n_field, issue_fraction=fraction,
        note='Rendered injection truth; smoothed real background may retain other unlabelled sources.')


def save_scene(root, entry, image, patches, labels):
    d = root/'inputs'/entry['id']; d.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, pixels in [('image.png', image)]+[(f'patch_{i+1:02}.png', q) for i, q in enumerate(patches)]:
        p = d/filename
        if not cv2.imwrite(str(p), pixels):
            raise RuntimeError(f'Cannot write {p}')
        paths.append(dict(path=str(p.relative_to(root)), sha256=sha(p)))
    label_path = root/'labels'/f'{entry["id"]}.json'
    write_json(label_path, labels)
    entry.update(files=paths, labels=str(label_path.relative_to(root)), labels_sha256=sha(label_path),
                 shape=list(image.shape), n_queries=len(patches))
    return entry


def build(data, output, config):
    cv2.setNumThreads(1)
    data, output = Path(data).resolve(), Path(output).resolve()
    if not (data/'train').exists() and (data/'participant/train').exists():
        data = data/'participant'
    files = sorted((data/'train').glob('*/*_image.png'))
    if not files:
        raise ValueError('No training skies found')
    images = {str(p.relative_to(data)): cv2.imread(str(p), 0) for p in files}
    if any(x is None for x in images.values()):
        raise ValueError('Unreadable source image')
    patterns = extract_patterns(data/'patterns')
    if len(patterns) != 48:
        raise ValueError('Expected all 48 supplied references')
    sources = {str(p.relative_to(data)): sha(p) for p in files}
    model = json.loads(Path(config.source_model).read_text()) if config.source_model else None
    if model and (model['fit_split'] != 'development' or model['source_sha256'] != sources):
        raise ValueError('Source model must be fitted on development regions of these source images')
    pattern_hashes = {p.name: sha(p) for p in sorted((data/'patterns').glob('*_pattern.png'))}
    spec = dict(version=VERSION, config=asdict(config), sources=sources, patterns=pattern_hashes,
                generator_sha256=sha(__file__), source_model=model,
                realism_sha256=sha(Path(__file__).with_name('realism.py')), numpy=np.__version__, opencv=cv2.__version__)
    key = digest(spec)
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing['build_id'] != key:
            raise ValueError('Output contains a different build; choose a new output directory')
        manifest = existing
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Output must be empty or contain a compatible manifest')
        output.mkdir(parents=True, exist_ok=True)
        (output/'patterns').mkdir(exist_ok=True)
        for p in sorted((data/'patterns').glob('*_pattern.png')):
            (output/'patterns'/p.name).write_bytes(p.read_bytes())
        manifest = dict(build_id=key, spec=spec, python=platform.python_version(),
                        regions=source_regions(images, config.guard), scenes=[], status='building',
                        scope='Region-held-out real imagery; not independent real scenes. Synthetic labels are known by construction.')
        write_json(manifest_path, manifest)
    if config.size < 512 or config.real_queries < 6 or config.rendered_per_split < 1:
        raise ValueError('Require size>=512, real_queries>=6, rendered_per_split>=1')
    existing = {e['id']: e for e in manifest['scenes']}
    start = time.perf_counter()
    for split_index, split in enumerate(SPLITS):
        regions = [r for r in manifest['regions'] if r['split'] == split]
        jobs = [('real', i) for i in range(len(regions))]+[('rendered', i) for i in range(config.rendered_per_split)]
        for track, index in jobs:
            sid = f'{track}_{split}_{index:04}'
            if sid in existing:
                e = existing[sid]
                if all((output/f['path']).exists() and sha(output/f['path']) == f['sha256'] for f in e['files']) and sha(output/e['labels']) == e['labels_sha256']:
                    continue
                raise ValueError(f'Existing scene corrupted: {sid}; use a fresh output')
            rng = rng_for(config.seed, track, split, index)
            region = regions[index % len(regions)]
            if track == 'real':
                # Cross-sky donor from same split, never a crop in the target input.
                donor = next(r for r in regions if r['source'] != region['source'])
                image, patches, labels = real_scene(images, region, donor, config, rng, model)
            else:
                name = sorted(patterns)[index % len(patterns)]
                profile = tuple(PROFILES)[(index//len(patterns)+index+split_index) % 3]
                image, patches, labels = rendered_scene(images, region, patterns, name, config.size, rng, profile, model)
            entry = dict(id=sid, split=split, track=track, index=index)
            manifest['scenes'].append(save_scene(output, entry, image, patches, labels))
            write_json(manifest_path, manifest)
            print(f'built {sid} ({len(patches)} queries)', flush=True)
    manifest.update(status='complete', generation_seconds=time.perf_counter()-start)
    write_json(manifest_path, manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('.'))
    p.add_argument('--output', type=Path, default=Path('outputs/imagebench/v1'))
    p.add_argument('--seed', type=int, default=610120)
    p.add_argument('--source-model', type=str)
    p.add_argument('--rendered-per-split', type=int, default=48)
    p.add_argument('--real-queries', type=int, default=24)
    p.add_argument('--size', type=int, default=3000)
    args = p.parse_args()
    build(args.data, args.output, BuildConfig(seed=args.seed, rendered_per_split=args.rendered_per_split,
                                            real_queries=args.real_queries, size=args.size, source_model=args.source_model))

if __name__ == '__main__':
    main()
