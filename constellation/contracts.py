from dataclasses import dataclass, field
import ast
import csv
import numpy as np

@dataclass
class ScenePrediction:
    patches: list
    constellation: str = 'unknown'
    diagnostics: dict = field(default_factory=dict)

def parse_cell(value):
    if str(value).strip() == '-1':
        return None
    p = ast.literal_eval(value)
    if len(p) != 3 or p[2] not in (0, 1) or not np.isfinite(p).all():
        raise ValueError(f'Invalid patch: {value}')
    return tuple(p)

def read_truth(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return {r['Id']: ScenePrediction([parse_cell(r[f'patch_{i:02}']) for i in range(1, int(r['n_patches'])+1)], r['constellation']) for r in rows}

def reward(distance):
    return np.clip((36 - np.asarray(distance)) / 24, 0, 1)

def evaluate(predictions, ground_truth):
    results = {}
    for name, truth in ground_truth.items():
        pred = predictions[name]
        if len(pred.patches) != len(truth.patches):
            raise ValueError('Query count mismatch')
        y = np.array([p is not None for p in truth.patches])
        z = np.array([p is not None for p in pred.patches])
        f1 = []
        for cls in (False, True):
            tp = np.sum((y == cls) & (z == cls))
            den = np.sum(y == cls) + np.sum(z == cls)
            f1.append(2*tp/den if den else 1.0)
        loc = [float(reward(np.linalg.norm(np.array(t[:2])-p[:2]))) if p is not None else 0.0 for p,t in zip(pred.patches, truth.patches) if t is not None]
        figure = np.array([t[:2] for t in truth.patches if t is not None and t[2] == 1]).reshape(-1,2)
        points = np.array([p[:2] for p in pred.patches if p is not None]).reshape(-1,2)
        total = 0.
        if len(figure) and len(points):
            d = np.linalg.norm(figure[:,None,:]-points[None,:,:], axis=2)
            used_f, used_p = set(), set()
            for flat in np.argsort(d, axis=None, kind='stable'):
                i,j = np.unravel_index(flat, d.shape)
                if i not in used_f and j not in used_p:
                    total += float(reward(d[i,j]))
                    used_f.add(i); used_p.add(j)
        m = dict(presence=float(np.mean(f1)), localization=float(np.mean(loc)) if loc else 1., recovery=total/len(figure) if len(figure) else 1., identification=float(pred.constellation==truth.constellation))
        m['score'] = sum(m[k]*w for k,w in [('presence',.25),('localization',.2),('recovery',.25),('identification',.3)])
        results[name] = m
    return {'scenes':results, 'mean':{k:float(np.mean([r[k] for r in results.values()])) for k in next(iter(results.values()))}, 'worst_score':min(r['score'] for r in results.values()), 'conventions':'Unofficial: empty class F1 and empty localization/recovery = 1; greedy distance ties use stable row-major order.'}

def write_submission(predictions, sample_submission, output_path):
    with open(sample_submission, newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
    if set(predictions) != {r['Id'] for r in rows}:
        raise ValueError('Scene set mismatch')
    for row in rows:
        p = predictions[row['Id']]
        if len(p.patches) != int(row['n_patches']):
            raise ValueError('Query count mismatch')
        for col in fields:
            if col.startswith('patch_'):
                idx = int(col.split('_')[1])-1
                point = p.patches[idx] if idx < len(p.patches) else None
                row[col] = '-1' if point is None else str((round(float(point[0]),3),round(float(point[1]),3),int(point[2])))
        row['constellation'] = p.constellation
    with open(output_path,'w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
