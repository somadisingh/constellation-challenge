from __future__ import annotations
import math
import numpy as np
from .assignment import unique_assignment
from .correspondences import correspondence_pool
from .transforms import apply_transform, fit_transform, transform_diagnostics

MINIMAL={"similarity":2,"reflected_similarity":2,"anisotropic":3,"affine":3}

def required_trials(p,w,s,maximum=20000):
    if not 0<p<1 or s<1: raise ValueError("invalid p or sample size")
    if w<=0: return maximum
    if w>=1: return 1
    den=math.log1p(-(w**s))
    return maximum if not np.isfinite(den) or den==0 else max(1,min(maximum,int(math.ceil(math.log1p(-p)/den))))

def _sample_valid(pool,candidates,size,rng,n_prefix):
    for _ in range(60):
        ids=rng.choice(min(n_prefix,len(pool)),size,replace=False); rows=[pool[int(i)] for i in ids]
        cs=[candidates[x[1]] for x in rows]
        if len({x[0] for x in rows})==size and len({c.query_id for c in cs})==size and len({c.physical_star_cluster_id for c in cs})==size:
            return rows
    return None

def run_ransac(nodes,candidates,family="similarity",method="prosac",seed=0,
               discovery_radius=18.,final_radius=12.,max_trials=1200,refit_iterations=3,
               duplicate_safe=True,refit=True):
    nodes=np.asarray(nodes,float); size=MINIMAL[family]; rng=np.random.default_rng(seed)
    pool=correspondence_pool(nodes,candidates); best=None; degenerate=duplicate_rejected=0
    adaptive=max_trials
    for trial in range(max_trials):
        if trial>=adaptive: break
        prefix=len(pool) if method=="uniform" else min(len(pool),max(size+2,20+trial//4))
        rows=_sample_valid(pool,candidates,size,rng,prefix)
        if rows is None: duplicate_rejected+=1; continue
        ni=[x[0] for x in rows]; ci=[x[1] for x in rows]
        m=fit_transform(nodes[ni],[candidates[i].coordinate for i in ci],family)
        if m is None: degenerate+=1; continue
        diag=transform_diagnostics(m)
        if family=="affine" and (diag["condition_number"]>8 or diag["shear"]>.8): continue
        mapped=apply_transform(nodes,m); disc=unique_assignment(mapped,candidates,discovery_radius,duplicate_safe)
        key=(disc["unique_assignment_inlier_count"],-np.median(disc["residuals"]) if disc["residuals"] else -1e9)
        if best is None or key>best[0]:
            best=(key,m,disc,set(ni)); w=disc["unique_assignment_inlier_count"]/max(len(pool),1)
            adaptive=min(adaptive,max(trial+1,required_trials(.99,w,size,max_trials)))
    if best is None: return {"status":"no_hypothesis","family":family,"trials":min(max_trials,adaptive)}
    _,m,disc,seeds=best; pre=disc["unique_assignment_inlier_count"]
    collapse=False
    if refit:
        for _ in range(refit_iterations):
            pairs=disc["pairs"]
            if len(pairs)<size: collapse=True; break
            new=fit_transform(nodes[[a for a,_ in pairs]],[candidates[b].coordinate for _,b in pairs],family)
            if new is None: collapse=True; break
            nd=unique_assignment(apply_transform(nodes,new),candidates,discovery_radius,duplicate_safe)
            if nd["pairs"]==pairs: m=new; disc=nd; break
            m=new; disc=nd
    final=unique_assignment(apply_transform(nodes,m),candidates,final_radius,duplicate_safe)
    residual=np.asarray(final["residuals"],float); matched={a for a,_ in final["pairs"]}
    return {"status":"ok","family":family,"matrix":np.asarray(m).tolist(),"trials":trial+1,
      "raw_inliers":final["raw_inlier_count"],"unique_inliers":final["unique_assignment_inlier_count"],
      "held_out_inliers":sum(a not in seeds for a in matched),"support_fraction":len(matched)/max(len(nodes),1),
      "median_residual":float(np.median(residual)) if len(residual) else None,
      "p90_residual":float(np.quantile(residual,.9)) if len(residual) else None,
      "max_residual":float(residual.max()) if len(residual) else None,"pairs":final["pairs"],
      "pre_refit_support":pre,"post_refit_support":final["unique_assignment_inlier_count"],
      "support_collapsed":collapse,"degenerate_samples":degenerate,"duplicate_rejected_samples":duplicate_rejected,
      "diagnostics":transform_diagnostics(m),"mapped_nodes":apply_transform(nodes,m).tolist()}
