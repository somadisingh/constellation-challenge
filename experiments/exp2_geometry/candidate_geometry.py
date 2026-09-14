"""Candidate-level learned evidence in the production geometric recognizer."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from constellation.contracts import ScenePrediction
from constellation.finalize import COARSE_CUTOFF, auxiliary_map
from constellation.joint import recognize_joint
from constellation.references import extract_patterns
from experiments.exp1.evaluation import _aligned_set, load_arm, load_real_aligned
from experiments.exp1.data import load_scene
from experiments.exp1.scoring import score_descriptor
from experiments.exp1.stages import Paths
from experiments.exp1.env import read_json, write_json
from . import OUT, ROOT
from .integration import learned_slate


def score_union_bank(scene: str, checkpoint: Path, device: str = "mps") -> list[list[float]]:
    """Score the frozen union bank with one frozen learned checkpoint."""
    model, _ = load_arm("hardnet", "trained", device, checkpoint)
    node = load_real_aligned(Paths(ROOT / "outputs/exp1"), scene)
    return [score_descriptor(model, _aligned_set(node, i), device).tolist()
            for i in range(len(node["counts"]))]


def cached_scores(fold: str, scene: str, seed: int, device: str = "mps") -> list:
    path = OUT / "candidate_scores" / f"seed_{seed}" / f"{fold}_{scene}.json"
    if path.exists():
        return read_json(path)["scores"]
    checkpoint = ROOT / "outputs/exp1b/runs" / fold / f"B_s{seed}" / "best.pt"
    scores = score_union_bank(scene, checkpoint, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"fold": fold, "scene": scene, "seed": seed,
                      "checkpoint": str(checkpoint.relative_to(ROOT)), "scores": scores})
    return scores


def rerun_geometry(scene: str, score_rows: list, rank_weight: float) -> dict:
    """Rerun both C0 branches with fixed calibration, membership and seed anchors."""
    doc = read_json(ROOT / "outputs/joint_train" / f"{scene}.json")
    union = load_real_aligned(Paths(ROOT / "outputs/exp1"), scene)
    patterns = extract_patterns(ROOT / "patterns")
    image = load_scene(scene).image
    aux = auxiliary_map(image)
    competing, audits = [], []
    for stage, field, cutoff in (("refined", "candidates", .72),
                                 ("coarse", "coarse_candidates", COARSE_CUTOFF)):
        ids, slates = [], []
        for i, q in enumerate(doc["diagnostics"]["queries"]):
            candidates = q[field]
            if not candidates or candidates[0][2] < cutoff:
                continue
            k = int(union["counts"][i])
            slate, audit = learned_slate(candidates, union["xy"][i, :k], score_rows[i])
            ids.append(i); slates.append(slate)
            audits.append({"stage": stage, "query": i, **audit})
        name, chosen, diag = recognize_joint(
            slates, patterns, seed=6643, tolerance=18., gap=.03, top_k=8,
            cap=80000, quad_share=0., aux_weight=3., models=("affine",),
            shear_penalty=2., auxiliary_map=aux, pool_by="calib",
            rank_weight=rank_weight, diag_top=48)
        best = diag["hypotheses"][0] if diag.get("hypotheses") else {"score": -1e9}
        competing.append({"stage": stage, "name": name,
                          "score": float(best.get("score", -1e9)),
                          "chosen": {ids[int(k)]: v for k, v in chosen.items()},
                          "diagnostics": diag})
    competing.sort(key=lambda x: (-x["score"], x["name"], x["stage"]))
    winner = competing[0]
    return {"name": winner["name"], "chosen": winner["chosen"],
            "rank_weight": float(rank_weight),
            "stage": winner["stage"], "geometry": winner["diagnostics"],
            "competing": [{k: x[k] for k in ("stage", "name", "score")}
                          for x in competing],
            "mapping_audit": {"n": sum(a["mapped"] for a in audits),
                              "max_distance": max((a["max_distance"] for a in audits), default=0.)}}


def apply_geometry(learned: ScenePrediction, result: dict, *, rescue: bool,
                   keep_classical_name: str | None = None) -> ScenePrediction:
    """Apply learned-scored correspondences to learned presence decisions."""
    chosen = {int(k): v for k, v in result["chosen"].items()}
    hyps = result["geometry"].get("hypotheses") or []
    nodes = np.asarray(hyps[0].get("nodes", []), float).reshape(-1, 2) if hyps else np.empty((0, 2))
    patches, actions = [], []
    for i, lp in enumerate(learned.patches):
        if i in chosen and (lp is not None or rescue):
            x, y = map(float, chosen[i])
            member = int(bool(len(nodes)) and np.linalg.norm(nodes - [x, y], axis=1).min() < 18.)
            patches.append((x, y, member)); actions.append("geometry_rescue" if lp is None else "geometry_snap")
        else:
            patches.append(lp); actions.append("learned")
    return ScenePrediction(patches, keep_classical_name or result["name"], {
        "integration": "candidate_level_learned_geometry",
        "rank_weight": result["rank_weight"],
        "geometry_stage": result["stage"], "actions": actions,
        "candidate_mapping": result["mapping_audit"]})
