"""Execute Experiment 2 with frozen Exp1B checkpoints and C0 geometry."""
from __future__ import annotations

import json
from constellation.contracts import ScenePrediction, evaluate
from experiments.exp1.data import load_truth
from experiments.exp1.env import read_json, write_json
from . import OUT, ROOT, SCENES
from .candidate_geometry import apply_geometry, cached_scores, rerun_geometry
from .integration import hybrid_prediction

RULES = (
    ("learned_raw", "none", "none"),
    ("snap_relocated", "relocated", "none"),
    ("rescue_relocated", "none", "relocated"),
    ("snap_and_rescue_relocated", "relocated", "relocated"),
    ("snap_and_rescue_member", "member", "member"),
)


def prediction(doc: dict) -> ScenePrediction:
    return ScenePrediction(doc["patches"], doc["constellation"], doc.get("diagnostics", {}))


def load_inputs(seed: int = 31004):
    classical = {s: prediction(read_json(ROOT / "outputs/joint_train" / f"{s}.json"))
                 for s in SCENES}
    learned = {}
    sources = {}
    for fold in SCENES:
        d = read_json(ROOT / "outputs/exp1b/folds" / fold / "evaluation.json")
        arm = "SELECTED" if seed == 31004 else "SELECTED_s31005"
        if arm not in d["arms"]:
            raise FileNotFoundError(f"{fold}: missing {arm}")
        node = d["arms"][arm]
        learned[fold] = prediction(node["predictions"][fold])
        sources[fold] = {"arm": arm, "checkpoint_sha256": node.get("checkpoint_sha256"),
                         "threshold": node.get("threshold")}
    return classical, learned, sources


def evaluate_rules(seed: int = 31004) -> dict:
    classical, learned, sources = load_inputs(seed)
    truth = load_truth()
    result = {}
    for name, snap, rescue in RULES:
        preds = {s: (learned[s] if name == "learned_raw" else
                     hybrid_prediction(learned[s], classical[s], snap=snap, rescue=rescue))
                 for s in SCENES}
        result[name] = {"snap": snap, "rescue": rescue,
                        "metrics": evaluate(preds, truth),
                        "predictions": {s: {"patches": preds[s].patches,
                                             "constellation": preds[s].constellation,
                                             "diagnostics": preds[s].diagnostics}
                                        for s in SCENES}}
    return {"seed": seed, "sources": sources, "rules": result}


def evaluate_candidate_geometry(seed: int = 31004, device: str = "mps") -> dict:
    """Complete OOF candidate-level experiment; each sky uses its held-out fold model."""
    classical, learned, _ = load_inputs(seed)
    truth = load_truth()
    modes = {}
    for weight in (0., .05, .10, .20, .40):
        variants = {"snap": {}, "snap_rescue": {}, "snap_keep_name": {},
                    "snap_rescue_keep_name": {}}
        audit = {}
        for scene in SCENES:
            scores = cached_scores(scene, scene, seed, device)
            geom = rerun_geometry(scene, scores, weight)
            audit[scene] = {"name": geom["name"], "stage": geom["stage"],
                            "mapping": geom["mapping_audit"],
                            "competing": geom["competing"]}
            for rescue, keep, key in ((False, False, "snap"),
                                      (True, False, "snap_rescue"),
                                      (False, True, "snap_keep_name"),
                                      (True, True, "snap_rescue_keep_name")):
                variants[key][scene] = apply_geometry(
                    learned[scene], geom, rescue=rescue,
                    keep_classical_name=classical[scene].constellation if keep else None)
        for key, preds in variants.items():
            name = f"rank_weight_{weight:.2f}:{key}"
            modes[name] = {"rank_weight": weight, "mode": key,
                           "metrics": evaluate(preds, truth), "audit": audit,
                           "predictions": {s: {"patches": p.patches,
                                                "constellation": p.constellation,
                                                "diagnostics": p.diagnostics}
                                           for s, p in preds.items()}}
    return {"seed": seed, "modes": modes,
            "controls": {"pool_membership": "classical calibration",
                         "ambiguity": "classical calibration gap",
                         "seed": "classical candidate index 0",
                         "candidate_identity": "coordinate-mapped to frozen union bank",
                         "only_changed_geometry_signal": "learned within-query rank cost"}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    primary = evaluate_rules(31004)
    repeat = evaluate_rules(31005)
    candidate_primary = evaluate_candidate_geometry(31004)
    candidate_repeat = evaluate_candidate_geometry(31005)
    c0 = read_json(ROOT / "outputs/joint_train/metrics.json")
    selected = "snap_and_rescue_relocated"
    m = primary["rules"][selected]["metrics"]
    m2 = repeat["rules"][selected]["metrics"]
    gate = {
        "selected": selected,
        "gain_over_C0": m["mean"]["score"] - c0["mean"]["score"],
        "primary_above_C0": m["mean"]["score"] > c0["mean"]["score"],
        "repeat_above_C0": m2["mean"]["score"] > c0["mean"]["score"],
        "recovery_at_least_C0_minus_0.05": m["mean"]["recovery"] >= c0["mean"]["recovery"] - .05,
        "no_scene_regression_over_0.07": all(
            m["scenes"][s]["score"] >= c0["scenes"][s]["score"] - .07 for s in SCENES),
    }
    gate["pass"] = all(v for k, v in gate.items() if k not in {"selected", "gain_over_C0"})
    record = {"experiment": "learned scores integrated with frozen geometric recovery",
              "c0": c0, "primary": primary, "repeat": repeat, "gate": gate,
              "candidate_geometry_primary": candidate_primary,
              "candidate_geometry_repeat": candidate_repeat,
              "selection_note": ("Rule chosen from the causal primary-development comparison, "
                                  "then held fixed for the independent training-seed repeat: "
                                  "learned presence by default; only production-verified "
                                  "relocations may snap or rescue."),
              "limitations": ("Three repeatedly used labelled scenes are development evidence, "
                              "not an untouched generalization estimate.")}
    write_json(OUT / "record.json", record)
    from .report import write_report
    write_report(record)
    print(json.dumps({"selected": selected, "mean": m["mean"],
                      "repeat_mean": m2["mean"], "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
