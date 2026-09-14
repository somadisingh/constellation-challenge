"""Final completeness, pairing, and protected-artifact audit."""
from pathlib import Path

from experiments.exp1 import SCENES
from experiments.exp1.env import read_json, sha256_file, write_json

from .data import OUT


def run():
    metrics = read_json(OUT / "metrics.json")
    required = ["C1", "A", "B", "C", "D", "E", "SELECTED"]
    complete = {arm: bool(metrics["oof"].get(arm, {}).get("complete"))
                for arm in required}
    pairing = {}
    checkpoints = {}
    for fold in SCENES:
        d = read_json(OUT / "runs" / fold / "D_s31004" / "result.json")
        e = read_json(OUT / "runs" / fold / "E_s31004" / "result.json")
        pairing[fold] = {
            "D_anchor_digest": d["counts"]["anchor_digest"],
            "E_anchor_digest": e["counts"]["anchor_digest"],
            "identical": d["counts"]["anchor_digest"] == e["counts"]["anchor_digest"],
            "D_presentations": d["counts"]["presentations"],
            "E_presentations": e["counts"]["presentations"],
        }
        checkpoints[fold] = {}
        for arm in "ABCDE":
            root = OUT / "runs" / fold / f"{arm}_s31004"
            checkpoints[fold][arm] = {
                str(step): (root / f"step_{step}.pt").exists()
                for step in (500, 1000, 1500, 2000)
            }

    before = read_json(OUT / "protected_before.json")
    current = {name: sha256_file(name) for name in before if Path(name).exists()}
    missing = sorted(set(before) - set(current))
    changed = sorted(name for name, digest in current.items()
                     if digest != before[name])
    audit = {
        "complete_required_arms": complete,
        "all_required_arms_complete": all(complete.values()),
        "paired_D_E": pairing,
        "all_D_E_anchors_identical": all(x["identical"] for x in pairing.values()),
        "common_step_checkpoints": checkpoints,
        "all_common_step_checkpoints_present": all(
            present for fold in checkpoints.values() for arm in fold.values()
            for present in arm.values()),
        "protected_files_checked": len(before),
        "protected_missing": missing,
        "protected_changed": changed,
        "production_and_exp1_unchanged": not missing and not changed,
    }
    write_json(OUT / "integrity.json", audit)
    if not (audit["all_required_arms_complete"] and
            audit["all_D_E_anchors_identical"] and
            audit["all_common_step_checkpoints_present"] and
            audit["production_and_exp1_unchanged"]):
        raise RuntimeError(audit)
    return audit


if __name__ == "__main__":
    print(run())
