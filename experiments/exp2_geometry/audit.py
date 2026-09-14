"""Integrity checks scoped to Experiment 2's intentional documentation changes."""
import hashlib
import json

from experiments.exp1.env import read_json, sha256_file, write_json
from . import OUT, ROOT

EXPECTED_PROTECTED_CHANGES = {"README.md", "FINDINGS.md"}


def run():
    baseline = read_json(ROOT / "outputs/exp1b/protected_before.json")
    current = {name: sha256_file(ROOT / name) for name in baseline
               if (ROOT / name).exists()}
    missing = sorted(set(baseline) - set(current))
    changed = {name for name, digest in current.items() if digest != baseline[name]}
    unexpected = sorted(changed - EXPECTED_PROTECTED_CHANGES)
    record = read_json(OUT / "record.json")
    mapping = [node["mapping"]["max_distance"]
               for mode in record["candidate_geometry_primary"]["modes"].values()
               for node in mode["audit"].values()]
    prediction_hashes = {}
    for name, node in record["candidate_geometry_primary"]["modes"].items():
        mode = name.split(":", 1)[1]
        scored_outputs = {scene: {"patches": pred["patches"],
                                  "constellation": pred["constellation"]}
                          for scene, pred in node["predictions"].items()}
        digest = hashlib.sha256(json.dumps(
            scored_outputs, sort_keys=True).encode()).hexdigest()
        prediction_hashes.setdefault(mode, set()).add(digest)
    audit = {
        "protected_files": len(baseline),
        "expected_documentation_changes": sorted(EXPECTED_PROTECTED_CHANGES),
        "actual_protected_changes": sorted(changed),
        "unexpected_protected_changes": unexpected,
        "protected_missing": missing,
        "computational_protected_files_unchanged": not missing and not unexpected,
        "exact_candidate_mapping": max(mapping, default=0.) == 0.,
        "candidate_mapping_max_distance": max(mapping, default=0.),
        "primary_rank_weights_prediction_inert": all(
            len(v) == 1 for v in prediction_hashes.values()),
        "source_hashes": {str(p.relative_to(ROOT)): sha256_file(p)
                          for p in sorted((ROOT / "experiments/exp2_geometry").glob("*.py"))},
    }
    audit["pass"] = (audit["computational_protected_files_unchanged"] and
                     audit["exact_candidate_mapping"] and
                     audit["primary_rank_weights_prediction_inert"])
    write_json(OUT / "integrity.json", audit)
    if not audit["pass"]:
        raise RuntimeError(audit)
    return audit


if __name__ == "__main__":
    print(run())
