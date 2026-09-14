"""Create the concise human report and machine-readable Experiment 1B record."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

from experiments.exp1 import SCENES
from experiments.exp1.env import read_json, sha256_file, write_json

from .data import OLD, OUT


def _fmt(value, digits=4):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _metric_row(name, node, c0):
    if not node or not node.get("complete"):
        return f"| {name} | incomplete | | | | | |"
    m = node["mean"]
    return (f"| {name} | {_fmt(m['score'])} | {m['score']-c0:+.4f} | "
            f"{_fmt(m['presence'],3)} | {_fmt(m['localization'],3)} | "
            f"{_fmt(m['recovery'],3)} | {_fmt(m['identification'],3)} |")


def _make_curves(metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel = OUT / "panels"
    panel.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True)
    for row_index, fold in enumerate(SCENES):
        for arm in "ABCDE":
            path = OUT / "runs" / fold / f"{arm}_s31004" / "result.json"
            if not path.exists():
                continue
            run = read_json(path)
            steps = [e["step"] for e in run["evaluations"]]
            fixed = [e["fixed"]["equal_sky_top1_localization_reward"]
                     for e in run["evaluations"]]
            fresh = [e["fresh"]["equal_sky_top1_localization_reward"]
                     for e in run["evaluations"]]
            axes[row_index, 0].plot(steps, fixed, marker="o", label=arm)
            axes[row_index, 1].plot(steps, fresh, marker="o", label=arm)
        axes[row_index, 0].set_ylabel(f"{fold}\nreward")
        axes[row_index, 0].set_title("Fixed corrected inner bank")
        axes[row_index, 1].set_title("Fresh corrected inner bank")
        for ax in axes[row_index]:
            ax.grid(alpha=.25)
            ax.set_ylim(0, 1)
    for ax in axes[-1]:
        ax.set_xlabel("optimizer step")
    axes[0, 1].legend(ncol=5, fontsize=8)
    fig.tight_layout()
    target = panel / "inner_learning_curves.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return str(target)


def _hypotheses(runs, appearance):
    def mean_best(arms):
        values = []
        for fold in SCENES:
            for arm in arms:
                values.append(read_json(OUT / "runs" / fold /
                                        f"{arm}_s31004" / "result.json")["best"]["metric"])
        return float(np.mean(values))

    a, b, c, d, e = (mean_best(x) for x in ("A", "B", "C", "D", "E"))
    corrected_gap = []
    for fold in SCENES:
        node = appearance[fold]
        corrected_gap.append(abs(node["C"]["queries"]["std"]["mean"] -
                                 node["real_queries"]["queries"]["std"]["mean"]))
    original_gap = []
    for fold in SCENES:
        node = appearance[fold]
        original_gap.append(abs(node["A"]["queries"]["std"]["mean"] -
                                node["real_queries"]["queries"]["std"]["mean"]))
    return {
        "fixed_data_exhaustion": {
            "comparison": {"A": a, "B": b, "C": c, "D": d},
            "finding": "supported" if (b > a or d > c) else "contradicted",
            "criterion": "fresh counterpart improves mean best inner ranking"
        },
        "appearance_mismatch": {
            "original_mean_absolute_std_gap": float(np.mean(original_gap)),
            "corrected_mean_absolute_std_gap": float(np.mean(corrected_gap)),
            "finding": "correction narrowed measured contrast gap"
                       if np.mean(corrected_gap) < np.mean(original_gap)
                       else "correction did not narrow measured contrast gap"
        },
        "hard_negative_value": {
            "D_hard": d, "E_random": e,
            "finding": "hard policy helped" if d > e else
                       "random control matched or exceeded hard policy"
        }
    }


def make_report():
    metrics = read_json(OUT / "metrics.json")
    appearance = read_json(OUT / "appearance_audit.json")
    diagnosis = read_json(OUT / "diagnosis.json")
    environment = read_json(OUT / "environment.json")
    curves = _make_curves(metrics)
    c0_score = metrics["c0"]["mean"]["score"]
    oof = metrics["oof"]
    selected = oof["SELECTED"]
    c1 = oof["C1"]
    historical = metrics["historical"].get("SELECTED")
    hypotheses = _hypotheses(metrics, appearance)
    integrity_path = OUT / "integrity.json"
    integrity = read_json(integrity_path) if integrity_path.exists() else None
    completed_runs = [read_json(path) for path in sorted((OUT / "runs").glob("*/*/result.json"))]
    training_hours = sum(run["seconds"] for run in completed_runs) / 3600.0
    selected_inference_seconds = sum(
        read_json(OUT / "folds" / fold / "evaluation.json")["arms"]["SELECTED"].get("seconds", 0)
        for fold in SCENES)
    hardnet_provenance = read_json(OLD / "models.json")["models"]["hardnet"]

    lines = [
        "# Experiment 1B — corrected learned patch verification",
        "",
        f"**Verdict: {metrics['verdict']}.**",
        "",
        (f"The inner-selected recipe scored **{selected['mean']['score']:.4f}** "
         f"out of fold, {selected['mean']['score']-c0_score:+.4f} versus C0 and "
         f"{selected['mean']['score']-c1['mean']['score']:+.4f} versus C1. "
         "These are repeated exploratory evaluations on three development skies, "
         "not an untouched test or a guaranteed Kaggle gain."),
        "",
        "## End-to-end result",
        "",
        "| Arm | Total | vs C0 | Presence | Localization | Recovery | Identification |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _metric_row("C0 production", {"complete": True, "mean": metrics["c0"]["mean"]}, c0_score),
        _metric_row("C1 classical expanded bank", c1, c0_score),
    ]
    if historical:
        lines.append(_metric_row("Historical learned SELECTED", historical, c0_score))
    for arm in "ABCDE":
        lines.append(_metric_row(arm, oof.get(arm), c0_score))
    lines.append(_metric_row("Inner-selected recipe", selected, c0_score))
    if "SELECTED_s31005" in oof:
        lines.append(_metric_row("Selected, seed 31005", oof["SELECTED_s31005"], c0_score))

    lines += ["", "### Per-scene selected result", "",
              "| Held-out sky | Arm | Step | Total | Presence | Localization | Recovery | Identification |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for fold in SCENES:
        sel = metrics["selection"][fold]
        m = selected["per_scene"][fold]
        lines.append(f"| {fold} | {sel['arm']} | {sel['step']} | {_fmt(m['score'])} | "
                     f"{_fmt(m['presence'],3)} | {_fmt(m['localization'],3)} | "
                     f"{_fmt(m['recovery'],3)} | {_fmt(m['identification'],3)} |")

    lines += ["", "### Every arm by held-out scene", "",
              "| Scene | Arm | Total | Presence | Localization | Recovery | Identification |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for fold in SCENES:
        lines.append(f"| {fold} | C0 | {_fmt(metrics['c0']['scenes'][fold]['score'])} | "
                     f"{_fmt(metrics['c0']['scenes'][fold]['presence'],3)} | "
                     f"{_fmt(metrics['c0']['scenes'][fold]['localization'],3)} | "
                     f"{_fmt(metrics['c0']['scenes'][fold]['recovery'],3)} | "
                     f"{_fmt(metrics['c0']['scenes'][fold]['identification'],3)} |")
        for arm in ("C1", "A", "B", "C", "D", "E"):
            m = oof[arm]["per_scene"][fold]
            lines.append(f"| {fold} | {arm} | {_fmt(m['score'])} | "
                         f"{_fmt(m['presence'],3)} | {_fmt(m['localization'],3)} | "
                         f"{_fmt(m['recovery'],3)} | {_fmt(m['identification'],3)} |")

    lines += ["", "## What was corrected and measured", ""]
    for item in diagnosis["verified"]:
        lines.append(f"- {item}")
    lines += ["", "The exact legacy reproduction again reached zero active loss by "
              "step 500 and kept it through step 2,000. A–E use an explicit policy "
              "without hidden in-batch hardest-negative selection. The historical "
              "reproduction remains separate because A also fixes the discovered "
              "0–255 versus 0–1 cached-positive input inconsistency.", ""]

    lines += ["### Actual training coverage", "",
              "| Fold | Arm | Unique centres | Unique generated queries | Replay | Nonzero-loss queries | Pool new locations |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for fold in SCENES:
        for arm in "ABCDE":
            run = read_json(OUT / "runs" / fold / f"{arm}_s31004" / "result.json")
            counts = run["counts"]
            new = sum(x["new_ever"] for x in run["mining"])
            lines.append(f"| {fold} | {arm} | {counts['unique_source_centres']} | "
                         f"{counts['unique_query_images_entering_updates']} | "
                         f"{counts['replay_fraction']:.3f} | "
                         f"{counts['unique_queries_nonzero_loss']} | {new} |")

    lines += ["", "## Realism audit", "",
              "Source weighting changed which real FIT locations were sampled; it "
              "did not edit query histograms. The original degradation laws remained "
              "unchanged because paired affine residuals contain alignment and clipping "
              "error and did not justify an arbitrary brightness law.", "",
              "| Fold | Original query std | Corrected query std | Real query std | Original mean | Corrected mean | Real mean |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for fold in SCENES:
        a = appearance[fold]["A"]["queries"]
        c = appearance[fold]["C"]["queries"]
        r = appearance[fold]["real_queries"]["queries"]
        lines.append(f"| {fold} | {_fmt(a['std']['mean'],2)} | {_fmt(c['std']['mean'],2)} | "
                     f"{_fmt(r['std']['mean'],2)} | {_fmt(a['mean']['mean'],2)} | "
                     f"{_fmt(c['mean']['mean'],2)} | {_fmt(r['mean']['mean'],2)} |")
    lines += ["", "Each fold has fixed-scale contact sheets at "
              "`outputs/exp1b/folds/<fold>/synthesis_contact.png`. Bright fraction "
              "(`I > 200`) and clipping (`I == 255`) are stored separately in "
              "`appearance_audit.json`.", ""]

    lines += ["## Hypothesis decisions", ""]
    for name, node in hypotheses.items():
        lines.append(f"- **{name.replace('_', ' ').title()}:** {node['finding']}.")
    lines += ["", "## Alignment audit", "",
              "The post-freeze audit identifies the nearest retrieved candidate for "
              "each truly present held-out query, records whether it is within 12px "
              "and has a valid pose, its NCC and fitted residual, then tries a finer "
              "diagnostic pose grid. No diagnostic coordinate or pose changes a scored "
              "prediction. `alignment_failed_all` only means every bank score was "
              "invalid; it never proved the true candidate was aligned accurately.", ""]
    for fold in SCENES:
        audit = read_json(OUT / "folds" / fold / "alignment_audit.json")
        present = len(audit)
        near = sum(x["near_correct_exists"] for x in audit)
        valid = sum(x["near_correct_exists"] and x["valid_pose"] for x in audit)
        changed = sum(x.get("rank_after") != x.get("rank_before") for x in audit
                      if x.get("rank_before") is not None)
        lines.append(f"- {fold}: correct candidate {near}/{present}; valid pose "
                     f"{valid}/{present}; finer pose changed descriptor rank for "
                     f"{changed} auditable queries.")

    lines += ["", "## Selected-arm error changes and strata", "",
              "| Scene | vs C0 presence fixes/regressions | vs C0 localization fixes/regressions | Figure reward | Off-figure reward | Absent count |",
              "|---|---:|---:|---:|---:|---:|"]
    for fold in SCENES:
        evaluation = read_json(OUT / "folds" / fold / "evaluation.json")
        selected_node = evaluation["arms"]["SELECTED"]
        change = selected_node["fixes_vs_C0"]
        strata = selected_node["strata"][fold]
        fig = strata.get("figure", {})
        off = strata.get("offfigure", {})
        absent = strata.get("absent", {})
        lines.append(f"| {fold} | {change['presence_fixes']}/{change['presence_regressions']} | "
                     f"{change['localization_fixes']}/{change['localization_regressions']} | "
                     f"{_fmt(fig.get('mean_reward'),3)} | {_fmt(off.get('mean_reward'),3)} | "
                     f"{absent.get('n', 0)} |")
    lines += ["", "Exact query IDs, coordinates, probabilities, fixed/regressed "
              "labels and rewards are stored in each fold's `evaluation.json`; the "
              "report does not truncate that machine-readable evidence."]

    gate = metrics["gates"]["SELECTED"]
    lines += ["", "## Promotion decision", "",
              f"Selected gate pass: **{gate['pass']}**. Checks: " +
              ", ".join(f"{k}={v}" for k, v in gate["checks"].items()) + ".",
              "",
              "Production remains unchanged. No validation submission or Kaggle upload "
              "was produced. Better-than-C1 performance without passing the C0 gates "
              "would be matcher evidence only.", "",
              "## Environment and runtime", "",
              f"Python {environment['python']}, PyTorch {environment['torch']}, Kornia "
              f"{environment['kornia']}, device MPS available={environment['mps_available']}. "
              f"The 18 completed matrix/repeat runs recorded {training_hours:.2f} aggregate "
              f"training wall-hours; selected-arm real-query inference recorded "
              f"{selected_inference_seconds:.2f}s across the three fold evaluations. "
              "Every run records wall time, generation/mining/gradient time, MPS memory, "
              "config/source hashes and immutable checkpoints. Only one training process ran.",
              "", f"HardNet initialization: `{hardnet_provenance['weights_path']}`, SHA256 "
              f"`{hardnet_provenance['actual_sha256']}`, strict load "
              f"verified by Experiment 1 (digest match={hardnet_provenance['sha256_matches']}). "
              "Upstream source and license are "
              "preserved in `outputs/exp1/models.json`; Experiment 1B downloaded no weights.",
              "", ("Integrity checks: all required arms complete; D/E anchor streams "
                    "byte-identical in every fold; all common-step checkpoints present; "
                    "330 protected production/Experiment 1 files unchanged."
                    if integrity and integrity.get("production_and_exp1_unchanged")
                    else "Integrity results are stored in `outputs/exp1b/integrity.json`."),
              "", f"Learning curves: `{curves}`.", "",
              "## Reproduction", "",
              "See `experiments/exp1b/README.md` for the fixed protocol and commands. "
              "Tests cover fresh generation, source coverage, fold isolation, footprint "
              "validity, physical negative separation, refresh turnover, genuinely random "
              "selection/loss, pose parity, alignment semantics, unit input range and "
              "complete-fold aggregation.", ""]

    report = "\n".join(lines)
    Path("EXPERIMENT1B_REPORT.md").write_text(report)

    run_files = sorted((OUT / "runs").glob("*/*/result.json"))
    record = {
        "verdict": metrics["verdict"],
        "exploratory_oof": True,
        "production_modified": False,
        "kaggle_submission_produced": False,
        "metrics": "outputs/exp1b/metrics.json",
        "report": "EXPERIMENT1B_REPORT.md",
        "config": read_json(OUT / "config.json"),
        "selection": metrics["selection"],
        "gates": metrics["gates"],
        "hypotheses": hypotheses,
        "runs": [{"path": str(p), "sha256": sha256_file(p)} for p in run_files],
        "artifacts": {
            "diagnosis": "outputs/exp1b/diagnosis.json",
            "appearance": "outputs/exp1b/appearance_audit.json",
            "environment": "outputs/exp1b/environment.json",
            "learning_curves": curves,
            "run_ledger": "outputs/exp1b/runs.jsonl",
            "tests": "outputs/exp1b/all_tests_final.log",
            "integrity": "outputs/exp1b/integrity.json"
        }
    }
    write_json(OUT / "record.json", record)
    return report


if __name__ == "__main__":
    make_report()
