"""
Aggregate + present prompt-sensitivity ablation results produced by run_ablation.py.

Reads every outputs/ablation/**/metrics.json (and judge_metrics.json for OEQ),
attaches the task group from datasets.txt, and emits:

  outputs/ablation/summary_runs.csv     one row per (exp, condition, model, dataset)
  outputs/ablation/summary_A.csv        Exp A aggregated per (model, task group)
  outputs/ablation/summary_B.csv        Exp B aggregated per (model, task group, NOTA rate)
  outputs/ablation/summary_C.csv        Exp C aggregated per (model, task group, arm, name)
  outputs/ablation/table_A.tex          LaTeX: mean +/- std across 5 seeds, per task
  outputs/ablation/table_B.tex          LaTeX: accuracy per NOTA rate, per task
  outputs/ablation/fig_C.{png,pdf}      Bar chart: common vs scientific, per task
  outputs/ablation/fig_C_data.csv       Underlying numbers for the chart (replot-friendly)

Everything reads from saved intermediate files, so it can be rerun any time
(e.g. after more runs finish) without touching the models.

Usage:
  python scripts/plot_ablation.py
  python scripts/plot_ablation.py --ablation-dir outputs/ablation
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

TASK_GROUP_ORDER = ["disease", "pest_damage", "weeds"]
TASK_GROUP_LABEL = {"disease": "Disease", "pest_damage": "Pest/Damage", "weeds": "Weeds"}

# Two-category, colorblind-safe (Okabe-Ito): blue = common, orange = scientific.
COLOR_COMMON = "#0072B2"
COLOR_SCIENTIFIC = "#E69F00"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_task_groups(datasets_file: Path) -> dict:
    tg = {"disease": "disease", "pest/damage": "pest_damage", "crops/weeds": "weeds"}
    out = {}
    for line in datasets_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            out[parts[0]] = tg.get(parts[2], "other")
    return out


def collect_runs(ablation_dir: Path, task_groups: dict) -> pd.DataFrame:
    rows = []
    for metrics_path in ablation_dir.rglob("metrics.json"):
        with open(metrics_path) as f:
            m = json.load(f)
        run_dir = metrics_path.parent
        # Judged accuracy for OEQ runs, if present.
        judge_acc = np.nan
        jpath = run_dir / "judge_metrics.json"
        if jpath.exists():
            with open(jpath) as f:
                jm = json.load(f)
            judge_acc = jm.get("judge_match_accuracy", np.nan)
        m["judge_accuracy"] = judge_acc
        m["task_group"] = task_groups.get(m.get("dataset"), "other")
        rows.append(m)
    if not rows:
        raise SystemExit(f"No metrics.json found under {ablation_dir}. Run run_ablation.py first.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment A — order randomization: mean +/- std across seeds
# ---------------------------------------------------------------------------

def summarize_A(df: pd.DataFrame) -> pd.DataFrame:
    a = df[df["experiment"] == "A"].copy()
    if a.empty:
        return pd.DataFrame()
    # Per (model, task_group, seed): average accuracy across datasets in the task.
    per_seed = (a.groupby(["model", "task_group", "seed"])["fuzzy_accuracy"]
                  .mean().reset_index())
    # Then mean +/- std across the seeds -> isolates option-order variance.
    agg = (per_seed.groupby(["model", "task_group"])["fuzzy_accuracy"]
           .agg(mean="mean", std="std", n_seeds="count").reset_index())
    return agg


def latex_table_A(agg: pd.DataFrame) -> str:
    if agg.empty:
        return "% Experiment A: no data\n"
    models = sorted(agg["model"].unique())
    lines = [
        "% Experiment A: MCQA answer-order randomization (mean +/- std across 5 seeds)",
        "\\begin{table}[t]", "\\centering",
        "\\caption{MCQA accuracy under answer-order randomization (options, including "
        "``None of the above'', shuffled to a random position). Mean $\\pm$ std.\\ across "
        "five random orderings, averaged over datasets within each task group.}",
        "\\label{tab:ablation_order}",
        "\\begin{tabular}{l" + "c" * len(models) + "}",
        "\\toprule",
        "Task group & " + " & ".join(models) + " \\\\",
        "\\midrule",
    ]
    for tg in TASK_GROUP_ORDER:
        cells = []
        for model in models:
            r = agg[(agg["task_group"] == tg) & (agg["model"] == model)]
            if r.empty:
                cells.append("--")
            else:
                mean = r["mean"].iloc[0] * 100
                std = r["std"].iloc[0] * 100
                std = 0.0 if np.isnan(std) else std
                cells.append(f"{mean:.1f} $\\pm$ {std:.1f}")
        lines.append(f"{TASK_GROUP_LABEL[tg]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment B — NOTA-rate sensitivity
# ---------------------------------------------------------------------------

def summarize_B(df: pd.DataFrame) -> pd.DataFrame:
    b = df[df["experiment"] == "B"].copy()
    if b.empty:
        return pd.DataFrame()
    # Per (model, task_group, nota_rate): mean +/- std ACROSS datasets in the task.
    agg = (b.groupby(["model", "task_group", "nota_rate"])["fuzzy_accuracy"]
           .agg(mean="mean", std="std", n_datasets="count").reset_index())
    return agg


def latex_table_B(agg: pd.DataFrame) -> str:
    if agg.empty:
        return "% Experiment B: no data\n"
    models = sorted(agg["model"].unique())
    rates = sorted(agg["nota_rate"].unique())
    lines = [
        "% Experiment B: 'None of the above' rate sensitivity",
        "\\begin{table}[t]", "\\centering",
        "\\caption{MCQA accuracy as the ``None of the above'' rate varies. "
        "Mean $\\pm$ std.\\ across datasets within each task group.}",
        "\\label{tab:ablation_nota}",
        "\\begin{tabular}{ll" + "c" * len(rates) + "}",
        "\\toprule",
        "Model & Task group & " + " & ".join(f"{r}\\%" for r in rates) + " \\\\",
        "\\midrule",
    ]
    for model in models:
        first = True
        for tg in TASK_GROUP_ORDER:
            cells = []
            for rate in rates:
                r = agg[(agg["task_group"] == tg) & (agg["model"] == model)
                        & (agg["nota_rate"] == rate)]
                if r.empty:
                    cells.append("--")
                else:
                    mean = r["mean"].iloc[0] * 100
                    std = r["std"].iloc[0] * 100
                    std = 0.0 if np.isnan(std) else std
                    cells.append(f"{mean:.1f} $\\pm$ {std:.1f}")
            model_cell = model if first else ""
            first = False
            lines.append(f"{model_cell} & {TASK_GROUP_LABEL[tg]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment C — common vs scientific names (bar chart)
# ---------------------------------------------------------------------------

def summarize_C(df: pd.DataFrame) -> pd.DataFrame:
    c = df[df["experiment"] == "C"].copy()
    if c.empty:
        return pd.DataFrame()
    # For OEQ prefer judged accuracy; for MCQA use fuzzy accuracy.
    c["score"] = np.where(
        (c["arm"] == "oeq") & c["judge_accuracy"].notna(),
        c["judge_accuracy"], c["fuzzy_accuracy"],
    )
    agg = (c.groupby(["model", "task_group", "arm", "name_kind"])["score"]
           .agg(mean="mean", std="std", n_datasets="count").reset_index())
    return agg


def plot_C(agg: pd.DataFrame, out_base: Path):
    if agg.empty:
        print("Experiment C: no data to plot.")
        return
    import matplotlib.pyplot as plt

    models = sorted(agg["model"].unique())
    arms = [a for a in ("mcqa", "oeq") if a in set(agg["arm"])]
    # C applies to disease + pest/damage only.
    task_groups = [tg for tg in ("disease", "pest_damage") if tg in set(agg["task_group"])]

    nrows, ncols = len(models), len(arms)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows),
                             squeeze=False)

    x = np.arange(len(task_groups))
    width = 0.38

    for ri, model in enumerate(models):
        for ci, arm in enumerate(arms):
            ax = axes[ri][ci]
            common_vals, common_err, sci_vals, sci_err = [], [], [], []
            for tg in task_groups:
                rc = agg[(agg.model == model) & (agg.arm == arm)
                         & (agg.task_group == tg) & (agg.name_kind == "common")]
                rs = agg[(agg.model == model) & (agg.arm == arm)
                         & (agg.task_group == tg) & (agg.name_kind == "scientific")]
                common_vals.append(rc["mean"].iloc[0] * 100 if not rc.empty else 0)
                sci_vals.append(rs["mean"].iloc[0] * 100 if not rs.empty else 0)
                common_err.append((rc["std"].iloc[0] * 100 if not rc.empty
                                   and not np.isnan(rc["std"].iloc[0]) else 0))
                sci_err.append((rs["std"].iloc[0] * 100 if not rs.empty
                                and not np.isnan(rs["std"].iloc[0]) else 0))

            b1 = ax.bar(x - width / 2, common_vals, width, yerr=common_err,
                        label="Common name", color=COLOR_COMMON, capsize=3)
            b2 = ax.bar(x + width / 2, sci_vals, width, yerr=sci_err,
                        label="Scientific name", color=COLOR_SCIENTIFIC, capsize=3)
            ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
            ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels([TASK_GROUP_LABEL[t] for t in task_groups])
            ax.set_ylim(0, 100)
            ax.set_ylabel("Accuracy (%)")
            metric = "judged" if arm == "oeq" else "fuzzy"
            ax.set_title(f"{model} — {arm.upper()} ({metric})", fontsize=10, fontweight="bold")
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax.set_axisbelow(True)
            if ri == 0 and ci == ncols - 1:
                ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Common vs scientific plant-type name (class labels unchanged)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved chart: {out_base.with_suffix('.png')} / .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-dir", default=str(REPO_ROOT / "outputs" / "ablation"))
    ap.add_argument("--datasets-file", default=str(REPO_ROOT / "datasets.txt"))
    args = ap.parse_args()

    ablation_dir = Path(args.ablation_dir)
    task_groups = load_task_groups(Path(args.datasets_file))
    df = collect_runs(ablation_dir, task_groups)

    df.to_csv(ablation_dir / "summary_runs.csv", index=False)
    print(f"Collected {len(df)} runs -> {ablation_dir / 'summary_runs.csv'}")

    # Experiment A
    agg_a = summarize_A(df)
    if not agg_a.empty:
        agg_a.to_csv(ablation_dir / "summary_A.csv", index=False)
        (ablation_dir / "table_A.tex").write_text(latex_table_A(agg_a))
        print(f"Experiment A -> summary_A.csv, table_A.tex")

    # Experiment B
    agg_b = summarize_B(df)
    if not agg_b.empty:
        agg_b.to_csv(ablation_dir / "summary_B.csv", index=False)
        (ablation_dir / "table_B.tex").write_text(latex_table_B(agg_b))
        print(f"Experiment B -> summary_B.csv, table_B.tex")

    # Experiment C
    agg_c = summarize_C(df)
    if not agg_c.empty:
        agg_c.to_csv(ablation_dir / "summary_C.csv", index=False)
        agg_c.to_csv(ablation_dir / "fig_C_data.csv", index=False)
        plot_C(agg_c, ablation_dir / "fig_C")
        print(f"Experiment C -> summary_C.csv, fig_C_data.csv, fig_C.png/pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()
