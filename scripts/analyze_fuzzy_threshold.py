"""
Analyze fuzzy match false positives across all predictions.csv files.

Addresses reviewer comment:
  "A 0.6 threshold is too low for fine-grained agricultural taxonomy.
   Briefly discuss if this low threshold triggered false positives
   prior to the LLM-judging phase."

Usage:
  python scripts/analyze_fuzzy_threshold.py --outputs-dir outputs/
  python scripts/analyze_fuzzy_threshold.py --outputs-dir outputs/ --model qwen_vl
"""

import argparse
import os
import json
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Helpers — replicate the containment logic from utils/utils.py
# ---------------------------------------------------------------------------

def _is_containment_match(generated_text: str, label: str) -> bool:
    """Return True if this match was won via the containment shortcut (lines 55-56)."""
    gl = generated_text.lower().strip()
    ll = label.lower().strip()
    # strip common prefixes (same as fuzzy_match_label)
    prefixes = ['the answer is', 'category:', 'class:', 'label:', 'answer:', 'it is', 'this is']
    for p in prefixes:
        if gl.startswith(p):
            gl = gl[len(p):].strip()
    gl = gl.rstrip('.,;:!?')
    return ll in gl or gl in ll


def _seq_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_predictions_csvs(outputs_dir: str, model_filter: str = None) -> list[Path]:
    root = Path(outputs_dir)
    csvs = sorted(root.rglob("predictions.csv"))
    if model_filter:
        csvs = [p for p in csvs if model_filter in str(p)]
    return csvs


def load_all(csvs: list[Path]) -> pd.DataFrame:
    frames = []
    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
            # Parse the path to extract model / seed / dataset
            parts = csv_path.parts
            # outputs/<model>/<seed_tag>/<dataset>/predictions.csv  (4 levels up)
            dataset = parts[-2]
            seed_tag = parts[-3] if len(parts) >= 4 else "unknown"
            model = parts[-4] if len(parts) >= 5 else "unknown"
            df["_model"] = model
            df["_seed"] = seed_tag
            df["_dataset"] = dataset
            df["_csv_path"] = str(csv_path)
            frames.append(df)
        except Exception as e:
            print(f"WARNING: could not load {csv_path}: {e}")
    if not frames:
        raise RuntimeError(f"No predictions.csv files found under {outputs_dir}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def classify_match_type(row) -> str:
    """Classify how the prediction was matched."""
    gt = str(row.get("generated_text", ""))
    pl = str(row.get("pred_label", ""))
    ms = row.get("match_score", float("nan"))

    if pd.isna(ms) or pl in ("None", "nan", ""):
        return "no_match"

    ms = float(ms)

    if pl == row["label"]:
        # correct — but still classify how it matched
        if ms == 1.0:
            if _is_containment_match(gt, pl):
                return "correct_containment"
            return "correct_exact_or_high"
        return "correct_fuzzy"
    else:
        # wrong
        if ms == 1.0 and _is_containment_match(gt, pl):
            return "fp_containment"   # false positive via containment shortcut
        if ms >= 0.6:
            return "fp_fuzzy"         # false positive via threshold fuzzy match
        return "no_match"


def similar_label_pairs(labels: list[str], sim_threshold: float = 0.5) -> list[tuple]:
    """Find pairs of class labels that share high character overlap."""
    pairs = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            s = _seq_similarity(a, b)
            if s >= sim_threshold:
                pairs.append((a, b, round(s, 3)))
    return sorted(pairs, key=lambda x: -x[2])


def analyze_fps(df: pd.DataFrame) -> dict:
    """Compute false positive breakdown and return summary dict."""
    df = df.copy()
    df["match_type"] = df.apply(classify_match_type, axis=1)

    total = len(df)
    n_correct = (df["label"] == df["pred_label"]).sum()
    n_no_match = (df["match_type"] == "no_match").sum()
    n_fp_containment = (df["match_type"] == "fp_containment").sum()
    n_fp_fuzzy = (df["match_type"] == "fp_fuzzy").sum()
    n_fp_total = n_fp_containment + n_fp_fuzzy

    n_model_errors = total - n_correct - n_no_match  # wrong but matched
    # Threshold-induced FP rate: wrong predictions caused by the 0.6 threshold
    # being too permissive (excludes containment matches, which are model errors
    # where the matcher correctly identified the class the model named)
    threshold_fp_rate = round(n_fp_fuzzy / total, 4) if total else 0

    return {
        "total_samples": total,
        "correct": int(n_correct),
        "accuracy": round(n_correct / total, 4) if total else 0,
        "no_match": int(n_no_match),
        "model_errors": int(n_model_errors),
        "model_error_rate": round(n_model_errors / total, 4) if total else 0,
        "fp_containment": int(n_fp_containment),
        "fp_fuzzy": int(n_fp_fuzzy),
        "fp_total": int(n_fp_total),
        "threshold_fp_rate": threshold_fp_rate,
        "fp_containment_pct_of_errors": round(
            n_fp_containment / max(n_model_errors, 1), 4
        ),
        "df_with_types": df,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(result: dict, df_all: pd.DataFrame):
    print("\n" + "=" * 60)
    print("FUZZY MATCH FALSE POSITIVE ANALYSIS")
    print("=" * 60)
    print(f"Total samples analysed : {result['total_samples']:,}")
    print(f"Correct predictions    : {result['correct']:,}  ({result['accuracy']*100:.1f}%)")
    print(f"No match (below 0.6)   : {result['no_match']:,}")
    print()
    print(f"Model error rate       : {result['model_error_rate']*100:.2f}%  "
          f"({result['model_errors']:,} wrong predictions matched to some class)")
    print(f"  └─ matched via containment : {result['fp_containment']:,}  "
          f"({result['fp_containment_pct_of_errors']*100:.1f}% of model errors)")
    print(f"     [model named the wrong class; matcher correctly identified it]")
    print(f"  └─ matched via fuzzy 0.6–1.0: {result['fp_fuzzy']:,}")
    print(f"     [ambiguous output pushed over threshold → potential threshold FPs]")
    print()
    print(f"Threshold FP rate      : {result['threshold_fp_rate']*100:.2f}%  "
          f"({result['fp_fuzzy']:,} / {result['total_samples']:,})")
    print(f"  [predictions wrong due to the 0.6 threshold being too permissive]")

    df_typed = result["df_with_types"]

    # --- containment FPs by class pair ---
    fp_cont = df_typed[df_typed["match_type"] == "fp_containment"].copy()
    if not fp_cont.empty:
        print(f"\n--- Top containment false positive pairs ---")
        pair_counts = (
            fp_cont.groupby(["label", "pred_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        print(pair_counts.head(20).to_string(index=False))

    # --- similar label pairs per dataset ---
    print("\n--- Similar label pairs (char similarity ≥ 0.50) ---")
    for ds, grp in df_all.groupby("_dataset"):
        labels = sorted(grp["label"].dropna().unique().tolist())
        pairs = similar_label_pairs(labels, sim_threshold=0.50)
        if pairs:
            print(f"\n  Dataset: {ds}")
            for a, b, s in pairs[:10]:
                print(f"    {a!r:40s} ↔  {b!r:40s}  (sim={s:.2f})")

    # --- match score distribution for wrong predictions ---
    wrong = df_typed[df_typed["label"] != df_typed["pred_label"]]
    if "match_score" in wrong.columns and not wrong.empty:
        ms = wrong["match_score"].dropna()
        bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.01]
        labels_b = ["<0.5", "0.5–0.6", "0.6–0.7", "0.7–0.8", "0.8–0.9", "0.9–1.0", "1.0"]
        counts, _ = np.histogram(ms, bins=bins)
        print("\n--- Match score distribution for wrong predictions ---")
        for label_b, count in zip(labels_b, counts):
            bar = "█" * min(count, 50)
            print(f"  {label_b:10s} | {bar:50s} {count}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(result: dict, df_all: pd.DataFrame, out_path: str):
    df_typed = result["df_with_types"]
    wrong = df_typed[df_typed["label"] != df_typed["pred_label"]].copy()

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Error type breakdown ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    type_counts = df_typed["match_type"].value_counts()
    colors = {
        "correct_exact_or_high": "#2ecc71",
        "correct_containment": "#27ae60",
        "correct_fuzzy": "#82e0aa",
        "fp_containment": "#e74c3c",
        "fp_fuzzy": "#f39c12",
        "no_match": "#95a5a6",
    }
    wedge_colors = [colors.get(k, "#bdc3c7") for k in type_counts.index]
    ax1.pie(type_counts.values, labels=type_counts.index, colors=wedge_colors,
            autopct="%1.1f%%", startangle=140, textprops={"fontsize": 8})
    ax1.set_title("Prediction outcome breakdown", fontsize=10, fontweight="bold")

    # ── 2. Match score histogram — wrong predictions ──────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if "match_score" in wrong.columns and not wrong.empty:
        ms = wrong["match_score"].dropna()
        ax2.hist(ms, bins=20, color="#e74c3c", edgecolor="white", alpha=0.85)
        ax2.axvline(0.6, color="black", linestyle="--", linewidth=1.2, label="threshold=0.6")
        ax2.set_xlabel("Match score")
        ax2.set_ylabel("Count")
        ax2.set_title("Match score distribution\n(wrong predictions only)", fontsize=10, fontweight="bold")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No wrong predictions found", ha="center", va="center")
        ax2.set_title("Match score distribution (wrong preds)", fontsize=10)

    # ── 3. FP rate per dataset ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    fp_mask = df_typed["match_type"].isin(["fp_containment", "fp_fuzzy"])
    per_ds = df_typed.groupby("_dataset").apply(
        lambda g: pd.Series({
            "fp_rate": fp_mask[g.index].sum() / len(g),
            "n": len(g),
        })
    ).reset_index()
    per_ds = per_ds.sort_values("fp_rate", ascending=True)
    y_pos = range(len(per_ds))
    ax3.barh(y_pos, per_ds["fp_rate"] * 100, color="#e67e22", alpha=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(per_ds["_dataset"], fontsize=7)
    ax3.set_xlabel("False positive rate (%)")
    ax3.set_title("FP rate per dataset", fontsize=10, fontweight="bold")
    ax3.axvline(0, color="black", linewidth=0.5)

    # ── 4. Top containment FP class pairs ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    fp_cont = df_typed[df_typed["match_type"] == "fp_containment"]
    if not fp_cont.empty:
        pair_counts = (
            fp_cont.groupby(["label", "pred_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(12)
        )
        pair_labels = [f"{r['label']}\n→ {r['pred_label']}" for _, r in pair_counts.iterrows()]
        y_pos2 = range(len(pair_labels))
        ax4.barh(y_pos2, pair_counts["count"].values, color="#c0392b", alpha=0.85)
        ax4.set_yticks(y_pos2)
        ax4.set_yticklabels(pair_labels, fontsize=7)
        ax4.set_xlabel("Count")
        ax4.set_title("Top containment false positive pairs\n(true → predicted)", fontsize=10, fontweight="bold")
    else:
        ax4.text(0.5, 0.5, "No containment FPs found", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=10)
        ax4.set_title("Containment FP class pairs", fontsize=10)

    fig.suptitle("Fuzzy Matching False Positive Analysis\n(threshold = 0.6)", fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse fuzzy match false positives.")
    parser.add_argument("--outputs-dir", default="outputs/", help="Root outputs directory.")
    parser.add_argument("--model", default=None, help="Filter to a specific model name.")
    parser.add_argument("--plot", default="outputs/fuzzy_fp_analysis.png",
                        help="Where to save the output plot.")
    parser.add_argument("--csv", default="outputs/fuzzy_fp_summary.csv",
                        help="Where to save per-row annotated CSV.")
    args = parser.parse_args()

    csvs = find_predictions_csvs(args.outputs_dir, args.model)
    if not csvs:
        print(f"ERROR: no predictions.csv files found under '{args.outputs_dir}'")
        return

    print(f"Found {len(csvs)} predictions.csv file(s):")
    for p in csvs:
        print(f"  {p}")

    df_all = load_all(csvs)
    print(f"\nLoaded {len(df_all):,} rows total.")

    result = analyze_fps(df_all)
    print_report(result, df_all)

    # Save annotated CSV
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    df_typed = result["df_with_types"]
    df_typed.to_csv(args.csv, index=False)
    print(f"\nAnnotated CSV saved to: {args.csv}")

    # Save plot
    os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
    make_plots(result, df_all, args.plot)


if __name__ == "__main__":
    main()
