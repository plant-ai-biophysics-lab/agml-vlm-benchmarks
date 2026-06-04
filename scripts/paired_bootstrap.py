"""
Paired bootstrap resampling for between-model accuracy comparisons.

Accepts either a folder (containing dataset subfolders with predictions.csv)
or a single predictions.csv file for each model.

Usage — folder mode (all datasets):
  python scripts/paired_bootstrap.py \
    --model-a outputs/gpt-5/seed_42/ \
    --model-b outputs/gemini-3-pro-preview/seed_42/ \
    --label-a "GPT-5" \
    --label-b "Gemini-3 Pro"

Usage — single file mode:
  python scripts/paired_bootstrap.py \
    --model-a outputs/gpt-5/seed_42/bean_disease_ethiopia/predictions.csv \
    --model-b outputs/gemini-3-pro-preview/seed_42/bean_disease_ethiopia/predictions.csv \
    --label-a "GPT-5" \
    --label-b "Gemini-3 Pro"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def paired_bootstrap(correct_a, correct_b, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    diffs = [
        correct_b[idx].mean() - correct_a[idx].mean()
        for idx in (rng.integers(0, n, size=n) for _ in range(n_boot))
    ]
    diffs = np.array(diffs)
    observed_diff = correct_b.mean() - correct_a.mean()
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    p_value = np.mean(diffs <= 0) if observed_diff > 0 else np.mean(diffs >= 0)
    p_value = min(p_value * 2, 1.0)
    return observed_diff, ci_low, ci_high, p_value


def load_and_align(path_a, path_b, dataset_name=""):
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    merged = df_a[["image_path", "label", "pred_label"]].merge(
        df_b[["image_path", "label", "pred_label"]],
        on="image_path",
        suffixes=("_a", "_b"),
    )

    if len(merged) == 0:
        raise ValueError(f"No matching image_path values for dataset '{dataset_name}'.")

    if len(merged) < max(len(df_a), len(df_b)):
        print(f"  WARNING [{dataset_name}]: {len(df_a)} samples in A, "
              f"{len(df_b)} in B, {len(merged)} matched.")

    correct_a = (merged["label_a"] == merged["pred_label_a"]).to_numpy()
    correct_b = (merged["label_b"] == merged["pred_label_b"]).to_numpy()
    return correct_a, correct_b


def find_dataset_pairs(dir_a, dir_b):
    """Return list of (dataset_name, path_a, path_b) for matching subfolders."""
    datasets_a = {p.parent.name: p for p in Path(dir_a).rglob("predictions.csv")}
    datasets_b = {p.parent.name: p for p in Path(dir_b).rglob("predictions.csv")}

    common = sorted(datasets_a.keys() & datasets_b.keys())
    only_a = sorted(datasets_a.keys() - datasets_b.keys())
    only_b = sorted(datasets_b.keys() - datasets_a.keys())

    if only_a:
        print(f"Datasets only in A (skipped): {only_a}")
    if only_b:
        print(f"Datasets only in B (skipped): {only_b}")

    return [(ds, datasets_a[ds], datasets_b[ds]) for ds in common]


def print_result(label_a, label_b, acc_a, acc_b, obs_diff, ci_low, ci_high, p_val, n, dataset=""):
    tag = f" [{dataset}]" if dataset else ""
    sig_str = "p < 0.05 *" if p_val < 0.05 else "n.s."
    print(f"{tag or 'POOLED':35s}  "
          f"{label_a}: {acc_a*100:5.2f}%  "
          f"{label_b}: {acc_b*100:5.2f}%  "
          f"diff: {obs_diff*100:+5.2f}pp  "
          f"95% CI [{ci_low*100:+.2f}, {ci_high*100:+.2f}]  "
          f"p={p_val:.4f}  {sig_str}  "
          f"n={n:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True,
                        help="Folder of dataset subfolders or single predictions.csv for model A.")
    parser.add_argument("--model-b", required=True,
                        help="Folder of dataset subfolders or single predictions.csv for model B.")
    parser.add_argument("--label-a", default="Model A")
    parser.add_argument("--label-b", default="Model B")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv-out", default=None,
                        help="Optional path to save per-dataset results as CSV.")
    args = parser.parse_args()

    path_a = Path(args.model_a)
    path_b = Path(args.model_b)

    # Single-file mode
    if path_a.is_file() and path_b.is_file():
        correct_a, correct_b = load_and_align(path_a, path_b)
        obs_diff, ci_low, ci_high, p_val = paired_bootstrap(
            correct_a, correct_b, args.n_boot, args.seed
        )
        print("\n" + "=" * 70)
        print("PAIRED BOOTSTRAP RESAMPLING RESULTS")
        print("=" * 70)
        print_result(args.label_a, args.label_b,
                     correct_a.mean(), correct_b.mean(),
                     obs_diff, ci_low, ci_high, p_val, len(correct_a))
        return

    # Folder mode
    pairs = find_dataset_pairs(path_a, path_b)
    if not pairs:
        print("ERROR: No matching dataset subfolders found.")
        return

    print(f"\nFound {len(pairs)} matching datasets.")
    print("=" * 70)
    print("PAIRED BOOTSTRAP RESAMPLING — PER DATASET")
    print("=" * 70)

    rows = []
    all_correct_a, all_correct_b = [], []

    for dataset, csv_a, csv_b in pairs:
        try:
            correct_a, correct_b = load_and_align(csv_a, csv_b, dataset)
        except ValueError as e:
            print(f"  SKIP {dataset}: {e}")
            continue

        obs_diff, ci_low, ci_high, p_val = paired_bootstrap(
            correct_a, correct_b, args.n_boot, args.seed
        )
        print_result(args.label_a, args.label_b,
                     correct_a.mean(), correct_b.mean(),
                     obs_diff, ci_low, ci_high, p_val,
                     len(correct_a), dataset)

        rows.append({
            "dataset": dataset,
            f"acc_{args.label_a}": round(correct_a.mean() * 100, 4),
            f"acc_{args.label_b}": round(correct_b.mean() * 100, 4),
            "obs_diff_pp": round(obs_diff * 100, 4),
            "ci_low_pp": round(ci_low * 100, 4),
            "ci_high_pp": round(ci_high * 100, 4),
            "p_value": round(p_val, 4),
            "significant": p_val < 0.05,
            "n_paired": len(correct_a),
        })

        all_correct_a.append(correct_a)
        all_correct_b.append(correct_b)

    # Pooled across all datasets
    if all_correct_a:
        pooled_a = np.concatenate(all_correct_a)
        pooled_b = np.concatenate(all_correct_b)
        obs_diff, ci_low, ci_high, p_val = paired_bootstrap(
            pooled_a, pooled_b, args.n_boot, args.seed
        )
        print("\n" + "=" * 70)
        print("POOLED ACROSS ALL DATASETS")
        print("=" * 70)
        print_result(args.label_a, args.label_b,
                     pooled_a.mean(), pooled_b.mean(),
                     obs_diff, ci_low, ci_high, p_val, len(pooled_a))

        n_sig = sum(r["significant"] for r in rows)
        print(f"\nDatasets with significant difference (p < 0.05): {n_sig} / {len(rows)}")

        # LaTeX summary
        sig = "significantly" if p_val < 0.05 else "not significantly"
        direction = "outperforms" if obs_diff > 0 else "underperforms"
        print(f"\nLaTeX summary (pooled):")
        print(
            f"  {args.label_b} ({pooled_b.mean()*100:.1f}\\%) {direction} "
            f"{args.label_a} ({pooled_a.mean()*100:.1f}\\%) "
            f"by {abs(obs_diff)*100:.1f}pp (95\\% CI [{ci_low*100:+.1f}, "
            f"{ci_high*100:+.1f}]pp, $p={p_val:.3f}$, {len(rows)} datasets), "
            f"a {sig} difference."
        )

    # Save CSV
    if args.csv_out and rows:
        pd.DataFrame(rows).to_csv(args.csv_out, index=False)
        print(f"\nPer-dataset results saved to: {args.csv_out}")


if __name__ == "__main__":
    main()
