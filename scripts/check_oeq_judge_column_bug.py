"""
Check whether LLMJudge.evaluate_predictions() judged raw generated_text or the
already fuzzy-matched pred_label, for every OEQ predictions.csv under a root
directory.

evaluate_predictions() only uses generated_text if pred_label is "mostly empty"
(>50% NaN). If pred_label is populated for most rows (because upstream fuzzy
matching filled it in even for wrong matches), the judge silently re-confirms
the fuzzy match instead of doing real semantic judging on the raw text -- the
judge then adds no information (judge_match_accuracy == exact_match_accuracy).

This scans real production outputs (default: outputs/, excluding outputs/ablation)
for OEQ-mode predictions.csv (files WITHOUT MCQA-specific columns like
mcqa_correct_answer) and reports the pred_label empty ratio, so you can tell
which historical judge runs are trustworthy.

Usage:
  python scripts/check_oeq_judge_column_bug.py
  python scripts/check_oeq_judge_column_bug.py --root outputs/gpt-5
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--exclude", default="ablation",
                    help="Subdirectory name to skip (default: ablation).")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for preds in sorted(root.rglob("predictions.csv")):
        if args.exclude and args.exclude in preds.parts:
            continue
        try:
            df = pd.read_csv(preds)
        except Exception as e:
            print(f"SKIP (read error): {preds} -- {e}")
            continue

        is_mcqa = "mcqa_correct_answer" in df.columns
        if is_mcqa:
            continue  # MCQA is supposed to use pred_label; not affected.

        if "pred_label" not in df.columns:
            continue

        total = len(df)
        empty = df["pred_label"].isna().sum() + (df["pred_label"] == "").sum()
        empty_ratio = empty / total if total else 0.0
        would_use = "generated_text" if empty_ratio > 0.5 else "pred_label (BUG RISK)"

        has_judge = (preds.parent / "judge_metrics.json").exists()

        rows.append({
            "path": str(preds.parent),
            "n": total,
            "pred_label_empty_ratio": round(empty_ratio, 3),
            "judge_would_use": would_use,
            "already_judged": has_judge,
        })

    if not rows:
        print(f"No OEQ predictions.csv found under {root} (excluding '{args.exclude}').")
        return

    df_out = pd.DataFrame(rows)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print(df_out.to_string(index=False))

    n_risk = (df_out["judge_would_use"] == "pred_label (BUG RISK)").sum()
    n_judged_at_risk = ((df_out["judge_would_use"] == "pred_label (BUG RISK)")
                         & df_out["already_judged"]).sum()
    print(f"\n{n_risk} / {len(df_out)} OEQ runs would have judged pred_label "
          f"instead of generated_text.")
    print(f"{n_judged_at_risk} of those already have a judge_metrics.json on "
          f"disk -- those judged results are likely unreliable and should be "
          f"treated as re-confirmed exact-match, not real semantic judging.")


if __name__ == "__main__":
    main()
