import pandas as pd
import argparse
import os
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Manually evaluate LLM judge decisions.")
    parser.add_argument("csv_path", help="Path to predictions_with_judge.csv")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to evaluate")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Filter for cases where Exact Match failed (pred_label is null or doesn't match ground truth)
    # These are the cases where the LLM judge had to interpret the context.
    df_eval = df[df['label'] != df['pred_label']].dropna(subset=['judge_is_match']).copy()

    if len(df_eval) == 0:
        print("No mismatched/evaluated rows found for manual review.")
        return

    sample_size = min(args.num_samples, len(df_eval))
    df_sample = df_eval.sample(n=sample_size).reset_index()

    results = []
    agreements = 0

    print(f"\n--- Evaluating {sample_size} Judge Decisions ---")
    for idx, row in df_sample.iterrows():
        print(f"\n========================================================")
        print(f"[{idx+1}/{sample_size}] ID: {row.get('id', 'N/A')}")
        print(f"Ground Truth Label : {row['label']}")
        print(f"Model Generated    : {row.get('generated_text', '')}")
        print(f"--------------------------------------------------------")
        print(f"Judge Decision     : {'MATCH' if row['judge_is_match'] else 'NO MATCH'} (Confidence: {row.get('judge_confidence', 'N/A')})")
        print(f"Judge Reasoning    : {row.get('judge_reasoning', '')}")
        print(f"========================================================")
        
        while True:
            ans = input("Do you agree with the judge's decision? (y/n/q to quit): ").strip().lower()
            if ans in ['y', 'n', 'q']:
                break
            print("Please enter 'y', 'n', or 'q'.")
        
        if ans == 'q':
            print("Quitting early...")
            break
            
        agreed = (ans == 'y')
        if agreed:
            agreements += 1
            
        results.append({
            'id': row.get('id', ''),
            'label': row['label'],
            'model_generated_text': row.get('generated_text', ''),
            'judge_decision': row['judge_is_match'],
            'judge_reasoning': row.get('judge_reasoning', ''),
            'human_agreement': agreed
        })

    if not results:
        print("No evaluations completed.")
        return

    # Save report
    report_df = pd.DataFrame(results)
    report_csv_path = csv_path.parent / "judge_report_manual.csv"
    report_txt_path = csv_path.parent / "judge_report_manual.txt"
    report_df.to_csv(report_csv_path, index=False)
    
    # Save a text summary as well
    with open(report_txt_path, 'w') as f:
        f.write("=== Manual Judge Evaluation Report ===\n")
        f.write(f"Source file: {csv_path.name}\n")
        f.write(f"Total Evaluated: {len(results)}\n")
        f.write(f"Human Agreements: {agreements}\n")
        f.write(f"Human Disagreements: {len(results) - agreements}\n\n")
        
        f.write("--- Disagreements Details ---\n")
        disagreements = [r for r in results if not r['human_agreement']]
        if not disagreements:
            f.write("None! You agreed with the judge 100% of the time.\n")
        else:
            for r in disagreements:
                f.write(f"- ID: {r['id']} | Ground truth: {r['label']} | Judge said: {r['judge_decision']}\n")
    
    # Output summary to terminal
    print(f"\n--- Evaluation Complete ---")
    print(f"Total Evaluated: {len(results)}")
    print(f"Human Agreements: {agreements}/{len(results)} ({(agreements/len(results))*100:.1f}%)")
    print(f"Human Disagreements: {len(results) - agreements}/{len(results)}")
    print(f"\nDetailed reports saved to:")
    print(f"  - {report_csv_path}")
    print(f"  - {report_txt_path}")

if __name__ == "__main__":
    main()
