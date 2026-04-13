#!/usr/bin/env python3
"""
Update judge metrics to calculate accuracy on full dataset instead of only judged samples.

This script recalculates the metrics in judge_metrics.json files by accounting for
samples that were skipped (no predictions). This makes judge accuracy directly
comparable to the original zero-shot accuracy.

Usage:
    python experiments/update_judge_metrics.py <path_to_results_dir>
    python experiments/update_judge_metrics.py --all  # Update all judge results
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd


def update_single_judge_metrics(result_dir: str, dry_run: bool = False) -> bool:
    """
    Update judge_metrics.json for a single result directory.
    
    Args:
        result_dir: Directory containing predictions.csv and judge_metrics.json
        dry_run: If True, only print what would be updated without modifying files
        
    Returns:
        True if updated, False if skipped
    """
    result_path = Path(result_dir)
    
    # Check required files exist
    predictions_csv = result_path / "predictions.csv"
    judge_metrics_json = result_path / "judge_metrics.json"
    predictions_with_judge_csv = result_path / "predictions_with_judge.csv"
    
    if not predictions_csv.exists():
        print(f"  ⚠️  Skipping {result_dir} - predictions.csv not found")
        return False
    
    if not judge_metrics_json.exists():
        print(f"  ⚠️  Skipping {result_dir} - judge_metrics.json not found")
        return False
    
    if not predictions_with_judge_csv.exists():
        print(f"  ⚠️  Skipping {result_dir} - predictions_with_judge.csv not found")
        return False
    
    # Load existing metrics
    with open(judge_metrics_json, 'r') as f:
        metrics = json.load(f)
    
    # Check if already updated (has 'total_samples' key)
    if 'total_samples' in metrics:
        print(f"  ✓  Already updated: {result_dir}")
        return False
    
    # Load dataframes
    df = pd.read_csv(predictions_csv)
    df_judged = pd.read_csv(predictions_with_judge_csv)
    
    # Recalculate metrics on full dataset
    total_dataset = len(df)
    total_judged = len(df_judged)
    total_skipped = total_dataset - total_judged
    
    # Get counts from judged samples
    exact_match_count = metrics['exact_match_count']
    judge_match_count = metrics['judge_match_count']
    
    # Recalculate accuracies on full dataset
    exact_match_accuracy = exact_match_count / total_dataset
    judge_match_accuracy = judge_match_count / total_dataset
    accuracy_gain = (judge_match_count - exact_match_count) / total_dataset
    
    # Calculate accuracy on judged samples only (for reference)
    judge_match_accuracy_on_judged = judge_match_count / total_judged if total_judged > 0 else 0.0
    
    # Update metrics
    old_total = metrics.get('total_evaluated', total_judged)
    old_exact_acc = metrics['exact_match_accuracy']
    old_judge_acc = metrics['judge_match_accuracy']
    
    metrics['total_samples'] = total_dataset
    metrics['total_judged'] = total_judged
    metrics['total_skipped'] = total_skipped
    metrics['exact_match_accuracy'] = exact_match_accuracy
    metrics['judge_match_accuracy'] = judge_match_accuracy
    metrics['accuracy_gain'] = accuracy_gain
    metrics['judge_match_accuracy_on_judged'] = judge_match_accuracy_on_judged
    
    # Remove old 'total_evaluated' key if present
    if 'total_evaluated' in metrics:
        del metrics['total_evaluated']
    
    # Print changes
    print(f"  📊 {result_dir}")
    print(f"     Total samples: {old_total} → {total_dataset} (skipped: {total_skipped})")
    print(f"     Exact match accuracy: {old_exact_acc:.4f} → {exact_match_accuracy:.4f}")
    print(f"     Judge match accuracy: {old_judge_acc:.4f} → {judge_match_accuracy:.4f}")
    if total_skipped > 0:
        print(f"     Judge accuracy (on judged only): {judge_match_accuracy_on_judged:.4f}")
    
    # Save updated metrics
    if not dry_run:
        with open(judge_metrics_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"     ✅ Updated!")
    else:
        print(f"     [DRY RUN - not saved]")
    
    return True


def find_all_judge_results(base_dir: str) -> list:
    """
    Find all directories containing judge results.
    
    This works for:
    - Single result directory (e.g., bean_disease_uganda/)
    - Model directory (e.g., gpt-5/ containing multiple datasets)
    - Top-level directory (e.g., oeq_1/ containing multiple models)
    
    Args:
        base_dir: Base directory to search from
        
    Returns:
        List of paths to result directories
    """
    results = []
    base_path = Path(base_dir)
    
    # Find all judge_metrics.json files recursively
    for judge_metrics in base_path.rglob("judge_metrics.json"):
        result_dir = judge_metrics.parent
        results.append(str(result_dir))
    
    return sorted(results)


def main():
    parser = argparse.ArgumentParser(
        description="Update judge metrics to calculate accuracy on full dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update a single result directory
  python experiments/update_judge_metrics.py zero_shot_classification/oeq_1/gpt-5/bean_disease_uganda
  
  # Update all datasets for a single model (auto-detects multiple datasets)
  python experiments/update_judge_metrics.py /path/to/zero_shot_classification/oeq_1/gpt-5
  
  # Update all models and datasets
  python experiments/update_judge_metrics.py /path/to/zero_shot_classification/oeq_1
  
  # Dry run to see what would be updated
  python experiments/update_judge_metrics.py --dry-run /path/to/oeq_1/gpt-5
        """
    )
    
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to single result dir, model dir (e.g., gpt-5/), or top-level dir (default: current directory)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Update all judge results found recursively (automatic if path contains multiple datasets)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without modifying files"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - no files will be modified\n")
    
    if not os.path.isdir(args.path):
        print(f"❌ Error: {args.path} is not a directory")
        return
    
    # Check if this is a single result directory or needs recursive search
    has_judge_metrics = (Path(args.path) / "judge_metrics.json").exists()
    
    if args.all or not has_judge_metrics:
        # Find and update all judge results recursively
        print(f"🔍 Searching for judge results in: {args.path}\n")
        result_dirs = find_all_judge_results(args.path)
        
        if not result_dirs:
            print("❌ No judge results found!")
            return
        
        print(f"Found {len(result_dirs)} result directories\n")
        
        updated_count = 0
        for result_dir in result_dirs:
            if update_single_judge_metrics(result_dir, dry_run=args.dry_run):
                updated_count += 1
            print()  # blank line between results
        
        print(f"\n{'[DRY RUN] Would update' if args.dry_run else 'Updated'} {updated_count}/{len(result_dirs)} directories")
    
    else:
        # Update single directory
        update_single_judge_metrics(args.path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
