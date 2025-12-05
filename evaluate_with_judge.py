#!/usr/bin/env python3
"""
Evaluate classification predictions using LLM judge.

This script takes predictions from a model evaluation and uses an LLM to judge
whether the predicted labels semantically match the ground truth, even when
they differ textually.

Usage:
    python evaluate_with_judge.py <predictions_csv> [options]

Example:
    python evaluate_with_judge.py outputs/qwen_vl/dataset_name/predictions.csv --threshold 1
    
    # Process multiple datasets
    python evaluate_with_judge.py outputs/qwen_vl/*/predictions.csv --threshold 2
"""

import argparse
import sys
import glob
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.llm_judge import LLMJudge


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate classification predictions using LLM judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Evaluate with default settings (threshold=1, gpt-4o-mini)
            python evaluate_with_judge.py outputs/model_name/dataset/predictions.csv
            
            # Use stricter threshold (only very confident matches)
            python evaluate_with_judge.py outputs/model_name/dataset/predictions.csv --threshold 2
            
            # Process multiple datasets with glob pattern
            python evaluate_with_judge.py "outputs/qwen_vl/*/predictions.csv" --threshold 1
            
            # Use different LLM model
            python evaluate_with_judge.py predictions.csv --model gpt-4o --threshold 1
            
            Confidence Levels:
            0 = Very unsure / labels clearly refer to different things
            1 = Could possibly be the same / uncertain match
            2 = Very confident / labels clearly refer to the same thing
            
            Setting threshold=1 means confidence scores of 1 or 2 count as matches.
            Setting threshold=2 means only confidence score of 2 counts as a match.
        """
    )
    
    parser.add_argument(
        "predictions_csv",
        help="Path to predictions.csv file (supports glob patterns)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model name (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic", "hf"],
        help="API provider: openai, anthropic, or hf (Hugging Face local model) (default: openai)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Confidence threshold for match (default: 1)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of parallel API calls for API models (default: 10)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for local models: auto, cuda, cpu (default: auto)"
    )
    parser.add_argument(
        "--reasoning-level",
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort for gpt-oss models: low (fast), medium (balanced), high (detailed) (default: medium)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: same as input file)"
    )
    parser.add_argument(
        "--context",
        help="Additional context about the classification task"
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Re-evaluate even if results already exist (default: skip completed)"
    )
    
    args = parser.parse_args()
    
    # Handle glob patterns
    prediction_files = glob.glob(args.predictions_csv)
    
    if not prediction_files:
        print(f"Error: No files found matching: {args.predictions_csv}")
        sys.exit(1)
    
    print(f"Found {len(prediction_files)} prediction file(s) to evaluate")
    print()
    
    # Initialize judge once for all files
    print("Initializing LLM Judge...")
    print(f"  Model: {args.model}")
    print(f"  Provider: {args.provider}")
    print(f"  Confidence Threshold: {args.threshold}")
    print(f"  Max Workers: {args.max_workers}")
    print(f"  Skip Completed: {not args.no_skip_completed}")
    print()
    
    judge = LLMJudge(
        model_name=args.model,
        api_provider=args.provider,
        confidence_threshold=args.threshold,
        max_workers=args.max_workers,
        context_info=args.context,
        device=args.device,
        reasoning_level=args.reasoning_level
    )
    
    # Process each file
    all_metrics = []
    skipped_count = 0
    for i, pred_file in enumerate(prediction_files, 1):
        print(f"\n{'='*80}")
        print(f"Processing file {i}/{len(prediction_files)}: {pred_file}")
        print(f"{'='*80}\n")
        
        try:
            metrics = judge.evaluate_predictions(
                predictions_csv=pred_file,
                output_dir=args.output_dir,
                skip_completed=not args.no_skip_completed
            )
            
            # check if it was skipped (metrics loaded from existing file)
            if not args.no_skip_completed:
                output_dir = args.output_dir if args.output_dir else Path(pred_file).parent
                judge_metrics_json = Path(output_dir) / "judge_metrics.json"
                if judge_metrics_json.exists():
                    # if metrics match what was already there, it was skipped
                    with open(judge_metrics_json, 'r') as f:
                        existing_metrics = json.load(f)
                    if existing_metrics == metrics:
                        skipped_count += 1
            all_metrics.append({
                'file': pred_file,
                'metrics': metrics
            })
        except Exception as e:
            print(f"Error processing {pred_file}: {e}")
            continue
    
    # Summary
    if len(all_metrics) > 0:
        print(f"\n\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}\n")
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} already completed dataset(s)")
            print(f"Evaluated {len(all_metrics) - skipped_count} dataset(s)")
            print()
        
        if len(all_metrics) > 1:
            for result in all_metrics:
                m = result['metrics']
                print(f"{Path(result['file']).parent.name}:")
                print(f"  Exact: {m['exact_match_accuracy']:.4f} -> Judge: {m['judge_match_accuracy']:.4f} (gain: {m['accuracy_gain']:+.4f})")
            
            # Overall averages
            avg_exact = sum(r['metrics']['exact_match_accuracy'] for r in all_metrics) / len(all_metrics)
            avg_judge = sum(r['metrics']['judge_match_accuracy'] for r in all_metrics) / len(all_metrics)
            avg_gain = sum(r['metrics']['accuracy_gain'] for r in all_metrics) / len(all_metrics)
            
            print(f"\nAverage across {len(all_metrics)} datasets:")
            print(f"  Exact Match: {avg_exact:.4f}")
            print(f"  Judge Match: {avg_judge:.4f}")
            print(f"  Average Gain: {avg_gain:+.4f}")


if __name__ == "__main__":
    main()
