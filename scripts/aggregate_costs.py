#!/usr/bin/env python3
"""
Utility script to aggregate token usage and costs across multiple experiments.

Usage:
    python scripts/aggregate_costs.py ./output
    python scripts/aggregate_costs.py ./output --by-model
    python scripts/aggregate_costs.py ./output --by-dataset
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

def format_cost(cost):
    """Format cost with appropriate precision."""
    if cost < 0.01:
        return f"${cost:.6f}"
    return f"${cost:.2f}"

def aggregate_costs(output_dir, group_by=None):
    """Aggregate costs from all token_usage.json files."""
    
    # Find all token usage files
    token_files = list(Path(output_dir).rglob("token_usage.json"))
    
    if not token_files:
        print(f"No token usage files found in {output_dir}")
        return
    
    print(f"Found {len(token_files)} token usage file(s)\n")
    
    # Aggregate data
    if group_by:
        grouped_data = defaultdict(lambda: {
            'cost': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'images': 0,
            'runs': 0
        })
    
    total_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_images = 0
    
    for token_file in token_files:
        try:
            with open(token_file) as f:
                data = json.load(f)
            
            cost = data['total_cost_usd']
            input_tokens = data['total_input_tokens']
            output_tokens = data['total_output_tokens']
            images = data['num_images']
            
            total_cost += cost
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_images += images
            
            # Group by model or dataset if requested
            if group_by:
                key = data.get(group_by, 'unknown')
                grouped_data[key]['cost'] += cost
                grouped_data[key]['input_tokens'] += input_tokens
                grouped_data[key]['output_tokens'] += output_tokens
                grouped_data[key]['images'] += images
                grouped_data[key]['runs'] += 1
        
        except Exception as e:
            print(f"Warning: Could not process {token_file}: {e}")
    
    # Print results
    print("=" * 80)
    print("TOTAL USAGE SUMMARY")
    print("=" * 80)
    print(f"Total images processed:   {total_images:,}")
    print(f"Total input tokens:       {total_input_tokens:,}")
    print(f"Total output tokens:      {total_output_tokens:,}")
    print(f"Total cost:               {format_cost(total_cost)}")
    print(f"Average cost per image:   {format_cost(total_cost / total_images if total_images > 0 else 0)}")
    print("=" * 80)
    
    # Print grouped results if requested
    if group_by and grouped_data:
        print(f"\nBREAKDOWN BY {group_by.upper()}")
        print("=" * 80)
        
        # Sort by cost (descending)
        sorted_items = sorted(grouped_data.items(), key=lambda x: x[1]['cost'], reverse=True)
        
        for key, stats in sorted_items:
            print(f"\n{key}:")
            print(f"  Runs:          {stats['runs']}")
            print(f"  Images:        {stats['images']:,}")
            print(f"  Input tokens:  {stats['input_tokens']:,}")
            print(f"  Output tokens: {stats['output_tokens']:,}")
            print(f"  Cost:          {format_cost(stats['cost'])}")
            print(f"  Cost per img:  {format_cost(stats['cost'] / stats['images'] if stats['images'] > 0 else 0)}")
        
        print("=" * 80)
    
    # Print individual file details if not grouping
    if not group_by:
        print("\nINDIVIDUAL RUNS")
        print("=" * 80)
        
        # Sort files by cost
        file_costs = []
        for token_file in token_files:
            try:
                with open(token_file) as f:
                    data = json.load(f)
                file_costs.append((token_file, data))
            except Exception:
                pass
        
        file_costs.sort(key=lambda x: x[1]['total_cost_usd'], reverse=True)
        
        for token_file, data in file_costs:
            rel_path = token_file.relative_to(output_dir)
            print(f"\n{rel_path}")
            print(f"  Model:   {data.get('model', 'unknown')}")
            print(f"  Dataset: {data.get('dataset', 'unknown')}")
            print(f"  Images:  {data['num_images']}")
            print(f"  Cost:    {format_cost(data['total_cost_usd'])}")
        
        print("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate token usage and costs from multiple experiments'
    )
    parser.add_argument(
        'output_dir',
        help='Directory containing experiment outputs with token_usage.json files'
    )
    parser.add_argument(
        '--by-model',
        action='store_true',
        help='Group results by model'
    )
    parser.add_argument(
        '--by-dataset',
        action='store_true',
        help='Group results by dataset'
    )
    
    args = parser.parse_args()
    
    # Determine grouping
    group_by = None
    if args.by_model:
        group_by = 'model'
    elif args.by_dataset:
        group_by = 'dataset'
    
    aggregate_costs(args.output_dir, group_by=group_by)

if __name__ == '__main__':
    main()
