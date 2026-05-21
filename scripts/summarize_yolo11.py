import pandas as pd
import argparse
from pathlib import Path
import numpy as np

def load_datasets(filepath):
    task_map = {}
    if not Path(filepath).exists():
        return task_map
    with open(filepath, 'r') as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line: continue
            if clean_line.startswith('#'):
                clean_line = clean_line.lstrip('#').strip()
            if ',' not in clean_line or "dataset_name" in clean_line: continue
            parts = [p.strip() for p in clean_line.split(',')]
            if len(parts) >= 3:
                dataset_name, _, task = parts[0], parts[1], parts[2]
                task_map[dataset_name] = task
    return task_map

def summarize_yolo(yolo_dir):
    yolo_path = Path(yolo_dir)
    if not yolo_path.exists():
        print(f"Error: Directory not found - {yolo_dir}")
        return
    
    # Lists for Task Metrics (accuracy, f1-score)
    metrics_dict = {'disease': {'acc': [], 'f1': []}, 'pest/damage': {'acc': [], 'f1': []}, 'crops/weeds': {'acc': [], 'f1': []}, 'overall': {'acc': [], 'f1': []}}
    dataset_to_task = load_datasets('/group/jmearlesgrp/scratch/eranario/vlm-investigation/datasets.txt')

    dataset_dirs = sorted([d for d in yolo_path.iterdir() if d.is_dir()])
    
    found_metrics = 0
    missing_metrics = 0
    
    for dataset_dir in dataset_dirs:
        # Skip plant_village if it exists
        if dataset_dir.name == 'plant_village_classification':
            continue
            
        # --- Gather Task Metrics from custom_metrics.csv ---
        metrics_csv = dataset_dir / "custom_metrics.csv"
        task = dataset_to_task.get(dataset_dir.name, "unknown")
        
        if metrics_csv.exists() and task in metrics_dict:
            try:
                df_m = pd.read_csv(metrics_csv)
                df_m.columns = [c.strip() for c in df_m.columns]
                # Scale by 100.0, handling YOLO's custom metric output format
                acc = float(df_m['accuracy'].iloc[0]) * 100.0
                f1 = float(df_m['f1_macro'].iloc[0])
                
                metrics_dict[task]['acc'].append(acc)
                metrics_dict[task]['f1'].append(f1)
                metrics_dict['overall']['acc'].append(acc)
                metrics_dict['overall']['f1'].append(f1)
                found_metrics += 1
            except Exception as e:
                print(f"Error reading {metrics_csv}: {e}")
                pass
        elif task in metrics_dict and not metrics_csv.exists():
            missing_metrics += 1

    # Output Task Metrics Summary
    task_summary_data = []
    print("\n" + "="*80)
    print(f"--- Task-Level Metrics Summary for YOLO11 ({yolo_path.name}) ---")
    print(f"{'Task':<15} | {'Acc (Mean ± SD)':<20} | {'F1-Macro (Mean ± SD)':<25}")
    print("-" * 80)
    
    for task in ['disease', 'pest/damage', 'crops/weeds', 'overall']:
        acc_list = metrics_dict[task]['acc']
        f1_list = metrics_dict[task]['f1']
        if len(acc_list) > 0:
            mean_acc = np.mean(acc_list)
            # Sample std div (ddof=1) if > 1 sample, else 0.0
            std_acc = np.std(acc_list, ddof=1) if len(acc_list) > 1 else 0.0
            mean_f1 = np.mean(f1_list)
            std_f1 = np.std(f1_list, ddof=1) if len(f1_list) > 1 else 0.0
            
            print(f"{task.capitalize():<15} | {mean_acc:.1f}% ± {std_acc:.1f}%       | {mean_f1:.4f} ± {std_f1:.4f}")
            
            task_summary_data.append({
                'model': 'yolo11',
                'task': task,
                'averaged_accuracy': mean_acc,
                'accuracy_standard_deviation': std_acc,
                'averaged_f1_score': mean_f1,
                'f1_standard_deviation': std_f1
            })
            
    if len(task_summary_data) > 0:
        out_metrics_csv = yolo_path / "task_metrics_summary.csv"
        pd.DataFrame(task_summary_data).to_csv(out_metrics_csv, index=False)
        print("="*80)
        print(f"✅ Saved task-level metrics report to: {out_metrics_csv}")
        
    print(f"\nStats: Processed {found_metrics} datasets. {missing_metrics} datasets are still missing 'custom_metrics.csv'.")
    if missing_metrics > 0:
        print("Hint: To generate missing metrics, make sure you ran `python3 scripts/calc_yolo_f1.py` successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Summarize YOLO11 Task Metrics')
    parser.add_argument('yolo_dir', help='Path to the yolo directory containing dataset folders')
    args = parser.parse_args()
    summarize_yolo(args.yolo_dir)
