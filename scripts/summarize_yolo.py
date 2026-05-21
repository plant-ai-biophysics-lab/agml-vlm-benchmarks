
import pandas as pd
import argparse
from pathlib import Path
import numpy as np

def load_datasets(filepath):
    task_map = {}
    if not Path(filepath).exists():
        return task_map
    with open(filepath, "r") as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith("#") or "dataset_name" in clean_line: continue
            parts = [p.strip() for p in clean_line.split(",")]
            if len(parts) >= 3:
                task_map[parts[0]] = parts[2]
    return task_map

def summarize_yolo(yolo_dir):
    yolo_path = Path(yolo_dir)
    if not yolo_path.exists():
        print(f"Error: Directory not found - {yolo_dir}")
        return
    
    metrics = {"disease": {"acc":[], "f1":[]}, "pest/damage": {"acc":[], "f1":[]}, "crops/weeds": {"acc":[], "f1":[]}, "overall": {"acc":[], "f1":[]}}
    dataset_to_task = load_datasets("/group/jmearlesgrp/scratch/eranario/vlm-investigation/datasets.txt")
    
    dataset_dirs = sorted([d for d in yolo_path.iterdir() if d.is_dir()])
    found = 0

    for d in dataset_dirs:
        if d.name == "plant_village_classification": continue
        task = dataset_to_task.get(d.name, "unknown")
        csv_m = d / "custom_metrics.csv"
        
        if csv_m.exists() and task in metrics:
            df = pd.read_csv(csv_m)
            df.columns = [c.strip() for c in df.columns]
            acc = float(df["accuracy"].iloc[0]) * 100.0
            f1 = float(df["f1_macro"].iloc[0])
            metrics[task]["acc"].append(acc)
            metrics[task]["f1"].append(f1)
            metrics["overall"]["acc"].append(acc)
            metrics["overall"]["f1"].append(f1)
            found += 1
            
    summary = []
    print("\n===")
    print(f"Task-Level Metrics Summary for YOLO11 ({yolo_path.name})")
    print("Task | Acc (Mean +/- SD) | F1-Macro (Mean +/- SD)")
    for task in ["disease", "pest/damage", "crops/weeds", "overall"]:
        acc_l, f1_l = metrics[task]["acc"], metrics[task]["f1"]
        if acc_l:
            m_acc, s_acc = np.mean(acc_l), np.std(acc_l, ddof=1) if len(acc_l)>1 else 0.0
            m_f1, s_f1 = np.mean(f1_l), np.std(f1_l, ddof=1) if len(f1_l)>1 else 0.0
            print(f"{task.capitalize():<15} | {m_acc:.1f}% ± {s_acc:.1f}%       | {m_f1:.4f} ± {s_f1:.4f}")
            summary.append({"model": "yolo11", "task": task, "averaged_accuracy": m_acc, "accuracy_standard_deviation": s_acc, "averaged_f1_score": m_f1, "f1_standard_deviation": s_f1})
    
    if summary:
        out = yolo_path / "task_metrics_summary.csv"
        pd.DataFrame(summary).to_csv(out, index=False)
        print(f"\nSaved to: {out}")
    print(f"Processed {found} datasets.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yolo_dir")
    args = parser.parse_args()
    summarize_yolo(args.yolo_dir)
