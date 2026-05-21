import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

MODELS = {
    "LLaVA-NeXT-8B": "llava_next",
    "Qwen-VL-7B": "qwen_vl",
    "Gemma_3-4B": "gemma_3",
    "Deepseek-VL-7B": "deepseek_vl",
    "Qwen-VL-72B*": "qwen_vl_72b",
    "GPT-5-Nano*": "gpt-5-nano",
    "GPT-5*": "gpt-5",
    "Gemini-3 Pro*": "gemini-3-pro-preview",
    "Claude Haiku 4.5*": "claude-haiku-4-5",
    "YOLO11 (SFT)": "yolo"
}

METRICS = ["accuracy", "f1_macro"]

def load_datasets(filepath):
    task_map = {}
    with open(filepath, 'r') as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line: continue
            if clean_line.startswith('#'):
                clean_line = clean_line.lstrip('#').strip()
            if ',' not in clean_line or "dataset_name" in clean_line: continue
            parts = [p.strip() for p in clean_line.split(',')]
            if len(parts) >= 3:
                dataset_name, plant_type, task = parts[0], parts[1], parts[2]
                if dataset_name == "plant_village_classification": continue
                task_map[dataset_name] = task
    return task_map

def load_baselines(filepath):
    df = pd.read_csv(filepath)
    
    # Calculate F1 Macro for majority class predictor:
    # F1_majority_class = 2 * (precision * recall) / (precision + recall)
    # precision = majority_baseline, recall = 1.0
    # F1_macro = F1_majority_class / num_classes
    df['f1_macro_baseline'] = (2 * df['majority_baseline'] / (df['majority_baseline'] + 1)) / df['num_classes']
    
    task_baselines_acc = df.groupby('task')['majority_baseline'].mean().to_dict()
    task_baselines_acc['overall'] = df['majority_baseline'].mean()

    task_baselines_f1 = df.groupby('task')['f1_macro_baseline'].mean().to_dict()
    task_baselines_f1['overall'] = df['f1_macro_baseline'].mean()

    return {"accuracy": task_baselines_acc, "f1_macro": task_baselines_f1}

def get_yolo_metric(yolo_dir, dataset_name, metric_name):
    # Check for custom generated metrics first
    custom_metrics_path = os.path.join(yolo_dir, dataset_name, "custom_metrics.csv")
    if os.path.exists(custom_metrics_path):
        df = pd.read_csv(custom_metrics_path)
        if metric_name in df.columns:
            return df[metric_name].iloc[0]

    # YOLO results are typically saved in yolo11_train, yolo11_train2, etc.
    # Note: Ultralytics classification only logs 'metrics/accuracy_top1' and 'metrics/accuracy_top5'
    # F1 macro is natively NOT tracked for image classification in results.csv
    for i in ["", "2", "3", "4", "5"]:
        train_dir = "yolo11_train" + i
        metrics_path = os.path.join(yolo_dir, dataset_name, train_dir, "results.csv")
        
        if os.path.exists(metrics_path):
            try:
                df = pd.read_csv(metrics_path)
                df.columns = df.columns.str.strip()  # Clean up ultralytics spaces
                
                if metric_name == "accuracy" and "metrics/accuracy_top1" in df.columns:
                    return df["metrics/accuracy_top1"].iloc[-1]
                elif metric_name in df.columns:
                    return df[metric_name].iloc[-1]
            except Exception as e:
                print(f"Error reading YOLO metrics for {dataset_name}: {e}")
                
    return None

def get_metric(model_key, model_folder, dataset_name, metric_name, zero_shot_dir, yolo_dir):
    if model_key == "YOLO11 (SFT)":
        return get_yolo_metric(yolo_dir, dataset_name, metric_name)
    
    metrics_path = os.path.join(zero_shot_dir, model_folder, dataset_name, "metrics.csv")
    if not os.path.exists(metrics_path): return None
    try:
        df = pd.read_csv(metrics_path)
        if metric_name in df.columns:
            return df[metric_name].iloc[0]
    except:
        pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate accuracy and f1_macro graphs with baselines.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the generated plots")
    parser.add_argument("--datasets-file", default="/group/jmearlesgrp/scratch/eranario/vlm-investigation/datasets.txt", help="Path to datasets.txt")
    parser.add_argument("--baselines-file", default="/group/jmearlesgrp/scratch/eranario/vlm-investigation/outputs/baselines_results.csv", help="Path to baselines_results.csv")
    parser.add_argument("--zero-shot-dir", default="/group/jmearlesgrp/intermediate_data/eranario/vlm-investigation/zero_shot_classification/mcqa_1", help="Zero-shot data dir")
    parser.add_argument("--yolo-dir", default="/group/jmearlesgrp/intermediate_data/eranario/vlm-investigation/fine_tune_classification/yolo", help="YOLO data dir")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_to_task = load_datasets(args.datasets_file)
    
    tasks = ["overall", "disease", "pest/damage", "crops/weeds"]
    task_baselines = load_baselines(args.baselines_file) if os.path.exists(args.baselines_file) else {"accuracy": {}, "f1_macro": {}}

    results = {m: {t: {model: [] for model in MODELS} for t in tasks} for m in METRICS}
    
    for dataset_name, task in dataset_to_task.items():
        if task not in tasks: continue
        for model_key, model_folder in MODELS.items():
            for metric in METRICS:
                val = get_metric(model_key, model_folder, dataset_name, metric, args.zero_shot_dir, args.yolo_dir)
                if val is not None:
                    results[metric][task][model_key].append(val)
                    results[metric]["overall"][model_key].append(val)
                    
    # Generate a plot for each metric
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        task_names = []
        for t in tasks:
            n_datasets = len([d for d, t_c in dataset_to_task.items() if t_c == t])
            if t == "overall":
                n_datasets = len(dataset_to_task)
            task_names.append(f"{t.capitalize()} (n={n_datasets})")
            
        x = np.arange(len(tasks))
        width = 0.08
        multiplier = 0
        
        for model_key in MODELS:
            means = []
            for t in tasks:
                vals = results[metric][t][model_key]
                means.append(np.mean(vals) if vals else 0)
                
            offset = width * multiplier
            ax.bar(x + offset, means, width, label=model_key)
            multiplier += 1
            
        # Add baselines
        baselines_for_metric = task_baselines.get(metric, {})
        for i, t in enumerate(tasks):
            baseline = baselines_for_metric.get(t, None)
            if baseline is not None:
                ax.hlines(y=baseline, xmin=x[i]-width, xmax=x[i]+width*len(MODELS), 
                         color='black', linestyle='--', alpha=0.7)
                         
                if i == 0:
                     ax.hlines(y=baseline, xmin=x[i]-width, xmax=x[i]+width*len(MODELS), 
                         color='black', linestyle='--', alpha=0.7, label="Naive Baseline")
        
        metric_label = metric.replace('_', ' ').title()
        ax.set_ylabel(f'{metric_label} (Avg. Across Datasets)')
        ax.set_xlabel('Agricultural Task')
        ax.set_title(f'{metric_label} vs Agricultural Task')
        ax.set_xticks(x + width * (len(MODELS) - 1) / 2)
        ax.set_xticklabels(task_names)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, f"{metric}_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved {plot_path}")

if __name__ == '__main__':
    main()
