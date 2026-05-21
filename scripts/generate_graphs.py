import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Adjust base paths as needed based on the workspace structure
ZERO_SHOT_DIR = "/group/jmearlesgrp/intermediate_data/eranario/vlm-investigation/zero_shot_classification/mcqa_1"
YOLO_DIR = "/group/jmearlesgrp/intermediate_data/eranario/vlm-investigation/fine_tune_classification/yolo"
OUTPUT_DIR = "/group/jmearlesgrp/scratch/eranario/vlm-investigation/outputs/plots"

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

METRICS = [
    "accuracy", "precision_weighted", "recall_weighted", "f1_weighted", 
    "precision_macro", "recall_macro", "f1_macro"
]

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
    # Average majority baseline per task
    task_baselines = df.groupby('task')['majority_baseline'].mean().to_dict()
    task_baselines['overall'] = df['majority_baseline'].mean()
    return task_baselines

def get_yolo_metric(dataset_name, metric_name):
    # Just an approximation, for YOLO it's metrics/results.csv 
    # Usually metric for YOLO object classification top1 accuracy is in results.csv 
    # Might need a better mapping if metrics names are totally different in YOLO.
    metrics_path = os.path.join(YOLO_DIR, dataset_name, "yolo11_val", "results.csv")
    if not os.path.exists(metrics_path):
        metrics_path = os.path.join(YOLO_DIR, dataset_name, "yolo11_val2", "results.csv")
    if not os.path.exists(metrics_path):
        return None
    try:
        df = pd.read_csv(metrics_path)
        # Assuming YOLO has 'metrics/accuracy_top1' or similar; falling back to 0 if not found
        col_name = "metrics/accuracy_top1" if metric_name == "accuracy" else None
        if col_name and col_name in df.columns:
            return df[col_name].iloc[-1]
        elif metric_name in df.columns:
            return df[metric_name].iloc[-1]
    except:
        pass
    return None

def get_metric(model_key, model_folder, dataset_name, metric_name):
    if model_key == "YOLO11 (SFT)":
        return get_yolo_metric(dataset_name, metric_name)
    
    metrics_path = os.path.join(ZERO_SHOT_DIR, model_folder, dataset_name, "metrics.csv")
    if not os.path.exists(metrics_path): return None
    try:
        df = pd.read_csv(metrics_path)
        if metric_name in df.columns:
            return df[metric_name].iloc[0]
    except:
        pass
    return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    datasets_file = "/group/jmearlesgrp/scratch/eranario/vlm-investigation/datasets.txt"
    dataset_to_task = load_datasets(datasets_file)
    
    tasks = ["overall", "disease", "pest/damage", "crops/weeds"]
    
    baselines_file = "/group/jmearlesgrp/scratch/eranario/vlm-investigation/outputs/baselines_results.csv"
    task_baselines = load_baselines(baselines_file) if os.path.exists(baselines_file) else {}

    results = {m: {t: {model: [] for model in MODELS} for t in tasks} for m in METRICS}
    
    for dataset_name, task in dataset_to_task.items():
        if task not in tasks: continue
        for model_key, model_folder in MODELS.items():
            for metric in METRICS:
                val = get_metric(model_key, model_folder, dataset_name, metric)
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
            
        # Add baselines as horizontal lines or scatter points depending on visual preference
        # We can add a horizontal line for each task grouping
        for i, t in enumerate(tasks):
            baseline = task_baselines.get(t, None)
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
        plot_path = os.path.join(OUTPUT_DIR, f"{metric}_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved {plot_path}")

if __name__ == '__main__':
    main()
