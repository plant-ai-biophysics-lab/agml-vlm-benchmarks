import os
import sys
import pandas as pd

from tasks.classification import load_agml_dataset, agml_to_df

def main():
    datasets_file = "datasets.txt"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv = os.path.join(output_dir, "baselines_results.csv")
    output_txt = os.path.join(output_dir, "baselines_summary.txt")
    
    # Parse datasets.txt
    tasks_map = {}
    with open(datasets_file, 'r') as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line:
                continue
            
            # Remove comment hashes to process all lines, including commented datasets (e.g. crops/weeds)
            if clean_line.startswith('#'):
                clean_line = clean_line.lstrip('#').strip()
            
            if ',' not in clean_line or "dataset_name" in clean_line:
                continue
                
            parts = [p.strip() for p in clean_line.split(',')]
            if len(parts) >= 3:
                dataset_name, plant_type, task = parts[0], parts[1], parts[2]
                
                # Explicitly skip plant_village
                if dataset_name == "plant_village_classification":
                    continue
                    
                if task not in tasks_map:
                    tasks_map[task] = []
                tasks_map[task].append(dataset_name)

    task_overall_stats = {}
    all_stats = []

    with open(output_txt, 'w') as f_out:
        header = f"\n--- Baselines per Task and Dataset ---\n"
        header += f"{'Task':<15} | {'Dataset':<45} | {'Classes':<7} | {'Samples':<7} | {'Uniform':<16} | {'Majority':<17}\n"
        header += "-" * 120 + "\n"
        print(header, end="")
        f_out.write(header)

        for task_category, datasets in tasks_map.items():
            task_overall_stats[task_category] = []
                
            for dataset_name in datasets:
                try:
                    # directly load from agml (creates the test combinations directly)
                    dataset_path = load_agml_dataset(dataset_name)
                    val_dir = os.path.join(dataset_path, "val")
                    
                    if not os.path.exists(val_dir):
                        msg = f"{task_category:<15} | {dataset_name:<45} | Error: val split not found\n"
                        print(msg, end="")
                        f_out.write(msg)
                        continue
                    
                    df = agml_to_df(val_dir)
                    total_samples = len(df)
                    if total_samples == 0:
                        continue
                    
                    class_counts = df['label'].value_counts()
                    num_classes = len(class_counts)
                    
                    uniform_baseline = 1.0 / num_classes if num_classes > 0 else 0
                    majority_baseline = class_counts.max() / total_samples
                    
                    msg = f"{task_category:<15} | {dataset_name:<45} | {num_classes:<7} | {total_samples:<7} | {uniform_baseline:.2%} ({uniform_baseline:.4f})  | {majority_baseline:.2%} ({majority_baseline:.4f})\n"
                    print(msg, end="")
                    f_out.write(msg)
                    
                    stat_entry = {
                        "task": task_category,
                        "dataset": dataset_name,
                        "uniform_baseline": uniform_baseline,
                        "majority_baseline": majority_baseline,
                        "num_classes": num_classes,
                        "total_samples": total_samples
                    }
                    task_overall_stats[task_category].append(stat_entry)
                    all_stats.append(stat_entry)
                    
                except Exception as e:
                    msg = f"{task_category:<15} | {dataset_name:<45} | Error: {str(e)}\n"
                    print(msg, end="")
                    f_out.write(msg)

        agg_header = "\n--- Overall Aggregated Baselines per Task ---\n"
        agg_header += f"{'Task':<15} | {'Total Datasets':<14} | {'Avg Uniform Baseline':<20} | {'Avg Majority Baseline':<22}\n"
        agg_header += "-" * 80 + "\n"
        print(agg_header, end="")
        f_out.write(agg_header)
        
        for task, stats in task_overall_stats.items():
            if not stats: continue
            avg_uniform = sum(s["uniform_baseline"] for s in stats) / len(stats)
            avg_majority = sum(s["majority_baseline"] for s in stats) / len(stats)
            msg = f"{task:<15} | {len(stats):<14} | {avg_uniform:.2%} ({avg_uniform:.4f})       | {avg_majority:.2%} ({avg_majority:.4f})\n"
            print(msg, end="")
            f_out.write(msg)

    # Save to CSV
    if all_stats:
        df_out = pd.DataFrame(all_stats)
        df_out.to_csv(output_csv, index=False)
        print(f"\nResults successfully saved to:")
        print(f"  - CSV: {output_csv}")
        print(f"  - TXT: {output_txt}")

if __name__ == '__main__':
    main()
