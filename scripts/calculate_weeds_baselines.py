import os
import sys

from tasks.classification import load_agml_dataset, agml_to_df

def main():
    datasets = [
        "crop_weeds_greece",
        "plant_seedlings_aarhus",
        "rangeland_weeds_australia",
        "soybean_weed_uav_brazil"
    ]
    
    print(f"\n--- Baselines for Crops/Weeds Datasets ---")
    print(f"{'Dataset':<35} | {'Classes':<7} | {'Samples':<7} | {'Uniform Baseline':<20} | {'Majority Baseline':<22}")
    print("-" * 100)

    task_stats = []

    for dataset_name in datasets:
        try:
            dataset_path = load_agml_dataset(dataset_name)
            val_dir = os.path.join(dataset_path, "val")
            if not os.path.exists(val_dir):
                print(f"{dataset_name:<35} | Error: val split not found")
                continue
            
            df = agml_to_df(val_dir)
            total_samples = len(df)
            if total_samples == 0:
                print(f"{dataset_name:<35} | Error: 0 samples found")
                continue
            
            class_counts = df['label'].value_counts()
            num_classes = len(class_counts)
            
            uniform_baseline = 1.0 / num_classes if num_classes > 0 else 0
            majority_baseline = class_counts.max() / total_samples
            
            print(f"{dataset_name:<35} | {num_classes:<7} | {total_samples:<7} | {uniform_baseline:.2%} ({uniform_baseline:.4f})       | {majority_baseline:.2%} ({majority_baseline:.4f})")
            
            task_stats.append({
                "dataset": dataset_name,
                "uniform": uniform_baseline,
                "majority": majority_baseline,
                "num_classes": num_classes,
                "total_samples": total_samples
            })
            
        except Exception as e:
            print(f"{dataset_name:<35} | Error: {str(e)}")

    print("\n--- Aggregated Baselines for Crops/Weeds ---")
    print(f"{'Total Datasets':<14} | {'Avg Uniform Baseline':<20} | {'Avg Majority Baseline':<22}")
    print("-" * 65)
    
    if task_stats:
        avg_uniform = sum(s["uniform"] for s in task_stats) / len(task_stats)
        avg_majority = sum(s["majority"] for s in task_stats) / len(task_stats)
        print(f"{len(task_stats):<14} | {avg_uniform:.2%} ({avg_uniform:.4f})       | {avg_majority:.2%} ({avg_majority:.4f})")

if __name__ == '__main__':
    main()
