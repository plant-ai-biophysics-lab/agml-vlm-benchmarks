import os
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import f1_score, accuracy_score

DATASETS_FILE = "/group/jmearlesgrp/scratch/eranario/vlm-investigation/datasets.txt"
YOLO_DIR = "/group/jmearlesgrp/intermediate_data/eranario/vlm-investigation/fine_tune_classification/yolo"

def load_datasets(filepath):
    datasets = []
    with open(filepath, 'r') as f:
        for line in f:
            clean = line.strip()
            if not clean: continue
            if clean.startswith('#'):
                clean = clean.lstrip('#').strip()
            parts = [p.strip() for p in clean.split(',')]
            if len(parts) >= 3 and 'dataset_name' not in clean:
                if parts[0] != "plant_village_classification":
                    datasets.append(parts[0])
    return datasets

def get_latest_weights(dataset_name):
    dataset_dir = os.path.join(YOLO_DIR, dataset_name)
    if not os.path.exists(dataset_dir): return None
    best_weights = None
    max_num = -1
    for d in os.listdir(dataset_dir):
        if d.startswith('yolo11_train'):
            num = d.replace('yolo11_train', '')
            val = int(num) if num.isdigit() else 0
            w = os.path.join(dataset_dir, d, 'weights', 'best.pt')
            if os.path.exists(w) and val >= max_num:
                max_num = val
                best_weights = w
    return best_weights

def main():
    datasets = load_datasets(DATASETS_FILE)
    
    for ds in datasets:
        weights_path = get_latest_weights(ds)
        if not weights_path:
            print(f"Skip {ds}: No weights found")
            continue
            
        out_csv = os.path.join(YOLO_DIR, ds, "custom_metrics.csv")
        if os.path.exists(out_csv):
            print(f"Skip {ds}: Metrics already exist")
            continue
            
        val_dir = os.path.expanduser(f"~/.agml/datasets/{ds}_split/val")
        if not os.path.exists(val_dir):
            print(f"Skip {ds}: Validation directory missing {val_dir}")
            continue
            
        print(f"Processing {ds}...")
        try:
            model = YOLO(weights_path)
            # Find index mapping
            name_to_idx = {name: idx for idx, name in model.names.items()}
            
            img_paths = []
            true_labels = []
            for name in os.listdir(val_dir):
                if name not in name_to_idx: continue
                class_dir = os.path.join(val_dir, name)
                for img in os.listdir(class_dir):
                    if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_paths.append(os.path.join(class_dir, img))
                        true_labels.append(name_to_idx[name])
                        
            if not img_paths:
                print(f"Skip {ds}: No images found")
                continue
                
            import torch
            
            # Fundamentals of ML memory management:
            # 1. Do not pass the entire dataset into the predict function if it's large.
            # 2. Iterate through batches explicitly.
            # 3. Pull predictions off the GPU using .item() or similar native python types immediately.
            
            pred_labels = []
            batch_size = 8
            
            # Predict in isolated, strictly sized chunks
            with torch.no_grad():
                for i in range(0, len(img_paths), batch_size):
                    batch_paths = img_paths[i:i + batch_size]
                    results = model.predict(batch_paths, verbose=False, batch=batch_size)
                    
                    for r in results:
                        # Extract the integer immediately without holding on to the Results object
                        pred_labels.append(r.probs.top1)
            
            # Aggressively delete the model & wipe the GPU cache
            del model
            torch.cuda.empty_cache()
            
            acc = accuracy_score(true_labels, pred_labels)
            f1_m = f1_score(true_labels, pred_labels, average='macro')
            
            pd.DataFrame([{"accuracy": acc, "f1_macro": f1_m}]).to_csv(out_csv, index=False)
            print(f"  -> Acc: {acc:.4f}, F1 Macro: {f1_m:.4f}")
        except Exception as e:
            print(f"  -> Error: {e}")

if __name__ == '__main__':
    main()
